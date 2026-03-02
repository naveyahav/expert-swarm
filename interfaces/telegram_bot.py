"""
interfaces/telegram_bot.py
ExpertSwarm Telegram bot — production-grade async interface.

Privacy guarantees:
  - Raw Telegram user_id is NEVER stored. It is hashed with a
    daily-rotating HMAC salt before reaching PrivacyMiddleware.
  - No message text is logged. Audit records contain only timestamp,
    expert name, credit cost, and response length.

UX features:
  - Free-text messages auto-routed — no /ask command required.
  - Inline keyboard for expert selection (persistent per-session).
  - Typing indicator + edit-in-place "thinking…" placeholder.
  - Per-user concurrency lock — concurrent requests rejected gracefully.
  - 120-second inference timeout with user-friendly error message.
  - Input length validation (1 000 char limit).
  - Telegram menu populated via set_my_commands on startup.

Run:
    TELEGRAM_BOT_TOKEN=<your_token> python interfaces/telegram_bot.py
"""

import asyncio
import functools
import hashlib
import hmac
import logging
import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from credits.ledger import CreditLedger, DEMO_MINT_AMOUNT
from privacy.middleware import PrivacyMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_PROMPT_LEN   = 1_000   # chars; longer inputs rejected with a clear message
_INFERENCE_TIMEOUT = 120    # seconds; exceeded → friendly timeout notice
_MAX_RESPONSE_LEN  = 4_000  # Telegram hard limit is 4 096; we leave 96 chars headroom

# ---------------------------------------------------------------------------
# Privacy: daily-rotating salt for Telegram user-ID hashing
# ---------------------------------------------------------------------------

_SALT_SECRET: bytes = os.urandom(32)


def _hash_user_id(telegram_user_id: int) -> bytes:
    """
    Derive a daily-rotating opaque identifier.
    The raw integer is never stored; only this digest enters the session layer.
    """
    daily_salt = _SALT_SECRET + date.today().isoformat().encode()
    return hmac.new(daily_salt, str(telegram_user_id).encode(), hashlib.sha256).digest()


# ---------------------------------------------------------------------------
# Shared application state (in-process, ephemeral)
# ---------------------------------------------------------------------------

ledger     = CreditLedger()
middleware = PrivacyMiddleware(credit_cost_per_request=1)

_user_sessions: dict[bytes, str]            = {}   # hashed_id → session token
_user_experts:  dict[bytes, str | None]     = {}   # hashed_id → active expert (None = auto)
_user_locks:    dict[bytes, asyncio.Lock]   = {}   # hashed_id → per-user concurrency lock


def _get_or_create_session(user_id: int) -> str:
    hashed = _hash_user_id(user_id)
    if hashed not in _user_sessions:
        token = middleware.create_session(opaque_identifier=hashed)
        _user_sessions[hashed] = token
    return _user_sessions[hashed]


def _get_user_lock(user_id: int) -> asyncio.Lock:
    hashed = _hash_user_id(user_id)
    if hashed not in _user_locks:
        _user_locks[hashed] = asyncio.Lock()
    return _user_locks[hashed]


def _get_active_expert(user_id: int) -> str | None:
    return _user_experts.get(_hash_user_id(user_id))


def _set_active_expert(user_id: int, expert: str | None) -> None:
    _user_experts[_hash_user_id(user_id)] = expert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expert_keyboard(manifest: dict) -> InlineKeyboardMarkup:
    """Build an inline keyboard: one button per enabled expert + auto-route."""
    buttons = [[InlineKeyboardButton("🔍 Auto-route", callback_data="expert:__auto__")]]
    for name, entry in manifest.get("experts", {}).items():
        if entry.get("enabled"):
            buttons.append([InlineKeyboardButton(f"🤖 {name}", callback_data=f"expert:{name}")])
    return InlineKeyboardMarkup(buttons)


def _truncate(text: str, limit: int = _MAX_RESPONSE_LEN) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n_[… response truncated]_"


async def _safe_edit(message, text: str) -> None:
    """Edit a message, ignoring Telegram errors (e.g. text unchanged)."""
    try:
        await message.edit_text(text)
    except TelegramError:
        pass


async def _run_inference(token: str, prompt: str, expert: str | None) -> str:
    """
    Execute blocking middleware.handle() in the default ThreadPoolExecutor,
    wrapped in asyncio.wait_for so the event loop is never stalled.
    """
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(
        None,
        functools.partial(
            middleware.handle,
            session_token=token,
            prompt=prompt,
            expert=expert,
            credit_check=ledger.check_and_deduct,
        ),
    )
    return await asyncio.wait_for(task, timeout=_INFERENCE_TIMEOUT)


# ---------------------------------------------------------------------------
# Core inference entry point (shared by handle_message and cmd_ask)
# ---------------------------------------------------------------------------

async def _handle_prompt(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    prompt: str,
    expert: str | None,
) -> None:
    """
    Full inference flow:
      1. Input validation (non-empty, length limit)
      2. Credit pre-check (informational; hard gate is inside middleware)
      3. Acquire per-user concurrency lock — reject duplicate requests politely
      4. Send typing indicator + "thinking…" placeholder message
      5. Run inference in thread pool with timeout
      6. Edit placeholder with response (or structured error message)
    """
    user_id = update.effective_user.id

    # --- 1. Input validation -------------------------------------------------
    prompt = prompt.strip()
    if not prompt:
        await update.message.reply_text("Please send a non-empty message.")
        return
    if len(prompt) > _MAX_PROMPT_LEN:
        await update.message.reply_text(
            f"⚠️ Message too long ({len(prompt):,} chars). "
            f"Please keep it under {_MAX_PROMPT_LEN:,} characters."
        )
        return

    token = _get_or_create_session(user_id)

    # --- 2. Credit pre-check -------------------------------------------------
    if ledger.balance(token) < 1:
        await update.message.reply_text(
            "⚠️ You're out of credits.\n"
            "Use /start to receive a fresh batch of demo credits."
        )
        return

    # --- 3. Concurrency guard ------------------------------------------------
    lock = _get_user_lock(user_id)
    if lock.locked():
        await update.message.reply_text(
            "⏳ Still processing your previous request — please wait."
        )
        return

    async with lock:
        # --- 4. Immediate visual feedback ------------------------------------
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
        label = expert if expert else "auto"
        thinking_msg = await update.message.reply_text(f"[{label}] thinking…")

        # --- 5. Inference ----------------------------------------------------
        try:
            response = await _run_inference(token, prompt, expert)
        except asyncio.TimeoutError:
            await _safe_edit(
                thinking_msg,
                "⚠️ Request timed out (>120 s). Try a shorter prompt.",
            )
            return
        except Exception as exc:
            log.exception("Unexpected inference error: %s", exc)
            await _safe_edit(thinking_msg, "❌ Unexpected error. Please try again.")
            return

        # --- 6. Surface structured errors from middleware --------------------
        if response in ("Invalid or expired session.", "Insufficient credits."):
            await _safe_edit(thinking_msg, f"⚠️ {response}")
            return

        await _safe_edit(thinking_msg, _truncate(response))


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message + expert selector keyboard + demo credit mint."""
    import router
    manifest = router.load_manifest()
    token = _get_or_create_session(update.effective_user.id)
    balance = ledger.mint(token, DEMO_MINT_AMOUNT)
    await update.message.reply_text(
        f"🐝 *ExpertSwarm* — local-first modular AI\n\n"
        f"You have *{balance} credits*. Each query costs 1 credit.\n\n"
        f"Select an expert below, or just send a message to auto-route:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=_expert_keyboard(manifest),
    )


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show credit balance and currently active expert."""
    token  = _get_or_create_session(update.effective_user.id)
    balance = ledger.balance(token)
    expert  = _get_active_expert(update.effective_user.id) or "auto-route"
    await update.message.reply_text(
        f"⚡ *{balance} credits* remaining\n"
        f"Active expert: *{expert}*",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_experts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show expert selector keyboard."""
    import router
    manifest = router.load_manifest()
    await update.message.reply_text(
        "Select your expert:",
        reply_markup=_expert_keyboard(manifest),
    )


async def cmd_setexpert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set active expert: /setexpert <name>"""
    import router
    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: /setexpert <name>\n"
            "Example: /setexpert coder\n"
            "Use /experts to see all available experts."
        )
        return

    expert = args[0].lower()
    manifest = router.load_manifest()
    enabled = {n for n, e in manifest.get("experts", {}).items() if e.get("enabled")}

    if expert not in enabled:
        await update.message.reply_text(
            f"Unknown expert '{expert}'.\n"
            f"Available: {', '.join(sorted(enabled))}"
        )
        return

    _set_active_expert(update.effective_user.id, expert)
    await update.message.reply_text(
        f"✅ Active expert set to *{expert}*. Send your message.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset to auto-routing."""
    _set_active_expert(update.effective_user.id, None)
    await update.message.reply_text("🔍 Switched to auto-routing. Send your message.")


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Legacy command: /ask <expert> <message>. Kept for backwards compatibility."""
    args = context.args
    if not args or len(args) < 2:
        await update.message.reply_text(
            "Usage: /ask <expert> <message>\n"
            "Tip: you can skip /ask entirely — just send any message."
        )
        return
    expert = args[0].lower()
    prompt = " ".join(args[1:])
    await _handle_prompt(update, context, prompt=prompt, expert=expert)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*ExpertSwarm commands*\n\n"
        "  /start — welcome screen + expert selector\n"
        "  /experts — pick an expert (inline buttons)\n"
        "  /setexpert `<name>` — pin a specific expert\n"
        "  /clear — switch back to auto-routing\n"
        "  /balance — show credit balance\n"
        "  /ask `<expert>` `<msg>` — legacy query command\n"
        "  /help — this message\n\n"
        "Or *just send any message* — it will be routed automatically.\n"
        "Each query costs 1 credit.",
        parse_mode=ParseMode.MARKDOWN,
    )


# ---------------------------------------------------------------------------
# Inline keyboard callback
# ---------------------------------------------------------------------------

async def callback_expert_select(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle expert selection from the inline keyboard."""
    query = update.callback_query
    await query.answer()

    chosen = query.data.split(":", 1)[1]
    if chosen == "__auto__":
        _set_active_expert(query.from_user.id, None)
        await query.edit_message_text("🔍 Auto-routing enabled. Send your message.")
    else:
        _set_active_expert(query.from_user.id, chosen)
        await query.edit_message_text(
            f"✅ Expert set to *{chosen}*. Send your message.",
            parse_mode=ParseMode.MARKDOWN,
        )


# ---------------------------------------------------------------------------
# Free-text message handler
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route plain-text messages through the inference pipeline."""
    prompt = (update.message.text or "").strip()
    expert = _get_active_expert(update.effective_user.id)
    await _handle_prompt(update, context, prompt=prompt, expert=expert)


# ---------------------------------------------------------------------------
# Startup hook — register commands in Telegram's menu
# ---------------------------------------------------------------------------

async def _post_init(application: Application) -> None:
    await application.bot.set_my_commands([
        BotCommand("start",     "Welcome + expert selector"),
        BotCommand("experts",   "Pick an expert (inline buttons)"),
        BotCommand("setexpert", "Pin a specific expert"),
        BotCommand("clear",     "Switch to auto-routing"),
        BotCommand("balance",   "Show credit balance"),
        BotCommand("ask",       "Legacy: /ask <expert> <msg>"),
        BotCommand("help",      "Show help"),
    ])
    log.info("Bot commands registered.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log.error("TELEGRAM_BOT_TOKEN environment variable is not set.")
        sys.exit(1)

    app = (
        Application.builder()
        .token(token)
        .post_init(_post_init)
        .build()
    )

    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("balance",   cmd_balance))
    app.add_handler(CommandHandler("experts",   cmd_experts))
    app.add_handler(CommandHandler("setexpert", cmd_setexpert))
    app.add_handler(CommandHandler("clear",     cmd_clear))
    app.add_handler(CommandHandler("ask",       cmd_ask))
    app.add_handler(CommandHandler("help",      cmd_help))
    app.add_handler(CallbackQueryHandler(callback_expert_select, pattern=r"^expert:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("ExpertSwarm Telegram bot starting…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
