"""
interfaces/telegram_bot.py
ExpertSwarm Telegram bot — privacy-by-design interface.

Privacy guarantees:
  - The raw Telegram user_id is NEVER stored. It is hashed with a
    daily-rotating HMAC salt before being passed to PrivacyMiddleware,
    which derives a further ephemeral session token.
  - No message text is logged. The audit record contains only
    timestamp, expert name, credit cost, and response length.

Commands:
  /start              — Create session and mint demo credits.
  /balance            — Show current credit balance.
  /ask <expert> <msg> — Route prompt through credit gate and router.
  /experts            — List available experts.
  /help               — Show command reference.

Run:
    TELEGRAM_BOT_TOKEN=<your_token> python interfaces/telegram_bot.py
"""

import hashlib
import hmac
import logging
import os
import sys
from datetime import date
from pathlib import Path

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from credits.ledger import CreditLedger, DEMO_MINT_AMOUNT
from privacy.middleware import PrivacyMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Privacy: daily-rotating salt for user ID hashing
# ---------------------------------------------------------------------------

# A process-lifetime secret combined with the current date produces a
# salt that rotates every midnight, bounding the linkability window.
_SALT_SECRET: bytes = os.urandom(32)


def _hash_user_id(telegram_user_id: int) -> bytes:
    """
    Derive a daily-rotating opaque identifier from a Telegram user ID.
    The raw integer is never stored; only this digest reaches the middleware.
    """
    daily_salt = _SALT_SECRET + date.today().isoformat().encode()
    return hmac.new(daily_salt, str(telegram_user_id).encode(), hashlib.sha256).digest()


# ---------------------------------------------------------------------------
# Shared application state (in-process, ephemeral)
# ---------------------------------------------------------------------------

ledger = CreditLedger()
middleware = PrivacyMiddleware(credit_cost_per_request=1)

# Map hashed user bytes → session token. Cleared on process restart.
_user_sessions: dict[bytes, str] = {}


def _get_or_create_session(user_id: int) -> str:
    """Return an existing session token or create a fresh one."""
    hashed = _hash_user_id(user_id)
    if hashed not in _user_sessions:
        token = middleware.create_session(opaque_identifier=hashed)
        _user_sessions[hashed] = token
    return _user_sessions[hashed]


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Create session and mint demo credits."""
    token = _get_or_create_session(update.effective_user.id)
    balance = ledger.mint(token, DEMO_MINT_AMOUNT)
    await update.message.reply_text(
        f"Welcome to ExpertSwarm.\n"
        f"Your session is active. Demo credits minted: {DEMO_MINT_AMOUNT}\n"
        f"Balance: {balance} credits\n\n"
        f"Use /ask <expert> <message> to query an expert.\n"
        f"Use /experts to see available experts."
    )


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current credit balance."""
    token = _get_or_create_session(update.effective_user.id)
    balance = ledger.balance(token)
    await update.message.reply_text(f"Credit balance: {balance}")


async def cmd_experts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List available experts from manifest."""
    import router
    manifest = router.load_manifest()
    lines = []
    for name, entry in manifest.get("experts", {}).items():
        status = "enabled" if entry.get("enabled") else "disabled"
        lines.append(f"  {name} [{status}] — {entry.get('description', '')}")
    await update.message.reply_text("Available experts:\n" + "\n".join(lines))


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route a prompt: /ask <expert> <message>"""
    args = context.args
    if not args or len(args) < 2:
        await update.message.reply_text(
            "Usage: /ask <expert> <message>\nExample: /ask coder Write a binary search in Python"
        )
        return

    expert = args[0].lower()
    prompt = " ".join(args[1:])

    token = _get_or_create_session(update.effective_user.id)

    await update.message.reply_text(f"Routing to {expert} expert…")

    response = middleware.handle(
        session_token=token,
        prompt=prompt,
        expert=expert,
        credit_check=ledger.check_and_deduct,
    )

    await update.message.reply_text(response)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "ExpertSwarm commands:\n"
        "  /start           — Initialise session and get demo credits\n"
        "  /balance         — Show credit balance\n"
        "  /experts         — List available experts\n"
        "  /ask <e> <msg>   — Query expert e with message\n"
        "  /help            — This message"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log.error("TELEGRAM_BOT_TOKEN environment variable is not set.")
        sys.exit(1)

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("balance", cmd_balance))
    app.add_handler(CommandHandler("experts", cmd_experts))
    app.add_handler(CommandHandler("ask",     cmd_ask))
    app.add_handler(CommandHandler("help",    cmd_help))

    log.info("ExpertSwarm Telegram bot starting…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
