"""
interfaces/web_app.py
ExpertSwarm — production Streamlit web UI.

Run:
    streamlit run interfaces/web_app.py

Features:
  - Dark slate theme (configured in .streamlit/config.toml)
  - Streaming responses via router.route_stream() + st.write_stream()
  - Sidebar: expert selector, live credit balance, top-up, clear chat,
    backend type indicator
  - Input validation: 1 000-char limit with live counter
  - Inference timing displayed per response
  - CreditLedger + PrivacyMiddleware — same security gates as bot
  - No prompt or response text is logged or persisted beyond the session
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from credits.ledger import CreditLedger, DEMO_MINT_AMOUNT
from privacy.middleware import PrivacyMiddleware

_MAX_PROMPT_LEN = 1_000

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ExpertSwarm",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 1rem; }

.expert-badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
    background: #7C3AED22;
    color: #A78BFA;
    border: 1px solid #7C3AED55;
}

.credit-pill {
    text-align: center;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    background: #1E293B;
    border: 1px solid #334155;
    font-size: 1.1rem;
    font-weight: 700;
    color: #A78BFA;
    margin-bottom: 0.5rem;
}

.meta-chip {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 999px;
    font-size: 0.65rem;
    color: #64748B;
    border: 1px solid #334155;
    margin-right: 4px;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Shared resources — one instance per Streamlit server process
# ---------------------------------------------------------------------------

@st.cache_resource
def get_ledger() -> CreditLedger:
    return CreditLedger()

@st.cache_resource
def get_middleware() -> PrivacyMiddleware:
    return PrivacyMiddleware(credit_cost_per_request=1)

@st.cache_resource
def get_manifest() -> dict:
    import router
    return router.load_manifest()

def enabled_experts(manifest: dict) -> list[str]:
    return [n for n, e in manifest.get("experts", {}).items() if e.get("enabled")]

# ---------------------------------------------------------------------------
# Per-browser session state
# ---------------------------------------------------------------------------

ledger     = get_ledger()
middleware = get_middleware()
manifest   = get_manifest()
experts    = enabled_experts(manifest)

if "session_token" not in st.session_state:
    st.session_state.session_token = middleware.create_session()
    ledger.mint(st.session_state.session_token, DEMO_MINT_AMOUNT)

if "messages" not in st.session_state:
    st.session_state.messages = []

token = st.session_state.session_token

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🐝 ExpertSwarm")
    st.divider()

    st.markdown("**Active Expert**")
    selected_expert = st.selectbox(
        label="expert",
        options=experts,
        label_visibility="collapsed",
    )

    meta = manifest["experts"].get(selected_expert, {})
    st.caption(meta.get("description", ""))
    st.divider()

    # Credit balance
    balance = ledger.balance(token)
    credit_color = "#A78BFA" if balance > 2 else "#EF4444"
    st.markdown(
        f'<div class="credit-pill" style="color:{credit_color}">⚡ {balance} credits</div>',
        unsafe_allow_html=True,
    )

    if balance <= 2:
        st.warning("Low credits — top up to continue.", icon="⚠️")

    if st.button("＋ Add 10 credits", use_container_width=True):
        ledger.mint(token, 10)
        st.rerun()

    st.divider()

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Backend + session info
    backend_name = type(ledger.backend).__name__
    st.markdown(
        f'<span class="meta-chip">backend: {backend_name}</span>'
        f'<span class="meta-chip">session: {token[:8]}…</span>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main — header + chat
# ---------------------------------------------------------------------------

st.markdown("# ExpertSwarm")
st.caption("Local-first modular AI · Privacy-by-design · Powered by phi-2 + LoRA")
st.divider()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            expert_label = msg.get("expert", "base")
            elapsed      = msg.get("elapsed_s")
            timing       = f" · {elapsed:.1f}s" if elapsed else ""
            st.markdown(
                f'<span class="expert-badge">{expert_label}{timing}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input(f"Ask the swarm… (max {_MAX_PROMPT_LEN:,} chars)"):

    # Input validation
    if len(prompt) > _MAX_PROMPT_LEN:
        st.warning(
            f"Message too long ({len(prompt):,} chars). "
            f"Please keep it under {_MAX_PROMPT_LEN:,} characters."
        )
        st.stop()

    # Credit pre-check
    balance = ledger.balance(token)
    if balance < 1:
        st.warning(
            "⚠️ Out of credits. Use **＋ Add 10 credits** in the sidebar.",
            icon="⚠️",
        )
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        st.markdown(
            f'<span class="expert-badge">{selected_expert}</span>',
            unsafe_allow_html=True,
        )
        with st.spinner(f"{selected_expert} is thinking…"):
            # Hard credit gate
            if not ledger.check_and_deduct(token, 1):
                st.warning("Insufficient credits.")
                st.stop()

            import router
            t0 = time.perf_counter()
            try:
                response = st.write_stream(
                    router.route_stream(prompt, expert=selected_expert)
                )
            except Exception as exc:
                st.error(f"Inference error: {exc}")
                st.stop()
            elapsed = time.perf_counter() - t0

    # Persist to history with timing
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   response,
        "expert":    selected_expert,
        "elapsed_s": elapsed,
    })

    st.rerun()
