"""
interfaces/web_app.py
ExpertSwarm — modern dark-theme web UI.

Replaces interfaces/desktop_app.py (tkinter).

Run:
    streamlit run interfaces/web_app.py

Features:
  - Dark slate theme (configured in .streamlit/config.toml)
  - Streaming responses via router.route_stream() + st.write_stream()
  - Sidebar: expert selector, live credit balance, top-up, clear chat
  - CreditLedger + PrivacyMiddleware wired in — same gates as desktop app
  - No prompt or response text is logged or persisted beyond the session
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from credits.ledger import CreditLedger, DEMO_MINT_AMOUNT
from privacy.middleware import PrivacyMiddleware

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
# Custom CSS — polish on top of the dark theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Tighten up the top padding */
.block-container { padding-top: 2rem; padding-bottom: 1rem; }

/* Expert badge chip in the chat bubble */
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

/* Credit balance pill */
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

/* Remove the default footer */
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

    # Expert selector
    st.markdown("**Active Expert**")
    selected_expert = st.selectbox(
        label="expert",
        options=experts,
        label_visibility="collapsed",
    )

    # Expert description
    meta = manifest["experts"].get(selected_expert, {})
    st.caption(meta.get("description", ""))
    st.divider()

    # Credit balance
    balance = ledger.balance(token)
    st.markdown(
        f'<div class="credit-pill">⚡ {balance} credits</div>',
        unsafe_allow_html=True,
    )

    if st.button("＋ Add 10 credits", use_container_width=True):
        ledger.mint(token, 10)
        st.rerun()

    st.divider()

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Session: `{token[:12]}…`")

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
            st.markdown(
                f'<span class="expert-badge">{msg.get("expert", "base")}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])

# Chat input — stays pinned to the bottom by Streamlit's layout
if prompt := st.chat_input("Ask the swarm…"):

    # Guard: check credits before doing anything
    balance = ledger.balance(token)
    if balance < 1:
        st.warning("Out of credits. Use **＋ Add 10 credits** in the sidebar.")
        st.stop()

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the assistant response
    with st.chat_message("assistant"):
        st.markdown(
            f'<span class="expert-badge">{selected_expert}</span>',
            unsafe_allow_html=True,
        )
        with st.spinner(f"{selected_expert} is thinking…"):
            # Deduct credit before inference (hard gate)
            if not ledger.check_and_deduct(token, 1):
                st.warning("Insufficient credits.")
                st.stop()

            import router
            response = st.write_stream(
                router.route_stream(prompt, expert=selected_expert)
            )

    # Persist to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "expert": selected_expert,
    })

    # Trigger sidebar balance refresh
    st.rerun()
