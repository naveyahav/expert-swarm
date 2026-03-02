"""
ui/dashboard.py
ExpertSwarm — Streamlit dashboard.

Run from the project root:
    streamlit run ui/dashboard.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `router` and `core` are importable
# regardless of the working directory Streamlit was launched from.
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import router

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ExpertSwarm",
    page_icon="🐝",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Manifest — cached for the lifetime of the Streamlit server process.
# Reloads automatically when the user presses 'Clear cache' or restarts.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_manifest() -> dict:
    return router.load_manifest()


def enabled_experts(manifest: dict) -> list[str]:
    """Return names of all enabled experts in manifest order."""
    return [
        name
        for name, entry in manifest.get("experts", {}).items()
        if entry.get("enabled", False)
    ]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("ExpertSwarm")
st.caption("Local-first modular AI — powered by Llama 3 + LoRA adapters")
st.divider()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    try:
        manifest = get_manifest()
        experts = enabled_experts(manifest)
    except FileNotFoundError as exc:
        st.error(f"Could not load manifest:\n{exc}")
        st.stop()

    if not experts:
        st.warning("No enabled experts found in manifest.json.")
        st.stop()

    selected_expert = st.selectbox(
        "Active Expert",
        options=experts,
        help="Selects which LoRA adapter the router will load for your next message.",
    )

    st.divider()
    meta = manifest["experts"][selected_expert]
    st.caption(f"**{selected_expert}**")
    st.caption(meta.get("description", ""))
    st.caption(f"v{meta.get('version', '—')}")

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
st.subheader("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input.
if prompt := st.chat_input(f"Ask the {selected_expert} expert…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Loading {selected_expert} expert…"):
            try:
                response = router.route(prompt, expert=selected_expert)
            except Exception as exc:
                st.exception(exc)
                response = f"Unexpected error: {exc}"

        # Surface security failures as a visible warning, not just text.
        if response.startswith("Security check failed"):
            st.warning(response)
        else:
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
