"""
interfaces/desktop_app.py
ExpertSwarm desktop application — tkinter-based native UI.

Same privacy and credit guarantees as the Telegram bot:
  - No prompt or response text is logged or persisted.
  - Session token is ephemeral and clears on window close.
  - Credit gate blocks inference when balance reaches zero.

Run:
    python interfaces/desktop_app.py
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credits.ledger import CreditLedger, DEMO_MINT_AMOUNT
from privacy.middleware import PrivacyMiddleware


class ExpertSwarmApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("ExpertSwarm")
        self.resizable(True, True)
        self.minsize(640, 480)

        # Privacy and credit layers.
        self._ledger = CreditLedger()
        self._middleware = PrivacyMiddleware(credit_cost_per_request=1)
        self._session = self._middleware.create_session()
        self._ledger.mint(self._session, DEMO_MINT_AMOUNT)

        self._build_ui()
        self._refresh_balance()
        self._log(f"Session started. {DEMO_MINT_AMOUNT} demo credits minted.")
        self._load_experts()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ---- Top bar ----
        top = tk.Frame(self, pady=6, padx=10)
        top.pack(fill=tk.X)

        tk.Label(top, text="ExpertSwarm", font=("Helvetica", 14, "bold")).pack(side=tk.LEFT)

        self._balance_var = tk.StringVar(value="Credits: —")
        tk.Label(top, textvariable=self._balance_var, font=("Helvetica", 11)).pack(side=tk.RIGHT)

        # ---- Chat display ----
        self._chat = scrolledtext.ScrolledText(
            self, state=tk.DISABLED, wrap=tk.WORD,
            font=("Courier", 10), padx=8, pady=8,
        )
        self._chat.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        # ---- Input area ----
        bottom = tk.Frame(self, padx=10, pady=6)
        bottom.pack(fill=tk.X)

        tk.Label(bottom, text="Expert:").pack(side=tk.LEFT)
        self._expert_var = tk.StringVar()
        self._expert_combo = ttk.Combobox(
            bottom, textvariable=self._expert_var, state="readonly", width=12,
        )
        self._expert_combo.pack(side=tk.LEFT, padx=(4, 10))

        self._input = tk.Entry(bottom, font=("Helvetica", 11))
        self._input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._input.bind("<Return>", lambda _: self._send())

        self._send_btn = tk.Button(bottom, text="Send", width=8, command=self._send)
        self._send_btn.pack(side=tk.LEFT, padx=(6, 0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_experts(self) -> None:
        """Populate the expert dropdown from manifest."""
        try:
            import router
            manifest = router.load_manifest()
            enabled = [
                name for name, entry in manifest.get("experts", {}).items()
                if entry.get("enabled", False)
            ]
            self._expert_combo["values"] = enabled
            if enabled:
                self._expert_var.set(enabled[0])
        except Exception as exc:
            self._log(f"[error] Could not load manifest: {exc}")

    def _refresh_balance(self) -> None:
        bal = self._ledger.balance(self._session)
        self._balance_var.set(f"Credits: {bal}")

    def _log(self, text: str) -> None:
        """Append a line to the chat display."""
        self._chat.config(state=tk.NORMAL)
        self._chat.insert(tk.END, text + "\n")
        self._chat.see(tk.END)
        self._chat.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    def _send(self) -> None:
        prompt = self._input.get().strip()
        if not prompt:
            return

        expert = self._expert_var.get() or None
        self._input.delete(0, tk.END)
        self._log(f"\nYou [{expert}]: {prompt}")

        # Disable input while inference runs to prevent double-sends.
        self._input.config(state=tk.DISABLED)
        self._send_btn.config(state=tk.DISABLED)

        # Run inference in a background thread so the UI stays responsive.
        import threading
        threading.Thread(target=self._infer, args=(prompt, expert), daemon=True).start()

    def _infer(self, prompt: str, expert: str | None) -> None:
        response = self._middleware.handle(
            session_token=self._session,
            prompt=prompt,
            expert=expert,
            credit_check=self._ledger.check_and_deduct,
        )
        # Update UI from the main thread.
        self.after(0, self._on_response, response)

    def _on_response(self, response: str) -> None:
        self._log(f"ExpertSwarm: {response}")
        self._refresh_balance()
        self._input.config(state=tk.NORMAL)
        self._send_btn.config(state=tk.NORMAL)
        self._input.focus_set()


def main() -> None:
    app = ExpertSwarmApp()
    app.mainloop()


if __name__ == "__main__":
    main()
