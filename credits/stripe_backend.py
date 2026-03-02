"""
credits/stripe_backend.py
Stripe-verified credit top-up layer built on top of SQLiteBackend.

How the payment flow works:
  1. Client calls POST /create-payment-intent (your web/bot layer) to get
     a Stripe PaymentIntent client_secret.
  2. User completes payment via Stripe.js or the Stripe mobile SDK.
  3. Stripe fires a webhook event (payment_intent.succeeded) to your
     HTTPS endpoint.
  4. The endpoint calls StripeBackend.handle_webhook(payload, sig_header)
     which verifies the signature and credits the session.

Alternatively, for server-side confirmation (e.g. after redirects):
  - Call StripeBackend.verify_stripe_payment(payment_intent_id, session_token)
    after the user returns to your app.

Required environment variables:
    STRIPE_SECRET_KEY      — sk_live_... or sk_test_...
    STRIPE_WEBHOOK_SECRET  — whsec_... (from Stripe dashboard)

Install:
    pip install stripe>=7.0.0

Rates (configurable via _CENTS_PER_CREDIT):
    100 cents ($1.00) = 10 credits  →  $0.10 per credit
"""

import logging
import os

from credits.sqlite_backend import SQLiteBackend

log = logging.getLogger(__name__)

# Conversion rate: how many cents equal one credit.
_CENTS_PER_CREDIT: int = 10  # $0.10 per credit


class StripeBackend(SQLiteBackend):
    """
    SQLiteBackend extended with Stripe PaymentIntent verification.

    All credit balances are persisted in SQLite; Stripe is only called to
    verify that a payment actually succeeded before crediting the account.
    """

    def __init__(self, db_path: str | None = None) -> None:
        super().__init__(db_path=db_path)
        self._secret_key = os.environ.get("STRIPE_SECRET_KEY", "")
        self._webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        if not self._secret_key:
            log.warning(
                "STRIPE_SECRET_KEY is not set — Stripe payment verification will fail. "
                "Set it via environment variable before accepting real payments."
            )

    # ------------------------------------------------------------------
    # Payment verification
    # ------------------------------------------------------------------

    def verify_stripe_payment(self, payment_intent_id: str, session_token: str) -> int:
        """
        Retrieve a Stripe PaymentIntent by ID, verify it succeeded,
        compute credits from the amount, and top up the session.

        Args:
            payment_intent_id: The `pi_...` ID returned by Stripe.
            session_token:     The ExpertSwarm session to credit.

        Returns:
            New credit balance after top-up.

        Raises:
            EnvironmentError: STRIPE_SECRET_KEY not set.
            ValueError:       PaymentIntent not in 'succeeded' state.
            stripe.error.*:   Network or API errors from Stripe.
        """
        try:
            import stripe
        except ImportError as exc:
            raise ImportError(
                "stripe package not installed. Run: pip install stripe>=7.0.0"
            ) from exc

        if not self._secret_key:
            raise EnvironmentError("STRIPE_SECRET_KEY environment variable is not set.")

        stripe.api_key = self._secret_key
        pi = stripe.PaymentIntent.retrieve(payment_intent_id)

        if pi["status"] != "succeeded":
            raise ValueError(
                f"PaymentIntent {payment_intent_id} is not succeeded "
                f"(status={pi['status']!r}). No credits granted."
            )

        credits = pi["amount"] // _CENTS_PER_CREDIT
        if credits < 1:
            raise ValueError(
                f"PaymentIntent amount ({pi['amount']} cents) is below the "
                f"minimum ({_CENTS_PER_CREDIT} cents = 1 credit)."
            )

        new_balance = self.mint(session_token, credits)
        log.info(
            "Stripe payment verified: pi=%s amount=%d cents → %d credits "
            "for token %s… (new balance=%d)",
            payment_intent_id, pi["amount"], credits,
            session_token[:8], new_balance,
        )
        return new_balance

    def handle_webhook(self, payload: bytes, sig_header: str) -> dict | None:
        """
        Verify a Stripe webhook signature and process the event.

        Currently handles:
            payment_intent.succeeded → credits the session stored in
                                       event.data.object.metadata.session_token

        Args:
            payload:    Raw request body bytes.
            sig_header: Value of the Stripe-Signature HTTP header.

        Returns:
            The constructed Stripe event dict, or None if the event type
            is not handled.

        Raises:
            EnvironmentError: STRIPE_WEBHOOK_SECRET not set.
            stripe.error.SignatureVerificationError: Signature invalid.
        """
        try:
            import stripe
        except ImportError as exc:
            raise ImportError(
                "stripe package not installed. Run: pip install stripe>=7.0.0"
            ) from exc

        if not self._webhook_secret:
            raise EnvironmentError(
                "STRIPE_WEBHOOK_SECRET environment variable is not set."
            )

        stripe.api_key = self._secret_key
        event = stripe.Webhook.construct_event(payload, sig_header, self._webhook_secret)

        if event["type"] == "payment_intent.succeeded":
            pi = event["data"]["object"]
            session_token = pi.get("metadata", {}).get("session_token")
            if session_token:
                credits = pi["amount"] // _CENTS_PER_CREDIT
                self.mint(session_token, credits)
                log.info(
                    "Webhook: credited %d credits to token %s…",
                    credits, session_token[:8],
                )
            else:
                log.warning(
                    "payment_intent.succeeded received but metadata.session_token "
                    "is missing — no credits granted."
                )
            return event

        log.debug("Unhandled Stripe event type: %s", event["type"])
        return None
