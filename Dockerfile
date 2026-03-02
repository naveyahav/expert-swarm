# =============================================================================
# ExpertSwarm — Multi-stage Dockerfile
#
# Stage 1 (builder): install Python deps into an isolated virtualenv.
# Stage 2 (runtime): copy the venv + app code; run as non-root.
#
# Build:
#   docker build -t expertswarm .
#
# Run web UI:
#   docker run -p 8501:8501 --env-file .env expertswarm
#
# Run Telegram bot:
#   docker run --env-file .env expertswarm python interfaces/telegram_bot.py
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — dependency builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Build-time system packages (gcc needed by some wheels; absent in runtime).
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv so the runtime stage can copy it cleanly.
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Minimal runtime system packages (curl for healthcheck only).
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder — compiler toolchain is NOT in production image.
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# ---------------------------------------------------------------------------
# Non-root user (principle of least privilege)
# ---------------------------------------------------------------------------
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy application code; owned by appuser from the start.
COPY --chown=appuser:appgroup . .

# Writable directories for SQLite DB and HuggingFace model cache.
RUN mkdir -p /data /home/appuser/.cache \
 && chown -R appuser:appgroup /data /home/appuser/.cache

USER appuser

# ---------------------------------------------------------------------------
# Runtime environment — no secrets here; pass via --env-file .env
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    EXPERTSWARM_BACKEND=sqlite \
    EXPERTSWARM_DB_PATH=/data/credits.db

EXPOSE 8501

# Streamlit's built-in health endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: start the web UI.
# Override CMD for the Telegram bot:
#   docker run --env-file .env expertswarm python interfaces/telegram_bot.py
CMD ["python", "-m", "streamlit", "run", "interfaces/web_app.py"]
