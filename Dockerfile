# ============================================================
# ExpertSwarm — Security-hardened Dockerfile
# Build:  docker build -t expertswarm .
# Run:    docker run --gpus all -p 8501:8501 expertswarm
# ============================================================

# Use an official PyTorch base image with CUDA support.
# Pin to a specific digest in production to prevent supply-chain attacks.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ------------------------------------------------------------
# System hardening
# ------------------------------------------------------------
# Disable package caching and install only what is strictly needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Non-root user (Zero Trust: AI process must not run as root)
# ------------------------------------------------------------
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# ------------------------------------------------------------
# Application setup
# ------------------------------------------------------------
WORKDIR /app

# Copy requirements first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application as root, then lock down ownership.
COPY . .
RUN chown -R appuser:appgroup /app

# ------------------------------------------------------------
# Drop to non-root user for all subsequent commands and at runtime
# ------------------------------------------------------------
USER appuser

# Expose Streamlit port.
EXPOSE 8501

# Streamlit expects these env vars for headless/server mode.
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Default entrypoint — run the Streamlit dashboard.
# Override CMD to run router.py directly for CLI/batch inference.
CMD ["streamlit", "run", "ui/dashboard.py"]
