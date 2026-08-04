# Dockerfile for Spamlyser Streamlit App
# ─────────────────────────────────────────────────────────────────────
# Hardened production image:
#   • Non-root user for security
#   • Health check for orchestrator integration
#   • OCI-standard labels for image metadata
# ─────────────────────────────────────────────────────────────────────
FROM python:3.13-slim

# OCI / opencontainers metadata labels
LABEL org.opencontainers.image.title="Spamlyser Pro" \
      org.opencontainers.image.description="Ensemble SMS spam classifier powered by Streamlit and Hugging Face Transformers" \
      org.opencontainers.image.source="https://github.com/theeccentriccoder01/Spamlyser" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install dependencies first (layer caching optimisation)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create a non-root user and switch to it
RUN addgroup --system spamlyser \
    && adduser --system --ingroup spamlyser spamlyser \
    && chown -R spamlyser:spamlyser /app
USER spamlyser

# Expose the default Streamlit port
EXPOSE 8501

# Health check — Streamlit exposes /_stcore/health by default
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
