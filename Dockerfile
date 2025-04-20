FROM python:3.9-bullseye

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libgomp1 \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    MAX_WORKERS=4 \
    PATH="/home/appuser/.local/bin:${PATH}"

# Create app directory and user
RUN mkdir /app && \
    useradd -m appuser && \
    chown -R appuser /app

WORKDIR /app

# Install dependencies as root first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Switch to appuser and copy files
USER appuser
COPY --chown=appuser:appuser . .

# Healthcheck and run
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]