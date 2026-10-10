# ── base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS python-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
FROM python-base AS deps

COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ── Frontend build ─────────────────────────────────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --ignore-scripts 2>/dev/null || npm install --ignore-scripts
COPY frontend/ ./
RUN npm run build

# ── Final image ───────────────────────────────────────────────────────────────
FROM deps AS final

WORKDIR /app

# Copy source
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY results/ ./results/
COPY data/ ./data/

# Copy built frontend
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "src.qcad.api:app", "--host", "0.0.0.0", "--port", "8000"]
