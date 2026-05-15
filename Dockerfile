# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Bibliometría · IA Generativa
# Universidad del Quindío · Análisis de Algoritmos
# Requerimiento 6: Despliegue Docker de la aplicación Streamlit
# ─────────────────────────────────────────────────────────────────────────────

# ── Etapa 1: Construcción de dependencias ────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Etapa 2: Descarga de modelos de IA ───────────────────────────────────────
FROM python:3.11-slim AS model-downloader

WORKDIR /models

COPY --from=builder /install /usr/local

RUN python -m spacy download en_core_web_md

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# ── Etapa 3: Imagen final de producción ──────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="calderon@vivexia.co"
LABEL description="Analisis Bibliometrico de IA Generativa - Universidad del Quindio"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY --from=model-downloader /root/.cache /app/.cache
COPY --from=model-downloader /usr/local/lib/python3.11/site-packages/en_core_web_md \
     /usr/local/lib/python3.11/site-packages/en_core_web_md
COPY --from=model-downloader /usr/local/lib/python3.11/site-packages/en_core_web_md-3.8.0.dist-info \
     /usr/local/lib/python3.11/site-packages/en_core_web_md-3.8.0.dist-info

COPY app.py .
COPY src/ ./src/
COPY data/ ./data/

RUN mkdir -p /app/data/processed /app/data/duplicates /app/.streamlit

COPY .streamlit/ ./.streamlit/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
