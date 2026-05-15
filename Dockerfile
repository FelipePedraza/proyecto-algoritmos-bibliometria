# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Bibliometría · IA Generativa
# Universidad del Quindío · Análisis de Algoritmos
#
# Requerimiento 6: Despliegue Docker de la aplicación Streamlit
# ─────────────────────────────────────────────────────────────────────────────

# ── Etapa 1: Construcción de dependencias ────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Instalar compiladores necesarios para algunas dependencias nativas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo el archivo de dependencias primero (cache layer)
COPY requirements.txt .

# Instalar dependencias en /install para copiarlas a la imagen final
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Etapa 2: Descarga de modelos de IA ───────────────────────────────────────
FROM python:3.11-slim AS model-downloader

WORKDIR /models

# Copiar dependencias instaladas desde el builder
COPY --from=builder /install /usr/local

# Descargar modelo spaCy en_core_web_md (300-dim GloVe vectors, ~45 MB)
RUN python -m spacy download en_core_web_md

# Pre-descargar modelo Sentence-BERT all-MiniLM-L6-v2 (~90 MB)
# Se guarda en el directorio de cache de transformers
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"


# ── Etapa 3: Imagen final de producción ──────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="calderon@vivexia.co"
LABEL description="Análisis Bibliométrico de IA Generativa — Universidad del Quindío"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_BASE=light \
    # Directorio de modelos de transformers
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    # Directorio de modelos de spaCy
    SPACY_DATA=/app/.cache/spacy

WORKDIR /app

# Instalar solo las dependencias del sistema que necesita la app en runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias Python instaladas
COPY --from=builder /install /usr/local

# Copiar modelos de IA descargados
COPY --from=model-downloader /root/.cache /app/.cache
COPY --from=model-downloader /usr/local/lib/python3.11/site-packages/en_core_web_md \
     /usr/local/lib/python3.11/site-packages/en_core_web_md
COPY --from=model-downloader /usr/local/lib/python3.11/site-packages/en_core_web_md-3.8.0.dist-info \
     /usr/local/lib/python3.11/site-packages/en_core_web_md-3.8.0.dist-info 2>/dev/null || true

# Copiar código fuente de la aplicación
COPY app.py .
COPY src/ ./src/
COPY data/ ./data/

# Crear directorio para datos de sesión
RUN mkdir -p /app/data/processed /app/data/duplicates /app/.streamlit

# Configuración de Streamlit
COPY .streamlit/ ./.streamlit/ 2>/dev/null || true

# Puerto expuesto
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Comando de inicio
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
