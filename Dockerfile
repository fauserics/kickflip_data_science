# ===== Build base =====
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Sistema: paquetes mínimos (build y runtime ligeros)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# ===== Install deps =====
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===== App =====
# Copiamos todo (si tu repo tiene archivos pesados, ajusta .dockerignore)
COPY . .

# Puerto por defecto de Streamlit
EXPOSE 8501

# Healthcheck: UI debe responder algo en /
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Entrypoint: usa variables para puerto y address
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
