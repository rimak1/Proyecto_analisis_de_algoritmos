# Dockerfile — Bibliometrics AI Analyzer
FROM python:3.10-slim

LABEL maintainer="Bibliometrics Project"
LABEL description="Bibliometric analysis system for Generative AI literature"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Descargar recursos NLTK
RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('wordnet', quiet=True); \
"

# Copiar el código fuente
COPY . .

# Crear directorios de datos
RUN mkdir -p data/raw data/processed data/sample docs

# Exponer el puerto
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de inicio
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none", \
     "--theme.base=dark", \
     "--theme.primaryColor=#6C63FF", \
     "--theme.backgroundColor=#0E1117", \
     "--theme.secondaryBackgroundColor=#1c1e26"]
