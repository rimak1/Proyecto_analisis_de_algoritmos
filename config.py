"""
config.py — Configuración global del proyecto Bibliometría IA Generativa
"""
import os
from pathlib import Path

# ─── Directorios base ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
DOCS_DIR = BASE_DIR / "docs"

# Crear directorios si no existen
for d in [RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Archivos principales ──────────────────────────────────────────────────────
UNIFIED_CSV = PROCESSED_DIR / "unified_dataset.csv"
DUPLICATES_CSV = PROCESSED_DIR / "duplicates_removed.csv"

# ─── Columnas estándar del dataset unificado ──────────────────────────────────
STANDARD_COLUMNS = [
    "id",
    "title",
    "authors",
    "year",
    "abstract",
    "keywords",
    "journal",
    "doi",
    "source_db",        # Base de datos de origen
    "country",          # País del primer autor
    "citations",
    "url",
]

# ─── Fuentes de datos disponibles ─────────────────────────────────────────────
# Bases de datos compatibles con biblioteca UIQ: https://library.uniquindio.edu.co/databases
DATA_SOURCES = {
    # ── UIQ Library databases ──────────────────────────────────────────────────
    "acm": {
        "name": "ACM Digital Library",
        "base_url": "https://api.openalex.org/works",   # OpenAlex filtrado por ACM
        "enabled": True,
        "uiq_library": True,
    },
    "sage": {
        "name": "SAGE Journals",
        "base_url": "https://api.openalex.org/works",   # OpenAlex filtrado por SAGE
        "enabled": True,
        "uiq_library": True,
    },
    "sciencedirect": {
        "name": "ScienceDirect (Elsevier)",
        "base_url": "https://api.elsevier.com/content/search/sciencedirect",
        "fallback_url": "https://api.openalex.org/works",
        "enabled": True,
        "uiq_library": True,
    },
    "ebsco": {
        "name": "EBSCO Discovery (Biblioteca UniQ)",
        "base_url": "https://search.ebscohost.com/login.aspx",
        "proxy_url": "https://login.crai.referencistas.com/login?url=",
        "custid": "ns004363",
        "enabled": True,
        "uiq_library": True,
    },
    # ── Fuentes complementarias ────────────────────────────────────────────────
    "crossref": {
        "name": "CrossRef",
        "base_url": "https://api.crossref.org/works",
        "enabled": True,
        "uiq_library": False,
    },
    "semantic_scholar": {
        "name": "Semantic Scholar",
        "base_url": "https://api.semanticscholar.org/graph/v1/paper/search",
        "enabled": True,
        "uiq_library": False,
    },
}

# ─── Query de búsqueda por defecto ────────────────────────────────────────────
DEFAULT_QUERY = "generative artificial intelligence"
MAX_RESULTS_PER_SOURCE = 100

# ─── Términos predefinidos — Req 3 ────────────────────────────────────────────
PREDEFINED_TERMS = [
    "generative models",
    "prompting",
    "machine learning",
    "multimodality",
    "fine-tuning",
    "training data",
    "algorithmic bias",
    "explainability",
    "transparency",
    "ethics",
    "privacy",
    "personalization",
    "human-ai interaction",
    "ai literacy",
    "co-creation",
]

# ─── Modelos de IA ────────────────────────────────────────────────────────────
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PARAPHRASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ─── Configuración de clustering ─────────────────────────────────────────────
CLUSTERING_METHODS = ["ward", "complete", "average"]
DEFAULT_N_CLUSTERS = 5

# ─── Colores y estética de la app ────────────────────────────────────────────
APP_TITLE = "Bibliometrics AI Analyzer"
APP_SUBTITLE = "Analisis de Literatura Cientifica — Inteligencia Artificial Generativa"
# Paleta institucional — Marrón Chocolate / Verde Oliva / Naranja Terracota / Beige Arena
PRIMARY_COLOR     = "#3E2723"   # Marrón Chocolate — fondos oscuros, encabezados
ACCENT_COLOR      = "#D84315"   # Naranja Terracota — botones, CTAs
OLIVE_COLOR       = "#558B2F"   # Verde Oliva — elementos secundarios, iconos
BG_COLOR          = "#D7CCC8"   # Beige Arena — fondo general
SURFACE_COLOR     = "#FFFFFF"   # Blanco — cards / inputs
BORDER_COLOR      = "#BCAAA4"   # Beige oscuro — bordes / divisores
TEXT_PRIMARY      = "#1B0000"   # Texto principal (oscuro)
TEXT_SECONDARY    = "#558B2F"   # Verde Oliva — texto secundario
COLOR_SUCCESS     = "#558B2F"   # Exito (verde oliva)
COLOR_ERROR       = "#D84315"   # Error (terracota)
COLOR_WARNING     = "#F57F17"   # Aviso (ambar)

# ─── PDF Export ──────────────────────────────────────────────────────────────
PDF_OUTPUT_PATH = PROCESSED_DIR / "bibliometric_report.pdf"
