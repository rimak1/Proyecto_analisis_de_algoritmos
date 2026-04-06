"""
data_extraction/fetcher.py — Req 1
Automatización de extracción de datos desde múltiples fuentes bibliográficas.

Fuentes implementadas:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ COMPATIBLES con la biblioteca de la Universidad del Quindío        │
  │ https://library.uniquindio.edu.co/databases                        │
  │                                                                     │
  │ 1. ACM Digital Library   → OpenAlex (filtro: host_organization=ACM)│
  │ 2. SAGE Journals         → OpenAlex (filtro: host_organization=SAGE)│
  │ 3. ScienceDirect/Elsevier→ Elsevier Developer API (dev.elsevier.com)│
  │    + OpenAlex (filtro Elsevier)                                     │
  │ 4. CrossRef              → API REST gratuita (todos los editores)   │
  │ 5. Semantic Scholar      → API REST gratuita (AI-centric)           │
  └─────────────────────────────────────────────────────────────────────┘

Nota sobre ACM y SAGE:
  ACM y SAGE no ofrecen APIs de búsqueda bibliográfica abiertas de forma
  gratuita. Sin embargo, OpenAlex indexa el 100% de los metadatos de sus
  publicaciones y permite filtrar exactamente por editorial. Se usa ese
  canal como "API compatible" según lo permite el enunciado:
  "simulando la conexión o usando APIs de fuentes compatibles".

  La columna `source_db` en el dataset indica claramente "ACM", "SAGE" o
  "ScienceDirect" según la editorial del artículo recuperado.

Nota sobre ScienceDirect/Elsevier:
  Elsevier ofrece una API REST oficial (https://dev.elsevier.com) que
  requiere una API key gratuita. El fetcher la usa si está configurada en
  .env (ELSEVIER_API_KEY). Sin ella, usa el canal OpenAlex como fallback.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    DATA_SOURCES,
    DEFAULT_QUERY,
    MAX_RESULTS_PER_SOURCE,
    RAW_DIR,
    STANDARD_COLUMNS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Elsevier API key (opcional — configurar en .env o variable de entorno)
ELSEVIER_API_KEY = os.getenv("ELSEVIER_API_KEY", "")

# IDs de organizaciones en OpenAlex para filtrar por editorial
OPENALEX_PUBLISHER_IDS = {
    "ACM": "P4310313ull",          # Association for Computing Machinery
    "SAGE": "P4310316181",         # SAGE Publications
    "Elsevier": "P4310319965",     # Elsevier (ScienceDirect)
    "IEEE": "P4310312979",         # IEEE (bonus)
    "Springer": "P4310315589",     # Springer Nature (bonus)
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_get(d: dict, *keys, default=""):
    """Navegación segura en diccionarios anidados."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, {})
    return d if d != {} else default


# ─── Clase principal ───────────────────────────────────────────────────────────

class DataFetcher:
    """
    Descarga registros bibliográficos desde múltiples fuentes.

    Fuentes principales (UIQ-compatibles):
      • fetch_acm()           → artículos de ACM via OpenAlex (filtro editorial)
      • fetch_sage()          → artículos de SAGE via OpenAlex (filtro editorial)
      • fetch_sciencedirect() → artículos de Elsevier/ScienceDirect via:
                                   1. Elsevier API (si hay API key)
                                   2. OpenAlex (filtro Elsevier) como fallback

    Fuentes complementarias:
      • fetch_crossref()      → CrossRef API (todos los editores)
      • fetch_semantic_scholar() → Semantic Scholar (AI-centric)

    Uso:
        fetcher = DataFetcher(query="generative artificial intelligence")
        dfs = fetcher.fetch_all(sources=["acm", "sage", "sciencedirect"])
    """

    def __init__(
        self,
        query: str = DEFAULT_QUERY,
        max_results: int = MAX_RESULTS_PER_SOURCE,
        progress_callback=None,
        reset_ebsco: bool = False,
    ):
        self.query = query
        self.max_results = max_results
        self.progress_callback = progress_callback
        self.reset_ebsco = reset_ebsco
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "BibliometricsAnalyzer/1.0 (UniQuindio academic-research)"
        })

    # ══════════════════════════════════════════════════════════════════════════
    # ── BASE: OpenAlex con filtro de editorial ─────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def _fetch_openalex_filtered(
        self,
        publisher_name: str,
        publisher_id: str,
        source_label: str,
    ) -> pd.DataFrame:
        """
        Extrae artículos de OpenAlex filtrando por editorial específica.

        OpenAlex indexa el 100% del catálogo de ACM, SAGE y Elsevier.
        El filtro `primary_location.source.publisher_lineage` permite
        obtener SOLO artículos publicados por esa editorial.

        Args:
            publisher_name: Nombre legible (ej: "ACM")
            publisher_id:   ID de OpenAlex del publisher (ej: "P4310313ull")
            source_label:   Etiqueta para la columna source_db

        Returns:
            DataFrame con artículos de esa editorial.
        """
        records = []
        base_url = "https://api.openalex.org/works"
        page = 1
        per_page = min(50, self.max_results)

        logger.info(f"Extrayendo desde OpenAlex → {publisher_name}...")

        # Filtro compuesto: query de búsqueda + editorial específica
        openalex_filter = f"language:en,primary_location.source.publisher_lineage:{publisher_id}"

        with tqdm(total=self.max_results, desc=f"OpenAlex/{publisher_name}") as pbar:
            while len(records) < self.max_results:
                params = {
                    "search": self.query,
                    "per-page": per_page,
                    "page": page,
                    "filter": openalex_filter,
                    "select": (
                        "id,title,authorships,publication_year,"
                        "abstract_inverted_index,primary_location,keywords,doi,cited_by_count"
                    ),
                }
                try:
                    resp = self.session.get(base_url, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    logger.warning(f"OpenAlex/{publisher_name} error pág {page}: {e}")
                    break

                items = data.get("results", [])
                if not items:
                    break

                for item in items:
                    record = self._parse_openalex(item, source_label)
                    if record:
                        records.append(record)

                pbar.update(len(items))
                page += 1
                if len(records) >= self.max_results:
                    break
                time.sleep(0.25)

        df = pd.DataFrame(records, columns=STANDARD_COLUMNS)
        logger.info(f"OpenAlex/{publisher_name}: {len(df)} registros.")
        return df

    def _parse_openalex(self, item: dict, source_label: str = "OpenAlex") -> list | None:
        """Parsea un registro OpenAlex al formato estándar."""
        title = (item.get("title") or "").strip()
        if not title:
            return None

        abstract = self._reconstruct_abstract(item.get("abstract_inverted_index", {}))

        authorships = item.get("authorships", [])
        authors = "; ".join(
            [a.get("author", {}).get("display_name", "") for a in authorships[:10]]
        )

        country = ""
        if authorships:
            insts = authorships[0].get("institutions", [])
            if insts:
                country = insts[0].get("country_code", "")

        kws = item.get("keywords", [])
        keywords = "; ".join([k.get("display_name", "") for k in kws])

        loc = item.get("primary_location", {}) or {}
        source = loc.get("source", {}) or {}
        journal = source.get("display_name", "")

        year = item.get("publication_year") or ""
        doi = item.get("doi", "") or ""
        citations = item.get("cited_by_count", 0) or 0
        oa_id = item.get("id", "")

        return [
            oa_id, title, authors, year, abstract, keywords,
            journal, doi, source_label, country, citations, oa_id
        ]

    @staticmethod
    def _reconstruct_abstract(inverted_index: dict) -> str:
        """Reconstruye el abstract desde el índice invertido de OpenAlex."""
        if not inverted_index:
            return ""
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join([w for _, w in word_positions])

    # ══════════════════════════════════════════════════════════════════════════
    # ── Fuente 1: ACM Digital Library (via OpenAlex filtrado) ─────────────
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_acm(self) -> pd.DataFrame:
        """
        Extrae artículos de ACM Digital Library via OpenAlex.

        ACM no ofrece una API pública gratuita de búsqueda, pero OpenAlex
        indexa completamente su catálogo. Al filtrar por publisher_lineage
        del ID de ACM, obtenemos ÚNICAMENTE artículos publicados en:
          - ACM Digital Library
          - ACM Transactions on...
          - Proceedings of ACM conferences (SIGCHI, CSCW, CHI, etc.)

        Equivalencia con biblioteca UIQ:
          → Compatible con la suscripción institucional a ACM de UniQ.
          → Los metadatos son los mismos; el acceso al full-text PDF
            requiere la VPN/EZProxy de la universidad.
        """
        return self._fetch_openalex_filtered(
            publisher_name="ACM",
            publisher_id=OPENALEX_PUBLISHER_IDS["ACM"],
            source_label="ACM",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ── Fuente 2: SAGE Journals (via OpenAlex filtrado) ───────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_sage(self) -> pd.DataFrame:
        """
        Extrae artículos de SAGE Journals via OpenAlex.

        SAGE no tiene API pública de búsqueda gratuita. OpenAlex indexa
        su catálogo completo. El filtro recupera artículos de:
          - SAGE Open
          - British Journal of Educational Technology (BJET)
          - Journal of Information Technology
          - Y 900+ revistas SAGE más

        Equivalencia con biblioteca UIQ:
          → Compatible con la suscripción de UniQ a SAGE.
        """
        return self._fetch_openalex_filtered(
            publisher_name="SAGE",
            publisher_id=OPENALEX_PUBLISHER_IDS["SAGE"],
            source_label="SAGE",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ── Fuente 3: ScienceDirect (Elsevier Developer API + OpenAlex fallback)
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_sciencedirect(self) -> pd.DataFrame:
        """
        Extrae artículos de ScienceDirect/Elsevier.

        Estrategia dual:
          A) Si ELSEVIER_API_KEY está configurada:
             Usa la API oficial de Elsevier (dev.elsevier.com):
             GET https://api.elsevier.com/content/search/sciencedirect
             Requiere API key gratuita en: https://dev.elsevier.com/apikey/manage

          B) Si no hay API key (modo de compatibilidad):
             Usa OpenAlex filtrado por publisher Elsevier.
             Cubre todas las revistas de ScienceDirect con acceso abierto.

        Equivalencia con biblioteca UIQ:
          → Compatible con la suscripción de UniQ a ScienceDirect.
        """
        if ELSEVIER_API_KEY:
            logger.info("Usando Elsevier Developer API (ScienceDirect)...")
            return self._fetch_elsevier_api()
        else:
            logger.info(
                "ELSEVIER_API_KEY no configurada. "
                "Usando OpenAlex/Elsevier como fuente compatible para ScienceDirect. "
                "Para activar la API oficial: set ELSEVIER_API_KEY=<tu_key> en .env"
            )
            return self._fetch_openalex_filtered(
                publisher_name="Elsevier/ScienceDirect",
                publisher_id=OPENALEX_PUBLISHER_IDS["Elsevier"],
                source_label="ScienceDirect",
            )

    def _fetch_elsevier_api(self) -> pd.DataFrame:
        """
        Extrae artículos directamente de la Elsevier ScienceDirect Search API.

        Endpoint: GET https://api.elsevier.com/content/search/sciencedirect
        Docs: https://dev.elsevier.com/documentation/ScienceDirectSearchAPI.wadl

        Requiere:
          - API key gratuita en https://dev.elsevier.com/apikey/manage
          - Ser parte de una institución con acceso a ScienceDirect (UniQ)
        """
        records = []
        base_url = "https://api.elsevier.com/content/search/sciencedirect"
        start = 0
        count = min(25, self.max_results)  # Elsevier max 25 por request

        self.session.headers.update({
            "X-ELS-APIKey": ELSEVIER_API_KEY,
            "Accept": "application/json",
        })

        logger.info("Extrayendo desde ScienceDirect (Elsevier API)...")

        with tqdm(total=self.max_results, desc="ScienceDirect") as pbar:
            while len(records) < self.max_results:
                params = {
                    "query": self.query,
                    "count": count,
                    "start": start,
                    "field": "dc:title,dc:description,dc:creator,prism:publicationName,"
                             "prism:coverDate,dc:identifier,prism:doi,prism:volume",
                }
                try:
                    resp = self.session.get(base_url, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    logger.error(f"Elsevier API error: {e}")
                    break

                entries = (
                    data.get("search-results", {})
                    .get("entry", [])
                )
                if not entries:
                    break

                for entry in entries:
                    record = self._parse_elsevier(entry)
                    if record:
                        records.append(record)

                pbar.update(len(entries))
                start += count
                total_results = int(
                    data.get("search-results", {})
                    .get("opensearch:totalResults", 0)
                )
                if start >= total_results or len(records) >= self.max_results:
                    break
                time.sleep(0.5)

        df = pd.DataFrame(records, columns=STANDARD_COLUMNS)
        logger.info(f"ScienceDirect (Elsevier API): {len(df)} registros.")
        return df

    def _parse_elsevier(self, entry: dict) -> list | None:
        """Parsea un registro de la Elsevier ScienceDirect API."""
        title = (entry.get("dc:title") or "").strip()
        if not title:
            return None

        abstract = (entry.get("dc:description") or "").strip()
        abstract = re.sub(r"<[^>]+>", " ", abstract).strip()

        # Autores (campo puede ser str o lista)
        authors_raw = entry.get("dc:creator") or ""
        if isinstance(authors_raw, list):
            authors = "; ".join(authors_raw[:10])
        else:
            authors = str(authors_raw)

        doi = (entry.get("prism:doi") or "").strip()
        year_raw = entry.get("prism:coverDate", "")
        year = year_raw[:4] if year_raw else ""
        journal = (entry.get("prism:publicationName") or "").strip()
        record_id = f"SD_{doi}" if doi else f"SD_{hash(title)}"
        url = f"https://www.sciencedirect.com/science/article/pii/{entry.get('pii', '')}"

        return [
            record_id, title, authors, year, abstract, "",
            journal, doi, "ScienceDirect", "", 0, url
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # ── Fuente 4: CrossRef ────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_crossref(self) -> pd.DataFrame:
        """
        Extrae artículos de CrossRef usando su API REST paginada.
        Endpoint: GET https://api.crossref.org/works?query=<query>
        Cubre artículos de ACM, SAGE, Elsevier, IEEE, Springer, etc.
        """
        records = []
        base_url = DATA_SOURCES["crossref"]["base_url"]
        offset = 0
        rows = min(50, self.max_results)

        logger.info("Iniciando extracción desde CrossRef...")

        with tqdm(total=self.max_results, desc="CrossRef") as pbar:
            while len(records) < self.max_results:
                params = {
                    "query": self.query,
                    "rows": rows,
                    "offset": offset,
                    "select": "DOI,title,author,published,abstract,subject,container-title,is-referenced-by-count,URL",
                    "filter": "type:journal-article,has-abstract:true",
                    "mailto": "bibliometrics@uniquindio.edu.co",
                }
                try:
                    resp = self.session.get(base_url, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    logger.warning(f"CrossRef error en offset {offset}: {e}")
                    break

                items = data.get("message", {}).get("items", [])
                if not items:
                    break

                for item in items:
                    record = self._parse_crossref(item)
                    if record:
                        records.append(record)

                pbar.update(len(items))
                offset += rows
                if len(records) >= self.max_results:
                    break
                time.sleep(0.3)

        df = pd.DataFrame(records, columns=STANDARD_COLUMNS)
        logger.info(f"CrossRef: {len(df)} registros obtenidos.")
        return df

    def _parse_crossref(self, item: dict) -> list | None:
        title_list = item.get("title", [])
        title = title_list[0].strip() if title_list else ""
        if not title:
            return None

        abstract = item.get("abstract", "")
        abstract = re.sub(r"<[^>]+>", " ", abstract).strip()

        authors_raw = item.get("author", [])
        authors = "; ".join(
            [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors_raw[:10]]
        )

        country = ""
        if authors_raw:
            aff = authors_raw[0].get("affiliation", [])
            if aff:
                country = aff[0].get("name", "")

        pub_date = item.get("published", {})
        date_parts = pub_date.get("date-parts", [[""]])
        year = date_parts[0][0] if date_parts and date_parts[0] else ""

        keywords_list = item.get("subject", [])
        keywords = "; ".join(keywords_list)

        journal_list = item.get("container-title", [])
        journal = journal_list[0] if journal_list else ""

        doi = item.get("DOI", "")
        citations = item.get("is-referenced-by-count", 0) or 0
        url = item.get("URL", "")
        record_id = f"CR_{doi}" if doi else f"CR_{hash(title)}"

        return [
            record_id, title, authors, year, abstract, keywords,
            journal, doi, "CrossRef", country, citations, url
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # ── Fuente 5: Semantic Scholar ────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_semantic_scholar(self) -> pd.DataFrame:
        """
        Extrae artículos de Semantic Scholar.
        Endpoint: GET https://api.semanticscholar.org/graph/v1/paper/search
        """
        records = []
        base_url = DATA_SOURCES["semantic_scholar"]["base_url"]
        offset = 0
        limit = min(100, self.max_results)

        logger.info("Iniciando extracción desde Semantic Scholar...")

        with tqdm(total=self.max_results, desc="Semantic Scholar") as pbar:
            while len(records) < self.max_results:
                params = {
                    "query": self.query,
                    "offset": offset,
                    "limit": limit,
                    "fields": "paperId,title,authors,year,abstract,venue,externalIds,citationCount,publicationTypes",
                }
                try:
                    resp = self.session.get(base_url, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    logger.warning(f"Semantic Scholar error offset {offset}: {e}")
                    break

                items = data.get("data", [])
                if not items:
                    break

                for item in items:
                    record = self._parse_semantic_scholar(item)
                    if record:
                        records.append(record)

                pbar.update(len(items))
                offset += limit
                if len(records) >= self.max_results or offset >= data.get("total", 0):
                    break
                time.sleep(0.5)

        df = pd.DataFrame(records, columns=STANDARD_COLUMNS)
        logger.info(f"Semantic Scholar: {len(df)} registros.")
        return df

    def _parse_semantic_scholar(self, item: dict) -> list | None:
        title = (item.get("title") or "").strip()
        if not title:
            return None

        abstract = (item.get("abstract") or "").strip()
        authors_raw = item.get("authors", [])
        authors = "; ".join([a.get("name", "") for a in authors_raw[:10]])
        year = item.get("year", "")
        journal = item.get("venue", "")
        citations = item.get("citationCount", 0) or 0
        ext_ids = item.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "")
        paper_id = item.get("paperId", "")
        url = f"https://www.semanticscholar.org/paper/{paper_id}"

        return [
            f"SS_{paper_id}", title, authors, year, abstract, doi,
            journal, doi, "SemanticScholar", "", citations, url
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # ── fetch_all & save ──────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_all(self, sources: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """
        Ejecuta la extracción de las fuentes seleccionadas.

        Fuentes disponibles:
          "acm"              → ACM Digital Library (OpenAlex filtrado)
          "sage"             → SAGE Journals (OpenAlex filtrado)
          "sciencedirect"    → ScienceDirect (Elsevier API o OpenAlex filtrado)
          "crossref"         → CrossRef (todos los editores)
          "semantic_scholar" → Semantic Scholar

        Args:
            sources: lista de fuentes. None = todas las 5 fuentes.

        Returns:
            Dict {nombre_fuente: DataFrame}
        """
        available = {
            "ebsco": self.fetch_ebsco,
            "acm": self.fetch_acm,
            "sage": self.fetch_sage,
            "sciencedirect": self.fetch_sciencedirect,
            "crossref": self.fetch_crossref,
            "semantic_scholar": self.fetch_semantic_scholar,
        }

        selected = sources or list(available.keys())
        results = {}

        # Procesar EBSCO primero si está seleccionado (requiere login
        # manual en navegador — el usuario debe verlo de inmediato).
        ordered = selected
        if "ebsco" in selected:
            ordered = ["ebsco"] + [s for s in selected if s != "ebsco"]

        for name in ordered:
            if name not in available:
                logger.warning(f"Fuente desconocida: {name}. Disponibles: {list(available)}")
                continue
            logger.info(f"── Extrayendo desde: {name.upper()} ──")
            df = available[name]()
            results[name] = df
            if self.progress_callback:
                self.progress_callback(name, len(df))

        return results

    def save_raw(self, dataframes: dict[str, pd.DataFrame]) -> None:
        """Guarda los DataFrames crudos en archivos CSV en data/raw/."""
        for source_name, df in dataframes.items():
            if df.empty:
                continue
            out_path = RAW_DIR / f"{source_name}_raw.csv"
            df.to_csv(out_path, index=False, encoding="utf-8")
            logger.info(f"Guardado: {out_path} ({len(df)} registros)")


    # ══════════════════════════════════════════════════════════════════════════
    # ── Fuente 6: EBSCO Discovery (Biblioteca UIQ — scraping directo) ────
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_ebsco(self) -> pd.DataFrame:
        """
        Extrae artículos de EBSCO EDS usando Playwright en un subproceso.

        Playwright no puede ejecutarse dentro del event loop de Streamlit
        en Windows (NotImplementedError en asyncio). Se lanza como subproceso
        con logs en tiempo real en la terminal de Streamlit.
        """
        import subprocess
        import sys

        try:
            from data_extraction.ebsco_scraper import PLAYWRIGHT_AVAILABLE
            if not PLAYWRIGHT_AVAILABLE:
                logger.warning(
                    "Playwright no instalado. "
                    "Ejecute: pip install playwright && playwright install chromium"
                )
                return pd.DataFrame(columns=STANDARD_COLUMNS)
        except ImportError:
            logger.warning("ebsco_scraper no disponible.")
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        # CSV temporal donde el subproceso guardará los resultados
        tmp_csv = RAW_DIR / "ebsco_results.csv"
        project_root = str(Path(__file__).resolve().parent.parent)

        # Script Python que ejecuta el scraper
        script = (
            f"import sys; sys.path.insert(0, r'{project_root}'); "
            f"from data_extraction.ebsco_scraper import EBSCOScraper; "
            f"s = EBSCOScraper("
            f"query=r'''{self.query}''', "
            f"max_results={self.max_results}, "
            f"headless=False, "
            f"reset_session={self.reset_ebsco}); "
            f"df = s.fetch(); "
            f"df.to_csv(r'{str(tmp_csv)}', index=False, encoding='utf-8'); "
            f"print(f'EBSCO_OK:{{len(df)}}')"
        )

        logger.info(
            "Lanzando EBSCO scraper — se abrirá un navegador Chromium.\n"
            "Si es la primera vez, inicie sesión con su cuenta Google institucional.\n"
            "Los logs del scraper aparecerán aquí abajo:"
        )

        try:
            # Popen: NO capturamos stderr para que los logs aparezcan en tiempo real.
            # Sí capturamos stdout para leer el resultado final.
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=None,   # stderr va directo a la terminal de Streamlit
                text=True,
                cwd=project_root,
            )

            # Esperar a que termine (timeout de 10 minutos)
            try:
                stdout, _ = proc.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
                logger.error(
                    "EBSCO: Timeout de 10 minutos. "
                    "¿Completó el login en el navegador?"
                )
                return pd.DataFrame(columns=STANDARD_COLUMNS)

            if stdout and "EBSCO_OK:" in stdout:
                count = stdout.strip().split("EBSCO_OK:")[-1]
                logger.info(f"Subproceso EBSCO terminó: {count} registros.")

            if proc.returncode != 0:
                logger.error(f"EBSCO subproceso terminó con código {proc.returncode}")
                return pd.DataFrame(columns=STANDARD_COLUMNS)

            # Leer resultados
            if tmp_csv.exists() and tmp_csv.stat().st_size > 0:
                df = pd.read_csv(tmp_csv, encoding="utf-8")
                logger.info(f"EBSCO: {len(df)} registros extraídos.")
                return df

            logger.warning("EBSCO: No se generaron resultados.")
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        except Exception as e:
            logger.error(f"Error en EBSCO subproceso: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=STANDARD_COLUMNS)


# ─── Ejecución directa ────────────────────────────────────────────────────────
if __name__ == "__main__":
    fetcher = DataFetcher(query=DEFAULT_QUERY, max_results=50)
    # Usar las 3 fuentes UIQ explícitas + las 2 complementarias
    dfs = fetcher.fetch_all(sources=["acm", "sage", "sciencedirect", "crossref", "semantic_scholar"])
    fetcher.save_raw(dfs)
    print("\n✅ Extracción completada.")
    for name, df in dfs.items():
        print(f"  {name}: {len(df)} registros")
