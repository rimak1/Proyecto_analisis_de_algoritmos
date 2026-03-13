"""
data_extraction/deduplicator.py — Req 1
Elimina registros duplicados del dataset unificado.

Criterios de deduplicación (en orden de prioridad):
  1. DOI exacto (si está presente en ambos registros)
  2. Título normalizado (lowercase, sin puntuación, sin stopwords cortas)
  3. Similitud fuzzy de título ≥ 0.92 (último recurso, más costoso)

El proceso es totalmente automático y genera un archivo separado con
todos los duplicados eliminados (con razón de eliminación) en:
  data/processed/duplicates_removed.csv
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
from difflib import SequenceMatcher

from config import DUPLICATES_CSV, STANDARD_COLUMNS, UNIFIED_CSV

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Detecta y elimina duplicados del dataset bibliográfico.

    Atributos:
        clean_df:   DataFrame sin duplicados (resultado final)
        dupes_df:   DataFrame con los registros eliminados y razón
    """

    SIMILARITY_THRESHOLD = 0.92  # para comparación fuzzy de títulos

    def __init__(
        self,
        input_path: Path = UNIFIED_CSV,
        output_clean_path: Path = UNIFIED_CSV,
        output_dupes_path: Path = DUPLICATES_CSV,
    ):
        self.input_path = input_path
        self.output_clean_path = output_clean_path
        self.output_dupes_path = output_dupes_path
        self.clean_df: pd.DataFrame = pd.DataFrame()
        self.dupes_df: pd.DataFrame = pd.DataFrame()

    # ── Normalización de clave de deduplicación ───────────────────────────────

    @staticmethod
    def _normalize_title(title: str) -> str:
        """
        Genera una clave canónica del título para comparación:
          - Minúsculas
          - Elimina caracteres no alfanuméricos
          - Elimina artículos/preposiciones muy cortos (≤ 2 letras)
          - Colapsa espacios
        """
        if not isinstance(title, str):
            return ""
        title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
        title = title.lower()
        title = re.sub(r"[^a-z0-9\s]", " ", title)
        tokens = [w for w in title.split() if len(w) > 2]
        return " ".join(sorted(tokens))  # orden canónico para comparación

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        """Normaliza DOI a lowercase sin trailing slash."""
        if not isinstance(doi, str) or not doi.strip():
            return ""
        doi = doi.strip().lower()
        doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
        return doi.rstrip("/")

    @staticmethod
    def _fuzzy_similarity(a: str, b: str) -> float:
        """Ratio de similitud de secuencias entre dos strings (0–1)."""
        return SequenceMatcher(None, a, b).ratio()

    # ── Pipeline de deduplicación ─────────────────────────────────────────────

    def deduplicate(
        self,
        df: pd.DataFrame | None = None,
        save: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta el pipeline completo de deduplicación en tres pasos:

          Paso 1 — DOI exacto:
            Si dos registros comparten el mismo DOI normalizado (no vacío),
            se considera que son el mismo producto. Se mantiene el primero
            encontrado (fuente de mayor prioridad: OpenAlex > CrossRef > SS).

          Paso 2 — Título canónico:
            Después de filtrar por DOI, se genera una clave canónica del
            título. Si dos registros comparten la misma clave, se elimina
            el duplicado posterior.

          Paso 3 — Similitud fuzzy de título:
            Para los títulos que pasaron los filtros anteriores, se compara
            por pares con SequenceMatcher. Si similitud ≥ 0.92, el segundo
            registro se considera duplicado.

        Args:
            df: DataFrame a deduplicar. Si es None, lo lee desde UNIFIED_CSV.
            save: Si True, guarda los archivos de salida.

        Returns:
            (clean_df, dupes_df) — dataset limpio y dataset de duplicados.
        """
        if df is None:
            if not self.input_path.exists():
                raise FileNotFoundError(
                    f"No se encontró el archivo: {self.input_path}\n"
                    "Ejecuta primero DataFetcher y DataUnifier."
                )
            df = pd.read_csv(self.input_path, encoding="utf-8", low_memory=False)

        total_inicial = len(df)
        logger.info(f"Iniciando deduplicación: {total_inicial} registros de entrada")

        duplicates_log = []  # Acumula todos los registros eliminados

        # ── Paso 1: DOI exacto ────────────────────────────────────────────────
        df["_doi_norm"] = df["doi"].apply(self._normalize_doi)
        mask_doi = df["_doi_norm"] != ""

        doi_dupes_mask = mask_doi & df.duplicated(subset=["_doi_norm"], keep="first")
        doi_dupes = df[doi_dupes_mask].copy()
        doi_dupes["_duplicate_reason"] = "DOI exacto duplicado"
        doi_dupes["_kept_id"] = ""  # Se podría rellenar con el ID del mantenido

        if not doi_dupes.empty:
            duplicates_log.append(doi_dupes)
            df = df[~doi_dupes_mask].copy()
            logger.info(f"  Paso 1 (DOI exacto): eliminados {len(doi_dupes)} duplicados")

        # ── Paso 2: Título canónico ───────────────────────────────────────────
        df["_title_key"] = df["title"].apply(self._normalize_title)
        title_dupes_mask = df.duplicated(subset=["_title_key"], keep="first")
        title_dupes = df[title_dupes_mask].copy()
        title_dupes["_duplicate_reason"] = "Título canónico duplicado"

        if not title_dupes.empty:
            duplicates_log.append(title_dupes)
            df = df[~title_dupes_mask].copy()
            logger.info(f"  Paso 2 (título canónico): eliminados {len(title_dupes)} duplicados")

        # ── Paso 3: Similitud fuzzy (solo si hay registros restantes) ─────────
        df, fuzzy_dupes = self._fuzzy_dedup(df)
        if not fuzzy_dupes.empty:
            duplicates_log.append(fuzzy_dupes)
            logger.info(f"  Paso 3 (fuzzy similarity): eliminados {len(fuzzy_dupes)} duplicados")

        # ── Resultado ─────────────────────────────────────────────────────────
        clean_cols = STANDARD_COLUMNS
        df_clean = df[clean_cols].copy().reset_index(drop=True)

        # Construir DataFrame de duplicados con metadatos adicionales
        if duplicates_log:
            df_dupes = pd.concat(duplicates_log, ignore_index=True)
            # Mantener columnas informativas
            dupe_cols = STANDARD_COLUMNS + ["_duplicate_reason"]
            existing_dupe_cols = [c for c in dupe_cols if c in df_dupes.columns]
            df_dupes = df_dupes[existing_dupe_cols].copy()
        else:
            df_dupes = pd.DataFrame(columns=STANDARD_COLUMNS + ["_duplicate_reason"])

        total_final = len(df_clean)
        total_eliminados = len(df_dupes)
        logger.info(
            f"\n{'─'*50}\n"
            f"  Registros iniciales : {total_inicial}\n"
            f"  Duplicados eliminados: {total_eliminados}\n"
            f"  Registros únicos    : {total_final}\n"
            f"{'─'*50}"
        )

        if save:
            self.output_clean_path.parent.mkdir(parents=True, exist_ok=True)
            df_clean.to_csv(self.output_clean_path, index=False, encoding="utf-8")
            df_dupes.to_csv(self.output_dupes_path, index=False, encoding="utf-8")
            logger.info(f"  Dataset limpio: {self.output_clean_path}")
            logger.info(f"  Duplicados: {self.output_dupes_path}")

        self.clean_df = df_clean
        self.dupes_df = df_dupes
        return df_clean, df_dupes

    def _fuzzy_dedup(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Deduplicación fuzzy por similaridad de título.
        Complejidad: O(n²) — usar solo para datasets ≤ 5000 registros.
        Para datasets grandes, el paso de título canónico ya cubre la mayoría.
        """
        if len(df) > 2000:
            logger.warning(
                "Dataset grande (>2000). El paso fuzzy se omite por rendimiento. "
                "Los pasos 1 y 2 cubrirán la mayoría de duplicados."
            )
            return df, pd.DataFrame(columns=STANDARD_COLUMNS + ["_duplicate_reason"])

        titles = df["_title_key"].tolist()
        indices_to_remove = set()
        fuzzy_dupes = []

        for i in range(len(titles)):
            if i in indices_to_remove:
                continue
            for j in range(i + 1, len(titles)):
                if j in indices_to_remove:
                    continue
                if len(titles[i]) < 5 or len(titles[j]) < 5:
                    continue
                sim = self._fuzzy_similarity(titles[i], titles[j])
                if sim >= self.SIMILARITY_THRESHOLD:
                    indices_to_remove.add(j)

        if indices_to_remove:
            dupes_df = df.iloc[list(indices_to_remove)].copy()
            dupes_df["_duplicate_reason"] = f"Similitud fuzzy de título ≥ {self.SIMILARITY_THRESHOLD}"
            clean_df = df.drop(index=df.index[list(indices_to_remove)]).copy()
        else:
            dupes_df = pd.DataFrame(columns=df.columns)
            clean_df = df.copy()

        return clean_df, dupes_df

    # ── Estadísticas ──────────────────────────────────────────────────────────

    def duplication_stats(self) -> dict:
        """Retorna estadísticas del proceso de deduplicación."""
        if self.clean_df.empty and self.dupes_df.empty:
            return {"error": "Ejecute deduplicate() primero."}

        total = len(self.clean_df) + len(self.dupes_df)
        reason_counts = {}
        if "_duplicate_reason" in self.dupes_df.columns:
            reason_counts = self.dupes_df["_duplicate_reason"].value_counts().to_dict()

        return {
            "total_original": total,
            "total_unique": len(self.clean_df),
            "total_removed": len(self.dupes_df),
            "removal_rate_pct": round(len(self.dupes_df) / total * 100, 2) if total > 0 else 0,
            "duplicates_by_reason": reason_counts,
        }


# ─── Ejecución directa ────────────────────────────────────────────────────────
if __name__ == "__main__":
    dedup = Deduplicator()
    clean, dupes = dedup.deduplicate()
    print("\n✅ Deduplicación completada.")
    print(dedup.duplication_stats())
