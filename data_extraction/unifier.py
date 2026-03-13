"""
data_extraction/unifier.py — Req 1
Unifica los DataFrames de múltiples fuentes en un único dataset CSV maestro.
Normaliza columnas, limpia texto y produce el archivo unified_dataset.csv.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, STANDARD_COLUMNS, UNIFIED_CSV

logger = logging.getLogger(__name__)


class DataUnifier:
    """
    Combina datasets de distintas fuentes bibliográficas en un único CSV.

    Flujo:
      1. Cargar CSVs crudos desde data/raw/ (o recibir DataFrames directamente)
      2. Normalizar columnas al esquema estándar
      3. Limpiar texto (abstracts, títulos, autores)
      4. Concatenar todos en un DataFrame maestro
      5. Guardar en data/processed/unified_dataset.csv
    """

    def __init__(self, raw_dir: Path = RAW_DIR, output_path: Path = UNIFIED_CSV):
        self.raw_dir = raw_dir
        self.output_path = output_path

    # ── Carga ─────────────────────────────────────────────────────────────────

    def load_raw_csvs(self) -> dict[str, pd.DataFrame]:
        """Lee todos los CSVs crudos desde data/raw/."""
        dfs = {}
        for csv_file in self.raw_dir.glob("*_raw.csv"):
            source_name = csv_file.stem.replace("_raw", "")
            df = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)
            dfs[source_name] = df
            logger.info(f"Cargado {csv_file.name}: {len(df)} registros")
        return dfs

    # ── Normalización ─────────────────────────────────────────────────────────

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalización básica de texto:
        - Elimina espacios múltiples
        - Convierte a escape de caracteres Unicode -> ASCII equivalente
        - Elimina caracteres de control
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        # Normalizar caracteres Unicode a su forma canónica
        text = unicodedata.normalize("NFKC", text)
        # Eliminar caracteres de control
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
        # Colapsar espacios múltiples
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def normalize_year(year_val) -> int | None:
        """Convierte el año a entero, retorna None si no es válido."""
        try:
            year = int(float(str(year_val)))
            if 1900 <= year <= 2100:
                return year
        except (ValueError, TypeError):
            pass
        return None

    def normalize_dataframe(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Asegura que el DataFrame tiene exactamente las columnas estándar
        y aplica limpieza de texto a todos sus campos.
        """
        # Agregar columnas faltantes
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        # Seleccionar solo columnas estándar
        df = df[STANDARD_COLUMNS].copy()

        # Limpiar texto
        for col in ["title", "abstract", "keywords", "authors", "journal", "country"]:
            df[col] = df[col].apply(self.normalize_text)

        # Normalizar año
        df["year"] = df["year"].apply(self.normalize_year)

        # Asegurar que source_db está marcado
        df["source_db"] = df["source_db"].fillna(source_name).replace("", source_name)

        # Filtrar registros sin título ni abstract (no aportan información)
        before = len(df)
        df = df[df["title"].str.len() > 3].copy()
        after = len(df)
        if before != after:
            logger.info(f"  Eliminados {before - after} registros sin título útil en {source_name}")

        return df

    # ── Concatenación ─────────────────────────────────────────────────────────

    def unify(
        self,
        dataframes: dict[str, pd.DataFrame] | None = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Carga (o recibe) DataFrames de múltiples fuentes, los normaliza y une.

        Args:
            dataframes: Si se proporciona, use estos DataFrames directamente.
                        Si es None, lee los CSVs crudos de data/raw/.
            save: Si True, guarda el resultado en UNIFIED_CSV.

        Returns:
            DataFrame unificado con todos los registros de todas las fuentes.
        """
        if dataframes is None:
            dataframes = self.load_raw_csvs()

        if not dataframes:
            logger.error("No hay datos para unificar. Ejecute primero el fetcher.")
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        normalized_dfs = []
        for source_name, df in dataframes.items():
            logger.info(f"Normalizando fuente: {source_name} ({len(df)} registros)")
            norm_df = self.normalize_dataframe(df.copy(), source_name)
            normalized_dfs.append(norm_df)

        unified = pd.concat(normalized_dfs, ignore_index=True)
        logger.info(f"Total antes de deduplicación: {len(unified)} registros")

        if save:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            unified.to_csv(self.output_path, index=False, encoding="utf-8")
            logger.info(f"Dataset unificado guardado en: {self.output_path}")

        return unified


# ─── Ejecución directa ────────────────────────────────────────────────────────
if __name__ == "__main__":
    unifier = DataUnifier()
    df = unifier.unify()
    print(f"\n✅ Unificación completada: {len(df)} registros en el dataset.")
    print(df[["title", "source_db", "year"]].head(10))
