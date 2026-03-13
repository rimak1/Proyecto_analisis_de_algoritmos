"""
clustering/preprocessor.py — Req 4
Preprocesamiento de abstracts para clustering jerárquico.

Pipeline:
  1. Tokenización y limpieza (NLTK)
  2. Eliminación de stopwords
  3. Stemming (SnowballStemmer) o Lematización opcional
  4. Vectorización TF-IDF
  5. Reducción de dimensionalidad (TruncatedSVD / LSA) opcional
  6. Cálculo de matrices de distancia (1 - similitud coseno)
"""

from __future__ import annotations

import logging
import re
import string
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cophenet

from config import UNIFIED_CSV

logger = logging.getLogger(__name__)

# Recursos NLTK
for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STEMMER = SnowballStemmer("english")
STOP_WORDS = set(stopwords.words("english"))


class ClusteringPreprocessor:
    """
    Preprocesa abstracts y genera matrices de distancia para clustering.

    Atributos públicos (disponibles después de fit()):
        texts:          lista de abstracts procesados (strings limpios)
        tfidf_matrix:   matriz TF-IDF sparse (n_docs × n_features)
        feature_names:  array de nombres de características TF-IDF
        distance_matrix: matriz condensada de distancias (para scipy)
        labels:         títulos cortos de los artículos
    """

    def __init__(
        self,
        dataset_path: Path = UNIFIED_CSV,
        use_stemming: bool = True,
        n_components: int = 0,  # 0 = sin reducción SVD
        max_features: int = 2000,
        min_df: int = 2,
    ):
        self.dataset_path = dataset_path
        self.use_stemming = use_stemming
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df

        # Resultados públicos
        self.texts: list[str] = []
        self.tfidf_matrix = None
        self.feature_names: np.ndarray | None = None
        self.distance_matrix: np.ndarray | None = None
        self.labels: list[str] = []
        self._df: pd.DataFrame | None = None

    def load_data(self, n_docs: int | None = None) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_csv(self.dataset_path, encoding="utf-8")
            # Filtrar artículos con abstract sustancial
            df = df[df["abstract"].fillna("").str.len() > 100].copy()
            if n_docs:
                df = df.head(n_docs)
            self._df = df
        return self._df

    # ── Limpieza de texto ─────────────────────────────────────────────────────

    def _clean_token(self, token: str) -> str:
        """Limpia un token individual."""
        token = token.lower()
        token = token.strip(string.punctuation)
        return token

    def _preprocess_text(self, text: str) -> str:
        """
        Pipeline de limpieza completo para un abstract:
          1. Minúsculas
          2. Eliminar caracteres no alfanuméricos (preservar espacios)
          3. Tokenizar
          4. Eliminar stopwords y tokens cortos
          5. Stemming opcional (SnowballStemmer)
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1-2. Limpiar
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)  # HTML
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # 3. Tokenizar
        tokens = text.split()

        # 4. Filtrar
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

        # 5. Stemming
        if self.use_stemming:
            tokens = [STEMMER.stem(t) for t in tokens]

        return " ".join(tokens)

    # ── Vectorización ─────────────────────────────────────────────────────────

    def fit(self, n_docs: int | None = None) -> "ClusteringPreprocessor":
        """
        Ejecuta todo el pipeline de preprocesamiento.

        Args:
            n_docs: Número máximo de documentos a procesar.

        Returns:
            self (para encadenamiento)
        """
        df = self.load_data(n_docs)
        logger.info(f"Preprocesando {len(df)} abstracts...")

        # Textos procesados
        self.texts = [
            self._preprocess_text(str(row.get("abstract", "")))
            for _, row in df.iterrows()
        ]
        self.labels = [
            str(row.get("title", f"Art. {i}"))[:60]
            for i, (_, row) in enumerate(df.iterrows())
        ]

        # Filtrar abstracts vacíos
        valid_mask = [len(t) > 10 for t in self.texts]
        self.texts = [t for t, v in zip(self.texts, valid_mask) if v]
        self.labels = [l for l, v in zip(self.labels, valid_mask) if v]

        if len(self.texts) < 3:
            raise ValueError("Se necesitan al menos 3 abstracts válidos para clustering.")

        # TF-IDF
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=max(1, self.min_df if len(self.texts) > 10 else 1),
            sublinear_tf=True,
        )
        self.tfidf_matrix = vectorizer.fit_transform(self.texts)
        self.feature_names = vectorizer.get_feature_names_out()

        # Reducción de dimensionalidad SVD (LSA) opcional
        if self.n_components > 0 and self.n_components < self.tfidf_matrix.shape[1]:
            svd = TruncatedSVD(n_components=self.n_components, random_state=42)
            matrix_dense = svd.fit_transform(self.tfidf_matrix)
        else:
            matrix_dense = self.tfidf_matrix.toarray()

        # Matriz de distancia = 1 - similitud coseno
        sim_matrix = cosine_similarity(matrix_dense)
        # Asegurar valores en [0, 1] (puede haber pequeñas imprecisiones numéricas)
        sim_matrix = np.clip(sim_matrix, 0, 1)
        np.fill_diagonal(sim_matrix, 1.0)
        self.distance_matrix = 1.0 - sim_matrix

        logger.info(
            f"Preprocesamiento completado. Shape: {self.tfidf_matrix.shape}. "
            f"Labels: {len(self.labels)}"
        )
        return self

    def get_condensed_distance(self) -> np.ndarray:
        """
        Retorna la matriz de distancias en formato condensado (vector 1D)
        requerido por scipy.cluster.hierarchy.linkage.
        """
        if self.distance_matrix is None:
            raise RuntimeError("Ejecute fit() primero.")
        return squareform(self.distance_matrix, checks=False)
