"""
similarity/interface.py — Req 2
Interfaz unificada que orquesta los 6 algoritmos de similitud
y permite seleccionar artículos desde el dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import UNIFIED_CSV
from similarity.classical import ClassicalSimilarity
from similarity.ai_models import AISimilarity

logger = logging.getLogger(__name__)


ALGORITHM_DESCRIPTIONS = {
    "levenshtein": {
        "name": "Distancia de Levenshtein",
        "category": "Clásico",
        "description": (
            "Mide el mínimo número de operaciones de edición "
            "(inserción, eliminación, sustitución) necesarias para transformar "
            "un texto en otro. Opera sobre caracteres individuales."
        ),
        "range": "[0, 1] — 1=idénticos, 0=completamente diferentes",
        "complexity": "O(m × n)",
    },
    "jaccard": {
        "name": "Similaridad de Jaccard",
        "category": "Clásico",
        "description": (
            "Compara los conjuntos de tokens. Calcula la razón entre la "
            "intersección y la unión de los vocabularios de ambos textos."
        ),
        "range": "[0, 1]",
        "complexity": "O(|A| + |B|)",
    },
    "tfidf_cosine": {
        "name": "Coseno TF-IDF",
        "category": "Clásico / Estadístico",
        "description": (
            "Convierte los textos en vectores TF-IDF (ponderando términos "
            "por su frecuencia e inversamente por su frecuencia en documentos) "
            "y calcula el ángulo coseno entre los vectores resultantes."
        ),
        "range": "[0, 1]",
        "complexity": "O(N×M) vectorización",
    },
    "ngram_bigram": {
        "name": "N-grama (bigramas de caracteres)",
        "category": "Clásico",
        "description": (
            "Extrae bigramas de caracteres de ambos textos y calcula el "
            "coeficiente de Dice sobre los multiconjuntos resultantes. "
            "Captura similitudes morfológicas y resistente a typos."
        ),
        "range": "[0, 1]",
        "complexity": "O(m + n)",
    },
    "sentence_bert": {
        "name": "Sentence-BERT (all-MiniLM-L6-v2)",
        "category": "IA / Transformers",
        "description": (
            "Codifica los textos en embeddings densos de 384 dimensiones "
            "usando un modelo Transformer de 6 capas con mean pooling. "
            "Captura semántica profunda incluyendo sinónimos y paráfrasis."
        ),
        "range": "[0, 1] (normalizado de coseno)",
        "complexity": "O(L × T²) por inferencia",
    },
    "paraphrase_miniLM": {
        "name": "Paraphrase-MiniLM-L12-v2",
        "category": "IA / Transformers",
        "description": (
            "Modelo Transformer de 12 capas fine-tuned específicamente para "
            "detección de paráfrasis (MRPC + Quora QP). Especializado en "
            "identificar reformulaciones semánticamente equivalentes."
        ),
        "range": "[0, 1] (normalizado de coseno)",
        "complexity": "O(L × T²) por inferencia",
    },
}


class SimilarityInterface:
    """
    Interfaz de alto nivel para:
      1. Cargar el dataset de artículos
      2. Seleccionar artículos por índice o ID
      3. Extraer sus abstracts
      4. Calcular los 6 algoritmos de similitud de forma individual o batch
    """

    def __init__(self, dataset_path: Path = UNIFIED_CSV):
        self.dataset_path = dataset_path
        self._df: pd.DataFrame | None = None
        self._classical = ClassicalSimilarity()
        self._ai = AISimilarity()

    # ── Carga del dataset ─────────────────────────────────────────────────────

    def load_dataset(self) -> pd.DataFrame:
        """Carga el dataset unificado."""
        if self._df is None:
            if not self.dataset_path.exists():
                raise FileNotFoundError(
                    f"Dataset no encontrado en: {self.dataset_path}\n"
                    "Por favor ejecute primero la extracción de datos."
                )
            self._df = pd.read_csv(self.dataset_path, encoding="utf-8")
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        return self.load_dataset()

    def get_article_list(self) -> list[dict]:
        """
        Retorna una lista de artículos con id, título y extracto del abstract.
        Útil para poblar un selector en la UI.
        """
        return [
            {
                "index": i,
                "id": row.get("id", i),
                "title": row.get("title", "Sin título")[:120],
                "year": row.get("year", ""),
                "journal": row.get("journal", ""),
                "abstract_preview": (str(row.get("abstract", "")) or "")[:200] + "...",
            }
            for i, row in self.df.iterrows()
        ]

    # ── Extracción de abstracts ───────────────────────────────────────────────

    def get_abstract(self, index: int) -> str:
        """Obtiene el abstract de un artículo por su índice en el DataFrame."""
        if index < 0 or index >= len(self.df):
            raise IndexError(f"Índice {index} fuera de rango.")
        abstract = str(self.df.iloc[index].get("abstract", "") or "")
        return abstract.strip()

    def get_selected_abstracts(self, indices: list[int]) -> dict[int, str]:
        """
        Retorna los abstracts de los artículos en los índices especificados.

        Args:
            indices: Lista de índices de artículos.

        Returns:
            Dict {índice: abstract}
        """
        return {i: self.get_abstract(i) for i in indices}

    # ── Cálculo de similitud ──────────────────────────────────────────────────

    def compute_similarity_pair(
        self,
        index_a: int,
        index_b: int,
        algorithms: list[str] | None = None,
    ) -> dict:
        """
        Calcula la similitud entre dos artículos usando los algoritmos
        especificados (por defecto, todos los 6).

        Args:
            index_a: Índice del primer artículo.
            index_b: Índice del segundo artículo.
            algorithms: Lista de nombres de algoritmos a usar.
                        None = todos los 6.

        Returns:
            Dict con metadata y scores por algoritmo.
        """
        abs_a = self.get_abstract(index_a)
        abs_b = self.get_abstract(index_b)

        row_a = self.df.iloc[index_a]
        row_b = self.df.iloc[index_b]

        available_algos = {
            "levenshtein": lambda: self._classical.levenshtein_similarity(abs_a, abs_b),
            "jaccard": lambda: self._classical.jaccard_similarity(abs_a, abs_b),
            "tfidf_cosine": lambda: self._classical.tfidf_cosine_similarity(abs_a, abs_b),
            "ngram_bigram": lambda: self._classical.ngram_similarity(abs_a, abs_b),
            "sentence_bert": lambda: self._ai.sbert_similarity(abs_a, abs_b),
            "paraphrase_miniLM": lambda: self._ai.paraphrase_similarity(abs_a, abs_b),
        }

        selected = algorithms or list(available_algos.keys())
        scores = {}
        for algo in selected:
            if algo in available_algos:
                try:
                    scores[algo] = available_algos[algo]()
                except Exception as e:
                    logger.error(f"Error en algoritmo {algo}: {e}")
                    scores[algo] = -1.0

        return {
            "article_a": {"index": index_a, "title": row_a.get("title", ""), "abstract": abs_a},
            "article_b": {"index": index_b, "title": row_b.get("title", ""), "abstract": abs_b},
            "scores": scores,
            "algorithm_info": {
                algo: ALGORITHM_DESCRIPTIONS.get(algo, {}) for algo in selected
            },
        }

    def compute_similarity_matrix(
        self,
        indices: list[int],
        algorithm: str = "tfidf_cosine",
    ) -> pd.DataFrame:
        """
        Calcula la matriz de similitud n×n para un conjunto de artículos.

        Args:
            indices: Lista de índices de artículos.
            algorithm: Algoritmo a usar.

        Returns:
            DataFrame n×n con índices y columnas siendo los títulos cortos.
        """
        abstracts = [self.get_abstract(i) for i in indices]
        titles = [str(self.df.iloc[i].get("title", f"Art. {i}"))[:50] for i in indices]
        n = len(abstracts)

        if algorithm == "tfidf_cosine":
            import numpy as np
            matrix = self._classical.tfidf_cosine_batch(abstracts)
        elif algorithm == "sentence_bert":
            matrix = self._ai.sbert_similarity_batch(abstracts)
        elif algorithm == "paraphrase_miniLM":
            matrix = self._ai.paraphrase_similarity_batch(abstracts)
        else:
            # Calcular par a par para algoritmos sin batch
            import numpy as np
            matrix = np.zeros((n, n))
            algo_fn = {
                "levenshtein": self._classical.levenshtein_similarity,
                "jaccard": self._classical.jaccard_similarity,
                "ngram_bigram": self._classical.ngram_similarity,
            }.get(algorithm, self._classical.jaccard_similarity)

            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        matrix[i][j] = 1.0
                    else:
                        score = algo_fn(abstracts[i], abstracts[j])
                        matrix[i][j] = score
                        matrix[j][i] = score

        return pd.DataFrame(matrix, index=titles, columns=titles)

    @staticmethod
    def get_algorithm_descriptions() -> dict:
        """Retorna la descripción completa de todos los algoritmos."""
        return ALGORITHM_DESCRIPTIONS
