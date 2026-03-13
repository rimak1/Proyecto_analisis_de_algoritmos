"""
nlp/frequency.py — Req 3
Calcula la frecuencia de aparición de 15 términos predefinidos en los abstracts
de la categoría "Concepts of Generative AI in Education".

Los términos son buscados como frases exactas (case-insensitive) con un
algoritmo de ventana deslizante sobre los tokens del abstract para capturar
tanto formas exactas como variaciones con/sin guión.
"""

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from config import PREDEFINED_TERMS, UNIFIED_CSV

# Variantes alternativas de los términos predefinidos (aliases)
TERM_ALIASES: dict[str, list[str]] = {
    "generative models": ["generative model", "generative modeling", "generative modelling"],
    "prompting": ["prompt engineering", "prompt", "prompts"],
    "machine learning": ["ml", "machine-learning"],
    "multimodality": ["multimodal", "multi-modal", "multi-modality"],
    "fine-tuning": ["fine tuning", "finetuning", "finetune"],
    "training data": ["training dataset", "training corpus", "training set"],
    "algorithmic bias": ["algorithm bias", "ai bias", "model bias"],
    "explainability": ["explainable ai", "xai", "model explainability"],
    "transparency": ["model transparency", "transparent ai"],
    "ethics": ["ethical", "ai ethics", "ethical ai"],
    "privacy": ["data privacy", "privacy preserving"],
    "personalization": ["personalized learning", "personalisation", "personalized"],
    "human-ai interaction": ["human-computer interaction", "human ai collaboration", "hci"],
    "ai literacy": ["digital literacy", "ai skills"],
    "co-creation": ["co creation", "human-ai co-creation", "collaborative creation"],
}


class TermFrequencyAnalyzer:
    """
    Analiza la frecuencia de los 15 términos de IA Generativa en los abstracts.

    Métricas calculadas:
      • Frecuencia absoluta: número total de ocurrencias del término
      • Frecuencia documental: número de artículos que contienen el término
      • Frecuencia relativa: ocurrencias / total de tokens en el corpus
      • TF promedio: media de TF por artículo
      • IDF implícito: log(N / df)
    """

    def __init__(
        self,
        dataset_path: Path = UNIFIED_CSV,
        terms: list[str] | None = None,
    ):
        self.dataset_path = dataset_path
        self.terms = terms or PREDEFINED_TERMS
        self._df: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.dataset_path, encoding="utf-8")
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        return self.load_data()

    # ── Utilitarios de matching ────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normaliza texto para búsqueda: minúsculas, sin puntuación extra."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Preservar guiones dentro de palabras compuestas
        text = re.sub(r"(?<=[a-z])-(?=[a-z])", "-", text)
        # Eliminar puntuación excepto guiones
        text = text.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
        return text

    def _count_term_in_text(self, text: str, term: str) -> int:
        """
        Cuenta ocurrencias de un término (y sus aliases) en un texto.
        Usa regex con word boundaries para evitar falsos positivos.
        """
        text = self._clean_text(text)
        all_variants = [term] + TERM_ALIASES.get(term, [])
        count = 0
        for variant in all_variants:
            variant_clean = self._clean_text(variant)
            # Escapar caracteres especiales en el patrón
            pattern = r'\b' + re.escape(variant_clean) + r'\b'
            count += len(re.findall(pattern, text))
        return count

    # ── Análisis principal ────────────────────────────────────────────────────

    def compute_frequencies(self, text_column: str = "abstract") -> pd.DataFrame:
        """
        Calcula las métricas de frecuencia de los 15 términos predefinidos.

        Returns:
            DataFrame con una fila por término y columnas:
            [term, absolute_freq, document_freq, relative_freq, tf_mean, idf, tf_idf_mean]
        """
        import math
        df = self.df.copy()
        texts = df[text_column].fillna("").tolist()
        N = len(texts)
        total_tokens = sum(len(t.split()) for t in texts)

        results = []
        for term in self.terms:
            # Frecuencia absoluta (todas las ocurrencias en todo el corpus)
            abs_freq = sum(self._count_term_in_text(t, term) for t in texts)

            # Frecuencia documental (artículos que lo contienen ≥ 1 vez)
            doc_freq = sum(1 for t in texts if self._count_term_in_text(t, term) > 0)

            # Frecuencia relativa (por total de tokens)
            relative_freq = abs_freq / total_tokens if total_tokens > 0 else 0

            # TF promedio por artículo
            tfs = []
            for t in texts:
                cnt = self._count_term_in_text(t, term)
                tokens_in_doc = len(t.split()) if t.strip() else 1
                tfs.append(cnt / tokens_in_doc)
            tf_mean = sum(tfs) / len(tfs) if tfs else 0

            # IDF
            idf = math.log(N / (1 + doc_freq)) if doc_freq > 0 else 0

            results.append({
                "term": term,
                "absolute_freq": abs_freq,
                "document_freq": doc_freq,
                "pct_documents": round(doc_freq / N * 100, 2) if N > 0 else 0,
                "relative_freq": round(relative_freq, 8),
                "tf_mean": round(tf_mean, 8),
                "idf": round(idf, 4),
                "tf_idf_mean": round(tf_mean * idf, 8),
            })

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("absolute_freq", ascending=False).reset_index(drop=True)
        result_df["rank"] = result_df.index + 1
        return result_df

    def frequency_per_article(
        self, text_column: str = "abstract"
    ) -> pd.DataFrame:
        """
        Retorna una matriz artículo × término con las frecuencias absolutas.
        Útil para análisis de co-ocurrencia.
        """
        df = self.df.copy()
        data = {}
        for term in self.terms:
            data[term] = [
                self._count_term_in_text(str(row.get(text_column, "")), term)
                for _, row in df.iterrows()
            ]
        result = pd.DataFrame(data, index=df.get("title", df.index))
        return result

    def co_occurrence_matrix(self, text_column: str = "abstract") -> pd.DataFrame:
        """
        Genera una matriz de co-ocurrencia simétrica entre términos.
        co_mat[i,j] = número de artículos donde aparecen juntos los términos i y j.
        """
        df = self.df.copy()
        texts = df[text_column].fillna("").tolist()
        n = len(self.terms)

        matrix = [[0] * n for _ in range(n)]
        for text in texts:
            presence = [self._count_term_in_text(text, t) > 0 for t in self.terms]
            for i in range(n):
                for j in range(i, n):
                    if presence[i] and presence[j]:
                        matrix[i][j] += 1
                        if i != j:
                            matrix[j][i] += 1

        return pd.DataFrame(matrix, index=self.terms, columns=self.terms)
