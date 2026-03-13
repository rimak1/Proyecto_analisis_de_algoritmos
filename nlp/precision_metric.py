"""
nlp/precision_metric.py — Req 3
Métrica de precisión para evaluar la calidad de las palabras clave extraídas.

La precisión se estima mediante 4 enfoques complementarios:
  1. Cobertura semántica (KeyBERT score): similitud vs. corpus
  2. Consenso multi-extractor: ¿aparece en YAKE, KeyBERT y TF-IDF?
  3. Relevancia de dominio: proximidad semántica al query principal
  4. Specificity (IDF-based): términos muy frecuentes en todo texto tienen
     menor puntuación de especificidad

Score final: promedio ponderado de los 4 indicadores → [0, 1]
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DEFAULT_QUERY, UNIFIED_CSV

logger = logging.getLogger(__name__)

# Pesos de cada indicador en el score final
METRIC_WEIGHTS = {
    "semantic_relevance": 0.40,  # similitud con el corpus (KeyBERT-like)
    "extractor_consensus": 0.25,  # aparece en múltiples extractores
    "domain_proximity": 0.20,    # proximidad semántica al query de dominio
    "specificity": 0.15,         # IDF-based specificity
}


class PrecisionMetric:
    """
    Evalúa la precisión de las palabras clave extraídas por el NLP pipeline.

    Uso:
        pm = PrecisionMetric()
        scored_df = pm.evaluate(extracted_keywords_df)
        print(pm.summary(scored_df))
    """

    def __init__(
        self,
        dataset_path: Path = UNIFIED_CSV,
        domain_query: str = DEFAULT_QUERY,
    ):
        self.dataset_path = dataset_path
        self.domain_query = domain_query
        self._df: pd.DataFrame | None = None
        self._corpus_texts: list[str] | None = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None

    def load_data(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.dataset_path, encoding="utf-8")
        return self._df

    def get_corpus(self) -> list[str]:
        if self._corpus_texts is None:
            df = self.load_data()
            self._corpus_texts = df["abstract"].fillna("").apply(str.lower).tolist()
        return self._corpus_texts

    def _build_tfidf(self) -> tuple[TfidfVectorizer, any]:
        if self._tfidf_vectorizer is None:
            corpus = self.get_corpus()
            self._tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                stop_words="english",
                min_df=1,
                max_df=0.9,
                sublinear_tf=True,
            )
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)
        return self._tfidf_vectorizer, self._tfidf_matrix

    # ── Indicador 1: Relevancia semántica ─────────────────────────────────────

    def _semantic_relevance(self, keyword: str) -> float:
        """
        Mide la similitud coseno TF-IDF entre el keyword y el corpus completo.
        Ratio de documentos del corpus que tienen TF-IDF > 0 para el keyword.
        """
        try:
            vectorizer, matrix = self._build_tfidf()
            kw_vec = vectorizer.transform([keyword.lower()])
            sims = cosine_similarity(kw_vec, matrix).flatten()
            # Proporción de documentos con similitud > 0
            non_zero = np.sum(sims > 0)
            coverage = non_zero / len(sims) if len(sims) > 0 else 0
            mean_sim = float(np.mean(sims[sims > 0])) if non_zero > 0 else 0
            return round((coverage * 0.6 + mean_sim * 0.4), 6)
        except Exception:
            return 0.0

    # ── Indicador 2: Consenso multi-extractor ─────────────────────────────────

    @staticmethod
    def _extractor_consensus(row: pd.Series) -> float:
        """
        Número de extractores que identificaron el término / total extractores.
        extractors_count es 1, 2 o 3.
        """
        count = int(row.get("extractors_count", 1))
        return count / 3.0

    # ── Indicador 3: Proximidad al dominio ────────────────────────────────────

    def _domain_proximity(self, keyword: str) -> float:
        """
        Calcula similitud coseno TF-IDF entre el keyword y el query de dominio
        ('generative artificial intelligence').
        Mide qué tan relacionado está el término con el dominio central.
        """
        try:
            vectorizer, _ = self._build_tfidf()
            vecs = vectorizer.transform([keyword.lower(), self.domain_query.lower()])
            sim = float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])
            return round(min(sim * 5, 1.0), 6)  # escalar porque es naturalmente bajo
        except Exception:
            return 0.0

    # ── Indicador 4: Especificidad (IDF-based) ────────────────────────────────

    def _specificity(self, keyword: str) -> float:
        """
        Calcula la especificidad del término usando su IDF en el corpus.
        Términos muy generales tienen IDF bajo → especificidad baja.

        specificity = IDF_normalizado = IDF(t) / log(N)
        """
        try:
            vectorizer, _ = self._build_tfidf()
            vocab = vectorizer.vocabulary_
            idf_values = vectorizer.idf_
            N = len(self.get_corpus())

            # Buscar el keyword en vocabulario del vectorizador
            kw_lower = keyword.lower()
            if kw_lower in vocab:
                idf = idf_values[vocab[kw_lower]]
            else:
                # Para términos fuera del vocabulario, calcular manualmente
                count = sum(1 for t in self.get_corpus() if kw_lower in t)
                idf = math.log(N / (1 + count)) if count > 0 else math.log(N)

            max_idf = math.log(N) if N > 1 else 1
            return round(min(idf / max_idf, 1.0), 6)
        except Exception:
            return 0.5

    # ── Score de precisión compuesto ─────────────────────────────────────────

    def evaluate(self, keywords_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el score de precisión para cada palabra clave extraída.

        Args:
            keywords_df: DataFrame con columnas: keyword, fused_score,
                         extractors_count, etc. (output de KeywordExtractor)

        Returns:
            DataFrame enriquecido con columnas de precisión:
              [semantic_relevance, extractor_consensus, domain_proximity,
               specificity, precision_score, precision_grade]
        """
        if keywords_df.empty:
            return keywords_df.copy()

        result = keywords_df.copy()

        sem_rel = []
        ext_con = []
        dom_prox = []
        spec = []

        for _, row in result.iterrows():
            kw = str(row.get("keyword", ""))
            sem_rel.append(self._semantic_relevance(kw))
            ext_con.append(self._extractor_consensus(row))
            dom_prox.append(self._domain_proximity(kw))
            spec.append(self._specificity(kw))

        result["semantic_relevance"] = sem_rel
        result["extractor_consensus"] = ext_con
        result["domain_proximity"] = dom_prox
        result["specificity"] = spec

        # Score final ponderado
        w = METRIC_WEIGHTS
        result["precision_score"] = (
            result["semantic_relevance"] * w["semantic_relevance"]
            + result["extractor_consensus"] * w["extractor_consensus"]
            + result["domain_proximity"] * w["domain_proximity"]
            + result["specificity"] * w["specificity"]
        ).round(6)

        # Grado cualitativo
        def grading(s: float) -> str:
            if s >= 0.75:
                return "🟢 Alta"
            elif s >= 0.50:
                return "🟡 Media"
            elif s >= 0.25:
                return "🟠 Baja"
            else:
                return "🔴 Muy baja"

        result["precision_grade"] = result["precision_score"].apply(grading)
        result = result.sort_values("precision_score", ascending=False).reset_index(drop=True)
        result["rank"] = result.index + 1

        return result

    def summary(self, evaluated_df: pd.DataFrame) -> dict:
        """Retorna un resumen estadístico de la evaluación de precisión."""
        if evaluated_df.empty or "precision_score" not in evaluated_df.columns:
            return {}
        scores = evaluated_df["precision_score"]
        grades = evaluated_df["precision_grade"].value_counts().to_dict()
        return {
            "n_keywords": len(evaluated_df),
            "mean_precision": round(float(scores.mean()), 4),
            "std_precision": round(float(scores.std()), 4),
            "min_precision": round(float(scores.min()), 4),
            "max_precision": round(float(scores.max()), 4),
            "grade_distribution": grades,
            "metric_weights": METRIC_WEIGHTS,
        }
