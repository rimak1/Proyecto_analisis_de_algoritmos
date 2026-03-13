"""
nlp/keyword_extractor.py — Req 3
Pipeline NLP para extraer automáticamente hasta 15 nuevas palabras clave
asociadas al tema de IA Generativa, desde los abstracts del corpus.

Algoritmos implementados en pipeline:
  1. YAKE  (Yet Another Keyword Extractor) — método estadístico
  2. KeyBERT — embeddings de BERT para extracción semántica
  3. TF-IDF sobre n-gramas de palabras (baseline estadístico)

Los resultados de los 3 métodos se fusionan mediante votación ponderada.
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import PREDEFINED_TERMS, UNIFIED_CSV

logger = logging.getLogger(__name__)

# Asegurar recursos NLTK
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from nltk.corpus import stopwords

ENGLISH_STOPWORDS = set(stopwords.words("english"))

# Pesos de cada extractor en la fusión
EXTRACTOR_WEIGHTS = {
    "yake": 0.35,
    "keybert": 0.45,
    "tfidf": 0.20,
}


class KeywordExtractor:
    """
    Extrae automáticamente nuevas palabras clave desde el corpus de abstracts.

    Proceso:
      1. Preprocesar y concatenar todos los abstracts
      2. Extraer candidatos con YAKE, KeyBERT y TF-IDF
      3. Fusionar y puntuar con votación ponderada
      4. Filtrar términos ya presentes en PREDEFINED_TERMS
      5. Retornar top-15 nuevas palabras
    """

    def __init__(
        self,
        dataset_path: Path = UNIFIED_CSV,
        max_keywords: int = 15,
        ngram_range: tuple[int, int] = (1, 3),
    ):
        self.dataset_path = dataset_path
        self.max_keywords = max_keywords
        self.ngram_range = ngram_range
        self._df: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.dataset_path, encoding="utf-8")
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        return self.load_data()

    # ── Preprocesamiento ──────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Limpieza básica para NLP."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)        # tags HTML
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)  # solo chars relevantes
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def get_corpus_texts(self, column: str = "abstract") -> list[str]:
        """Retorna lista de abstracts limpios."""
        return [self._clean_text(t) for t in self.df[column].fillna("").tolist()]

    def get_combined_corpus(self, column: str = "abstract") -> str:
        """Combina todos los abstracts en un solo texto."""
        return " ".join(self.get_corpus_texts(column))

    # ── Extractor 1: YAKE ────────────────────────────────────────────────────

    def _extract_yake(self, texts: list[str], top_n: int = 30) -> list[tuple[str, float]]:
        """
        YAKE (Yet Another Keyword Extractor):
          - Método no supervisado basado en estadísticas del texto
          - Propone candidatos por frecuencia, posición y dispersión
          - Score YAKE: menor valor = mayor relevancia
          - Se invierte para normalizar: 1 - score_norm
        """
        try:
            import yake
            kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=self.ngram_range[1],
                dedupLim=0.7,
                top=top_n,
                features=None,
            )
            # YAKE funciona mejor con textos individuales, luego agregar
            all_keywords: Counter = Counter()
            for text in texts:
                if len(text.split()) < 10:
                    continue
                kws = kw_extractor.extract_keywords(text)
                for kw, score in kws:
                    all_keywords[kw.lower().strip()] += (1.0 / (score + 1e-10))

            top = all_keywords.most_common(top_n)
            # Normalizar scores a [0, 1]
            max_score = max(s for _, s in top) if top else 1
            return [(kw, score / max_score) for kw, score in top]

        except ImportError:
            logger.warning("YAKE no instalado. Omitiendo extractor YAKE.")
            return []
        except Exception as e:
            logger.error(f"Error en YAKE: {e}")
            return []

    # ── Extractor 2: KeyBERT ──────────────────────────────────────────────────

    def _extract_keybert(self, texts: list[str], top_n: int = 30) -> list[tuple[str, float]]:
        """
        KeyBERT:
          - Usa embeddings BERT para el texto completo y candidatos
          - Candidatos: n-gramas extraídos del texto
          - Score: similitud coseno entre embedding del candidato y el documento
          - Selecciona top-n candidatos más representativos
        """
        try:
            from keybert import KeyBERT
            kw_model = KeyBERT(model="all-MiniLM-L6-v2")
            corpus = " ".join(texts[:200])  # limitar a 200 abstracts por rendimiento

            kws = kw_model.extract_keywords(
                corpus,
                keyphrase_ngram_range=self.ngram_range,
                stop_words="english",
                top_n=top_n,
                diversity=0.6,  # penaliza términos demasiado similares entre sí
                use_mmr=True,   # Maximal Marginal Relevance
            )
            return [(kw.lower().strip(), score) for kw, score in kws]

        except ImportError:
            logger.warning("KeyBERT no instalado. Omitiendo extractor KeyBERT.")
            return []
        except Exception as e:
            logger.error(f"Error en KeyBERT: {e}")
            return []

    # ── Extractor 3: TF-IDF n-gramas ─────────────────────────────────────────

    def _extract_tfidf(self, texts: list[str], top_n: int = 30) -> list[tuple[str, float]]:
        """
        TF-IDF sobre n-gramas de palabras:
          - Extrae los n-gramas con mayor puntaje TF-IDF promedio en el corpus
          - Filtra stopwords
        """
        if len(texts) < 2:
            return []
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                stop_words="english",
                min_df=2,
                max_df=0.85,
                max_features=500,
                sublinear_tf=True,
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = mean_scores.argsort()[::-1][:top_n]
            max_score = mean_scores[top_indices[0]] if top_indices.size > 0 else 1
            return [
                (feature_names[i].strip(), float(mean_scores[i]) / max_score)
                for i in top_indices
            ]
        except Exception as e:
            logger.error(f"Error en TF-IDF extractor: {e}")
            return []

    # ── Fusión de resultados ──────────────────────────────────────────────────

    def _is_predefined(self, term: str) -> bool:
        """Verifica si un término ya está en los predefined terms."""
        term_lower = term.lower().strip()
        for predefined in PREDEFINED_TERMS:
            if predefined in term_lower or term_lower in predefined:
                return True
        return False

    def _is_valid_term(self, term: str) -> bool:
        """
        Filtra términos no informativos:
          - Mínimo 3 caracteres
          - No solo stopwords
          - No es número puro
          - Al menos un token significativo
        """
        if len(term) < 3:
            return False
        tokens = term.split()
        content_tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 2]
        if not content_tokens:
            return False
        if re.match(r"^\d+$", term.replace(" ", "")):
            return False
        return True

    def _fuse_keywords(
        self,
        yake_kws: list[tuple[str, float]],
        keybert_kws: list[tuple[str, float]],
        tfidf_kws: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """
        Votación ponderada para fusionar los resultados de los 3 extractores.

        Fórmula de score fusionado:
          score_fused(t) = Σ_e [ w_e × score_e(t) ]
          donde w_e es el peso del extractor e (YAKE, KeyBERT, TF-IDF).

        Si un término no aparece en cierto extractor, su aportación es 0.
        """
        fused: dict[str, float] = {}

        for kws, extractor_name in [
            (yake_kws, "yake"),
            (keybert_kws, "keybert"),
            (tfidf_kws, "tfidf"),
        ]:
            weight = EXTRACTOR_WEIGHTS[extractor_name]
            for term, score in kws:
                term = term.lower().strip()
                if not self._is_valid_term(term) or self._is_predefined(term):
                    continue
                fused[term] = fused.get(term, 0.0) + weight * score

        # Ordenar por score descendente
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    # ── Extracción principal ──────────────────────────────────────────────────

    def extract(self, column: str = "abstract") -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de extracción de palabras clave.

        Returns:
            DataFrame con columnas: [rank, keyword, fused_score, yake_score,
                                     keybert_score, tfidf_score, extractors_count]
        """
        texts = self.get_corpus_texts(column)
        texts = [t for t in texts if len(t.split()) > 20]  # solo abstracts sustanciales

        logger.info(f"Extrayendo keywords de {len(texts)} abstracts...")

        # Ejecutar los 3 extractores
        yake_kws = self._extract_yake(texts, top_n=40)
        keybert_kws = self._extract_keybert(texts, top_n=40)
        tfidf_kws = self._extract_tfidf(texts, top_n=40)

        # Crear dicts de referencia para incluir scores por extractor en el resultado
        yake_dict = dict(yake_kws)
        keybert_dict = dict(keybert_kws)
        tfidf_dict = dict(tfidf_kws)

        # Fusionar
        fused = self._fuse_keywords(yake_kws, keybert_kws, tfidf_kws)

        # Construir DataFrame final
        rows = []
        for rank, (term, score) in enumerate(fused[: self.max_keywords], start=1):
            rows.append({
                "rank": rank,
                "keyword": term,
                "fused_score": round(score, 6),
                "yake_score": round(yake_dict.get(term, 0.0), 6),
                "keybert_score": round(keybert_dict.get(term, 0.0), 6),
                "tfidf_score": round(tfidf_dict.get(term, 0.0), 6),
                "extractors_count": sum([
                    term in yake_dict,
                    term in keybert_dict,
                    term in tfidf_dict,
                ]),
            })

        result_df = pd.DataFrame(rows)
        logger.info(f"Extraídas {len(result_df)} nuevas palabras clave.")
        return result_df
