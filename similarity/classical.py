"""
similarity/classical.py — Req 2
Implementación de 4 algoritmos clásicos de similitud textual.

Algoritmos:
  1. Distancia de Levenshtein (edición) → similitud normalizada
  2. Similitud de Jaccard sobre conjuntos de tokens
  3. Similitud Coseno con vectorización TF-IDF
  4. Similitud N-grama (bigramas de caracteres)

─────────────────────────────────────────────────────────────────────────
EXPLICACIÓN MATEMÁTICA DETALLADA DE CADA ALGORITMO
─────────────────────────────────────────────────────────────────────────

══ ALGORITMO 1: Distancia de Levenshtein (Edit Distance) ══════════════

Definición:
  La distancia de Levenshtein d(s, t) entre dos strings s y t es el
  número mínimo de operaciones elementales requeridas para transformar
  s en t. Las operaciones son:
    • Inserción   de un carácter
    • Eliminación de un carácter
    • Sustitución de un carácter por otro

Fórmula (programación dinámica):
  Sea m = len(s), n = len(t)
  d[0,j] = j  para j=0..n
  d[i,0] = i  para i=0..m

  Para i=1..m, j=1..n:
    d[i,j] = d[i-1,j-1]                    si s[i] == t[j]  (sin costo)
            = 1 + min(                       en otro caso:
                d[i-1,j],                    # eliminación
                d[i,j-1],                    # inserción
                d[i-1,j-1]                   # sustitución
              )

Conversión a similitud (0–1):
  sim_lev(s,t) = 1 - d(s,t) / max(len(s), len(t))

Complejidad: O(m × n) tiempo, O(m × n) espacio (reducible a O(min(m,n)))

Interpretación:
  • sim = 1.0 → cadenas idénticas
  • sim = 0.0 → no comparten ningún carácter en posición
  Se usa a nivel de carácter, por lo tanto es sensible a typos y
  variaciones ortográficas.

══ ALGORITMO 2: Similitud de Jaccard ══════════════════════════════════

Definición:
  Mide la similitud entre dos conjuntos como la razón entre la
  intersección y la unión.

  Sea A = conjunto de tokens de s, B = conjunto de tokens de t

Fórmula:
  J(A, B) = |A ∩ B| / |A ∪ B|

  donde:
    |A ∩ B| = número de tokens comunes
    |A ∪ B| = número de tokens totales únicos en ambos conjuntos

Rango: [0, 1]  — 0 = completamente diferentes, 1 = idénticos

Variante de distancia de Jaccard:
  d_J(A, B) = 1 - J(A, B)

Propiedades:
  • Simétrica: J(A,B) = J(B,A)
  • Invariante al orden de los tokens
  • No considera frecuencia de términos
  • Ideal para comparar vocabulario compartido

Complejidad: O(|A| + |B|)

══ ALGORITMO 3: Similitud Coseno con TF-IDF ══════════════════════════

Paso 1 — TF (Term Frequency):
  TF(t, d) = frecuencia del término t en el documento d
             / número total de términos en d

Paso 2 — IDF (Inverse Document Frequency):
  IDF(t, D) = log(N / (1 + df(t)))
  donde N = número total de documentos, df(t) = documentos que contienen t
  (el "+1" evita división por cero — suavizado de Laplace)

Paso 3 — TF-IDF:
  TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

  Esto genera un vector de pesos para cada documento en el espacio
  de todos los términos del corpus.

Paso 4 — Similitud Coseno:
  cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
            = Σᵢ(aᵢ × bᵢ) / √(Σᵢaᵢ²) × √(Σᵢbᵢ²)

  donde A, B son los vectores TF-IDF de los documentos.

Rango: [0, 1]  (ya que TF-IDF ≥ 0)

Interpretación:
  • cos = 1 → documentos paralelos (mismo ángulo en el espacio)
  • cos = 0 → documentos ortogonales (sin términos TF-IDF en común)
  Usa la **dirección** del vector, no su magnitud → resistente a longitud.

Complejidad: O(N × M) vectorización, O(M) por par de similitud

══ ALGORITMO 4: Similitud N-grama ════════════════════════════════════

Definición:
  Un n-grama es una subsecuencia contigua de n caracteres (char-ngrams)
  o n palabras (word-ngrams). Aquí usamos bigramas de caracteres (n=2).

Proceso:
  1. Extraer todos los bigramas de caracteres de s y t:
     Ex: "hello" → {"he", "el", "ll", "lo"}
  2. Aplicar similitud de Dice sobre los multiconjuntos:

Fórmula (Dice de bigramas):
  Dice_ngram(s, t) = 2 × |bigrams(s) ∩ bigrams(t)|
                     / (|bigrams(s)| + |bigrams(t)|)

  donde la intersección es de multiconjunto (cuenta ocurrencias).

Alternativa (Jaccard sobre n-gramas):
  J_ngram(s,t) = |bigrams(s) ∩ bigrams(t)| / |bigrams(s) ∪ bigrams(t)|

Rango: [0, 1]

Ventajas sobre Jaccard de palabras:
  • Captura similitud morfológica (raíces, sufijos)
  • Resistente a caracteres extra o typos
  • Útil para texto corto

Complejidad: O(m + n) para extracción de bigramas
"""

from __future__ import annotations

import re
import string
from collections import Counter
from math import log, sqrt
from typing import Sequence

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from Levenshtein import distance as lev_distance  # type: ignore
    _HAS_LEVENSHTEIN_LIB = True
except ImportError:
    _HAS_LEVENSHTEIN_LIB = False


# ─── Preprocesamiento básico ──────────────────────────────────────────────────

def _preprocess(text: str, lowercase: bool = True, remove_punct: bool = True) -> str:
    """Limpia y normaliza texto para comparaciones."""
    if not isinstance(text, str):
        return ""
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return [w for w in _preprocess(text).split() if len(w) > 1]


# ─── Clase principal ──────────────────────────────────────────────────────────

class ClassicalSimilarity:
    """
    Computa los 4 algoritmos clásicos de similitud textual.

    Uso:
        cs = ClassicalSimilarity()
        score = cs.levenshtein_similarity(text1, text2)
        all_scores = cs.compute_all(text1, text2)
    """

    # ── 1. Levenshtein ────────────────────────────────────────────────────────

    def levenshtein_similarity(self, s: str, t: str) -> float:
        """
        Similitud basada en distancia de edición (Levenshtein).
        Retorna valor normalizado entre 0 y 1.
        """
        s = _preprocess(s)
        t = _preprocess(t)
        if not s and not t:
            return 1.0
        if not s or not t:
            return 0.0

        if _HAS_LEVENSHTEIN_LIB:
            dist = lev_distance(s, t)
        else:
            dist = self._levenshtein_dp(s, t)

        max_len = max(len(s), len(t))
        return 1.0 - dist / max_len

    @staticmethod
    def _levenshtein_dp(s: str, t: str) -> int:
        """
        Implementación pura en Python de la distancia de Levenshtein
        mediante programación dinámica con optimización de memoria O(min(m,n)).
        """
        m, n = len(s), len(t)
        if m < n:
            s, t, m, n = t, s, n, m  # Asegurar m >= n

        prev = list(range(n + 1))
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                cost = 0 if s[i - 1] == t[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,      # eliminación
                    curr[j - 1] + 1,  # inserción
                    prev[j - 1] + cost  # sustitución
                )
            prev, curr = curr, prev

        return prev[n]

    # ── 2. Jaccard ────────────────────────────────────────────────────────────

    def jaccard_similarity(self, s: str, t: str) -> float:
        """
        Similitud de Jaccard sobre conjuntos de tokens.
        J(A,B) = |A∩B| / |A∪B|
        """
        tokens_s = set(_tokenize(s))
        tokens_t = set(_tokenize(t))

        if not tokens_s and not tokens_t:
            return 1.0
        if not tokens_s or not tokens_t:
            return 0.0

        intersection = len(tokens_s & tokens_t)
        union = len(tokens_s | tokens_t)
        return intersection / union

    # ── 3. TF-IDF Cosine ─────────────────────────────────────────────────────

    def tfidf_cosine_similarity(self, s: str, t: str) -> float:
        """
        Similitud coseno con representación TF-IDF.
        Eleva ambos textos a vectores TF-IDF y calcula el coseno del ángulo.
        """
        docs = [_preprocess(s), _preprocess(t)]
        if not docs[0].strip() or not docs[1].strip():
            return 0.0

        vectorizer = TfidfVectorizer(
            min_df=1,
            ngram_range=(1, 2),  # unigramas + bigramas de palabras
            sublinear_tf=True,   # aplica log(1 + TF) para suavizar frecuencias altas
        )
        try:
            matrix = vectorizer.fit_transform(docs)
            score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        except ValueError:
            score = 0.0
        return round(score, 6)

    @staticmethod
    def tfidf_cosine_batch(texts: list[str]) -> np.ndarray:
        """
        Calcula la matriz de similitud coseno TF-IDF para una lista de textos.
        Retorna matriz n×n de similitudes.
        Útil para el módulo de clustering.
        """
        processed = [_preprocess(t) for t in texts]
        vectorizer = TfidfVectorizer(
            min_df=1,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(processed)
        return cosine_similarity(matrix).astype(float)

    # ── 4. N-grama (bigramas de caracteres) ──────────────────────────────────

    def ngram_similarity(self, s: str, t: str, n: int = 2) -> float:
        """
        Similitud basada en n-gramas de caracteres (por defecto bigramas).
        Usa el coeficiente de Dice sobre multiconjuntos de n-gramas.

        Dice(s,t) = 2 × |bigrams(s) ∩ bigrams(t)| / (|bigrams(s)| + |bigrams(t)|)
        """
        s = _preprocess(s)
        t = _preprocess(t)

        def get_ngrams(text: str, n: int) -> Counter:
            return Counter(text[i:i+n] for i in range(len(text) - n + 1))

        ngrams_s = get_ngrams(s, n)
        ngrams_t = get_ngrams(t, n)

        total_s = sum(ngrams_s.values())
        total_t = sum(ngrams_t.values())

        if total_s == 0 and total_t == 0:
            return 1.0
        if total_s == 0 or total_t == 0:
            return 0.0

        # Intersección de multiconjunto
        intersection = sum((ngrams_s & ngrams_t).values())

        # Coeficiente de Dice
        dice = 2 * intersection / (total_s + total_t)
        return round(dice, 6)

    # ── Método unificado ──────────────────────────────────────────────────────

    def compute_all(self, s: str, t: str) -> dict[str, float]:
        """
        Calcula los 4 algoritmos clásicos entre dos textos.

        Returns:
            dict con scores de cada algoritmo (valores entre 0 y 1).
        """
        return {
            "levenshtein": round(self.levenshtein_similarity(s, t), 6),
            "jaccard": round(self.jaccard_similarity(s, t), 6),
            "tfidf_cosine": round(self.tfidf_cosine_similarity(s, t), 6),
            "ngram_bigram": round(self.ngram_similarity(s, t, n=2), 6),
        }
