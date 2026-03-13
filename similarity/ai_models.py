"""
similarity/ai_models.py — Req 2
Implementación de 2 algoritmos de similitud textual basados en IA.

Modelos:
  5. Sentence-BERT (all-MiniLM-L6-v2)  — modelo de propósito general
  6. Paraphrase-MiniLM-L12-v2          — optimizado para detectar paráfrasis

─────────────────────────────────────────────────────────────────────────
EXPLICACIÓN MATEMÁTICA — ALGORITMOS DE IA
─────────────────────────────────────────────────────────────────────────

══ ALGORITMO 5: Sentence-BERT (all-MiniLM-L6-v2) ════════════════════

Arquitectura base: BERT (Bidirectional Encoder Representations from Transformers)

Paso 1 — Tokenización y Embeddings:
  El texto de entrada se tokeniza usando WordPiece:
    • Se añaden tokens especiales: [CLS] al inicio, [SEP] al final
    • Se trunca/rellena a la longitud máxima del modelo (512 tokens)
    • Cada token i produce 3 embeddings sumados:
        e_total[i] = e_token[i] + e_segment[i] + e_position[i]
    donde cada embedding tiene dimensión d=384 (MiniLM-L6).

Paso 2 — Transformers (Atención Multi-Cabeza):
  El modelo tiene L=6 capas Transformer. En cada capa:
    • Atención multi-cabeza (h cabezas):
        Attention(Q,K,V) = softmax(QKᵀ / √d_k) × V
    • d_k = d_model / h (dimensión por cabeza)
    • Salida: representación contextualizada de cada token

Paso 3 — Mean Pooling (reducción a embedding de oración):
  Para convertir los embeddings de tokens en un único vector de oración:
    v = (Σᵢ tᵢ × mask[i]) / (Σᵢ mask[i])
  donde mask[i]=1 para tokens reales, 0 para tokens de padding.

Paso 4 — Generación de Sentence Embedding:
  El modelo fue entrenado con Siamese Networks (redes siamesas):
    • Toma pares de oraciones (A, B) con etiqueta de similitud
    • La función de pérdida es la cosine embedding loss:
        L = max(0, margin - cos(v_A, v_B)) para pares disímiles
        L = 1 - cos(v_A, v_B)             para pares similares

Paso 5 — Similitud Coseno Final:
  sim_BERT(A, B) = cos(v_A, v_B) = (v_A · v_B) / (‖v_A‖ × ‖v_B‖)

  El resultado se normaliza al rango [0, 1]:
    sim_norm = (cos_score + 1) / 2

Ventajas:
  • Captura semántica profunda (sinónimos, paráfrasis)
  • Contexto bidireccional (cada token entiende a todos los demás)
  • Preentrenado en millones de pares de oraciones

══ ALGORITMO 6: Paraphrase-MiniLM-L12-v2 ════════════════════════════

Misma arquitectura base que el Algoritmo 5 pero con 12 capas (L=12)
y entrenamiento especializado en detección de paráfrasis.

Diferencias clave:
  d_model = 384 (mismo que MiniLM-L6)
  L = 12 capas (vs. 6 del modelo anterior)
  
  Fine-tuning adicional:
    • Microsoft Paraphrase Corpus (MRPC)
    • Quora Question Pairs
    • ParaNMT-50M (pares de paráfrasis generados por traducción)

  Función de pérdida durante fine-tuning:
    L_paraphrase = L_cross_entropy(y_pred, y_true)   # clasificación binaria
                 + α × (1 - cos(v_A, v_B))            # regularización coseno

Proceso de inferencia (idéntico al Alg. 5):
  1. Tokenizar ambos textos → Transformers → Mean Pooling → v_A, v_B
  2. cos(v_A, v_B) → normalizar a [0,1]

Diferencia de aplicación:
  • all-MiniLM-L6-v2: énfasis en recuperación semántica general
  • paraphrase-MiniLM-L12-v2: énfasis en identificar reformulaciones
    del mismo contenido con vocabulario diferente

Ambos modelos son multilingües en la variante "-multilingual-".

Comparación de complejidad:
  Algoritmos 1-4: O(mn) hasta O(N²) en batch — deterministas
  Algoritmos 5-6: O(L × T²) por tokenización (L=capas, T=tokens)
                  → ~ms en GPU, ~5-50 ms en CPU por oración
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from config import SENTENCE_TRANSFORMER_MODEL, PARAPHRASE_MODEL

logger = logging.getLogger(__name__)


class AISimilarity:
    """
    Calcula similitud semántica entre textos usando modelos Sentence-BERT.
    Los modelos se cargan perezosamente (solo cuando se necesitan).

    Uso:
        ai = AISimilarity()
        score = ai.sbert_similarity(text1, text2)
        all_scores = ai.compute_all(text1, text2)
    """

    def __init__(self):
        self._model_sbert = None       # all-MiniLM-L6-v2
        self._model_paraphrase = None  # paraphrase-MiniLM-L12-v2

    # ── Carga perezosa de modelos ─────────────────────────────────────────────

    def _get_sbert_model(self):
        if self._model_sbert is None:
            logger.info(f"Cargando modelo: {SENTENCE_TRANSFORMER_MODEL}")
            from sentence_transformers import SentenceTransformer
            self._model_sbert = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        return self._model_sbert

    def _get_paraphrase_model(self):
        if self._model_paraphrase is None:
            logger.info(f"Cargando modelo: {PARAPHRASE_MODEL}")
            from sentence_transformers import SentenceTransformer
            self._model_paraphrase = SentenceTransformer(PARAPHRASE_MODEL)
        return self._model_paraphrase

    # ── Utilidades ────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Similitud coseno entre dos vectores numpy.
        Normaliza el resultado de [-1, 1] a [0, 1].
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos = float(np.dot(v1, v2) / (norm1 * norm2))
        # Normalizar de [-1,1] a [0,1]
        return round((cos + 1) / 2, 6)

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        """
        Aplica mean pooling sobre los token embeddings de BERT.
        Usa la máscara de atención para ignorar padding.
        """
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    # ── Algoritmo 5: Sentence-BERT ────────────────────────────────────────────

    def sbert_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similitud semántica usando Sentence-BERT (all-MiniLM-L6-v2).
        
        Proceso:
          1. Tokenización → mean pooling → embeddings de 384 dim.
          2. Similitud coseno normalizada a [0, 1].
        """
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            model = self._get_sbert_model()
            embeddings = model.encode([text1, text2], convert_to_numpy=True, normalize_embeddings=True)
            # Con normalize_embeddings=True, el producto punto es directamente el coseno
            cos = float(np.dot(embeddings[0], embeddings[1]))
            return round((cos + 1) / 2, 6)
        except Exception as e:
            logger.error(f"Error en SBERT: {e}")
            return -1.0  # Indica error

    def sbert_similarity_batch(self, texts: list[str]) -> np.ndarray:
        """
        Calcula la matriz de similitud SBERT para una lista de textos.
        Retorna matriz n×n.
        """
        model = self._get_sbert_model()
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        # Con normalized embeddings, la matriz de similitud = dot product
        sim_matrix = np.dot(embeddings, embeddings.T)
        # Normalizar [-1,1] → [0,1]
        sim_matrix = (sim_matrix + 1) / 2
        return sim_matrix.astype(float)

    # ── Algoritmo 6: Paraphrase Detection ────────────────────────────────────

    def paraphrase_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similitud semántica con modelo especializado en paráfrasis.
        Paraphrase-MiniLM-L12-v2: 12 capas, entrenado en MRPC + Quora QP.
        """
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            model = self._get_paraphrase_model()
            embeddings = model.encode([text1, text2], convert_to_numpy=True, normalize_embeddings=True)
            cos = float(np.dot(embeddings[0], embeddings[1]))
            return round((cos + 1) / 2, 6)
        except Exception as e:
            logger.error(f"Error en Paraphrase model: {e}")
            return -1.0

    def paraphrase_similarity_batch(self, texts: list[str]) -> np.ndarray:
        """Matriz de similitud usando el modelo de paráfrasis."""
        model = self._get_paraphrase_model()
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        sim_matrix = np.dot(embeddings, embeddings.T)
        return ((sim_matrix + 1) / 2).astype(float)

    # ── Método unificado ──────────────────────────────────────────────────────

    def compute_all(self, text1: str, text2: str) -> dict[str, float]:
        """
        Calcula los 2 algoritmos IA entre dos textos.
        """
        return {
            "sentence_bert": self.sbert_similarity(text1, text2),
            "paraphrase_miniLM": self.paraphrase_similarity(text1, text2),
        }
