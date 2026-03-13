"""
clustering/evaluator.py — Req 4
Evaluación automatizada de los 3 algoritmos de clustering jerárquico.

Métricas implementadas:
  1. Coeficiente de correlación cofenética (CCC) — mide qué tan bien
     el dendrograma preserva las distancias originales.
  2. Índice de silueta — mide cohesión y separación de los clusters.
  3. Índice de Calinski-Harabasz — razón de varianza inter/intraclúster.

La evaluación es completamente automática (determina el mejor método sin
intervención del usuario).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler

from config import CLUSTERING_METHODS, DEFAULT_N_CLUSTERS

logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """
    Evalúa automáticamente cuál de los 3 métodos de clustering jerárquico
    produce los agrupamientos más coherentes.

    Métricas de evaluación:
    ─────────────────────────────────────────────────────────────────────
    1. COEFICIENTE DE CORRELACIÓN COFENÉTICA (CCC):
       - Mide la correlación de Pearson entre la matriz de distancias
         original y la matriz cofenética (distancias del dendrograma).
       - CCC = corr(d_orig, d_coph)
       - Rango: [-1, 1] — valores > 0.75 indican buen ajuste
       - Interpretación: un CCC alto indica que el dendrograma
         representa fielmente la estructura de similitud del corpus.

    2. ÍNDICE DE SILUETA:
       - Para cada punto i, la silueta es:
           s(i) = (b(i) - a(i)) / max(a(i), b(i))
         donde:
           a(i) = distancia media a otros puntos del MISMO cluster
           b(i) = distancia media mínima al cluster MÁS CERCANO diferente
       - El índice de silueta global = media de s(i) para todos los puntos
       - Rango: [-1, 1] — valores > 0.5 son buenos
       - Selección automática del número óptimo de clusters: se prueba
         n ∈ {3, 4, 5, 6, 7} y se elige el que maximiza la silueta.

    3. ÍNDICE DE CALINSKI-HARABASZ (CH):
       - CH = [SS_B / (k-1)] / [SS_W / (n-k)]
         donde:
           SS_B = suma de cuadrados entre clusters (dispersión interclúster)
           SS_W = suma de cuadrados dentro de clusters (dispersión intraclúster)
           k    = número de clusters
           n    = número de puntos
       - Valores más altos son mejores (sin límite superior).
       - No requiere especificar n_clusters de antemano.

    Score compuesto de selección:
       final_score = 0.40 × CCC_norm + 0.35 × silhouette_norm + 0.25 × CH_norm
    ─────────────────────────────────────────────────────────────────────
    """

    # Pesos de cada métrica en el score compuesto
    METRIC_WEIGHTS = {
        "cophenetic": 0.40,
        "silhouette": 0.35,
        "calinski_harabasz": 0.25,
    }

    def __init__(
        self,
        feature_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        labels: list[str],
        n_clusters_range: list[int] | None = None,
    ):
        """
        Args:
            feature_matrix: Matriz densa n×m (para silueta y CH).
            distance_matrix: Matriz n×n de distancias (para CCC).
            labels: Etiquetas de cada documento.
            n_clusters_range: Rangos de n_clusters a evaluar.
        """
        self.feature_matrix = feature_matrix
        self.distance_matrix = distance_matrix
        self.labels = labels
        self.n_clusters_range = n_clusters_range or [3, 4, 5, 6, 7]
        self._results: dict[str, dict] = {}

    # ── Métrica 1: Coeficiente Cofenético ─────────────────────────────────────

    def cophenetic_correlation(self, Z: np.ndarray) -> float:
        """
        Calcula el coeficiente de correlación cofenética.
        CCC = correlación de Pearson(d_orig, d_coph)

        Args:
            Z: Matriz de linkage scipy.

        Returns:
            CCC en [-1, 1].
        """
        condensed = squareform(self.distance_matrix, checks=False)
        c, _ = cophenet(Z, condensed)
        return round(float(c), 6)

    # ── Métrica 2: Índice de Silueta ──────────────────────────────────────────

    def best_silhouette(self, Z: np.ndarray) -> tuple[float, int]:
        """
        Calcula el índice de silueta óptimo probando múltiples n_clusters.

        Returns:
            (mejor_silhouette, mejor_n_clusters)
        """
        from scipy.cluster.hierarchy import fcluster

        best_sil = -1.0
        best_n = self.n_clusters_range[0]

        for n in self.n_clusters_range:
            if n >= len(self.labels):
                continue
            cluster_labels = fcluster(Z, t=n, criterion="maxclust")
            if len(set(cluster_labels)) < 2:
                continue
            try:
                sil = silhouette_score(
                    self.distance_matrix,
                    cluster_labels,
                    metric="precomputed",
                )
                if sil > best_sil:
                    best_sil = sil
                    best_n = n
            except Exception as e:
                logger.warning(f"Silueta n={n}: {e}")

        return round(float(best_sil), 6), best_n

    # ── Métrica 3: Calinski-Harabász ──────────────────────────────────────────

    def best_calinski_harabasz(self, Z: np.ndarray) -> tuple[float, int]:
        """
        Calcula el índice Calinski-Harabász óptimo.

        Returns:
            (mejor_CH, mejor_n_clusters)
        """
        from scipy.cluster.hierarchy import fcluster

        best_ch = -1.0
        best_n = self.n_clusters_range[0]

        for n in self.n_clusters_range:
            if n >= len(self.labels):
                continue
            cluster_labels = fcluster(Z, t=n, criterion="maxclust")
            if len(set(cluster_labels)) < 2:
                continue
            try:
                # CH requiere la matriz de características, no de distancias
                ch = calinski_harabasz_score(self.feature_matrix, cluster_labels)
                if ch > best_ch:
                    best_ch = ch
                    best_n = n
            except Exception as e:
                logger.warning(f"CH n={n}: {e}")

        return round(float(best_ch), 6), best_n

    # ── Evaluación completa ───────────────────────────────────────────────────

    def evaluate(self, linkage_matrices: dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Evalúa automáticamente los 3 métodos de clustering.

        Args:
            linkage_matrices: Dict {method_name: Z_matrix}

        Returns:
            DataFrame con métricas por método, ordenado por score compuesto
            (el mejor método aparece primero).
        """
        rows = []
        for method, Z in linkage_matrices.items():
            logger.info(f"Evaluando método: {method}...")

            ccc = self.cophenetic_correlation(Z)
            sil, best_n_sil = self.best_silhouette(Z)
            ch, best_n_ch = self.best_calinski_harabasz(Z)

            rows.append({
                "method": method,
                "cophenetic_correlation": ccc,
                "silhouette_score": sil,
                "optimal_n_silhouette": best_n_sil,
                "calinski_harabasz": ch,
                "optimal_n_ch": best_n_ch,
            })

        if not rows:
            return pd.DataFrame()

        results_df = pd.DataFrame(rows)

        # ── Normalización min-max de cada métrica para score compuesto ─────────
        scaler = MinMaxScaler()
        metrics_to_normalize = ["cophenetic_correlation", "silhouette_score", "calinski_harabasz"]

        for col in metrics_to_normalize:
            vals = results_df[col].values.reshape(-1, 1)
            if vals.max() != vals.min():
                results_df[f"{col}_norm"] = scaler.fit_transform(vals).flatten()
            else:
                results_df[f"{col}_norm"] = 0.5  # empate

        # Score compuesto
        w = self.METRIC_WEIGHTS
        results_df["composite_score"] = (
            results_df["cophenetic_correlation_norm"] * w["cophenetic"]
            + results_df["silhouette_score_norm"] * w["silhouette"]
            + results_df["calinski_harabasz_norm"] * w["calinski_harabasz"]
        ).round(6)

        results_df = results_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        results_df["rank"] = results_df.index + 1
        results_df["best"] = results_df["rank"] == 1

        self._results = results_df.to_dict("records")
        return results_df

    def best_method(self) -> str:
        """Retorna el nombre del mejor método según el score compuesto."""
        if not self._results:
            raise RuntimeError("Ejecute evaluate() primero.")
        return self._results[0]["method"]

    def report(self) -> str:
        """Genera un texto resumen de la evaluación."""
        if not self._results:
            return "Evaluación no ejecutada aún."

        lines = [
            "=" * 60,
            "EVALUACIÓN AUTOMÁTICA DE ALGORITMOS DE CLUSTERING",
            "=" * 60,
        ]
        for r in self._results:
            lines.append(
                f"\n{'★ MEJOR' if r.get('best') else '  '} {r['method'].upper()}\n"
                f"  CCC:              {r['cophenetic_correlation']:.4f}\n"
                f"  Silueta:          {r['silhouette_score']:.4f} (n óptimo: {r['optimal_n_silhouette']})\n"
                f"  Calinski-Harabász:{r['calinski_harabasz']:.2f}\n"
                f"  Score compuesto:  {r['composite_score']:.4f}"
            )
        lines.append("\n" + "=" * 60)
        lines.append(f"CONCLUSIÓN: El mejor algoritmo es '{self.best_method().upper()}'")
        lines.append("=" * 60)
        return "\n".join(lines)
