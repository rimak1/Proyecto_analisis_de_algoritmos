"""
clustering/algorithms.py — Req 4
Implementación de 3 algoritmos de agrupamiento jerárquico con dendrogramas.

Algoritmos:
  1. Ward Linkage          — minimiza la varianza intraclúster
  2. Complete Linkage      — enlace máximo (peor caso)
  3. Average Linkage       — UPGMA (Unweighted Pair Group Method with Arithmetic Mean)

─────────────────────────────────────────────────────────────────────────
EXPLICACIÓN MATEMÁTICA DE CADA ALGORITMO
─────────────────────────────────────────────────────────────────────────

══ WARD LINKAGE ══════════════════════════════════════════════════════

Criterio: minimizar el incremento de la suma de cuadrados intraclúster
al fusionar dos grupos.

Sea D²(C_i, C_j) la "distancia de Ward" entre los clusters C_i y C_j:

  D²_ward(C_i, C_j) = (n_i × n_j) / (n_i + n_j) × ‖μ_i - μ_j‖²

  donde:
    n_i, n_j  = número de elementos en cada cluster
    μ_i, μ_j  = centroide de cada cluster
    ‖·‖²      = norma euclídea al cuadrado

Fórmula de actualización (Lance-Williams):
  D(C_k, C_i∪C_j) = [(n_k+n_i)/(n_k+n_i+n_j)] D(C_k,C_i)
                   + [(n_k+n_j)/(n_k+n_i+n_j)] D(C_k,C_j)
                   - [n_k/(n_k+n_i+n_j)] D(C_i,C_j)

Propiedad: produce clusters compactos y de tamaños equilibrados.
Sensible a outliers.

══ COMPLETE LINKAGE (Enlace Máximo) ══════════════════════════════════

Criterio: la distancia entre dos clusters es la distancia máxima
entre cualquier par de elementos de clusters distintos.

  D_comp(C_i, C_j) = max { d(x, y) : x ∈ C_i, y ∈ C_j }

Fórmula de actualización (Lance-Williams):
  D(C_k, C_i∪C_j) = max( D(C_k, C_i), D(C_k, C_j) )

Propiedad: tiende a producir clusters compactos y "esféricos".
Favorece clusters de tamaño similar. Robusto a ruido.

══ AVERAGE LINKAGE (UPGMA) ═══════════════════════════════════════════

Criterio: la distancia entre dos clusters es el promedio de todas
las distancias entre pares de elementos de clusters distintos.

  D_avg(C_i, C_j) = (1/(n_i × n_j)) Σ_{x∈C_i} Σ_{y∈C_j} d(x,y)

Fórmula de actualización (Lance-Williams):
  D(C_k, C_i∪C_j) = [n_i/(n_i+n_j)] D(C_k,C_i) + [n_j/(n_i+n_j)] D(C_k,C_j)

Propiedad: compromiso entre single y complete linkage.
UPGMA es el más usado en bibliometría por su estabilidad.

══ PROCESO GENERAL DE CLUSTERING AGLOMERATIVO ═══════════════════════

Algoritmo general (complejidad O(n³) naive → O(n² log n) con heap):
  1. Inicializar: cada elemento es un cluster {x_i}
  2. Calcular matriz de distancias D
  3. Repetir hasta n-1 fusiones:
     a. Encontrar el par (C_i, C_j) con mínima distancia en D
     b. Fusionar en nuevo cluster C_{ij}
     c. Actualizar D usando la fórmula de Lance-Williams del criterio
     d. Registrar la fusión para el dendrograma
  4. Construir dendrograma desde el registro de fusiones
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")  # Backend no-interactivo para Streamlit
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import (
    average,
    complete,
    dendrogram,
    fcluster,
    linkage,
    ward,
)
from scipy.spatial.distance import squareform

from config import CLUSTERING_METHODS

logger = logging.getLogger(__name__)

LinkageMethod = Literal["ward", "complete", "average"]


class HierarchicalClustering:
    """
    Aplica los 3 algoritmos de clustering jerárquico y genera dendrogramas.

    Uso:
        hc = HierarchicalClustering(distance_matrix, labels)
        Z_ward = hc.fit("ward")
        fig = hc.plot_dendrogram(Z_ward, "ward")
        cluster_labels = hc.cut_tree(Z_ward, n_clusters=5)
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        labels: list[str],
    ):
        """
        Args:
            distance_matrix: Matriz cuadrada n×n de distancias.
            labels: Etiquetas de cada documento (títulos cortos).
        """
        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("distance_matrix debe ser una matriz cuadrada n×n.")

        self.distance_matrix = distance_matrix
        self.labels = labels
        self.n = len(labels)
        self._linkage_matrices: dict[str, np.ndarray] = {}

    # ── Formato condensado ────────────────────────────────────────────────────

    def _get_condensed(self) -> np.ndarray:
        """Retorna la matriz de distancias en formato condensado scipy."""
        return squareform(self.distance_matrix, checks=False)

    # ── Ajuste de linkage ──────────────────────────────────────────────────────

    def fit(self, method: LinkageMethod = "ward") -> np.ndarray:
        """
        Calcula la matriz de linkage para el método especificado.

        Args:
            method: "ward", "complete" o "average"

        Returns:
            Matriz de linkage Z (numpy array n-1 × 4):
              Columnas: [cluster_1, cluster_2, distance, size_of_new_cluster]
        """
        if method in self._linkage_matrices:
            return self._linkage_matrices[method]

        condensed = self._get_condensed()

        if method == "ward":
            # Ward requiere distancias euclídeas; usamos la condensada directamente
            Z = linkage(condensed, method="ward", metric="euclidean", optimal_ordering=True)
        elif method == "complete":
            Z = linkage(condensed, method="complete", optimal_ordering=True)
        elif method == "average":
            Z = linkage(condensed, method="average", optimal_ordering=True)
        else:
            raise ValueError(f"Método no reconocido: {method}. Use: ward, complete, average")

        self._linkage_matrices[method] = Z
        logger.info(f"Linkage '{method}' calculado. Shape: {Z.shape}")
        return Z

    def fit_all(self) -> dict[str, np.ndarray]:
        """Calcula la matriz de linkage para los 3 métodos."""
        return {method: self.fit(method) for method in CLUSTERING_METHODS}

    # ── Dendrogramas ──────────────────────────────────────────────────────────

    def plot_dendrogram(
        self,
        Z: np.ndarray,
        method: str,
        n_clusters: int | None = None,
        figsize: tuple[int, int] | None = None,
        color_threshold: float | None = None,
        max_labels: int = 40,
    ) -> plt.Figure:
        """
        Genera una figura matplotlib del dendrograma.

        Args:
            Z: Matriz de linkage.
            method: Nombre del método (para el título).
            n_clusters: Si se especifica, dibuja línea de corte para n clusters.
            figsize: Tamaño de la figura.
            color_threshold: Umbral de color personalizado.
            max_labels: Máximo de etiquetas a mostrar.

        Returns:
            Figura matplotlib.
        """
        n_docs = len(self.labels)
        height = max(8, min(n_docs // 3, 25))
        fig_width = max(12, min(n_docs * 0.3, 30))
        fig, ax = plt.subplots(figsize=figsize or (fig_width, height))

        # Truncar etiquetas si hay demasiadas
        show_labels = n_docs <= max_labels
        label_rotation = 90 if n_docs > 20 else 45

        method_names = {
            "ward": "Ward (Varianza Mínima)",
            "complete": "Complete Linkage (Enlace Máximo)",
            "average": "Average Linkage (UPGMA)",
        }

        # Color threshold por defecto: 70% de la distancia máxima
        if color_threshold is None:
            color_threshold = 0.7 * np.max(Z[:, 2])

        dn = dendrogram(
            Z,
            labels=self.labels if show_labels else None,
            ax=ax,
            color_threshold=color_threshold,
            leaf_rotation=label_rotation,
            leaf_font_size=max(5, 11 - n_docs // 10),
            above_threshold_color="grey",
        )

        # Línea de corte para n_clusters
        if n_clusters and n_clusters > 1:
            # La altura de corte es la distancia del (n_clusters-1)-ésimo paso
            cut_height = Z[-(n_clusters - 1), 2]
            ax.axhline(
                y=cut_height,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Corte: {n_clusters} clusters",
            )
            ax.legend(fontsize=9)

        ax.set_title(
            f"Dendrograma — {method_names.get(method, method)}\n"
            f"(n={n_docs} artículos — Dominio: Generative AI)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Artículos", fontsize=10)
        ax.set_ylabel("Distancia", fontsize=10)

        # Estética
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#1c1e26")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        plt.tight_layout()
        return fig

    def plot_all_dendrograms(
        self, n_clusters: int | None = None
    ) -> dict[str, plt.Figure]:
        """Genera los 3 dendrogramas en un solo llamado."""
        figures = {}
        for method in CLUSTERING_METHODS:
            Z = self.fit(method)
            figures[method] = self.plot_dendrogram(Z, method, n_clusters=n_clusters)
        return figures

    # ── Corte del árbol ───────────────────────────────────────────────────────

    def cut_tree(self, Z: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Genera etiquetas de cluster cortando el dendrograma en n_clusters grupos.

        Returns:
            Array de enteros con la etiqueta de cluster de cada documento.
        """
        return fcluster(Z, t=n_clusters, criterion="maxclust")

    def get_cluster_summary(
        self, Z: np.ndarray, method: str, n_clusters: int
    ) -> pd.DataFrame:
        """
        Retorna un DataFrame con cada artículo y su cluster asignado.
        """
        import pandas as pd
        cluster_labels = self.cut_tree(Z, n_clusters)
        return pd.DataFrame({
            "title": self.labels,
            "cluster": cluster_labels,
            "method": method,
        }).sort_values("cluster")
