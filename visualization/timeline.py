"""
visualization/timeline.py — Req 5.3
Línea temporal interactiva de publicaciones filtrada por año y revista.
Implementada con Plotly para interactividad completa dentro de Streamlit.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import UNIFIED_CSV

logger = logging.getLogger(__name__)


class PublicationTimeline:
    """
    Genera líneas de tiempo interactivas de publicaciones.

    Funciones:
      - Timeline agregada por año
      - Timeline filtrada por revista/journal
      - Timeline acumulada (curva S de productividad)
      - Comparación por base de datos de origen
    """

    def __init__(self, dataset_path: Path = UNIFIED_CSV):
        self.dataset_path = dataset_path
        self._df: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_csv(self.dataset_path, encoding="utf-8")
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df[df["year"].notna() & (df["year"] >= 2000) & (df["year"] <= 2030)]
            df["year"] = df["year"].astype(int)
            self._df = df
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        return self.load_data()

    def get_available_journals(self) -> list[str]:
        """Retorna lista de revistas disponibles para el filtro."""
        journals = (
            self.df["journal"]
            .fillna("")
            .unique()
            .tolist()
        )
        return sorted([j for j in journals if j.strip()])

    def get_year_range(self) -> tuple[int, int]:
        """Retorna el rango de años disponible en el dataset."""
        years = self.df["year"].dropna()
        return int(years.min()), int(years.max())

    def plot_annual_count(
        self,
        year_start: int | None = None,
        year_end: int | None = None,
        selected_journals: list[str] | None = None,
        group_by_source: bool = False,
    ) -> go.Figure:
        """
        Genera línea temporal de publicaciones por año.

        Args:
            year_start: Año inicial del filtro.
            year_end: Año final del filtro.
            selected_journals: Lista de revistas a incluir (None = todas).
            group_by_source: Sí True, descompone por base de datos de origen.
        """
        df = self.df.copy()

        # Filtros
        if year_start:
            df = df[df["year"] >= year_start]
        if year_end:
            df = df[df["year"] <= year_end]
        if selected_journals:
            df = df[df["journal"].isin(selected_journals)]

        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Sin datos para los filtros seleccionados",
                paper_bgcolor="#0E1117",
                font_color="white",
            )
            return fig

        if group_by_source:
            agg = (
                df.groupby(["year", "source_db"])
                .size()
                .reset_index(name="count")
            )
            fig = px.line(
                agg,
                x="year",
                y="count",
                color="source_db",
                markers=True,
                title="Publicaciones por Ano — Generative AI (por Fuente)",
                labels={"year": "Ano", "count": "N. Publicaciones", "source_db": "Fuente"},
                color_discrete_sequence=["#008236","#FFD100","#DC3545","#6C757D","#4FC3F7"],
            )
        else:
            agg = (
                df.groupby("year")
                .size()
                .reset_index(name="count")
            )
            # Curva acumulada
            agg["cumulative"] = agg["count"].cumsum()

            # Dos trazas: barras + línea acumulada
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=agg["year"],
                y=agg["count"],
                name="Por ano",
                marker_color="#008236",
                opacity=0.85,
            ))

            fig.add_trace(go.Scatter(
                x=agg["year"],
                y=agg["cumulative"],
                mode="lines+markers",
                name="Acumulado",
                line=dict(color="#FFD100", width=2),
                yaxis="y2",
            ))

            fig.update_layout(
                title=dict(
                    text="Linea Temporal de Publicaciones — Generative AI",
                    font_size=15,
                ),
                yaxis=dict(title="Publicaciones por Ano", color="#212529"),
                yaxis2=dict(
                    title="Total Acumulado",
                    overlaying="y",
                    side="right",
                    color="#FFD100",
                ),
                legend=dict(x=0.02, y=0.95, bgcolor="rgba(255,255,255,0.85)"),
            )

        # Estilo global oscuro
        fig.update_layout(
            paper_bgcolor="#F8F9FA",
            plot_bgcolor="#ffffff",
            font=dict(color="#212529", family="Inter, sans-serif"),
            xaxis=dict(
                title="Ano",
                gridcolor="#E0E0E0",
                tickmode="linear",
                tick0=df["year"].min(),
                dtick=1,
            ),
            hovermode="x unified",
        )

        return fig

    def plot_journal_comparison(self, top_n: int = 10) -> go.Figure:
        """
        Muestra un gráfico de barras con los top-N journals por número de publicaciones.
        """
        df = self.df.copy()
        journal_counts = (
            df[df["journal"].fillna("") != ""]
            .groupby("journal")
            .size()
            .reset_index(name="count")
            .nlargest(top_n, "count")
            .sort_values("count")
        )

        if journal_counts.empty:
            fig = go.Figure()
            fig.update_layout(title="Sin datos de revistas disponibles")
            return fig

        fig = px.bar(
            journal_counts,
            x="count",
            y="journal",
            orientation="h",
            color="count",
            color_continuous_scale=[[0, "#d4edda"], [0.5, "#28a745"], [1, "#008236"]],
            title=f"Top {top_n} Revistas por Numero de Publicaciones",
            labels={"count": "Publicaciones", "journal": "Revista"},
        )

        fig.update_layout(
            paper_bgcolor="#F8F9FA",
            plot_bgcolor="#ffffff",
            font=dict(color="#212529", family="Inter, sans-serif"),
            showlegend=False,
            yaxis=dict(autorange=True),
        )

        return fig
