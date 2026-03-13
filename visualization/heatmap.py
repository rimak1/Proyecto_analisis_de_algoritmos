"""
visualization/heatmap.py — Req 5.1
Mapa de calor geográfico que muestra la distribución por país del primer autor.

Implementación:
  - Usa Plotly Choropleth para el mapa coroplético mundial (si países disponibles)
  - Fallback: Plotly Bar chart horizontal cuando los países no son ISO-3166
  - Integración con Streamlit via st.plotly_chart()
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import UNIFIED_CSV

logger = logging.getLogger(__name__)

# Mapa de nombre de país → código ISO-3166 alpha-3
# (subset de los más comunes en publicaciones científicas)
COUNTRY_TO_ISO3: dict[str, str] = {
    "us": "USA", "usa": "USA", "united states": "USA", "united states of america": "USA",
    "cn": "CHN", "china": "CHN",
    "gb": "GBR", "uk": "GBR", "united kingdom": "GBR",
    "de": "DEU", "germany": "DEU",
    "au": "AUS", "australia": "AUS",
    "ca": "CAN", "canada": "CAN",
    "in": "IND", "india": "IND",
    "kr": "KOR", "south korea": "KOR", "korea": "KOR",
    "es": "ESP", "spain": "ESP",
    "fr": "FRA", "france": "FRA",
    "it": "ITA", "italy": "ITA",
    "nl": "NLD", "netherlands": "NLD",
    "br": "BRA", "brazil": "BRA",
    "jp": "JPN", "japan": "JPN",
    "sg": "SGP", "singapore": "SGP",
    "se": "SWE", "sweden": "SWE",
    "ch": "CHE", "switzerland": "CHE",
    "pt": "PRT", "portugal": "PRT",
    "mx": "MEX", "mexico": "MEX",
    "ar": "ARG", "argentina": "ARG",
    "co": "COL", "colombia": "COL",
    "cl": "CHL", "chile": "CHL",
    "gr": "GRC", "greece": "GRC",
    "pl": "POL", "poland": "POL",
    "tr": "TUR", "turkey": "TUR",
    "eg": "EGY", "egypt": "EGY",
    "za": "ZAF", "south africa": "ZAF",
    "nz": "NZL", "new zealand": "NZL",
    "fi": "FIN", "finland": "FIN",
    "no": "NOR", "norway": "NOR",
    "dk": "DNK", "denmark": "DNK",
    "at": "AUT", "austria": "AUT",
    "be": "BEL", "belgium": "BEL",
    "ir": "IRN", "iran": "IRN",
    "pk": "PAK", "pakistan": "PAK",
    "tw": "TWN", "taiwan": "TWN",
    "hk": "HKG", "hong kong": "HKG",
    "my": "MYS", "malaysia": "MYS",
    "id": "IDN", "indonesia": "IDN",
    "th": "THA", "thailand": "THA",
    "ec": "ECU", "ecuador": "ECU",
    "pe": "PER", "peru": "PER",
    "ve": "VEN", "venezuela": "VEN",
}


class GeographicHeatmap:
    """
    Genera mapa de calor geográfico de publicaciones por país del primer autor.
    """

    def __init__(self, dataset_path: Path = UNIFIED_CSV):
        self.dataset_path = dataset_path
        self._df: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.dataset_path, encoding="utf-8")
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        return self.load_data()

    @staticmethod
    def _normalize_country(raw: str) -> str:
        """
        Normaliza el campo país a código ISO-3166 alpha-3.
        Acepta: códigos alpha-2, alpha-3, nombres completos, afiliaciones.
        """
        if not isinstance(raw, str) or not raw.strip():
            return "Unknown"
        raw = raw.strip().lower()

        # Intentar match directo
        if raw in COUNTRY_TO_ISO3:
            return COUNTRY_TO_ISO3[raw]

        # Intentar buscar un país en una cadena de afiliación más larga
        for key, iso3 in COUNTRY_TO_ISO3.items():
            if len(key) > 3 and key in raw:
                return iso3

        # Si parece un código de 2 letras no mapeado
        if re.match(r"^[a-z]{2}$", raw):
            return raw.upper()

        return "Unknown"

    def get_country_counts(self) -> pd.DataFrame:
        """
        Retorna DataFrame con conteos de publicaciones por país.
        """
        df = self.df.copy()
        df["country_iso3"] = df["country"].apply(self._normalize_country)

        counts = (
            df[df["country_iso3"] != "Unknown"]
            .groupby("country_iso3")
            .size()
            .reset_index(name="publications")
            .sort_values("publications", ascending=False)
        )
        return counts

    def plot_choropleth(self) -> go.Figure:
        """
        Genera un mapa coroplético mundial con Plotly.
        El color representa número de publicaciones.
        """
        counts = self.get_country_counts()

        if counts.empty:
            # Fallback: mensaje vacío
            fig = go.Figure()
            fig.update_layout(
                title="Sin datos de país disponibles",
                paper_bgcolor="#0E1117",
                font_color="white",
            )
            return fig

        fig = px.choropleth(
            counts,
            locations="country_iso3",
            color="publications",
            hover_name="country_iso3",
            color_continuous_scale=[[0, "#d4edda"], [0.5, "#28a745"], [1, "#008236"]],
            title="Distribucion Geografica de Publicaciones — Generative AI",
            labels={"publications": "N. Publicaciones"},
        )

        fig.update_layout(
            paper_bgcolor="#F8F9FA",
            plot_bgcolor="#F8F9FA",
            font=dict(color="#212529", family="Inter, sans-serif"),
            title_font_size=15,
            coloraxis_colorbar=dict(
                title="Publicaciones",
            ),
            geo=dict(
                bgcolor="#ffffff",
                landcolor="#e8f4ea",
                coastlinecolor="#cccccc",
                showframe=False,
            ),
        )

        return fig

    def plot_bar_fallback(self, top_n: int = 20) -> go.Figure:
        """
        Gráfico de barras horizontal como alternativa al mapa.
        Muestra los top N países con más publicaciones.
        """
        counts = self.get_country_counts().head(top_n)

        fig = px.bar(
            counts,
            x="publications",
            y="country_iso3",
            orientation="h",
            color="publications",
            color_continuous_scale=[[0, "#d4edda"], [0.5, "#28a745"], [1, "#008236"]],
            title=f"Top {top_n} Paises por Numero de Publicaciones",
            labels={"publications": "N. Publicaciones", "country_iso3": "Pais (ISO-3)"},
        )

        fig.update_layout(
            paper_bgcolor="#F8F9FA",
            plot_bgcolor="#ffffff",
            font=dict(color="#212529", family="Inter, sans-serif"),
            yaxis=dict(autorange="reversed"),
        )

        return fig

    def plot(self, mode: str = "choropleth") -> go.Figure:
        """
        Genera la visualización geográfica.

        Args:
            mode: "choropleth" o "bar"
        """
        if mode == "bar":
            return self.plot_bar_fallback()
        return self.plot_choropleth()
