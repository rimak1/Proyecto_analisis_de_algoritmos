"""
visualization/wordcloud_viz.py — Req 5.2
Nube de palabras dinámica generada desde abstracts y keywords del dataset.
Se actualiza automáticamente cuando se añaden nuevos estudios al CSV.
"""

from __future__ import annotations

import logging
import re
import string
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

from config import PREDEFINED_TERMS, UNIFIED_CSV

logger = logging.getLogger(__name__)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords as nltk_stopwords

CUSTOM_STOPWORDS = set(STOPWORDS) | set(nltk_stopwords.words("english")) | {
    "paper", "study", "research", "propose", "proposed", "using", "used",
    "approach", "method", "results", "result", "show", "shown", "based",
    "also", "however", "furthermore", "thus", "therefore", "et", "al",
    "fig", "figure", "table", "section", "abstract",
}

# Paletas de colores temáticas
# NOTA: PIL / Pillow requiere colores como tuplas de enteros (0–255),
#       NO como floats (0.0–1.0) ni como numpy.float64.
#       Por eso se convierte cada canal con int(x * 255).
COLOR_FUNCTIONS = {
    "viridis": lambda word, font_size, position, orientation, random_state=None, **kwargs: (
        tuple(int(c * 255) for c in plt.cm.viridis(np.random.uniform(0.2, 0.9))[:3])
    ),
    "plasma": lambda word, font_size, position, orientation, random_state=None, **kwargs: (
        tuple(int(c * 255) for c in plt.cm.plasma(np.random.uniform(0.2, 0.9))[:3])
    ),
    "purple_violet": lambda word, font_size, position, orientation, random_state=None, **kwargs:
        f"hsl({np.random.randint(260, 290)}, {np.random.randint(60, 100)}%, {np.random.randint(50, 85)}%)",
    "ocean": lambda word, font_size, position, orientation, random_state=None, **kwargs:
        f"hsl({np.random.randint(185, 215)}, {np.random.randint(50, 90)}%, {np.random.randint(40, 80)}%)",
}


class WordCloudViz:
    """
    Genera nubes de palabras dinámicas desde abstracts y keywords.
    Se recarga automáticamente al detectar cambios en el dataset.
    """

    def __init__(
        self,
        dataset_path: Path = UNIFIED_CSV,
        max_words: int = 150,
        color_theme: str = "purple_violet",
    ):
        self.dataset_path = dataset_path
        self.max_words = max_words
        self.color_theme = color_theme
        self._df = None
        self._file_mtime: float = 0.0

    def _needs_reload(self) -> bool:
        """Detecta si el dataset fue modificado (para actualización dinámica)."""
        if not self.dataset_path.exists():
            return False
        mtime = self.dataset_path.stat().st_mtime
        if mtime != self._file_mtime:
            self._file_mtime = mtime
            return True
        return False

    def load_data(self, force: bool = False):
        """Carga o recarga el dataset si ha cambiado."""
        import pandas as pd
        if self._df is None or force or self._needs_reload():
            self._df = pd.read_csv(self.dataset_path, encoding="utf-8")
        return self._df

    def _extract_text(self, include_keywords: bool = True) -> str:
        """Combina abstracts (y keywords si se pide) en un solo corpus."""
        df = self.load_data()
        texts = []

        abstracts = df["abstract"].fillna("").astype(str).tolist()
        texts.extend(abstracts)

        if include_keywords:
            kws = df["keywords"].fillna("").astype(str).tolist()
            # Reemplazar "; " por espacios para tratarlos como palabras individuales
            kws = [re.sub(r"[;,]", " ", k) for k in kws]
            texts.extend(kws)

        corpus = " ".join(texts)
        corpus = corpus.lower()
        corpus = re.sub(r"<[^>]+>", " ", corpus)      # HTML
        corpus = re.sub(r"[^a-z0-9\s]", " ", corpus)  # Solo alfanumérico
        corpus = re.sub(r"\b\d+\b", " ", corpus)        # Números solos
        corpus = re.sub(r"\s+", " ", corpus).strip()
        return corpus

    def _build_frequency_dict(self, corpus: str) -> dict[str, int]:
        """
        Genera diccionario de frecuencias boosteando los términos predefinidos.
        Los términos del dominio se multiplican × 3 para resaltarlos visualmente.
        """
        from collections import Counter
        tokens = corpus.split()
        freq = Counter(tokens)

        # Eliminar stopwords
        for sw in CUSTOM_STOPWORDS:
            freq.pop(sw.lower(), None)

        # Boost de términos del dominio
        for term in PREDEFINED_TERMS:
            for word in term.split():
                if word in freq:
                    freq[word] = freq[word] * 3

        # Eliminar términos muy cortos o muy frecuentes
        freq = {k: v for k, v in freq.items() if len(k) > 2 and v > 1}

        return freq

    def _get_color_func(self):
        """Retorna la función de color elegida."""
        if self.color_theme in COLOR_FUNCTIONS:
            return COLOR_FUNCTIONS[self.color_theme]
        return COLOR_FUNCTIONS["purple_violet"]

    def generate(
        self,
        include_keywords: bool = True,
        width: int = 900,
        height: int = 500,
        background_color: str = "#0E1117",
    ) -> plt.Figure:
        """
        Genera la nube de palabras y retorna una figura matplotlib.

        Args:
            include_keywords: Si incluir también los campos de keywords.
            width, height: Dimensiones del canvas de la nube.
            background_color: Color de fondo.

        Returns:
            Figura matplotlib con la nube de palabras.
        """
        corpus = self._extract_text(include_keywords=include_keywords)
        freq_dict = self._build_frequency_dict(corpus)

        if not freq_dict:
            logger.warning("No hay suficiente texto para generar nube de palabras.")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "Sin datos suficientes", ha="center", va="center",
                    fontsize=20, color="grey")
            ax.axis("off")
            fig.patch.set_facecolor(background_color)
            return fig

        wc = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=self.max_words,
            color_func=self._get_color_func(),
            stopwords=CUSTOM_STOPWORDS,
            collocations=True,          # detecta bigramas comunes
            collocation_threshold=30,
            prefer_horizontal=0.85,
            min_font_size=8,
            max_font_size=90,
            relative_scaling=0.5,
            normalize_plurals=True,
        )

        wc.generate_from_frequencies(freq_dict)

        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        try:
            ax.imshow(wc, interpolation="bilinear")
        except Exception as exc:
            logger.error(f"Error al renderizar la nube de palabras: {exc}")
            ax.text(0.5, 0.5, f"Error al generar la nube.\n{exc}",
                    ha="center", va="center", fontsize=12, color="red",
                    wrap=True, transform=ax.transAxes)
        ax.axis("off")

        ax.set_title(
            "Nube de Palabras — Generative Artificial Intelligence",
            fontsize=14,
            fontweight="bold",
            color="white",
            pad=10,
        )

        fig.patch.set_facecolor(background_color)
        plt.tight_layout(pad=0.2)
        return fig

    def to_bytes(self, fig: plt.Figure, fmt: str = "PNG") -> bytes:
        """Convierte la figura a bytes (para incrustar en PDF o descargar)."""
        buf = BytesIO()
        fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
        buf.seek(0)
        return buf.read()
