"""
visualization/report.py — Req 5.4
Exporta las visualizaciones bibliométricas a un reporte PDF profesional.

ESTRATEGIA DE IMÁGENES:
  Se evita deliberadamente el uso de plotly.to_image() (requiere kaleido).
  En su lugar, todas las gráficas del PDF se generan con matplotlib directamente
  a partir de los datos CSV. Esto garantiza que el PDF siempre funciona sin
  dependencias adicionales del sistema.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import PDF_OUTPUT_PATH, UNIFIED_CSV, PREDEFINED_TERMS

logger = logging.getLogger(__name__)

# ─── Estilo matplotlib para PDF ─────────────────────────────────────────
PDF_BG     = "#FFFFFF"          # Fondo blanco para PDF (legible impreso)
ACCENT1    = "#3E2723"          # Marrón Chocolate — encabezados de barras
ACCENT2    = "#D84315"          # Naranja Terracota — líneas acumuladas
GRAY_TEXT  = "#1B0000"          # Texto principal
BAR_COLORS = ["#3E2723", "#558B2F", "#D84315", "#795548", "#8D6E63",
              "#BF360C", "#33691E", "#4E342E", "#6D4C41", "#FF8A65"]


def _apply_clean_style(ax, title="", xlabel="", ylabel="", tight=True):
    """Aplica un estilo limpio y profesional a un eje matplotlib."""
    ax.set_facecolor("#F8F8FF")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors=GRAY_TEXT, labelsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#222222", pad=8)
    ax.set_xlabel(xlabel, fontsize=8, color=GRAY_TEXT)
    ax.set_ylabel(ylabel, fontsize=8, color=GRAY_TEXT)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#CCCCCC")
    if tight:
        plt.tight_layout()


def _fig_to_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    """Convierte figura matplotlib a PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", dpi=dpi,
                facecolor=PDF_BG, edgecolor="none")
    buf.seek(0)
    return buf.read()


# ─── Generadores de gráficas matplotlib para el PDF ──────────────────────────

def _build_country_chart(df: pd.DataFrame) -> plt.Figure:
    """Gráfica de barras horizontal: top-15 países por publicaciones."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(PDF_BG)

    if "country" not in df.columns or df["country"].dropna().empty:
        ax.text(0.5, 0.5, "Sin datos de país disponibles",
                ha="center", va="center", fontsize=11, color="grey")
        ax.axis("off")
        return fig

    counts = (
        df["country"].fillna("").str.strip()
        .replace("", pd.NA).dropna()
        .value_counts().head(15)
        .sort_values()
    )

    colors_bar = [ACCENT1 if i < len(counts) - 3 else ACCENT2
                  for i in range(len(counts))]
    bars = ax.barh(counts.index, counts.values, color=colors_bar, edgecolor="white",
                   linewidth=0.4, height=0.6)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=7, color=GRAY_TEXT)

    _apply_clean_style(ax,
        title="Distribución Geográfica de Publicaciones (Top 15 países)",
        xlabel="N° Publicaciones", ylabel="País")
    return fig


def _build_timeline_chart(df: pd.DataFrame) -> plt.Figure:
    """Gráfica combinada barras + línea acumulada por año."""
    fig, ax1 = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(PDF_BG)

    if "year" not in df.columns or df["year"].dropna().empty:
        ax1.text(0.5, 0.5, "Sin datos de año disponibles",
                 ha="center", va="center", fontsize=11, color="grey")
        ax1.axis("off")
        return fig

    year_counts = (
        df["year"].dropna()
        .astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .dropna()
        .astype(int)
        .pipe(lambda s: s[(s >= 2000) & (s <= 2030)])
        .value_counts()
        .sort_index()
    )

    if year_counts.empty:
        ax1.text(0.5, 0.5, "Sin datos de año en rango 2000-2030",
                 ha="center", va="center", color="grey")
        ax1.axis("off")
        return fig

    cumulative = year_counts.cumsum()
    years = year_counts.index.tolist()

    # Barras anuales
    ax1.bar(years, year_counts.values, color=ACCENT1, alpha=0.8,
            edgecolor="white", linewidth=0.4, label="Por año")
    ax1.set_xlabel("Año", fontsize=8, color=GRAY_TEXT)
    ax1.set_ylabel("Publicaciones / año", fontsize=8, color=ACCENT1)
    ax1.tick_params(axis="y", labelcolor=ACCENT1, labelsize=7)
    ax1.tick_params(axis="x", labelsize=7, rotation=45)

    # Línea acumulada
    ax2 = ax1.twinx()
    ax2.plot(years, cumulative.values, color=ACCENT2, linewidth=2,
             marker="o", markersize=3, label="Total acumulado")
    ax2.set_ylabel("Total acumulado", fontsize=8, color=ACCENT2)
    ax2.tick_params(axis="y", labelcolor=ACCENT2, labelsize=7)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
               loc="upper left", framealpha=0.7)

    ax1.set_facecolor("#F8F8FF")
    ax1.spines[["top"]].set_visible(False)
    ax1.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax2.spines[["top", "right"]].set_color("#CCCCCC")
    ax1.grid(axis="y", linestyle="--", alpha=0.3, color="#CCCCCC")
    ax1.set_title("Línea Temporal de Publicaciones — Generative AI",
                  fontsize=10, fontweight="bold", color="#222222", pad=8)
    fig.tight_layout()
    return fig


def _build_terms_chart(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Gráfica de frecuencia de los términos predefinidos (o TF-IDF simple)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(PDF_BG)

    corpus = " ".join(
        df["abstract"].fillna("").astype(str).tolist() +
        df.get("keywords", pd.Series([], dtype=str)).fillna("").astype(str).tolist()
    ).lower()

    term_counts = {}
    for term in PREDEFINED_TERMS:
        count = corpus.count(term.lower())
        if count > 0:
            term_counts[term] = count

    if not term_counts:
        ax.text(0.5, 0.5, "Sin datos de términos disponibles",
                ha="center", va="center", color="grey")
        ax.axis("off")
        return fig

    sorted_terms = sorted(term_counts.items(), key=lambda x: x[1])[-top_n:]
    terms, counts = zip(*sorted_terms)
    colors_t = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(terms))]

    bars = ax.barh(terms, counts, color=colors_t, edgecolor="white",
                   linewidth=0.4, height=0.6)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=7, color=GRAY_TEXT)

    _apply_clean_style(ax,
        title="Frecuencia de Términos — Generative AI in Education",
        xlabel="Ocurrencias en el corpus", ylabel="Término")
    return fig


def _build_source_pie(df: pd.DataFrame) -> plt.Figure:
    """Gráfica de torta: distribución de artículos por fuente/base de datos."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(PDF_BG)

    if "source_db" not in df.columns or df["source_db"].dropna().empty:
        ax.text(0.5, 0.5, "Sin datos de fuente", ha="center", va="center", color="grey")
        ax.axis("off")
        return fig

    src_counts = df["source_db"].value_counts()
    colors_pie = BAR_COLORS[:len(src_counts)]
    wedges, texts, autotexts = ax.pie(
        src_counts.values,
        labels=src_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors_pie,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=1.2),
    )
    for t in texts:
        t.set_fontsize(8)
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color("white")

    ax.set_title("Artículos por Fuente de Datos", fontsize=10,
                 fontweight="bold", color="#222222", pad=8)
    fig.tight_layout()
    return fig


def _build_wordcloud_chart(df: pd.DataFrame) -> plt.Figure:
    """Nube de palabras matplotlib si WordCloud está disponible; barras de frecuencia si no."""
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(PDF_BG)

    corpus = " ".join(df["abstract"].fillna("").astype(str).tolist()).lower()

    try:
        from wordcloud import WordCloud, STOPWORDS
        import nltk
        try:
            from nltk.corpus import stopwords as nltk_sw
            sw = set(nltk_sw.words("english"))
        except Exception:
            sw = set()

        sw = sw | set(STOPWORDS) | {
            "paper", "study", "research", "propose", "proposed", "using", "used",
            "approach", "results", "result", "show", "shown", "based", "also",
        }

        wc = WordCloud(
            width=900, height=400, background_color="white",
            max_words=100, stopwords=sw, colormap="viridis",
            prefer_horizontal=0.8,
        ).generate(corpus)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title("Nube de Palabras — Abstracts del Corpus",
                     fontsize=10, fontweight="bold", color="#222222", pad=8)
        ax.axis("off")

    except ImportError:
        # Fallback: top-20 palabras más frecuentes
        import re
        from collections import Counter
        words = re.findall(r"\b[a-z]{4,}\b", corpus)
        common_stopwords = {"that", "with", "this", "from", "have", "been", "which",
                            "also", "used", "such", "their", "more", "using", "than",
                            "into", "these", "they", "paper", "study", "results", "model"}
        words = [w for w in words if w not in common_stopwords]
        top = Counter(words).most_common(20)
        if top:
            labels, vals = zip(*top)
            y_pos = range(len(labels))
            ax.barh(y_pos, vals, color=ACCENT1, alpha=0.8)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(labels, fontsize=7)
            _apply_clean_style(ax, title="Términos más Frecuentes (Top 20)",
                               xlabel="Frecuencia")

    fig.tight_layout()
    return fig


# ─── Clase principal ───────────────────────────────────────────────────────────

class ReportExporter:
    """
    Genera un PDF completo con todas las visualizaciones bibliométricas.
    Genera las gráficas internamente desde el CSV — no depende de kaleido.
    """

    def __init__(self, output_path: Path = PDF_OUTPUT_PATH,
                 dataset_path: Path = UNIFIED_CSV):
        self.output_path = output_path
        self.dataset_path = dataset_path

    def _load_df(self) -> pd.DataFrame:
        if self.dataset_path.exists():
            return pd.read_csv(self.dataset_path, encoding="utf-8")
        return pd.DataFrame()

    def generate(
        self,
        heatmap_fig=None,       # Ignorado — se regenera desde datos
        wordcloud_fig=None,     # Ignorado — se regenera desde datos
        timeline_fig=None,      # Ignorado — se regenera desde datos
        freq_df=None,           # Tabla de frecuencias (opcional)
        cluster_eval_df=None,   # Tabla evaluación clustering (opcional)
        dataset_stats=None,     # Diccionario con estadísticas (opcional)
    ) -> bytes:
        """
        Genera el PDF con visualizaciones creadas directamente desde matplotlib.
        No requiere kaleido, no depende de session_state de Streamlit.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                Image, PageBreak, Paragraph, SimpleDocTemplate,
                Spacer, Table, TableStyle,
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError as e:
            logger.error(f"ReportLab no instalado: {e}")
            return b""

        df = self._load_df()
        buf = io.BytesIO()

        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
            title="Bibliometric Analysis Report — Generative AI",
            author="Bibliometrics AI Analyzer",
        )

        styles = getSampleStyleSheet()
        title_s = ParagraphStyle("titulo", parent=styles["Title"], fontSize=22,
                                 textColor=colors.HexColor("#3E2723"),
                                 spaceAfter=10, alignment=TA_CENTER)
        h1_s = ParagraphStyle("h1c", parent=styles["Heading1"], fontSize=14,
                               textColor=colors.HexColor("#3E2723"),
                               spaceBefore=12, spaceAfter=6)
        h2_s = ParagraphStyle("h2c", parent=styles["Heading2"], fontSize=11,
                               textColor=colors.HexColor("#D84315"),
                               spaceBefore=8, spaceAfter=4)
        body_s = ParagraphStyle("bodyc", parent=styles["Normal"], fontSize=9,
                                leading=13, textColor=colors.HexColor("#333333"))
        footer_s = ParagraphStyle("foot", parent=body_s, alignment=TA_CENTER, fontSize=7,
                                  textColor=colors.HexColor("#888888"))

        def _img_flowable(fig: plt.Figure, w_cm=15, h_cm=8) -> Image:
            img_bytes = _fig_to_bytes(fig)
            plt.close(fig)
            return Image(io.BytesIO(img_bytes), width=w_cm*cm, height=h_cm*cm)

        def _table(data, col_widths=None, header_color="#3E2723"):
            t = Table(data, colWidths=col_widths, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor(header_color)),
                ("TEXTCOLOR", (0,0), (-1,-1), colors.white),
                ("FONTSIZE", (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS", (0,1), (-1,-1),
                 [colors.HexColor("#F5F5FF"), colors.white]),
                ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#CCCCCC")),
                ("PADDING", (0,0), (-1,-1), 4),
                ("TEXTCOLOR", (0,1), (-1,-1), colors.HexColor("#222222")),
            ]))
            return t

        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        story = []

        # ── Portada ────────────────────────────────────────────────────────────
        story.append(Spacer(1, 1.5*cm))
        story.append(Paragraph("Bibliometrics AI Analyzer", title_s))
        story.append(Paragraph(
            "Análisis de Literatura Científica sobre Inteligencia Artificial Generativa",
            styles["Heading2"]))
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph(f"Generado: {now}", body_s))
        story.append(Paragraph("Dominio: <b>Generative Artificial Intelligence</b>", body_s))
        story.append(Paragraph(
            "Fuentes: ACM Digital Library | SAGE Journals | ScienceDirect | CrossRef", body_s))
        story.append(Spacer(1, 0.5*cm))

        # Estadísticas del dataset
        if not df.empty:
            story.append(Paragraph("Resumen del Dataset", h2_s))
            auto_stats = {
                "Total artículos": len(df),
                "Fuentes indexadas": df["source_db"].nunique() if "source_db" in df else "—",
                "Rango de años": (
                    f"{int(df['year'].min())}–{int(df['year'].max())}"
                    if "year" in df and df["year"].notna().any() else "—"
                ),
                "Países únicos": int(df["country"].nunique()) if "country" in df else "—",
                "Con abstract": int(df["abstract"].fillna("").str.len().gt(20).sum()),
            }
            if dataset_stats:
                auto_stats.update(dataset_stats)
            stats_data = [["Métrica", "Valor"]] + [[k, str(v)] for k, v in auto_stats.items()]
            story.append(_table(stats_data, col_widths=[9*cm, 6*cm]))

        story.append(PageBreak())

        # ── 1. Distribución geográfica ─────────────────────────────────────────
        story.append(Paragraph("1. Distribución Geográfica de Publicaciones", h1_s))
        story.append(Paragraph(
            "Países con mayor producción científica sobre IA Generativa "
            "según la afiliación del primer autor.", body_s))
        story.append(Spacer(1, 0.3*cm))
        story.append(_img_flowable(_build_country_chart(df), w_cm=15, h_cm=7))
        story.append(PageBreak())

        # ── 2. Nube de palabras ─────────────────────────────────────────────────
        story.append(Paragraph("2. Nube de Palabras — Términos más Frecuentes", h1_s))
        story.append(Paragraph(
            "Visualización de los términos más recurrentes en los abstracts. "
            "El tamaño es proporcional a la frecuencia de aparición.", body_s))
        story.append(Spacer(1, 0.3*cm))
        story.append(_img_flowable(_build_wordcloud_chart(df), w_cm=15, h_cm=7))
        story.append(PageBreak())

        # ── 3. Línea temporal ───────────────────────────────────────────────────
        story.append(Paragraph("3. Línea Temporal de Publicaciones", h1_s))
        story.append(Paragraph(
            "Evolución anual del número de publicaciones. "
            "La línea naranja muestra el total acumulado (curva de productividad).", body_s))
        story.append(Spacer(1, 0.3*cm))
        story.append(_img_flowable(_build_timeline_chart(df), w_cm=15, h_cm=7))
        story.append(PageBreak())

        # ── 4. Distribución por fuente ──────────────────────────────────────────
        story.append(Paragraph("4. Distribución por Base de Datos", h1_s))
        story.append(Paragraph(
            "Proporción de artículos recuperados de cada fuente bibliográfica.", body_s))
        story.append(Spacer(1, 0.3*cm))
        story.append(_img_flowable(_build_source_pie(df), w_cm=10, h_cm=6))
        story.append(PageBreak())

        # ── 5. Frecuencia de términos ───────────────────────────────────────────
        story.append(Paragraph("5. Frecuencia de Términos Predefinidos", h1_s))
        story.append(Paragraph(
            "Conteo de ocurrencias de los 15 términos de IA Generativa en el corpus.", body_s))
        story.append(Spacer(1, 0.3*cm))
        story.append(_img_flowable(_build_terms_chart(df), w_cm=15, h_cm=7))

        if freq_df is not None and not freq_df.empty:
            story.append(Spacer(1, 0.5*cm))
            story.append(Paragraph("Tabla de Frecuencias Detallada", h2_s))
            show_cols = [c for c in ["rank","term","absolute_freq","document_freq","pct_documents"]
                         if c in freq_df.columns]
            head = ["Rango", "Término", "Frec. Abs.", "N° Docs", "% Docs"][:len(show_cols)]
            tdata = [head] + [
                [str(v)[:40] for v in row]
                for row in freq_df[show_cols].head(15).values.tolist()
            ]
            story.append(_table(tdata, col_widths=[2*cm, 6*cm, 2.5*cm, 2.5*cm, 2.5*cm]))

        story.append(PageBreak())

        # ── 6. Evaluación de clustering ─────────────────────────────────────────
        if cluster_eval_df is not None and not cluster_eval_df.empty:
            story.append(Paragraph("6. Evaluación de Clustering Jerárquico", h1_s))
            story.append(Paragraph(
                "Comparación automática de Ward, Complete y Average "
                "mediante CCC, índice de silueta e índice Calinski-Harabász.", body_s))
            story.append(Spacer(1, 0.3*cm))
            ecols = [c for c in ["method","cophenetic_correlation","silhouette_score",
                                  "calinski_harabasz","composite_score","rank"]
                     if c in cluster_eval_df.columns]
            ehead = ["Método","CCC","Silueta","Calinski-H","Score","Rank"][:len(ecols)]
            edata = [ehead] + [
                [f"{v:.4f}" if isinstance(v, float) else str(v)[:20] for v in row]
                for row in cluster_eval_df[ecols].values.tolist()
            ]
            story.append(_table(edata, header_color="#D84315"))
            story.append(PageBreak())

        # ── Cierre ─────────────────────────────────────────────────────────────
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            "Generado por Bibliometrics AI Analyzer — "
            "Proyecto Final Análisis de Algoritmos — Universidad del Quindío",
            footer_s))

        # Build PDF
        doc.build(story)
        pdf_bytes = buf.getvalue()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info(f"PDF generado exitosamente: {self.output_path} ({len(pdf_bytes)} bytes)")

        return pdf_bytes
