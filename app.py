"""
app.py — Punto de entrada principal — Bibliometrics AI Analyzer
Aplicación Streamlit profesional con tema claro institucional.
"""

import sys
import io
import logging
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuracion de la pagina ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Bibliometrics AI Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Bibliometrics AI Analyzer — Universidad del Quindio"},
)

from config import (
    UNIFIED_CSV, DUPLICATES_CSV, PREDEFINED_TERMS,
    APP_TITLE, APP_SUBTITLE, DEFAULT_QUERY, CLUSTERING_METHODS,
    PRIMARY_COLOR, ACCENT_COLOR, BG_COLOR, SURFACE_COLOR,
    BORDER_COLOR, TEXT_PRIMARY, TEXT_SECONDARY,
    COLOR_SUCCESS, COLOR_ERROR, COLOR_WARNING,
)

# ── CSS institucional (tema Marrón/Terracota/Oliva/Beige) ───────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* Reset global */
  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT_PRIMARY};
  }}

  /* Fondo general */
  .stApp, .main {{
    background-color: {BG_COLOR};
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background-color: {PRIMARY_COLOR};
    border-right: 2px solid {BORDER_COLOR};
  }}
  /* Todos los textos de la sidebar en blanco/beige claro */
  [data-testid="stSidebar"] *,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] div,
  [data-testid="stSidebar"] small,
  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {{
    color: #EFEBE9 !important;
  }}
  /* Captions ligeramente más oscuros para jerarquía visual */
  [data-testid="stSidebar"] .stCaption,
  [data-testid="stSidebar"] small {{
    color: #BCAAA4 !important;
    opacity: 1 !important;
  }}
  /* Slider: etiqueta de valor actual */
  [data-testid="stSidebar"] [data-testid="stThumbValue"],
  [data-testid="stSidebar"] [data-testid="stTickBar"] {{
    color: #EFEBE9 !important;
  }}
  /* Inputs de texto dentro del sidebar */
  [data-testid="stSidebar"] input {{
    color: {TEXT_PRIMARY} !important;
    background-color: #EFEBE9 !important;
  }}
  /* Slider track color */
  [data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {{
    background-color: {ACCENT_COLOR} !important;
  }}

  /* alerts inside sidebar */
  [data-testid="stSidebar"] [data-testid="stNotification"] {{
    background-color: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
  }}
  [data-testid="stSidebar"] [data-testid="stNotificationContent"] p {{
    color: #FFFFFF !important;
  }}
  /* Especifico para success e info en sidebar */
  [data-testid="stSidebar"] .stSuccess,
  [data-testid="stSidebar"] .stInfo,
  [data-testid="stSidebar"] .stWarning,
  [data-testid="stSidebar"] .stError {{
    background-color: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #FFFFFF !important;
  }}
  [data-testid="stSidebar"] .stSuccess i,
  [data-testid="stSidebar"] .stInfo i {{
    color: #FFFFFF !important;
  }}


  /* Header banner principal */
  .header-banner {{
    background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, #5d4037 100%);
    border-radius: 10px;
    padding: 28px 36px;
    margin-bottom: 24px;
  }}
  .header-title {{
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.3px;
  }}
  .header-subtitle {{
    font-size: 0.95rem;
    color: rgba(255,255,255,0.85);
    margin-top: 6px;
  }}
  .header-domain {{
    display: inline-block;
    background: rgba(216,67,21,0.25);
    color: #FF8A65;
    border: 1px solid rgba(216,67,21,0.5);
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 10px;
    letter-spacing: 0.3px;
  }}

  /* Cards de metricas */
  .metric-card {{
    background: {SURFACE_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-left: 4px solid {ACCENT_COLOR};
    border-radius: 8px;
    padding: 18px 22px;
    text-align: center;
  }}
  .metric-value {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {PRIMARY_COLOR};
  }}
  .metric-label {{
    font-size: 0.82rem;
    color: {TEXT_SECONDARY};
    margin-top: 4px;
  }}

  /* Encabezados de seccion */
  .section-header {{
    border-left: 4px solid {ACCENT_COLOR};
    padding-left: 12px;
    margin: 20px 0 12px 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: {PRIMARY_COLOR};
  }}

  /* Cards de algoritmos */
  .algo-card {{
    background: {SURFACE_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
  }}
  .algo-title {{ font-weight: 600; color: {ACCENT_COLOR}; font-size: 0.95rem; }}
  .algo-category {{ color: {TEXT_SECONDARY}; font-size: 0.78rem; font-style: italic; }}
  .algo-desc {{ color: {TEXT_PRIMARY}; font-size: 0.85rem; margin-top: 6px; }}

  /* Botones primarios */
  .stButton > button[kind="primary"] {{
    background-color: {ACCENT_COLOR};
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    letter-spacing: 0.3px;
  }}
  .stButton > button[kind="primary"]:hover {{
    background-color: #BF360C;
  }}
  .stButton > button {{
    border-radius: 6px;
    border-color: {BORDER_COLOR};
  }}

  /* Tabs */
  [data-testid="stTabs"] [role="tab"] {{
    font-weight: 500;
    color: {TEXT_SECONDARY};
  }}
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {ACCENT_COLOR};
    border-bottom: 2px solid {ACCENT_COLOR};
    font-weight: 600;
  }}

  /* Inputs */
  .stTextInput > div > div > input,
  .stSelectbox > div > div,
  .stMultiSelect > div > div {{
    background-color: {SURFACE_COLOR};
    border-color: {BORDER_COLOR};
    color: {TEXT_PRIMARY};
  }}

  /* DataFrames */
  [data-testid="stDataFrame"] {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
  }}

  /* Divisor horizontal */
  hr {{ border-color: {BORDER_COLOR}; }}

  /* Alertas */
  .stSuccess {{ background-color: #dcedc8; border-color: {COLOR_SUCCESS}; }}
  .stWarning {{ background-color: #fff8e1; border-color: {COLOR_WARNING}; }}
  .stError   {{ background-color: #fbe9e7; border-color: {COLOR_ERROR};   }}
  .stInfo    {{ background-color: #efebe9; border-color: {PRIMARY_COLOR}; }}

  /* ─── RESPONSIVIDAD MÓVIL (≤ 768 px) ─────────────────────── */
  @media (max-width: 768px) {{
    /* Header más compacto */
    .header-banner {{
      padding: 16px 18px;
      border-radius: 6px;
      margin-bottom: 14px;
    }}
    .header-title {{
      font-size: 1.25rem;
    }}
    .header-subtitle {{
      font-size: 0.8rem;
    }}

    /* Columnas en vertical */
    [data-testid="stHorizontalBlock"] {{
      flex-direction: column !important;
    }}
    [data-testid="stHorizontalBlock"] > div {{
      width: 100% !important;
      min-width: 100% !important;
    }}

    /* Section header más pequeño */
    .section-header {{
      font-size: 0.95rem;
    }}

    /* Metric cards */
    .metric-card {{
      padding: 12px 14px;
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 1.4rem;
    }}

    /* Tabs en móvil: texto más pequeño */
    [data-testid="stTabs"] [role="tab"] {{
      font-size: 0.78rem;
      padding: 6px 8px;
    }}

    /* Slider y inputs full-width */
    .stSlider, .stTextInput, .stSelectbox, .stMultiSelect {{
      width: 100% !important;
    }}

    /* Sidebar en móvil: colapsada por defecto (Streamlit lo maneja),
       pero se reduce el padding */
    [data-testid="stSidebar"] > div:first-child {{
      padding: 1rem 0.75rem !important;
    }}
  }}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-banner">
  <p class="header-title">Bibliometrics AI Analyzer</p>
  <p class="header-subtitle">Analisis de Literatura Cientifica sobre Inteligencia Artificial Generativa</p>
  <span class="header-domain">Dominio: Generative Artificial Intelligence</span>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## Configuracion")
    query = st.text_input("Consulta de busqueda", value=DEFAULT_QUERY)
    max_results = st.slider("Resultados maximos por fuente", 10, 200, 50, step=10)

    st.markdown("---")
    st.markdown("### Estado del Dataset")
    if UNIFIED_CSV.exists():
        df_main = pd.read_csv(UNIFIED_CSV, encoding="utf-8")
        st.success(f"Dataset activo: {len(df_main)} articulos")
        if DUPLICATES_CSV.exists():
            df_dup = pd.read_csv(DUPLICATES_CSV, encoding="utf-8")
            st.info(f"Duplicados eliminados: {len(df_dup)}")
    else:
        df_main = pd.DataFrame()
        st.warning("Dataset no disponible. Vaya a Extraccion de Datos.")

    st.markdown("---")
    st.markdown("### Fuentes de Datos")
    st.caption("Bases de datos — Biblioteca UniQ")
    src_ebsco = st.checkbox("EBSCO Discovery (Conexion Directa)", value=False)
    src_acm = st.checkbox("ACM Digital Library", value=True)
    src_sage = st.checkbox("SAGE Journals", value=True)
    src_sd = st.checkbox("ScienceDirect (Elsevier)", value=True)

    st.caption("Fuentes complementarias")
    src_crossref = st.checkbox("CrossRef", value=True)
    src_ss = st.checkbox("Semantic Scholar", value=False)

    selected_sources = []
    if src_ebsco:    selected_sources.append("ebsco")
    if src_acm:      selected_sources.append("acm")
    if src_sage:     selected_sources.append("sage")
    if src_sd:       selected_sources.append("sciencedirect")
    if src_crossref: selected_sources.append("crossref")
    if src_ss:       selected_sources.append("semantic_scholar")

    if src_ebsco:
        st.warning(
            "EBSCO abrira un navegador Chromium. "
            "Si es la primera vez, debera iniciar sesion manualmente."
        )
        reset_ebsco = st.checkbox(
            "Forzar nuevo inicio de sesion (limpiar cache)", value=True
        )
    else:
        reset_ebsco = False

    if os.getenv("ELSEVIER_API_KEY", ""):
        st.success("Elsevier API key configurada")
    else:
        st.info("Sin ELSEVIER_API_KEY: ScienceDirect usa OpenAlex como canal compatible")


# ── Pestanas principales ──────────────────────────────────────────────────────
tabs = st.tabs([
    "Inicio",
    "Extraccion de Datos",
    "Similitud Textual",
    "Analisis NLP",
    "Clustering",
    "Visualizaciones",
])


# ══════════════════════════════════════════════════════════════════════════════
# PESTANA 0 — INICIO
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(
        '<div class="section-header">Bienvenido al Bibliometrics AI Analyzer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Esta plataforma implementa un pipeline completo de analisis bibliometrico "
        "sobre literatura de **Inteligencia Artificial Generativa**, cubriendo los "
        "6 requerimientos del proyecto."
    )

    col1, col2, col3 = st.columns(3)
    reqs = [
        ("Req 1", "Extraccion y unificacion automatica desde APIs publicas"),
        ("Req 2", "6 algoritmos de similitud textual (4 clasicos + 2 basados en IA)"),
        ("Req 3", "Frecuencia de terminos NLP + extraccion de palabras clave"),
        ("Req 4", "Clustering jerarquico (Ward, Complete, Average) + evaluacion"),
        ("Req 5", "4 visualizaciones interactivas + exportacion a PDF"),
        ("Req 6", "Arquitectura, Docker, documentacion y justificacion etica"),
    ]
    for i, (name, desc) in enumerate(reqs):
        with [col1, col2, col3][i % 3]:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-weight:700;color:{PRIMARY_COLOR};font-size:1rem">{name}</div>
              <div style="font-size:0.82rem;color:{TEXT_SECONDARY};margin-top:6px">{desc}</div>
            </div>
            <br>
            """, unsafe_allow_html=True)

    if not df_main.empty:
        st.markdown("---")
        st.markdown(
            '<div class="section-header">Estadisticas del Dataset Actual</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Articulos", len(df_main))
        c2.metric("Fuentes", df_main["source_db"].nunique() if "source_db" in df_main else "—")
        c3.metric(
            "Anos cubiertos",
            f"{int(df_main['year'].min())}–{int(df_main['year'].max())}"
            if "year" in df_main and df_main["year"].notna().any() else "—",
        )
        c4.metric("Paises unicos", df_main["country"].nunique() if "country" in df_main else "—")


# ══════════════════════════════════════════════════════════════════════════════
# PESTANA 1 — EXTRACCION (Req 1)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(
        '<div class="section-header">Req 1 — Extraccion y Unificacion de Datos</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    Descarga automatica desde **ACM**, **SAGE** y **ScienceDirect** con unificacion
    en esquema estandar y eliminacion de duplicados en tres etapas:
    1. **DOI exacto** — 2. **Titulo canonico** — 3. **Similitud fuzzy >= 0.92**
    """)

    col_run, col_info = st.columns([1, 2])

    with col_run:
        if st.button("Iniciar Extraccion Completa", type="primary", width='stretch'):
            if not selected_sources:
                st.error("Seleccione al menos una fuente en el panel lateral.")
            else:
                from data_extraction import DataFetcher, DataUnifier, Deduplicator

                prog = st.progress(0)
                status_box = st.empty()

                status_box.info("Extrayendo desde las fuentes seleccionadas...")
                fetcher = DataFetcher(
                    query=query,
                    max_results=max_results,
                    reset_ebsco=reset_ebsco,
                )
                dfs = fetcher.fetch_all(sources=selected_sources)
                fetcher.save_raw(dfs)
                prog.progress(40)

                status_box.info("Unificando datasets...")
                unifier = DataUnifier()
                unified_df = unifier.unify(dataframes=dfs, save=True)
                prog.progress(70)

                status_box.info("Eliminando duplicados...")
                dedup = Deduplicator()
                clean_df, dupes_df = dedup.deduplicate(df=unified_df, save=True)
                prog.progress(100)

                stats = dedup.duplication_stats()
                status_box.success(
                    f"Proceso completado: {stats.get('total_unique', len(clean_df))} articulos unicos "
                    f"| {stats.get('total_removed', len(dupes_df))} duplicados eliminados "
                    f"({stats.get('removal_rate_pct', 0)}%)"
                )
                st.rerun()

    with col_info:
        if UNIFIED_CSV.exists():
            st.dataframe(
                pd.read_csv(UNIFIED_CSV).head(20),
                width='stretch',
                height=300,
            )

    if DUPLICATES_CSV.exists():
        st.markdown("---")
        st.markdown("#### Registro de Duplicados Eliminados")
        df_dup = pd.read_csv(DUPLICATES_CSV)
        st.dataframe(df_dup.head(50), width='stretch', height=250)
        st.download_button(
            "Descargar duplicados (CSV)",
            df_dup.to_csv(index=False).encode(),
            "duplicates_removed.csv",
            "text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PESTANA 2 — SIMILITUD (Req 2)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(
        '<div class="section-header">Req 2 — Algoritmos de Similitud Textual</div>',
        unsafe_allow_html=True,
    )

    if df_main.empty:
        st.warning("Cargue el dataset desde la pestana de Extraccion de Datos.")
    else:
        from similarity import SimilarityInterface
        si = SimilarityInterface()
        articles = si.get_article_list()
        titles = [f"[{a['index']}] {a['title']}" for a in articles]

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            idx_a = st.selectbox("Articulo A", range(len(titles)),
                                 format_func=lambda i: titles[i])
        with col_sel2:
            idx_b = st.selectbox("Articulo B", range(len(titles)),
                                 index=min(1, len(titles) - 1),
                                 format_func=lambda i: titles[i])

        algos = st.multiselect(
            "Algoritmos a calcular",
            ["levenshtein", "jaccard", "tfidf_cosine", "ngram_bigram",
             "sentence_bert", "paraphrase_miniLM"],
            default=["levenshtein", "jaccard", "tfidf_cosine", "ngram_bigram"],
        )

        if st.button("Calcular Similitudes", type="primary"):
            with st.spinner("Calculando..."):
                result = si.compute_similarity_pair(idx_a, idx_b, algorithms=algos)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{result['article_a']['title'][:80]}...**")
                st.text_area("Abstract A", result["article_a"]["abstract"][:600], height=150)
            with c2:
                st.markdown(f"**{result['article_b']['title'][:80]}...**")
                st.text_area("Abstract B", result["article_b"]["abstract"][:600], height=150)

            st.markdown("#### Resultados de Similitud")
            score_cols = st.columns(max(len(result["scores"]), 1))
            for i, (algo, score) in enumerate(result["scores"].items()):
                info = result["algorithm_info"].get(algo, {})
                score_cols[i].metric(
                    label=info.get("name", algo),
                    value=f"{score:.4f}" if score >= 0 else "Error",
                    help=f"{info.get('description', '')}\nRango: {info.get('range', '')}",
                )

        st.markdown("---")
        st.markdown("#### Descripcion de los 6 Algoritmos")
        from similarity.interface import ALGORITHM_DESCRIPTIONS
        for algo, info in ALGORITHM_DESCRIPTIONS.items():
            with st.expander(f"{info['category']} — {info['name']}"):
                st.markdown(f"**Descripcion:** {info['description']}")
                st.markdown(f"**Rango de salida:** `{info['range']}`")
                st.markdown(f"**Complejidad:** `{info['complexity']}`")

        st.markdown("---")
        st.markdown("#### Matriz de Similitud")
        multi_indices = st.multiselect(
            "Seleccione articulos para la matriz",
            range(len(titles)),
            format_func=lambda i: titles[i],
            max_selections=15,
        )
        matrix_algo = st.selectbox(
            "Algoritmo para la matriz", algos or ["tfidf_cosine"]
        )

        if len(multi_indices) >= 2 and st.button("Calcular Matriz"):
            with st.spinner("Generando matriz..."):
                matrix_df = si.compute_similarity_matrix(multi_indices, algorithm=matrix_algo)

            import plotly.express as px
            fig_heat = px.imshow(
                matrix_df,
                color_continuous_scale="Blues",
                title=f"Matriz de Similitud — {matrix_algo}",
                zmin=0, zmax=1,
            )
            fig_heat.update_layout(
                paper_bgcolor="white",
                font_color=TEXT_PRIMARY,
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_heat, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PESTANA 3 — NLP (Req 3)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(
        '<div class="section-header">Req 3 — Frecuencia de Terminos y Extraccion NLP</div>',
        unsafe_allow_html=True,
    )

    if df_main.empty:
        st.warning("Cargue el dataset primero.")
    else:
        col_freq, col_extract = st.columns([1, 1])

        with col_freq:
            st.markdown("#### Frecuencia de los 15 Terminos Predefinidos")
            st.caption("Categoria: Conceptos de IA Generativa en Educacion")

            if st.button("Calcular Frecuencias", type="primary"):
                from nlp import TermFrequencyAnalyzer
                with st.spinner("Analizando corpus..."):
                    tfa = TermFrequencyAnalyzer()
                    freq_df = tfa.compute_frequencies()

                import plotly.express as px
                fig_freq = px.bar(
                    freq_df.head(15),
                    x="absolute_freq",
                    y="term",
                    orientation="h",
                    color="absolute_freq",
                    color_continuous_scale="Oranges",
                    title="Frecuencia Absoluta de Terminos",
                    labels={"absolute_freq": "Ocurrencias", "term": "Termino"},
                )
                fig_freq.update_layout(
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font_color=TEXT_PRIMARY,
                    yaxis=dict(autorange="reversed"),
                    showlegend=False,
                )
                st.plotly_chart(fig_freq, width='stretch')
                st.dataframe(freq_df, width='stretch', height=300)
                st.session_state["freq_df"] = freq_df

        with col_extract:
            st.markdown("#### Extraccion Automatica de Palabras Clave")
            st.caption("Metodos: YAKE + KeyBERT + TF-IDF (fusion ponderada)")

            max_kw = st.slider("Maximo de palabras clave nuevas", 5, 15, 15)

            if st.button("Extraer Palabras Clave", type="primary"):
                from nlp import KeywordExtractor, PrecisionMetric
                with st.spinner("Ejecutando pipeline NLP... (puede tardar ~30s)"):
                    extractor = KeywordExtractor(max_keywords=max_kw)
                    kw_df = extractor.extract()
                    pm = PrecisionMetric()
                    kw_evaluated = pm.evaluate(kw_df)
                    summary = pm.summary(kw_evaluated)

                st.dataframe(
                    kw_evaluated[[
                        "rank", "keyword", "fused_score",
                        "precision_score", "precision_grade", "extractors_count",
                    ]],
                    width='stretch',
                    height=350,
                )

                c_p1, c_p2, c_p3 = st.columns(3)
                c_p1.metric("Precision Media", f"{summary.get('mean_precision', 0):.4f}")
                c_p2.metric("Precision Maxima", f"{summary.get('max_precision', 0):.4f}")
                c_p3.metric("Palabras Extraidas", summary.get("n_keywords", 0))
                st.session_state["kw_df"] = kw_evaluated


# ══════════════════════════════════════════════════════════════════════════════
# PESTANA 4 — CLUSTERING (Req 4)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(
        '<div class="section-header">Req 4 — Agrupamiento Jerarquico</div>',
        unsafe_allow_html=True,
    )

    if df_main.empty:
        st.warning("Cargue el dataset primero.")
    else:
        c_params1, c_params2, c_params3 = st.columns(3)
        with c_params1:
            n_docs = st.slider("N de articulos", 20, min(300, len(df_main)), 80)
        with c_params2:
            n_clusters = st.slider("N de clusters", 2, 10, 5)
        with c_params3:
            use_stemming = st.checkbox("Aplicar stemming", value=True)

        if st.button("Ejecutar Clustering Jerarquico", type="primary"):
            from clustering import ClusteringPreprocessor, HierarchicalClustering, ClusteringEvaluator

            with st.spinner("Preprocesando abstracts y vectorizando..."):
                prep = ClusteringPreprocessor(use_stemming=use_stemming)
                prep.fit(n_docs=n_docs)

            st.success(
                f"{len(prep.labels)} documentos procesados. "
                f"Dimension TF-IDF: {prep.tfidf_matrix.shape}"
            )

            with st.spinner("Calculando linkage matrices..."):
                hc = HierarchicalClustering(prep.distance_matrix, prep.labels)
                linkage_mats = hc.fit_all()

            with st.spinner("Evaluando algoritmos..."):
                feature_dense = prep.tfidf_matrix.toarray()
                evaluator = ClusteringEvaluator(feature_dense, prep.distance_matrix, prep.labels)
                eval_df = evaluator.evaluate(linkage_mats)
                best = evaluator.best_method()

            st.markdown("#### Evaluacion Automatica de Metodos")
            st.dataframe(
                eval_df[[
                    "method", "cophenetic_correlation", "silhouette_score",
                    "calinski_harabasz", "composite_score", "rank", "best",
                ]],
                width='stretch',
            )
            st.success(f"Mejor algoritmo recomendado: **{best.upper()}**")
            st.session_state["cluster_eval_df"] = eval_df

            st.markdown("#### Dendrogramas")
            tab_w, tab_c, tab_a = st.tabs(["Ward", "Complete Linkage", "Average Linkage"])
            for tab_obj, method in zip([tab_w, tab_c, tab_a], CLUSTERING_METHODS):
                with tab_obj:
                    fig_dend = hc.plot_dendrogram(
                        linkage_mats[method], method, n_clusters=n_clusters
                    )
                    st.pyplot(fig_dend, width='stretch')
                    plt.close(fig_dend)

            st.markdown("#### Asignacion de Articulos por Cluster")
            best_Z = linkage_mats[best]
            cluster_summary = hc.get_cluster_summary(best_Z, best, n_clusters=n_clusters)
            st.dataframe(cluster_summary, width='stretch', height=300)


# ══════════════════════════════════════════════════════════════════════════════
# PESTANA 5 — VISUALIZACIONES (Req 5)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown(
        '<div class="section-header">Req 5 — Visualizaciones e Informe PDF</div>',
        unsafe_allow_html=True,
    )

    if df_main.empty:
        st.warning("Cargue el dataset primero.")
    else:
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Mapa Geografico",
            "Nube de Palabras",
            "Linea Temporal",
            "Exportar PDF",
        ])

        # ── Mapa Geografico ───────────────────────────────────────────────────
        with viz_tab1:
            from visualization import GeographicHeatmap
            ghm = GeographicHeatmap()

            map_mode = st.radio(
                "Tipo de visualizacion", ["choropleth", "bar"], horizontal=True
            )
            fig_map = ghm.plot(mode=map_mode)

            # Ajustar colores al tema claro
            fig_map.update_layout(
                paper_bgcolor=BG_COLOR,
                font_color=TEXT_PRIMARY,
                plot_bgcolor=SURFACE_COLOR,
            )
            st.plotly_chart(fig_map, width='stretch')

            counts = ghm.get_country_counts()
            st.dataframe(counts.head(20), width='stretch')
            st.session_state["heatmap_fig"] = fig_map

        # ── Nube de Palabras ──────────────────────────────────────────────────
        with viz_tab2:
            from visualization import WordCloudViz

            color_theme = st.selectbox(
                "Tema de color", ["viridis", "plasma", "purple_violet", "ocean"]
            )
            max_words = st.slider("Maximo de palabras", 50, 200, 120)
            include_kw = st.checkbox("Incluir campo keywords", True)

            if st.button("Generar Nube de Palabras", type="primary"):
                wc_viz = WordCloudViz(max_words=max_words, color_theme=color_theme)
                with st.spinner("Generando nube..."):
                    try:
                        fig_wc = wc_viz.generate(
                            include_keywords=include_kw,
                            background_color=SURFACE_COLOR,
                        )
                        st.pyplot(fig_wc, width='stretch')
                        st.session_state["wordcloud_fig"] = fig_wc
                    except Exception as e:
                        st.error(f"Error al generar la nube de palabras: {e}")

        # ── Linea Temporal ────────────────────────────────────────────────────
        with viz_tab3:
            from visualization import PublicationTimeline
            tl = PublicationTimeline()

            year_min, year_max = tl.get_year_range()
            col_y1, col_y2 = st.columns(2)
            with col_y1:
                yr_start = st.number_input(
                    "Ano inicio", min_value=year_min, max_value=year_max, value=year_min
                )
            with col_y2:
                yr_end = st.number_input(
                    "Ano fin", min_value=year_min, max_value=year_max, value=year_max
                )

            journals = tl.get_available_journals()
            sel_journals = st.multiselect("Filtrar por revista (vacio = todas)", journals)
            group_src = st.checkbox("Desglosar por fuente de datos", False)

            fig_timeline = tl.plot_annual_count(
                year_start=int(yr_start),
                year_end=int(yr_end),
                selected_journals=sel_journals or None,
                group_by_source=group_src,
            )
            fig_timeline.update_layout(
                paper_bgcolor=BG_COLOR,
                plot_bgcolor=SURFACE_COLOR,
                font_color=TEXT_PRIMARY,
            )
            st.plotly_chart(fig_timeline, width='stretch')

            fig_journals = tl.plot_journal_comparison()
            fig_journals.update_layout(
                paper_bgcolor=BG_COLOR,
                plot_bgcolor=SURFACE_COLOR,
                font_color=TEXT_PRIMARY,
            )
            st.plotly_chart(fig_journals, width='stretch')
            st.session_state["timeline_fig"] = fig_timeline

        # ── Exportar PDF ──────────────────────────────────────────────────────
        with viz_tab4:
            st.markdown("### Generar Reporte PDF Completo")
            st.markdown("""
            El reporte incluira:
            - Distribucion geografica por pais
            - Nube de palabras de los abstracts
            - Linea temporal de publicaciones
            - Distribucion por base de datos
            - Frecuencia de terminos predefinidos
            - Evaluacion de clustering (si fue calculado)
            """)

            if st.button("Generar PDF", type="primary", width='stretch'):
                from visualization import ReportExporter

                stats = {
                    "Total articulos": len(df_main),
                    "Fuentes indexadas": (
                        int(df_main["source_db"].nunique())
                        if "source_db" in df_main else "—"
                    ),
                    "Query de busqueda": DEFAULT_QUERY,
                    "Fecha de generacion": pd.Timestamp.now().strftime("%d/%m/%Y"),
                }

                with st.spinner("Generando PDF..."):
                    exporter = ReportExporter()
                    pdf_bytes = exporter.generate(
                        freq_df=st.session_state.get("freq_df"),
                        cluster_eval_df=st.session_state.get("cluster_eval_df"),
                        dataset_stats=stats,
                    )

                if pdf_bytes:
                    st.download_button(
                        label="Descargar Reporte PDF",
                        data=pdf_bytes,
                        file_name="bibliometric_report.pdf",
                        mime="application/pdf",
                    )
                    st.success("PDF generado exitosamente.")
                else:
                    st.error(
                        "Error al generar el PDF. "
                        "Verifique que reportlab este instalado: pip install reportlab"
                    )
