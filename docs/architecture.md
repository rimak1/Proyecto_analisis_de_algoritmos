# Documento de Diseño de Arquitectura
## Bibliometrics AI Analyzer — Análisis de Algoritmos en Bibliometría
**Dominio:** Generative Artificial Intelligence | **Versión:** 1.0 | **Fecha:** Marzo 2025

---

## 1. Descripción General del Sistema

El **Bibliometrics AI Analyzer** es un sistema modular de análisis de literatura científica que automatiza la recopilación, procesamiento y análisis de artículos académicos sobre **Inteligencia Artificial Generativa**. Combina métodos clásicos de bibliometría con técnicas avanzadas de aprendizaje automático y Procesamiento de Lenguaje Natural (NLP).

---

## 2. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CAPA DE PRESENTACIÓN                             │
│                    Streamlit Web App (app.py)                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Tab: Data│ │Tab: Sim. │ │ Tab: NLP │ │Tab:Clust.│ │ Tab: Viz │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ Llamadas a módulos Python
┌────────────────────────────────▼────────────────────────────────────────┐
│                         CAPA DE LÓGICA DE NEGOCIO                       │
│                                                                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ data_extraction/│  │   similarity/    │  │       nlp/            │  │
│  │  fetcher.py     │  │  classical.py    │  │  frequency.py         │  │
│  │  unifier.py     │  │  ai_models.py    │  │  keyword_extractor.py │  │
│  │  deduplicator.py│  │  interface.py    │  │  precision_metric.py  │  │
│  └────────┬────────┘  └────────┬─────────┘  └──────────┬────────────┘  │
│           │                   │                         │               │
│  ┌────────▼────────┐  ┌────────▼──────────────────────▼────────────┐  │
│  │  clustering/    │  │          visualization/                     │  │
│  │  preprocessor   │  │  heatmap.py     wordcloud_viz.py           │  │
│  │  algorithms.py  │  │  timeline.py    report.py                  │  │
│  │  evaluator.py   │  └────────────────────────────────────────────┘  │
│  └─────────────────┘                                                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                          CAPA DE DATOS                                  │
│                                                                         │
│  ┌─────────────┐  ┌──────────────────────┐  ┌────────────────────────┐ │
│  │  data/raw/  │  │  data/processed/     │  │   Modelos IA (cache)   │ │
│  │  (CSVs por  │  │  unified_dataset.csv │  │  all-MiniLM-L6-v2      │ │
│  │   fuente)   │  │  duplicates_.csv     │  │  paraphrase-MiniLM-L12 │ │
│  └─────────────┘  └──────────────────────┘  └────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTP / API REST
┌────────────────────────────────▼────────────────────────────────────────┐
│                       FUENTES EXTERNAS DE DATOS                         │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────────┐   │
│  │   OpenAlex   │  │   CrossRef   │  │      Semantic Scholar       │   │
│  │   (free API) │  │   (free API) │  │         (free API)          │   │
│  └──────────────┘  └──────────────┘  └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Descripción de Componentes y Decisiones de Diseño

### 3.1 Req 1 — Extracción y Unificación de Datos

#### `data_extraction/fetcher.py`
**Responsabilidad:** Automatizar la descarga desde 3 APIs públicas.

**Fuentes y justificación:**
| Fuente | Tipo | Cobertura | Razón de uso |
|--------|------|-----------|-------------|
| **OpenAlex** | API REST | 250M+ registros | Sustituto open-source de Scopus/WoS, compatible con ACM/IEEE/Springer |
| **CrossRef** | API REST | 140M+ DOIs | Amplia cobertura, excelente para metadata de journal articles |
| **Semantic Scholar** | API REST | 200M+ papers | Especializado en CS/AI, acceso gratuito |

**Patrón de diseño:** *Strategy Pattern* — cada fuente implementa el mismo contrato de método `fetch_X()` y retorna un `pd.DataFrame` con columnas `STANDARD_COLUMNS`.

**Limitación de velocidad (rate limiting):** `time.sleep(0.2-0.5)` entre requests para respetar los TOS de cada API.

#### `data_extraction/unifier.py`
**Responsabilidad:** Normalizar columnas heterogéneas de distintas fuentes a un esquema estándar de 12 columnas.

**Columnas estándar:** `id, title, authors, year, abstract, keywords, journal, doi, source_db, country, citations, url`

**Normalización de texto:** Uso de `unicodedata.normalize("NFKC")` + regex para garantizar consistencia UTF-8.

#### `data_extraction/deduplicator.py`
**Responsabilidad:** Garantizar una sola instancia por producto (sin duplicados).

**Pipeline de 3 etapas (cascada de filtros de creciente costo computacional):**

```
Paso 1: DOI exacto normalizado     → O(n)      — elimina ~40-60% de duplicados
Paso 2: Clave de título canónico   → O(n log n) — elimina otros ~20-30%
Paso 3: Similitud fuzzy (SequenceMatcher) → O(n²) — solo para ≤2000 docs
```

La elección de tres etapas permite eficiencia: los pasos más baratos filtran
la mayoría de duplicados antes del costoso paso fuzzy.

**Archivo secundario:** Todos los registros eliminados se guardan en `data/processed/duplicates_removed.csv` con columna `_duplicate_reason` que indica cuál de los 3 criterios lo detectó.

---

### 3.2 Req 2 — Algoritmos de Similitud Textual

**Decisión de diseño:** Los 6 algoritmos están separados en 2 módulos (`classical.py` y `ai_models.py`) por su naturaleza fundamentalmente diferente:

- Los 4 clásicos son **deterministas** y no requieren modelos entrenados.
- Los 2 de IA son **estadísticos** y requieren carga de modelos pesados (~90-200 MB cada uno).

La clase `SimilarityInterface` actúa como **Façade** que oculta la complejidad de ambos módulos al consumidor (la app Streamlit).

**Carga perezosa (Lazy Loading):** Los modelos Sentence-BERT se cargan solo cuando el usuario los solicita (`_model = None` hasta primer uso).

| Algoritmo | Módulo | Tiempo por par | Captura semántica |
|-----------|--------|---------------|-------------------|
| Levenshtein | classical | ~1 ms | No |
| Jaccard | classical | <1 ms | No |
| TF-IDF Cosine | classical | ~10 ms | Parcial (bag-of-words) |
| N-grama bigram | classical | ~1 ms | Morfológica |
| Sentence-BERT | ai_models | ~50 ms (CPU) | Alta |
| Paraphrase-MiniLM | ai_models | ~80 ms (CPU) | Alta (paráfrasis) |

---

### 3.3 Req 3 — Frecuencia de Términos y Extracción NLP

**`frequency.py`:** Búsqueda con regex `\b...\b` + aliases por término para evitar falsos negativos (ej: `fine-tuning` también detecta `finetuning`, `fine tuning`).

**`keyword_extractor.py`:** Pipeline de **fusión multi-extractor**:
- **YAKE:** score estadístico basado en posición, frecuencia y dispersión del término en el documento.
- **KeyBERT:** utiliza embeddings BERT para encontrar frases cuyo embedding es más cercano al embedding del documento completo.
- **TF-IDF n-gramas:** baseline estadístico que captura términos informativos por su rareza en el corpus.

Los 3 resultados se fusionan con **votación ponderada** (KeyBERT 45%, YAKE 35%, TF-IDF 20%) basada en la precisión reportada en la literatura para textos científicos en inglés.

**`precision_metric.py`:** Score compuesto de 4 indicadores:
1. **Relevancia semántica (40%):** proporción de documentos del corpus donde el término tiene TF-IDF > 0.
2. **Consenso de extractores (25%):** cuántos de los 3 extractores lo identificaron.
3. **Proximidad al dominio (20%):** similitud coseno del término vs. el query "generative artificial intelligence".
4. **Especificidad IDF (15%):** términos muy comunes tienen IDF bajo → baja especificidad.

---

### 3.4 Req 4 — Clustering Jerárquico

**`preprocessor.py`:** Pipeline TF-IDF + distancia coseno:
- Stemming con `SnowballStemmer("english")` para reducir variantes de palabras.
- Vectorización TF-IDF con n-gramas (1,2) y máximo 2000 features.
- Distancia: `d = 1 - similitud_coseno` (métrica de distancia válida ∈ [0, 1]).

**`algorithms.py`:** Los 3 métodos de linkage se eligen por:
- **Ward:** mejor para clusters compactos, minimiza varianza intraclúster. Más usado en bibliometría jerárquica.
- **Complete:** produce clusters compactos y redondos, robusto al ruido.
- **Average (UPGMA):** el más usado en análisis filogenético/bibliométrico; compromiso estable.

**`evaluator.py`:** Evaluación automática sin parámetros de usuario:
- Cophenetic: evalúa si el árbol de clustering representa fielmente las distancias originales.
- Silueta: busca el n_clusters óptimo en un rango {3,4,5,6,7} por búsqueda en grid.
- Calinski-Harabász: no requiere especificar n_clusters, evalúa la razón varianza inter/intra.

---

### 3.5 Req 5 — Visualización

**Decisión:** Plotly para visualizaciones interactivas (choropleth, línea de tiempo), Matplotlib para la nube de palabras (WordCloud library necesita surface bitmap), ReportLab para la generación de PDF composable.

| Visualización | Librería | Interactividad |
|---------------|----------|---------------|
| Mapa coroplético | Plotly | Alta (zoom, hover) |
| Nube de palabras | Matplotlib + WordCloud | Estática (configurable) |
| Línea temporal | Plotly | Alta (zoom, filtros) |
| PDF export | ReportLab | N/A |

**Actualización dinámica de la nube de palabras:** se detecta el `mtime` del archivo CSV. Si cambia (nuevos estudios), la nube se regenera automáticamente al volver a presionar el botón.

---

## 4. Flujo de Datos End-to-End

```
APIs Externas
    │
    ▼ HTTP GET (paginado)
[DataFetcher]  ──► data/raw/{source}_raw.csv
    │
    ▼
[DataUnifier]  ──► normalización de columnas
    │
    ▼
[Deduplicator] ──► data/processed/unified_dataset.csv
                    data/processed/duplicates_removed.csv
    │
    ├──► [SimilarityInterface] ──► scores 0-1 por algoritmo
    │
    ├──► [TermFrequencyAnalyzer] ──► tabla de frecuencias
    │    [KeywordExtractor] ──► top-15 palabras nuevas
    │    [PrecisionMetric] ──► score de precisión
    │
    ├──► [ClusteringPreprocessor] ──► matriz distancias
    │    [HierarchicalClustering] ──► dendrogramas
    │    [ClusteringEvaluator] ──► mejor método
    │
    └──► [GeographicHeatmap] → [WordCloudViz] → [PublicationTimeline]
             ▼
         [ReportExporter] ──► bibliometric_report.pdf
```

---

## 5. Principios Arquitectónicos Aplicados

| Principio | Implementación |
|-----------|----------------|
| **SRP** (Single Responsibility) | Cada módulo tiene UNA responsabilidad: fetching, unifying, deduplication, etc. |
| **Open/Closed** | Agregar una nueva fuente API solo requiere un nuevo método `fetch_X()` sin modificar el resto |
| **DRY** | `config.py` centraliza todas las constantes; `STANDARD_COLUMNS` se define una vez |
| **Fail Fast** | Validación de columnas y rangos al inicio de cada clase |
| **Lazy Loading** | Modelos BERT se cargan solo cuando se necesitan |
| **Separation of Concerns** | Presentación (app.py) separada de la lógica (módulos) |

---

## 6. Stack Tecnológico Detallado

```
Lenguaje:           Python 3.10+
UI / Despliegue:    Streamlit 1.32+
Datos:              Pandas, NumPy
NLP Clásico:        NLTK, Scikit-learn, SciPy
NLP con IA:         HuggingFace Transformers, Sentence-Transformers
Keyword Extract:    YAKE, KeyBERT
Visualización:      Plotly, Matplotlib, WordCloud, Folium
PDF:                ReportLab
Contenedores:       Docker, Docker Compose
Fuentes de datos:   OpenAlex API, CrossRef API, Semantic Scholar API
```
