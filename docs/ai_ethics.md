# Justificación del Uso de Modelos de IA y Consideraciones Éticas
## Bibliometrics AI Analyzer — Req 6.4
---

## 1. Justificación Técnica del Uso de Modelos de IA

### 1.1 Por qué modelos de IA para similitud textual

Los algoritmos clásicos de similitud textual (Levenshtein, Jaccard, TF-IDF) operan sobre la superficie del texto: comparan tokens, caracteres o frecuencias de palabras. Tienen una limitación fundamental: **no comprenden el significado semántico del lenguaje**.

Por ejemplo, los siguientes dos abstracts describen el mismo concepto con vocabulario diferente:

- *"We propose a generative model based on transformer architecture that produces synthetic text..."*
- *"A novel deep learning system capable of creating artificial text sequences was developed..."*

Un algoritmo Jaccard obtendría una similitud muy baja (vocabularios distintos), mientras que un modelo Sentence-BERT reconoce que ambas frases son semánticamente equivalentes, produciendo una similitud de ~0.87.

Esta distinción es crítica en bibliometría donde:
1. Los autores usan terminología variada para los mismos conceptos.
2. Los abstracts son reformulaciones del mismo hallazgo.
3. La detección de estudios duplicados o cercanos requiere comprensión semántica.

### 1.2 Selección de modelos: all-MiniLM-L6-v2 y paraphrase-MiniLM-L12-v2

Se eligieron estos modelos por las siguientes razones técnicas:

| Criterio | all-MiniLM-L6-v2 | paraphrase-MiniLM-L12-v2 |
|----------|-----------------|--------------------------|
| Tamaño | 22 MB | 118 MB |
| Velocidad (CPU) | ~14k oraciones/s | ~4k oraciones/s |
| Score MTEB (STS) | 0.567 | 0.595 |
| Licencia | Apache 2.0 | Apache 2.0 |
| Uso óptimo | Búsqueda semántica general | Detección de paráfrasis |

Ambos modelos son abiertos, auditables y se pueden ejecutar completamente **offline**, lo que garantiza privacidad de los datos del corpus. No se envía información a terceros.

### 1.3 Por qué KeyBERT para extracción de palabras clave

KeyBERT supera a los métodos estadísticos (YAKE, TF-IDF) en textos científicos porque:
- Considera el contexto semántico completo del abstract, no solo frecuencias.
- Usa la técnica MMR (Maximal Marginal Relevance) para evitar palabras clave redundantes.
- Captura términos compuestos significativos como "large language models" o "retrieval augmented generation".

---

## 2. Marco Ético del Proyecto

### 2.1 Principios Éticos Aplicados

El proyecto fue diseñado considerando los principios éticos de la IA establecidos por la **UNESCO (2021)** y la **Comisión Europea — AI Act**:

####  Transparencia y Explicabilidad
- Todos los algoritmos están documentados matemáticamente en el código fuente (inline documentation exhaustiva).
- Los scores de similitud se presentan en rangos [0,1] claramente interpretables.
- La métrica de precisión de palabras clave descompone el score en 4 indicadores auditables.
- Los modelos de IA usados son **open-weight** y sus arquitecturas son públicas.

####  Equidad y No-discriminación
- El sistema analiza publicaciones de cualquier país y región geográfica sin priorización.
- La visualización geográfica tiene el propósito explícito de **identificar brechas** en la producción científica global, no de jerarquizar países.
- Los modelos Sentence-BERT usan variantes multilingües opcionales para no sesgar hacia el inglés.

####  Sesgo Algorítmico (Algorithmic Bias)
- Los modelos de lenguaje preentrenados pueden tener sesgos de sus datos de entrenamiento. Se mitigó:
  - Usando 2 modelos diferentes para cruzar resultados.
  - Combinando con métodos clásicos (no sesgados de la misma manera).
  - La precisión de palabras extraídas se evalúa con un score compuesto, no un único modelo.
- El campo `source_db` en el dataset permite a los investigadores identificar si los resultados varían por fuente de datos.

####  Privacidad y Datos
- El sistema trabaja exclusivamente con **datos bibliográficos públicos** obtenidos de APIs abiertas.
- No se recopila ningún dato personal de usuarios.
- Los abstracts y metadatos son información pública de acceso abierto.
- Los modelos se ejecutan **localmente** — ningún abstract se envía a servicios externos.

####  Reproducibilidad y Rigor Científico
- Todas las versiones de paquetes están fijadas en `requirements.txt`.
- Los algoritmos de clustering usan `random_state=42` para resultados reproducibles.
- El dataset generado incluye la columna `source_db` para trazabilidad de origen.

### 2.2 Limitaciones y Riesgos

| Riesgo | Nivel | Mitigación |
|--------|-------|-----------|
| Sesgo de fuente (solo artículos en inglés) | Medio | Usar variante multilingual de los modelos |
| Dependencia de la calidad de APIs | Medio | 3 fuentes independientes compensan fallos parciales |
| Alucinaciones de KeyBERT en corpus pequeños | Bajo | Mínimo requerido de 20 tokens por abstract |
| Ward Linkage sensible a outliers | Bajo | Se proveen 3 algoritmos con evaluación automática |

### 2.3 Consideraciones de Uso Responsable

El sistema fue diseñado como **herramienta de apoyo** a la toma de decisiones académicas, NO como reemplazo del juicio de expertos. Sus resultados deben ser:

1. **Interpretados por especialistas** en el dominio de la IA generativa.
2. **Complementados** con búsquedas manuales en bases de datos institucionales (Scopus, WoS).
3. **Actualizados regularmente** ya que la literatura sobre IA generativa crece exponencialmente.

---

## 3. Impacto del Proyecto

La automatización bibliométrica con IA tiene impactos positivos documentados en la literatura:

- **Ahorro de tiempo:** Un Systematic Literature Review manual toma 100-200 horas; este sistema reduce ese proceso a horas.
- **Reproducibilidad:** El pipeline automatizado elimina errores de selección manual.
- **Descubrimiento:** Los algoritmos de clustering identifican sub-temas emergentes que un humano podría no percibir en cientos de abstracts.
- **Equidad geográfica:** La visualización de distribución geográfica hace visibles las asimetrías en la producción de conocimiento sobre IA generativa.

---

*Documento preparado en cumplimiento del Requerimiento 6 del proyecto académico "Análisis de Algoritmos en el Contexto de la Bibliometría".*
