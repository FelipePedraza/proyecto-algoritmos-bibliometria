# Documentación Técnica — Análisis Bibliométrico de IA Generativa
**Universidad del Quindío · Ingeniería de Sistemas y Computación · Análisis de Algoritmos**  
Cadena de búsqueda: `"generative artificial intelligence"`

---

## Tabla de Contenidos

1. [Descripción General del Proyecto](#1-descripción-general-del-proyecto)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [R1 · Automatización y Unificación Bibliométrica](#3-r1--automatización-y-unificación-bibliométrica)
4. [R2 · Similitud Textual entre Abstracts](#4-r2--similitud-textual-entre-abstracts)
5. [R3 · Análisis de Frecuencia de Términos](#5-r3--análisis-de-frecuencia-de-términos)
6. [R4 · Agrupamiento Jerárquico](#6-r4--agrupamiento-jerárquico)
7. [R5 · Visualización Científica](#7-r5--visualización-científica)
8. [R6 · Despliegue con Docker](#8-r6--despliegue-con-docker)
9. [Guía de Instalación y Ejecución](#9-guía-de-instalación-y-ejecución)
10. [Estructura del Repositorio](#10-estructura-del-repositorio)

---

## 1. Descripción General del Proyecto

Este sistema realiza análisis bibliométrico sobre publicaciones académicas relacionadas con **Inteligencia Artificial Generativa**. Las fuentes de datos son exportaciones CSV de tres bases de datos:

| Base de datos | Tipo | Uso en el proyecto |
|---|---|---|
| ACM Digital Library | Conferencias e industria de computación | Fuente principal obligatoria |
| ScienceDirect (Elsevier) | Revistas científicas | Fuente principal obligatoria |
| EBSCO (Business/Academic Source) | Multi-disciplinar | Fuente opcional — aporta datos geográficos |

El flujo completo parte de la carga manual de archivos CSV (las bases de datos bloquean scraping automatizado), los unifica en un dataset canónico, elimina duplicados y expone cinco módulos de análisis a través de una interfaz web Streamlit.

---

## 2. Arquitectura del Sistema

```
proyecto-algoritmos-bibliometria/
├── app.py                        # Punto de entrada Streamlit
├── requirements.txt              # Dependencias Python
├── Dockerfile                    # Imagen Docker multi-etapa
├── docker-compose.yml            # Orquestación de servicios
├── .streamlit/config.toml        # Configuración del servidor Streamlit
├── src/
│   ├── r1_scraping/              # R1: Parseo y deduplicación
│   │   ├── acm_parser.py
│   │   ├── sciencedirect_parser.py
│   │   ├── ebsco_parser.py
│   │   ├── unifier.py            # Union-Find + Levenshtein
│   │   └── page.py
│   ├── r2_similarity/            # R2: 6 algoritmos de similitud
│   │   ├── algorithms.py
│   │   ├── explanations.py
│   │   └── page.py
│   ├── r3_frequency/             # R3: Frecuencia de términos
│   ├── r4_clustering/            # R4: Clustering jerárquico
│   │   ├── algorithms.py
│   │   ├── explanations.py
│   │   └── page.py
│   ├── r5_visualization/         # R5: Visualización científica
│   │   ├── algorithms.py
│   │   ├── explanations.py
│   │   └── page.py
│   └── utils/
│       └── text_utils.py         # Normalización, Levenshtein helpers
├── data/
│   ├── processed/unified.csv     # Dataset unificado (generado en R1)
│   └── duplicates/duplicates.csv # Registros eliminados (generado en R1)
├── tests/
│   ├── test_r1.py
│   └── test_r2.py
└── docs/
    └── documentacion_tecnica.md  # Este documento
```

### Patrón de módulo

Cada requerimiento sigue el mismo patrón de tres capas:

```
src/rN_<nombre>/
├── algorithms.py    # Lógica computacional pura (sin UI, testeable)
├── explanations.py  # Trazas paso a paso para propósitos académicos
└── page.py          # Interfaz Streamlit (llama a algorithms.py)
```

Este patrón separa la lógica de la presentación, facilitando las pruebas unitarias y la reutilización.

### Flujo de datos entre requerimientos

```
CSV (ACM)      ─┐
CSV (SD)        ├─ R1 (unifier.py) ──► unified.csv ──► R2, R3, R4, R5
CSV (EBSCO) ───┘
```

---

## 3. R1 · Automatización y Unificación Bibliométrica

### Objetivo

Cargar exportaciones CSV de ACM, ScienceDirect y EBSCO, normalizar cada fuente a un esquema canónico, y eliminar duplicados cross-base usando similitud de títulos.

### Esquema canónico de salida

| Campo | Fuente ACM | Fuente ScienceDirect | Fuente EBSCO |
|---|---|---|---|
| `title` | `Title` | `Title` | `Title` |
| `authors` | `Authors` | `Authors` | `Author` |
| `year` | `Publication Year` | `Year` | `Publication Date` |
| `source` | `Publication Title` | `Source title` | `Source` |
| `abstract` | `Abstract` | `Abstract` | `Abstract` |
| `doi` | `DOI` | `DOI` | `DOI` |
| `keywords` | `Keywords` | `Author Keywords` | `Keywords` |
| `source_db` | `"ACM"` | `"ScienceDirect"` | `"EBSCO"` |
| `country` | — | — | `authorLocations` (parseado) |

### Algoritmo de deduplicación: Union-Find + Levenshtein

**¿Por qué Union-Find?** Las cadenas de duplicados son transitivas: si A ≈ B y B ≈ C, entonces A, B y C son el mismo artículo. Union-Find captura este comportamiento en O(α(n)) por operación.

**Proceso completo:**

1. **Normalización de títulos**: se eliminan acentos (Unidecode), se convierte a minúsculas y se colapsan espacios múltiples. Esto garantiza que `"AI in Healthcare"` y `"ai in healthcare"` sean comparados como iguales.

2. **Comparación por pares**: se calcula la similitud de Levenshtein entre cada par (i, j) de títulos normalizados. Complejidad: O(n² × L²), donde n es el número de registros y L es la longitud máxima del título.

3. **Clustering con Union-Find**: si la similitud ≥ umbral (defecto 0.92), se unen los dos elementos en el mismo componente.

4. **Selección del representante**: por cada componente con más de un elemento, se elige como representante el registro con mayor completitud de campos (más columnas no vacías).

**Fórmula de similitud de Levenshtein:**

```
similitud(s1, s2) = 1 - d_lev(s1, s2) / max(|s1|, |s2|)
```

Donde `d_lev(s1, s2)` es la distancia de edición mínima (inserciones + eliminaciones + sustituciones).

**Complejidad total del módulo:**

| Operación | Complejidad temporal | Complejidad espacial |
|---|---|---|
| Normalización (n títulos) | O(n × L) | O(n × L) |
| Comparación por pares | O(n² × L²) | O(n²) |
| Union-Find (α ≈ constante) | O(n² × α(n)) | O(n) |
| **Total** | **O(n² × L²)** | **O(n²)** |

**Umbral configurable (0.80 – 1.00):** El valor por defecto 0.92 balancea precisión (no unir artículos distintos con títulos similares) y cobertura (no dejar duplicados sin detectar). Valores más bajos generan más fusiones; valores más altos son más conservadores.

### Salidas

| Archivo | Contenido |
|---|---|
| `unified.csv` | Registros únicos con columnas canónicas |
| `duplicates.csv` | Registros descartados, con referencia al registro que los reemplaza |

---

## 4. R2 · Similitud Textual entre Abstracts

### Objetivo

Dado el dataset unificado, permitir seleccionar dos artículos y comparar sus abstracts con 6 algoritmos distintos de similitud textual, mostrando la traza matemática completa de cada uno.

### Algoritmos implementados

#### 4.1 Distancia de Levenshtein

Mide el número mínimo de operaciones de edición (inserción, eliminación, sustitución de un carácter) para transformar un texto en otro.

**Implementación:** Programación dinámica con matriz (n+1)×(m+1).

```
dp[i][j] = 0                         si i=0, j=0
dp[i][0] = i                         caso base: eliminar i caracteres
dp[0][j] = j                         caso base: insertar j caracteres
dp[i][j] = dp[i-1][j-1]             si s1[i] == s2[j]  (sin costo)
          min(dp[i-1][j] + 1,       eliminación
              dp[i][j-1] + 1,       inserción
              dp[i-1][j-1] + 1)     sustitución
```

**Similitud:** `1 - d / max(|s1|, |s2|)`  
**Complejidad:** O(n × m) temporal, O(n × m) espacial

#### 4.2 Similitud de Jaccard

Opera sobre conjuntos de tokens (palabras). Mide la proporción de palabras compartidas respecto al total de palabras distintas.

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

Donde A y B son conjuntos de palabras (tokens únicos) de cada texto.  
**Complejidad:** O(n + m) donde n, m son los conteos de tokens

#### 4.3 Coseno con TF-IDF

Vectoriza cada texto usando TF-IDF y mide el ángulo entre los vectores resultantes.

```
TF(t, d)    = freq(t, d) / |d|
IDF(t)      = log((N+1) / (df(t)+1)) + 1    [suavizado]
TF-IDF(t,d) = TF(t,d) × IDF(t)

similitud_coseno(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

**Complejidad:** O(V) donde V es el tamaño del vocabulario

#### 4.4 Distancia de Hamming Normalizada

Compara carácter a carácter. Para textos de distinta longitud, el más corto se rellena con espacios.

```
d_Hamming(s1, s2) = #{posiciones donde s1_padded[i] ≠ s2_padded[i]}
similitud = 1 - d_Hamming / max(|s1|, |s2|)
```

**Complejidad:** O(max(|s1|, |s2|)) temporal y espacial

#### 4.5 Sentence-BERT (IA — Transformers)

Modelo `all-MiniLM-L6-v2` (22M parámetros) que codifica cada texto como un vector denso de 384 dimensiones capturando significado semántico.

**Proceso:**
1. Tokenización subword (WordPiece)
2. Codificación mediante capas Transformer
3. Mean pooling sobre tokens para obtener el vector de oración
4. Similitud del coseno entre embeddings

Textos semánticamente equivalentes con palabras distintas ("AI applications" ≈ "artificial intelligence use cases") obtienen similitud alta.

#### 4.6 spaCy Word Vectors (IA — GloVe)

Modelo `en_core_web_md` (vectores GloVe, 300 dimensiones). Asigna un vector a cada palabra del vocabulario y calcula el vector promedio del texto.

**Diferencia con SBERT:** Los vectores GloVe son estáticos (la misma palabra siempre tiene el mismo vector, sin contexto), lo que lo hace más rápido pero menos preciso para frases con ambigüedad.

### Comparativa de algoritmos

| Algoritmo | Tipo | Captura semántica | Velocidad | Memoria |
|---|---|---|---|---|
| Levenshtein | Clásico — edición | No | Media | O(n×m) |
| Jaccard | Clásico — conjuntos | Parcial | Alta | O(V) |
| Coseno TF-IDF | Clásico — vectorial | Estadística | Alta | O(V) |
| Hamming | Clásico — edición | No | Alta | O(L) |
| Sentence-BERT | IA — Transformers | Alta | Baja | 90 MB |
| spaCy GloVe | IA — word vectors | Media | Media | 45 MB |

---

## 5. R3 · Análisis de Frecuencia de Términos

### Objetivo

Identificar los términos más frecuentes en los abstracts del dataset unificado para caracterizar el estado del arte de la IA Generativa en la literatura académica.

### Estado

El módulo de visualización (R5) ya incorpora el análisis de frecuencias mediante la función `compute_word_frequencies()` en `src/r5_visualization/algorithms.py`, que implementa el pipeline completo:

1. **Tokenización:** minúsculas → eliminación de URLs y números → eliminación de puntuación
2. **Filtrado:** tokens de longitud ≥ 3, exclusión de stop-words (inglés + español + términos bibliométricos genéricos)
3. **Ponderación de keywords:** los términos provenientes de keywords explícitas de los autores reciben un peso multiplicador (por defecto ×3) respecto a los del cuerpo del abstract
4. **Ranking:** se retorna el top-N de palabras por frecuencia acumulada

**Complejidad:** O(D × L) donde D = número de documentos, L = longitud promedio del abstract

La interfaz de R5 expone esta funcionalidad con parámetros configurables (top-N, fuentes incluidas, pesos, stop-words adicionales).

---

## 6. R4 · Agrupamiento Jerárquico

### Objetivo

Agrupar los artículos del dataset de acuerdo con la similitud de sus abstracts, usando tres algoritmos de clustering jerárquico aglomerativo implementados desde cero.

### Pipeline completo

```
abstracts (texto crudo)
    │
    ▼ preprocess_text()
tokens preprocesados
    │
    ▼ compute_tfidf_vectors()
matriz TF-IDF  [N × V]
    │
    ▼ compute_distance_matrix()
matriz de distancias coseno  [N × N]
    │
    ├──► complete_linkage()  ──► dendrograma + coef. cofenético
    ├──► average_linkage()   ──► dendrograma + coef. cofenético
    └──► ward_linkage()      ──► dendrograma + coef. cofenético
```

### Preprocesamiento de texto

| Paso | Transformación | Ejemplo |
|---|---|---|
| 1. Minúsculas | `"Generative AI"` → `"generative ai"` | |
| 2. Limpieza | Elimina no-alfabéticos | `"GPT-4,"` → `"gpt "` |
| 3. Tokenización | División por espacios | `["generative", "ai"]` |
| 4. Filtro longitud | Descarta tokens < 3 chars | `"ai"` eliminado |
| 5. Stop-words | Elimina 150+ palabras vacías | `"the"`, `"is"` eliminados |
| 6. Stemming | Sufijos comunes en inglés | `"learning"` → `"learn"` |

### Vectorización TF-IDF

```
TF(t, d)    = freq(t, d) / |d|
IDF(t)      = log((N+1) / (df(t)+1)) + 1
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

Complejidad: O(N × V) donde N = documentos, V = vocabulario

### Matriz de distancias coseno

```
d_coseno(A, B) = 1 - (A · B) / (‖A‖ × ‖B‖)    ∈ [0, 1]
```

- `d = 0`: abstracts idénticos (mismo vocabulario con mismos pesos)
- `d = 1`: sin vocabulario en común

### Algoritmos de Clustering Jerárquico Aglomerativo (HAC)

Los tres implementan el mismo esquema iterativo:

```
1. Inicializar: cada documento = su propio cluster
2. Repetir (N-1) veces:
   a. Encontrar el par de clusters más cercanos (búsqueda O(n²))
   b. Fusionarlos → nuevo cluster
   c. Actualizar matriz de distancias según la regla del método
3. Retornar linkage matrix [(cluster_a, cluster_b, dist, tamaño), ...]
```

#### Enlace Completo (Complete Linkage)

```
d(A∪B, C) = max(d(A,C), d(B,C))
```

Distancia entre clusters = distancia entre los puntos **más alejados** (vecino más lejano). Produce clusters compactos pero es sensible a outliers.

#### Enlace Promedio (Average Linkage — UPGMA)

```
d(A∪B, C) = (|A|·d(A,C) + |B|·d(B,C)) / (|A|+|B|)
```

Distancia = promedio ponderado por tamaño. Compromiso entre enlace completo y simple; generalmente produce el mejor coeficiente cofenético.

#### Método de Ward

Minimiza el incremento de varianza intra-cluster al fusionar. Usa la fórmula de Lance-Williams:

```
d(A∪B, C)² = [(nA+nC)·d(A,C)² + (nB+nC)·d(B,C)² − nC·d(A,B)²] / (nA+nB+nC)
```

Produce clusters de tamaño equilibrado. Opera internamente con distancias al cuadrado.

**Complejidad de los 3 algoritmos:** O(n³) — implementación naive para claridad académica

### Evaluación: Coeficiente de Correlación Cofenética

Mide qué tan bien el dendrograma representa las distancias originales (Sokal & Rohlf, 1962):

```
ccc = Pearson(distancias_originales, distancias_cofenéticas)
```

| Rango | Interpretación |
|---|---|
| > 0.90 | Representación excelente |
| 0.75 – 0.90 | Representación aceptable |
| < 0.75 | Dendrograma distorsionado |

La app selecciona automáticamente el algoritmo con mayor CCC como "mejor método".

---

## 7. R5 · Visualización Científica

### Objetivo

Generar visualizaciones interactivas del dataset para comunicar los hallazgos del análisis bibliométrico.

### Visualización 1 — Mapa de Calor Geográfico

Muestra la distribución mundial de publicaciones por país del primer autor.

**Pipeline:**
1. `detect_country_column(df)` — detecta la columna con información geográfica (prioridad: `country`, luego `authorLocations` de EBSCO)
2. `extract_country_from_text(text)` — normaliza texto libre → nombre canónico del país mediante regex sobre un diccionario de 80+ alias
3. `build_geo_dataframe(counts)` — mapea país → código ISO Alpha-3 para el choropleth de Plotly

Implementado en Plotly Express (`choropleth`) con fallback a barras horizontales Matplotlib para exportación PDF.

### Visualización 2 — Nube de Palabras

Términos más frecuentes en abstracts y keywords del dataset.

Las keywords de los autores reciben peso ×3 porque son términos seleccionados explícitamente para representar el trabajo.

Implementado con la librería `wordcloud`; fallback a gráfico de barras Matplotlib.

### Visualización 3 — Línea Temporal

Evolución del volumen de publicaciones por año, con panel secundario de distribución por revista/conferencia (top-8 fuentes).

### Exportación PDF

`generate_pdf_report()` genera un PDF de 4 páginas (portada + 3 visualizaciones) usando `matplotlib.backends.backend_pdf.PdfPages`.

---

## 8. R6 · Despliegue con Docker

### Objetivo

Empaquetar toda la aplicación — incluyendo los modelos de IA — en una imagen Docker reproducible, con instrucciones claras de despliegue.

### Estrategia: Build Multi-Etapa (Multi-Stage Build)

El `Dockerfile` usa 3 etapas para minimizar el tamaño de la imagen final:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Etapa 1: builder                                                    │
│  python:3.11-slim + build-essential                                  │
│  Instala todas las dependencias Python                               │
│  → /install (paquetes compilados)                                    │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ COPY --from=builder /install
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Etapa 2: model-downloader                                           │
│  python:3.11-slim (con dependencias del builder)                     │
│  Descarga en_core_web_md (~45 MB) y all-MiniLM-L6-v2 (~90 MB)       │
│  → /root/.cache/huggingface  y  site-packages/en_core_web_md         │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ COPY --from=model-downloader
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Etapa 3: production (imagen final)                                  │
│  python:3.11-slim (solo runtime, sin compiladores)                   │
│  Contiene: dependencias + modelos + código de la app                 │
│  EXPOSE 8501                                                         │
│  CMD streamlit run app.py                                            │
└──────────────────────────────────────────────────────────────────────┘
```

**Beneficio:** La imagen final no contiene `gcc`, `g++` ni artefactos de compilación, reduciéndose significativamente su tamaño.

### Variables de entorno relevantes

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `STREAMLIT_SERVER_PORT` | `8501` | Puerto del servidor |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Escucha en todas las interfaces |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Sin navegador automático |
| `TRANSFORMERS_CACHE` | `/app/.cache/huggingface` | Cache de modelos HuggingFace |
| `HF_HOME` | `/app/.cache/huggingface` | Home de HuggingFace |

### Volúmenes persistentes (docker-compose)

| Volumen | Ruta en contenedor | Propósito |
|---|---|---|
| `bibliometria_data` | `/app/data/processed` | Conserva `unified.csv` entre reinicios |
| `bibliometria_duplicates` | `/app/data/duplicates` | Conserva `duplicates.csv` entre reinicios |

---

## 9. Guía de Instalación y Ejecución

### Opción A — Docker (recomendado para despliegue)

**Prerrequisitos:** Docker Desktop instalado y en ejecución.

```bash
# 1. Clonar o descargar el repositorio
git clone <url-del-repositorio>
cd proyecto-algoritmos-bibliometria

# 2. Construir y levantar el contenedor
#    (la primera vez descarga ~150 MB de modelos de IA — tarda ~5 min)
docker compose up --build

# 3. Abrir en el navegador
#    http://localhost:8501

# 4. Detener el contenedor
docker compose down
```

**Comandos útiles:**

```bash
# Ver logs en tiempo real
docker compose logs -f bibliometria

# Reconstruir sin usar caché (tras cambios en requirements.txt)
docker compose build --no-cache

# Ejecutar en segundo plano
docker compose up -d

# Ver el estado del healthcheck
docker inspect bibliometria_app --format='{{.State.Health.Status}}'

# Acceder a la shell del contenedor
docker exec -it bibliometria_app /bin/bash
```

### Opción B — Entorno local (desarrollo)

**Prerrequisitos:** Python 3.10+ instalado.

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar modelo spaCy (requerido para R2)
python -m spacy download en_core_web_md

# 4. Iniciar la aplicación
streamlit run app.py

# 5. Ejecutar pruebas
pytest tests/ -v
```

### Notas importantes sobre los modelos de IA

- **Sentence-BERT** (`all-MiniLM-L6-v2`): se descarga automáticamente en el primer uso si no está en caché. En Docker ya está pre-descargado en la imagen.
- **spaCy** (`en_core_web_md`): debe instalarse explícitamente con `python -m spacy download en_core_web_md`. En Docker ya está incluido.
- **Tiempo de carga inicial:** la primera vez que se usa R2 en una sesión, los modelos toman ~10-30 segundos en cargar. Las siguientes llamadas son instantáneas (singleton pattern).

---

## 10. Estructura del Repositorio

```
proyecto-algoritmos-bibliometria/
│
├── app.py                          # Entrada principal — configuración y routing Streamlit
├── requirements.txt                # Dependencias Python con versiones mínimas
├── Dockerfile                      # Imagen Docker multi-etapa (builder/model-downloader/production)
├── docker-compose.yml              # Orquestación: servicio + volúmenes persistentes
├── .dockerignore                   # Exclusiones del contexto de build Docker
├── .streamlit/
│   └── config.toml                 # Puerto, tema, límite de upload (200 MB)
│
├── src/
│   ├── __init__.py
│   ├── r1_scraping/
│   │   ├── acm_parser.py           # Parser CSV de ACM Digital Library
│   │   ├── sciencedirect_parser.py # Parser CSV de ScienceDirect
│   │   ├── ebsco_parser.py         # Parser CSV de EBSCO + extracción de país
│   │   ├── unifier.py              # Union-Find + Levenshtein para deduplicación
│   │   └── page.py                 # UI de R1
│   ├── r2_similarity/
│   │   ├── algorithms.py           # 6 algoritmos: Levenshtein, Jaccard, Coseno TF-IDF,
│   │   │                           #   Hamming, Sentence-BERT, spaCy
│   │   ├── explanations.py         # Trazas matemáticas paso a paso
│   │   └── page.py                 # UI de R2
│   ├── r3_frequency/
│   │   └── __init__.py             # (integrado en R5 via compute_word_frequencies)
│   ├── r4_clustering/
│   │   ├── algorithms.py           # Complete/Average/Ward linkage + coef. cofenético
│   │   ├── explanations.py         # Trazas de clustering
│   │   └── page.py                 # UI de R4 con dendrogramas interactivos
│   ├── r5_visualization/
│   │   ├── algorithms.py           # Mapa geográfico, nube de palabras, línea temporal, PDF
│   │   ├── explanations.py         # Documentación interna de visualizaciones
│   │   └── page.py                 # UI de R5 con Plotly + Matplotlib
│   └── utils/
│       └── text_utils.py           # Normalización de texto, helpers de Levenshtein
│
├── data/
│   ├── processed/
│   │   └── unified.csv             # Dataset unificado (generado por R1)
│   └── duplicates/
│       └── duplicates.csv          # Registros eliminados (generado por R1)
│
├── tests/
│   ├── test_r1.py                  # Tests de parsers, Union-Find, Levenshtein
│   └── test_r2.py                  # Tests de los 6 algoritmos de similitud
│
└── docs/
    ├── documentacion_tecnica.md    # Este documento
    └── Proyecto Análisis de Algoritmos.pdf  # Enunciado original del proyecto
```

---

*Documentación para el proyecto de Análisis de Algoritmos — Universidad del Quindío.*  
*Última actualización: Mayo 2026.*
