"""
Requerimiento 5 — Explicaciones Educativas
===========================================
Textos paso a paso de los algoritmos y métodos usados en R5:
  1. Extracción de países (geocodificación bibliométrica)
  2. Nube de palabras (TF simplificado + layout de nube)
  3. Línea temporal (agregación por año y revista)
  4. Exportación a PDF
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MAPA DE CALOR GEOGRÁFICO
# ═══════════════════════════════════════════════════════════════════════════════

def explain_geo_heatmap() -> str:
    return """
###  Mapa de Calor Geográfico — Explicación del Algoritmo

#### Objetivo
Visualizar la distribución geográfica de la producción científica, identificando
cuántas publicaciones provienen del **primer autor** de cada país.

---

#### Paso 1 — Detección de columna geográfica

El sistema busca automáticamente columnas con información de país o afiliación:

| Prioridad | Nombres buscados |
|-----------|-----------------|
| 1ª | `country`, `countries` |
| 2ª | `affiliation`, `affiliations` |
| 3ª | `institution`, `organization`, `address` |

---

#### Paso 2 — Extracción del primer autor

Las columnas de afiliación típicamente listan **todos los autores** separados
por punto y coma. Sólo interesa el primero:

```
"MIT, USA; Oxford Univ., UK; TU Berlin, Germany"
          ↓
"MIT, USA"  ← primer autor
```

---

#### Paso 3 — Geocodificación por alias

Para cada texto de afiliación se aplica búsqueda de alias (de mayor a menor
longitud para evitar falsos positivos):

```python
# Orden de búsqueda (alias más largos primero):
"united states of america" → "United States"
"united states"             → "United States"
"usa"                       → "United States"
"us"                        → "United States"
```

El algoritmo usa expresiones regulares con `\\b` (límite de palabra) para
evitar que "us" coincida con "thus", "plus", etc.:

```python
pattern = r"\\b" + re.escape(alias) + r"\\b"
if re.search(pattern, text_clean):
    return COUNTRY_ALIASES[alias]
```

---

#### Paso 4 — Conteo y visualización

Se construye el diccionario `{país: n_publicaciones}` y se mapea a
código ISO Alpha-3 para el choropleth de Plotly:

```
"United States" → "USA"  → 42 publicaciones
"China"         → "CHN"  → 38 publicaciones
"Germany"       → "DEU"  → 15 publicaciones
...
```

**Mapa de calor (choropleth):** color proporcional al número de publicaciones
usando una escala secuencial (blanco → amarillo → naranja → rojo).

---

#### Complejidad
- **Tiempo:** O(n × A) donde n = artículos, A = número de alias (~120)
- **Espacio:** O(P) donde P = número de países únicos identificados
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NUBE DE PALABRAS
# ═══════════════════════════════════════════════════════════════════════════════

def explain_wordcloud() -> str:
    return """
###  Nube de Palabras — Explicación del Algoritmo

#### Objetivo
Identificar los **términos más representativos** del corpus bibliométrico,
combinando abstracts y keywords con pesos diferenciados.

---

#### Paso 1 — Preprocesamiento textual

Cada texto pasa por un pipeline de normalización:

```
Texto original:
"This paper proposes a novel approach to Generative AI (2024)..."

Pasos:
  1. Minúsculas      → "this paper proposes a novel approach..."
  2. Quitar URLs     → re.sub(r"https?://\\S+", "", ...)
  3. Quitar números  → re.sub(r"\\b\\d+\\b", "", ...)
  4. Quitar puntuación → re.sub(r"[^a-z\\s\\-]", " ", ...)
  5. Tokenizar       → ["paper", "proposes", "novel", "approach", ...]
  6. Filtrar stop-words y tokens cortos (< 3 chars)
```

---

#### Paso 2 — Ponderación diferenciada

| Fuente | Peso por token |
|--------|---------------|
| Abstracts | 1 (frecuencia natural) |
| Keywords | 3 (seleccionados por autores) |

Las keywords reciben mayor peso porque son términos **elegidos explícitamente**
por los autores para representar su trabajo.

---

#### Paso 3 — Nube de palabras (algoritmo de layout)

La librería `wordcloud` usa el algoritmo de **Spiraling Placement**:

1. Ordena palabras de mayor a menor frecuencia.
2. Para cada palabra, la coloca en el centro e intenta rotarla/escalarla.
3. Si hay colisión con palabras ya colocadas, espiral hacia afuera hasta
   encontrar posición libre.
4. El **tamaño de fuente** es proporcional a `log(frecuencia)` para evitar
   que las más comunes dominen visualmente.

```
frecuencia relativa = count_palabra / max(count_todas)
tamaño_fuente ∈ [8, 120] px  (escalado logarítmico)
```

---

#### Dinámica del corpus

La nube es **dinámica**: al agregar artículos al `unified.csv`, las
frecuencias se recalculan automáticamente al recargar el dataset en R5,
sin necesidad de reconfiguraciones.

---

#### Complejidad
- **Preprocesamiento:** O(T) donde T = total de tokens en el corpus
- **Layout de nube:** O(W × P) donde W = palabras únicas, P = posiciones
  probadas en la espiral (generalmente < 1000 por palabra)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LÍNEA TEMPORAL
# ═══════════════════════════════════════════════════════════════════════════════

def explain_timeline() -> str:
    return """
###  Línea Temporal — Explicación del Algoritmo

#### Objetivo
Mostrar la **evolución temporal** de la producción científica sobre
IA Generativa, tanto por año como por revista/conferencia.

---

#### Paso 1 — Normalización del campo `year`

```python
df["year"] = pd.to_numeric(df["year"], errors="coerce")  # NaN si no es número
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)
df = df[df["year"] >= 2000]                              # filtro de años válidos
```

---

#### Paso 2 — Agregación por año

```python
df_by_year = df.groupby("year").size().reset_index(name="count")
```

Esto produce la tabla:

| year | count |
|------|-------|
| 2019 | 3     |
| 2020 | 8     |
| 2021 | 15    |
| ...  | ...   |

---

#### Paso 3 — Desagregación por fuente

```python
df_by_year_source = (
    df.groupby(["year", "source"])
    .size()
    .reset_index(name="count")
)
```

Permite identificar qué revistas/conferencias lideran cada año.

---

#### Paso 4 — Selección de top fuentes

Para legibilidad, el panel de revistas muestra sólo las **top N fuentes**
por volumen total acumulado:

```python
source_totals = df_by_year_source.groupby("source")["count"].sum()
top_sources = source_totals.nlargest(N).index.tolist()
```

---

#### Interpretación de la visualización

- **Barras + línea (panel superior):** permiten ver tanto el valor
  absoluto por año como la tendencia de crecimiento.
- **Barras apiladas (panel inferior):** muestran qué revistas/conferencias
  concentran la producción en cada año. Colores distintos = fuentes distintas.
- **Crecimiento exponencial** es característico de temas emergentes como
  la IA Generativa.

---

#### Complejidad
- O(n log n) por el groupby + sort (donde n = número de artículos)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXPORTACIÓN A PDF
# ═══════════════════════════════════════════════════════════════════════════════

def explain_pdf_export() -> str:
    return """
###  Exportación a PDF — Explicación del Proceso

#### Herramienta utilizada: `matplotlib.backends.backend_pdf.PdfPages`

`PdfPages` es un gestor de contexto que acumula figuras Matplotlib en un
único archivo PDF multipágina, generando vectores PostScript internamente.

---

#### Pipeline de generación

```
1. Crear buffer en memoria (io.BytesIO)
2. Abrir PdfPages sobre el buffer
   ├── Página 0: Portada con estadísticas resumen
   ├── Página 1: Figura geo (barras horizontales, top 20 países)
   ├── Página 2: Nube de palabras (wordcloud o barras fallback)
   └── Página 3: Línea temporal (doble panel año + revistas)
3. Cerrar PdfPages → escribe el PDF al buffer
4. buf.seek(0) → listo para descarga con st.download_button
```

---

#### ¿Por qué barras en lugar del choropleth para el PDF?

Plotly genera gráficos interactivos (HTML/WebGL) que no se pueden embeber
directamente en un PDF estático sin dependencias externas (`kaleido`, `orca`).
Para el PDF se regenera la visualización geográfica como barras horizontales
con Matplotlib, que sí produce vectores portables.

---

#### Formato de salida

| Característica | Valor |
|----------------|-------|
| Formato | PDF/PostScript |
| Resolución | Vectorial (escalable sin pérdida) |
| Páginas | 4 (portada + 3 visualizaciones) |
| Tamaño aproximado | 300–800 KB |
"""
