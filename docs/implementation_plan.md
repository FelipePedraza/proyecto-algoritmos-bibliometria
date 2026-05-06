# Verificación R1 + Implementación R2 — Similitud Textual

## Verificación del Requerimiento 1

### Estado actual ✅

El R1 está **bien implementado** con los siguientes componentes:

| Archivo | Función |
|---------|---------|
| [acm_parser.py](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/src/r1_scraping/acm_parser.py) | Parsea CSV de ACM → esquema canónico |
| [sciencedirect_parser.py](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/src/r1_scraping/sciencedirect_parser.py) | Parsea CSV de ScienceDirect → esquema canónico |
| [unifier.py](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/src/r1_scraping/unifier.py) | Unificación con Union-Find + Levenshtein para duplicados |
| [page.py](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/src/r1_scraping/page.py) | UI Streamlit: carga, vista previa, unificación, descarga |
| [text_utils.py](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/src/utils/text_utils.py) | Normalización de títulos + Levenshtein similarity |
| [test_r1.py](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/src/tests/test_r1.py) | 13 tests unitarios cubriendo normalización, similitud y unificación |

### Cumplimiento de requisitos R1

| Requisito | Estado | Detalle |
|-----------|--------|---------|
| Unificar información de 2 bases de datos | ✅ | ACM + ScienceDirect |
| Una sola instancia del producto (sin duplicados por nombre) | ✅ | Union-Find + Levenshtein ≥ 0.92 |
| Archivo unificado con toda la información | ✅ | `unified.csv` con 11 campos |
| Proceso automático | ⚠️ | Ver nota abajo |
| Archivo separado con duplicados eliminados | ✅ | `duplicates.csv` con metadata del registro conservado |

> [!NOTE]
> **Sobre la automatización:** El requisito dice *"el proceso debe ser totalmente automático (búsqueda → archivo final)"*, lo cual implica scraping web automático. Sin embargo, ACM y ScienceDirect bloquean activamente el scraping automatizado (CAPTCHA, rate-limiting, términos de servicio). La implementación actual (upload manual de CSV → unificación automática) es la aproximación más práctica y es la que se usa en bibliometría real. **Se recomienda mantener como está.**

### Bug menor encontrado

Los parsers `acm_parser.py` y `sciencedirect_parser.py` usan `Path(filepath)` y verifican `.exists()`, pero en la UI de Streamlit se les pasa un `UploadedFile` (file-like object), no una ruta de archivo. Esto funciona porque Pandas puede leer de file-like objects, pero el check `Path(filepath).exists()` fallaría. Actualmente la página llama `pd.read_csv` directamente antes del parser para la vista previa, y luego pasa el mismo file-like al parser, que **sí funciona** porque `pd.read_csv` acepta file-like objects. Sin embargo, `Path(filepath)` sobre un `UploadedFile` no genera error porque simplemente crea un Path inválido pero nunca se evalúa como `.exists()` retornará `False`... 

**Corrección propuesta:** Hacer que los parsers acepten tanto rutas como file-like objects.

---

## Implementación del Requerimiento 2 — Similitud Textual

### Requisitos

- **4 algoritmos clásicos** (distancia de edición o vectorización estadística)
- **2 algoritmos con modelos de IA**
- Explicar cada algoritmo paso a paso (matemático y algorítmico)
- Permitir seleccionar artículos y analizar sus abstracts

### Algoritmos propuestos

#### 4 Clásicos

| # | Algoritmo | Tipo | Descripción |
|---|-----------|------|-------------|
| 1 | **Distancia de Levenshtein** | Distancia de edición | Operaciones mínimas (insertar, eliminar, sustituir) para transformar una cadena en otra |
| 2 | **Similitud de Jaccard** | Conjuntos | Intersección / Unión de conjuntos de palabras (tokens) |
| 3 | **Coseno TF-IDF** | Vectorización estadística | Representación vectorial TF-IDF + similitud del coseno |
| 4 | **Distancia de Hamming** | Distancia de edición | Número de posiciones donde los caracteres difieren (se adapta a cadenas de distinto largo con padding) |

#### 2 con IA

| # | Algoritmo | Modelo | Descripción |
|---|-----------|--------|-------------|
| 5 | **Sentence-BERT** | `all-MiniLM-L6-v2` | Embeddings semánticos de oraciones con transformers. Captura significado profundo |
| 6 | **spaCy Word Vectors** | `en_core_web_md` | Vectores de palabras promediados. Captura relaciones semánticas aprendidas |

### Estructura de archivos propuesta

```
src/r2_similarity/
├── __init__.py              (ya existe, vacío)
├── algorithms.py            [NEW] — Implementación de los 6 algoritmos
├── explanations.py          [NEW] — Explicaciones paso a paso en Markdown
├── page.py                  [NEW] — Interfaz Streamlit del R2
```

### Diseño de la interfaz (Streamlit)

1. **Selector de artículos**: El usuario carga (o reutiliza) el `unified.csv` del R1, y selecciona 2 artículos de un dropdown
2. **Visualización de abstracts**: Muestra ambos abstracts seleccionados
3. **Selección de algoritmos**: Checkboxes para elegir qué algoritmos ejecutar
4. **Resultados**: Tabla comparativa con scores de similitud + gráfico de barras
5. **Explicación paso a paso**: Expander por cada algoritmo con la explicación matemática y algorítmica detallada
6. **Matriz de similitud**: Opción para calcular similitud entre todos los artículos (NxN) con heatmap

---

### Archivos a modificar/crear

#### [FIX] [acm_parser.py](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/src/r1_scraping/acm_parser.py)
- Hacer que acepte tanto `Path` como file-like objects (para que funcione correctamente desde Streamlit)

#### [FIX] [sciencedirect_parser.py](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/src/r1_scraping/sciencedirect_parser.py)
- Mismo fix que acm_parser

#### [NEW] [algorithms.py](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/src/r2_similarity/algorithms.py)
- 6 funciones de similitud, cada una recibe 2 strings y retorna float [0, 1]
- Los algoritmos clásicos se implementan desde cero (sin librerías de similitud)
- Los de IA usan `sentence-transformers` y `spaCy`

#### [NEW] [explanations.py](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/src/r2_similarity/explanations.py)
- Funciones que retornan strings Markdown con la explicación paso a paso de cada algoritmo
- Incluyen fórmulas matemáticas (LaTeX para Streamlit) y pseudocódigo

#### [NEW] [page.py](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/src/r2_similarity/page.py)
- Interfaz Streamlit completa del R2

#### [MODIFY] [app.py](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/app.py)
- Conectar la ruta R2 al nuevo `page.py`

#### [MODIFY] [requirements.txt](file:///d:/Progra/Algoritmos/proyecto-algoritmos-bibliometria/requirements.txt)
- Agregar dependencias: `scikit-learn`, `sentence-transformers`, `spacy`, `matplotlib`

---

## Open Questions

> [!IMPORTANT]
> **Dependencias de IA:** Los modelos de IA (`sentence-transformers` y `spaCy`) requieren descargar modelos preentrenados (~100-400MB). `pyarrow` (dependencia de streamlit) falló en tu instalación anterior. ¿Tienes Rust/compilador C++ instalado, o prefieres que use versiones de streamlit que no requieran compilar pyarrow desde source? Alternativamente puedo usar `streamlit>=1.37` que usa `pyarrow` como opcional.

> [!IMPORTANT]
> **¿Quieres que corrija el bug de los parsers del R1** (Path vs file-like object) o prefieres dejarlo como está dado que funciona en la práctica?

---

## Verification Plan

### Automated Tests
- Tests unitarios para los 6 algoritmos de similitud (`test_r2.py`)
- Verificar que cada algoritmo retorne valores en [0, 1]
- Verificar que textos idénticos retornen 1.0
- Verificar que textos completamente diferentes retornen valores bajos

### Manual Verification  
- Ejecutar la app con `streamlit run app.py`
- Seleccionar R2, cargar unified.csv, seleccionar 2 artículos y verificar resultados
- Verificar que las explicaciones matemáticas se rendericen correctamente
