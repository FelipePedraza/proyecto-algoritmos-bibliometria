# Análisis de Algoritmos en Bibliometría: Inteligencia Artificial Generativa

## Descripción del Proyecto
Este proyecto implementa algoritmos computacionales para el análisis bibliométrico sobre el dominio de conocimiento de la inteligencia artificial generativa. El objetivo principal es extraer, procesar y analizar datos derivados de la producción científica utilizando métodos cuantitativos y cualitativos.

**Institución:** Universidad del Quindío  
**Programa:** Ingeniería de Sistemas y Computación  
**Cadena de búsqueda principal:** `"generative artificial intelligence"`  

---

## Características Principales (Requerimientos)

El desarrollo se fundamenta en los siguientes requerimientos funcionales:

* **1. Automatización de Descarga y Unificación:**  Descarga automatizada desde dos bases de datos institucionales.
  * Unificación de datos en un solo archivo garantizando una sola instancia por producto (eliminación de duplicados).
  * Almacenamiento de registros repetidos en un archivo independiente.
* **2. Análisis de Similitud Textual:**  Implementación de 4 algoritmos clásicos (distancia de edición o vectorización estadística) y 2 algoritmos basados en modelos de IA aplicados a los abstracts.
  * Explicación detallada del funcionamiento matemático y algorítmico de cada uno.
* **3. Frecuencia y Extracción de Términos:**  Cálculo de la frecuencia de palabras asociadas a la categoría *"Concepts of Generative AI in Education"*.
  * Extracción algorítmica de un listado de nuevas palabras asociadas (máximo 15) a partir de los abstracts y determinación de su precisión.
* **4. Agrupamiento Jerárquico (Clustering):**  Implementación de 3 algoritmos de agrupamiento para construir un dendrograma basado en la similitud de los abstracts preprocesados.
  * Evaluación de cuál algoritmo produce agrupamientos más coherentes.
* **5. Visualización Científica:** * Mapa de calor con distribución geográfica (basado en el primer autor).
  * Nube de palabras dinámica basada en abstracts y keywords.
  * Línea temporal de publicaciones por año y revista.
  * Exportación de las tres visualizaciones anteriores a formato PDF.
* **6. Despliegue y Documentación:**
  * Aplicación desplegada y soportada por un documento de diseño con la arquitectura y detalles de implementación técnica.

---

## Estructura del Repositorio

```text
proyecto-algoritmos-bibliometria/
│
├── data/
│   ├── raw/              # Archivos originales descargados de cada BD
│   ├── processed/        # Archivo unificado y limpio
│   └── duplicates/       # Registro de duplicados eliminados
│
├── src/
│   ├── r1_scraping/      # Lógica del Requerimiento 1 (Descarga y unificación)
│   ├── r2_similarity/    # Lógica del Requerimiento 2 (Similitud textual e IA)
│   ├── r3_frequency/     # Lógica del Requerimiento 3 (Frecuencia y términos)
│   ├── r4_clustering/    # Lógica del Requerimiento 4 (Agrupamiento jerárquico)
│   ├── r5_visualization/ # Lógica del Requerimiento 5 (Visualización de datos)
│   └── utils/            # Funciones compartidas (limpieza de texto, I/O)
│
├── docs/                 # Documentación técnica y de arquitectura
├── tests/                # Pruebas unitarias
├── app.py                # Punto de entrada de la aplicación (Streamlit/Flask/FastAPI)
└── requirements.txt      # Dependencias del proyecto

```

## Bases de datos disponibles en la Universidad del Quindio

| Base de datos | Formatos de exportación | Notas |
|---------------|------------------------|-------|
| ACM Digital Library | BibTeX, CSV | CSV tiene columna "Publication Year" |
| ScienceDirect | BibTeX, RIS, CSV | CSV tiene "Source title" e "Index Keywords" |
| SAGE Journals | RIS, BibTeX | — |
| Scopus | BibTeX, RIS, CSV | CSV similar a ScienceDirect |

Acceso: [library.uniquindio.edu.co/databases](https://library.uniquindio.edu.co/databases)
