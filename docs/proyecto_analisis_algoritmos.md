# Universidad del Quindío  
## Programa de Ingeniería de Sistemas y Computación  

# Análisis de algoritmos en el contexto de la bibliometría  
### Descripción inicial del proyecto  

---

## 1. Introducción  

La bibliometría es una disciplina que permite explorar y analizar volúmenes de datos derivados de la producción científica utilizando métodos cuantitativos y cualitativos. Se fundamenta en las matemáticas y la estadística, para establecer descripciones, relaciones, inferencias y presentaciones de la información suministrada por publicaciones científicas en diferentes áreas del conocimiento.  

En bibliometría se pueden identificar diferentes indicadores. Algunos de ellos se enfocan en la productividad de los autores, índices de impacto, países, tópicos según el área de conocimiento, relación visual a partir de diferentes variables bibliométricas y colaboración entre autores.  

---

## 2. Fuentes de información  

La Universidad del Quindío cuenta con bases de datos científicas disponibles en:  
https://library.uniquindio.edu.co/databases  

Algunas de las bases de datos son: ACM, SAGE y ScienceDirect. Cada una permite métodos de consulta, acceso y exportación de información. En este último aspecto, existen los formatos RIS, BibTex, CSV, texto plano, entre otros.  

Las bases de datos disponibles presentan diversas tipologías de productividad científica (artículos, conferencias, capítulos de libro, entre otros). Cada base de datos presenta limitantes en cuanto al acceso a la información y la calidad de los datos relacionados a la completitud.  

Para el proyecto del curso de análisis de algoritmos se plantea un dominio de conocimiento: **La inteligencia artificial generativa**.  
La cadena de búsqueda será: `"generative artificial intelligence"`.  

---

## 3. Propósito del proyecto  

Implementar algoritmos que permitan el análisis bibliométrico y computacional sobre un dominio de conocimiento a partir de las bases de datos disponibles en la Universidad del Quindío.  

El desarrollo del proyecto se fundamentará en requerimientos funcionales que contemplan la implementación de diversas técnicas bibliométricas y tipos de algoritmos. Para el proyecto es necesario el despliegue de la aplicación con la correspondiente documentación.  

---

## Requerimientos  

### Requerimiento 1. Automatización de proceso de descarga de datos  

Se debe automatizar la información de descarga sobre dos bases de datos. Posteriormente se debe unificar la información en un solo archivo garantizando una sola instancia del producto.  

- Si se identifica un producto repetido por su nombre, se debe tener un solo registro.  
- El archivo unificado debe contener toda la información (autores, título, palabras clave, resumen, etc.).  
- El proceso debe ser totalmente automático (búsqueda → archivo final).  

Además:  
- Se debe generar otro archivo con los registros repetidos eliminados.  

---

### Requerimiento 2. Similitud textual  

Se deben implementar:  
- 4 algoritmos clásicos (distancia de edición o vectorización estadística)  
- 2 algoritmos con modelos de IA  

Se debe:  
- Explicar cada algoritmo paso a paso (matemático y algorítmico)  
- Permitir seleccionar artículos y analizar sus abstracts  

---

### Requerimiento 3. Frecuencia de palabras  

Dada la categoría: **Concepts of Generative AI in Education**, se debe:  

1. Calcular la frecuencia de palabras en abstracts  
2. Generar nuevas palabras asociadas (máximo 15)  
3. Evaluar su precisión  

#### Palabras asociadas  

- Generative models  
- Prompting  
- Machine learning  
- Multimodality  
- Fine-tuning  
- Training data  
- Algorithmic bias  
- Explainability  
- Transparency  
- Ethics  
- Privacy  
- Personalization  
- Human-AI interaction  
- AI literacy  
- Co-creation  

---

### Requerimiento 4. Clustering jerárquico  

Se deben implementar 3 algoritmos de clustering jerárquico para generar un **dendrograma**.  

Proceso:  
1. Preprocesamiento de texto  
2. Cálculo de similitud  
3. Aplicación de clustering  
4. Representación gráfica  

Se debe determinar cuál algoritmo genera mejores agrupamientos.  

---

### Requerimiento 5. Análisis visual  

Se debe implementar:  

1. Mapa de calor (distribución geográfica por autor)  
2. Nube de palabras (abstracts y keywords)  
3. Línea temporal de publicaciones  
4. Exportación a PDF  

---

### Requerimiento 6. Despliegue  

El proyecto debe estar desplegado y documentado técnicamente.  

---

## Documento final  

Debe incluir:  
- Diseño y arquitectura  
- Explicación técnica de cada requerimiento  
- Uso fundamentado de IA  

---

## Nota  

La descripción del proyecto puede modificarse para mayor claridad, especialmente en los requerimientos funcionales.  
