"""
Requerimiento 4 — Explicaciones Paso a Paso
=============================================
Genera explicaciones matemáticas y algorítmicas en Markdown/LaTeX
para cada uno de los 3 algoritmos de clustering jerárquico,
el pipeline de preprocesamiento y el coeficiente cofenético.
"""

from __future__ import annotations

from typing import List


# ── Preprocesamiento ──────────────────────────────────────────────────────────


def explain_preprocessing(tokens_sample: List[List[str]], titles: List[str]) -> str:
    """Explicación del pipeline de preprocesamiento con ejemplos reales."""
    examples = ""
    for i in range(min(3, len(titles))):
        toks = tokens_sample[i]
        preview = ", ".join(f"`{t}`" for t in toks[:12])
        if len(toks) > 12:
            preview += f", ... (+{len(toks) - 12} más)"
        examples += f"**{i+1}. {titles[i][:60]}{'…' if len(titles[i])>60 else ''}**\n→ {preview}\n\n"

    return f"""### 🔧 Preprocesamiento de Texto

El preprocesamiento transforma los abstracts crudos en representaciones
limpias y normalizadas, eliminando ruido para mejorar la calidad del clustering.

#### Pipeline (5 pasos en orden)

| Paso | Operación | Qué elimina / transforma |
|------|-----------|--------------------------|
| 1 | **Minúsculas** | "Learning" → "learning" |
| 2 | **Limpieza** | Números, símbolos, puntuación → espacio |
| 3 | **Tokenización** | Texto → lista de palabras |
| 4 | **Stop words** | "the", "is", "paper", "study"... → eliminados |
| 5 | **Stemming** | "learning" → "learn", "proposed" → "propos" |

#### Stemmer simple (sufijos)

El stemmer busca el sufijo más largo de una lista ordenada y lo elimina,
siempre que la raíz resultante tenga ≥ 3 caracteres:

```
"algorithms" → "algorithm"   (quita -s)
"clustering"  → "cluster"    (quita -ing)
"proposed"    → "propos"     (quita -ed)
"generation"  → "generat"    (quita -ion)
```

#### Ejemplos con los datos cargados

{examples}

#### Impacto del preprocesamiento

Sin preprocesamiento, la similitud entre dos textos puede ser baja
simplemente porque usan formas morfológicas distintas ("generate",
"generated", "generating"), no porque traten temas diferentes.
Con preprocesamiento, todas las formas se reducen a la misma raíz,
capturando mejor el contenido temático.
"""


# ── TF-IDF ────────────────────────────────────────────────────────────────────


def explain_tfidf(vocab_size: int, n_docs: int, vocabulary_sample: List[str]) -> str:
    """Explicación de la vectorización TF-IDF."""
    vocab_preview = ", ".join(f"`{w}`" for w in vocabulary_sample[:15])
    if len(vocabulary_sample) > 15:
        vocab_preview += f", ... (+{vocab_size - 15} más)"

    return f"""### 📊 Vectorización TF-IDF

Cada abstract se convierte en un vector de **{vocab_size} dimensiones**
(una por término en el vocabulario).

#### Fórmulas

**TF — Term Frequency** (frecuencia de un término en un documento):

$$
TF(t, d) = \\frac{{\\text{{número de veces que aparece }} t \\text{{ en }} d}}{{\\text{{total de tokens en }} d}}
$$

**IDF — Inverse Document Frequency** (importancia global del término):

$$
IDF(t) = \\ln\\!\\left(\\frac{{N + 1}}{{df(t) + 1}}\\right) + 1
$$

Donde $N = {n_docs}$ documentos y $df(t)$ = número de documentos que contienen $t$.
El $+1$ en numerador y denominador es el suavizado (evita división por cero).

**Peso TF-IDF final:**

$$
w(t, d) = TF(t, d) \\times IDF(t)
$$

#### Interpretación del IDF

- Un término que aparece en **todos** los documentos tiene $IDF \\approx 1$ → peso bajo
- Un término que aparece en **pocos** documentos tiene $IDF$ alto → distingue mejor

#### Vocabulario ({vocab_size} términos)

Muestra: {vocab_preview}

#### Resultado

Cada abstract queda representado como un vector disperso de {vocab_size} dimensiones.
La mayoría de los valores son 0 (el abstract no usa ese término).
"""


# ── Distancia coseno ─────────────────────────────────────────────────────────


def explain_cosine_distance() -> str:
    """Explicación de la distancia coseno."""
    return """### 📐 Distancia Coseno

Para medir qué tan parecidos son dos abstracts, usamos la **distancia coseno**:

$$
d_{\\cos}(\\vec{A}, \\vec{B}) = 1 - \\frac{\\vec{A} \\cdot \\vec{B}}{\\|\\vec{A}\\| \\times \\|\\vec{B}\\|}
$$

Donde:
- $\\vec{A} \\cdot \\vec{B} = \\sum_i A_i \\cdot B_i$ — producto punto
- $\\|\\vec{A}\\| = \\sqrt{\\sum_i A_i^2}$ — norma L2

#### ¿Por qué coseno y no Euclídea?

| Criterio | Distancia Euclídea | Distancia Coseno |
|----------|-------------------|-----------------|
| Sensibilidad a longitud | Alta | Baja ✓ |
| Textos de diferente extensión | Penaliza ✗ | No penaliza ✓ |
| Uso en NLP | Moderado | Estándar ✓ |

La distancia coseno solo mide la **dirección** del vector (el ángulo entre ellos),
no su magnitud. Dos textos con las mismas palabras pero diferente longitud
tendrán distancia 0.

#### Rango de valores

$$
d_{\\cos} \\in [0, 1] \\quad \\begin{cases}
0 & \\text{abstracts con vocabulario idéntico} \\\\
1 & \\text{sin palabras en común (ortogonales)}
\\end{cases}
$$
"""


# ── Algoritmos de clustering ──────────────────────────────────────────────────


def explain_complete_linkage(trace: dict) -> str:
    """Explicación matemática del Enlace Completo con datos reales."""
    n = trace.get("n", 0)
    coph = trace.get("cophenetic", {}).get("complete", 0.0)
    steps = trace.get("linkage", {}).get("complete", [])
    first_merges = steps[:3] if steps else []

    merge_table = ""
    for k, (a, b, d, s) in enumerate(first_merges):
        label_a = f"Doc {int(a)}" if a < n else f"Cluster {int(a)}"
        label_b = f"Doc {int(b)}" if b < n else f"Cluster {int(b)}"
        merge_table += f"| {k+1} | {label_a} | {label_b} | {d:.4f} | {int(s)} |\n"

    return f"""### 🔴 Enlace Completo (Complete Linkage)

#### Regla de distancia

$$
d(A \\cup B,\\, C) = \\max\\bigl(d(A, C),\\; d(B, C)\\bigr)
$$

La distancia entre el cluster fusionado $A \\cup B$ y cualquier otro cluster $C$
es la **distancia máxima** entre cualquier par de puntos (uno de $A$ o $B$ y uno de $C$).
También se conoce como criterio del **vecino más lejano**.

#### Algoritmo paso a paso

```
1. Inicializar: cada documento = su propio cluster
   Clusters = {{0}}, {{1}}, ..., {{n-1}}

2. Repetir n-1 veces:
   a. Buscar par (A, B) con mínima distancia → O(n²)
   b. Fusionar: nuevo cluster C = A ∪ B
   c. Actualizar: d(C, K) = max(d(A, K), d(B, K)) para todo K activo

3. Retornar árbol de fusiones (linkage matrix)
```

#### Primeros {len(first_merges)} pasos sobre los datos cargados

| Paso | Cluster A | Cluster B | Distancia | Tamaño nuevo |
|------|-----------|-----------|-----------|--------------|
{merge_table}

#### Resultado

- **Coeficiente cofenético:** {coph:.4f}

#### Características

| Ventaja | Inconveniente |
|---------|---------------|
| Clusters compactos y globulares | Muy sensible a outliers |
| Bien separados entre sí | Puede romper clusters grandes |
| Determinístico | O(n³) — lento para n grande |

#### Complejidad

- **Temporal:** $O(n^3)$ — $n-1$ pasos × búsqueda $O(n^2)$
- **Espacial:** $O(n^2)$ — para la matriz de distancias
"""


def explain_average_linkage(trace: dict) -> str:
    """Explicación matemática del Enlace Promedio con datos reales."""
    n = trace.get("n", 0)
    coph = trace.get("cophenetic", {}).get("average", 0.0)
    steps = trace.get("linkage", {}).get("average", [])
    first_merges = steps[:3] if steps else []

    merge_table = ""
    for k, (a, b, d, s) in enumerate(first_merges):
        label_a = f"Doc {int(a)}" if a < n else f"Cluster {int(a)}"
        label_b = f"Doc {int(b)}" if b < n else f"Cluster {int(b)}"
        merge_table += f"| {k+1} | {label_a} | {label_b} | {d:.4f} | {int(s)} |\n"

    return f"""### 🟢 Enlace Promedio (Average Linkage / UPGMA)

#### Regla de distancia

$$
d(A \\cup B,\\, C) = \\frac{{|A| \\cdot d(A, C) + |B| \\cdot d(B, C)}}{{|A| + |B|}}
$$

La distancia al nuevo cluster es la **media ponderada por tamaño** de las distancias
a los clusters originales. UPGMA = *Unweighted Pair Group Method with Arithmetic Mean*.

#### Por qué funciona bien

Considerar el promedio da al algoritmo una visión "de conjunto":
no es dominado ni por los puntos más cercanos (como el enlace simple)
ni por los más lejanos (como el enlace completo).

#### Algoritmo paso a paso

```
1. Inicializar: cada documento = su propio cluster, tamaño = 1

2. Repetir n-1 veces:
   a. Buscar par (A, B) con mínima distancia → O(n²)
   b. Fusionar: C = A ∪ B, |C| = |A| + |B|
   c. Actualizar:
        d(C, K) = (|A|·d(A,K) + |B|·d(B,K)) / (|A| + |B|)

3. Retornar árbol de fusiones
```

#### Primeros {len(first_merges)} pasos sobre los datos cargados

| Paso | Cluster A | Cluster B | Distancia | Tamaño nuevo |
|------|-----------|-----------|-----------|--------------|
{merge_table}

#### Resultado

- **Coeficiente cofenético:** {coph:.4f}

#### Características

| Ventaja | Inconveniente |
|---------|---------------|
| Mejor coeficiente cofenético en promedio | Puede producir clusters desiguales |
| Menos sensible a outliers que completo | Intermedio en compacidad |
| Produce dendrogramas naturales | O(n³) |

#### Complejidad

- **Temporal:** $O(n^3)$ — idéntica a los otros métodos (naive)
- **Espacial:** $O(n^2)$
"""


def explain_ward_linkage(trace: dict) -> str:
    """Explicación matemática del Método de Ward con datos reales."""
    n = trace.get("n", 0)
    coph = trace.get("cophenetic", {}).get("ward", 0.0)
    steps = trace.get("linkage", {}).get("ward", [])
    first_merges = steps[:3] if steps else []

    merge_table = ""
    for k, (a, b, d, s) in enumerate(first_merges):
        label_a = f"Doc {int(a)}" if a < n else f"Cluster {int(a)}"
        label_b = f"Doc {int(b)}" if b < n else f"Cluster {int(b)}"
        merge_table += f"| {k+1} | {label_a} | {label_b} | {d:.4f} | {int(s)} |\n"

    return f"""### 🔵 Método de Ward

#### Criterio de fusión

Ward no define la distancia entre clusters directamente, sino que **minimiza
el incremento de inercia intra-cluster** al fusionar:

$$
\\Delta E(A, B) = \\frac{{|A| \\cdot |B|}}{{|A| + |B|}} \\cdot d(A, B)^2
$$

Los clusters $A$ y $B$ que minimizan $\\Delta E$ son los que se fusionan.

#### Fórmula de actualización (Lance-Williams)

Para actualizar distancias sin recalcular todo, se usa la fórmula de Lance-Williams:

$$
d(A \\cup B,\\, C)^2 = \\frac{{(n_A + n_C)\\, d(A,C)^2 + (n_B + n_C)\\, d(B,C)^2 - n_C\\, d(A,B)^2}}{{n_A + n_B + n_C}}
$$

El algoritmo opera internamente con **distancias al cuadrado** y reporta
$d = \\sqrt{{d^2}}$ como altura en el dendrograma.

#### Algoritmo paso a paso

```
1. Inicializar: dist²[i][j] = cosine_dist(i,j)², tamaño[i] = 1

2. Repetir n-1 veces:
   a. Buscar par (A, B) con mínimo dist²[A][B]  → O(n²)
   b. Fusionar: C = A ∪ B, |C| = |A| + |B|
   c. Altura de merge = √dist²[A][B]
   d. Actualizar usando Lance-Williams:
        dist²[C][K] = ((nA+nK)·d²AK + (nB+nK)·d²BK - nK·d²AB) / (nA+nB+nK)
```

> **Nota:** Ward fue diseñado para distancias Euclídeas. Aquí se aplica
> la fórmula de Lance-Williams a distancias coseno, lo que es válido como
> criterio de agrupamiento aunque no minimiza exactamente la varianza intra-cluster.

#### Primeros {len(first_merges)} pasos sobre los datos cargados

| Paso | Cluster A | Cluster B | Distancia | Tamaño nuevo |
|------|-----------|-----------|-----------|--------------|
{merge_table}

#### Resultado

- **Coeficiente cofenético:** {coph:.4f}

#### Características

| Ventaja | Inconveniente |
|---------|---------------|
| Clusters de tamaño equilibrado | Más sensible a outliers que promedio |
| El más popular en la práctica | Diseñado originalmente para Euclídeo |
| Minimiza varianza intra-cluster | O(n³) |

#### Complejidad

- **Temporal:** $O(n^3)$ — igual que los otros (naive)
- **Espacial:** $O(n^2)$
"""


# ── Coeficiente cofenético ────────────────────────────────────────────────────


def explain_cophenetic(coph_scores: dict, best_algo: str, algo_names: dict) -> str:
    """Explicación del coeficiente cofenético y conclusión sobre el mejor algoritmo."""
    rows = ""
    for key, score in sorted(coph_scores.items(), key=lambda x: -x[1]):
        is_best = "⭐ **Mejor**" if key == best_algo else ""
        rows += f"| {algo_names.get(key, key)} | {score:.4f} | {is_best} |\n"

    best_name = algo_names.get(best_algo, best_algo)
    best_score = coph_scores.get(best_algo, 0.0)

    quality_label = (
        "excelente (> 0.90)" if best_score > 0.90
        else "buena (0.75 – 0.90)" if best_score > 0.75
        else "aceptable (0.60 – 0.75)" if best_score > 0.60
        else "baja (< 0.60)"
    )

    return f"""### 📏 Coeficiente de Correlación Cofenética

#### Definición

Dado un dendrograma, la **distancia cofenética** entre dos puntos $i$ y $j$
es la altura $h_{{ij}}$ en la que son fusionados por primera vez.

El coeficiente cofenético es la **correlación de Pearson** entre:
- Las distancias originales $d(i, j)$ (de la matriz de distancias coseno)
- Las distancias cofenéticas $h(i, j)$ (del dendrograma)

$$
r_{{\\text{{coph}}}} = \\frac{{\\sum_{{i<j}} (d_{{ij}} - \\bar{{d}})(h_{{ij}} - \\bar{{h}})}}
{{\\sqrt{{\\sum_{{i<j}} (d_{{ij}} - \\bar{{d}})^2 \\cdot \\sum_{{i<j}} (h_{{ij}} - \\bar{{h}})^2}}}}
$$

#### Interpretación

| Valor | Calidad del dendrograma |
|-------|------------------------|
| > 0.90 | Excelente representación |
| 0.75 – 0.90 | Buena representación |
| 0.60 – 0.75 | Representación aceptable |
| < 0.60 | Representación pobre |

#### Resultados obtenidos

| Algoritmo | Coeficiente Cofenético | |
|-----------|------------------------|---|
{rows}

#### Conclusión

El algoritmo con mejor representación es **{best_name}** con un coeficiente
cofenético de **{best_score:.4f}** ({quality_label}).

Un valor de {best_score:.4f} significa que el dendrograma del Método {best_name}
preserva el **{best_score*100:.1f}%** de la estructura de distancias original
(en términos de correlación).

Esto indica que al cortar el dendrograma en grupos, los clusters obtenidos
con {best_name} son los más coherentes con las distancias reales entre abstracts.
"""


# ── Registro de funciones ─────────────────────────────────────────────────────

ALGO_EXPLANATION_FUNCS = {
    "complete": explain_complete_linkage,
    "average": explain_average_linkage,
    "ward": explain_ward_linkage,
}
