"""
Requerimiento 2 — Explicaciones Paso a Paso
=============================================
Genera explicaciones detalladas en Markdown (con LaTeX para Streamlit)
para cada algoritmo de similitud.
"""

from __future__ import annotations


def explain_levenshtein(trace: dict) -> str:
    """Explicación paso a paso del algoritmo de Levenshtein."""
    s1 = trace["s1"]
    s2 = trace["s2"]
    matrix = trace["matrix"]
    distance = trace["distance"]
    similarity = trace["similarity"]

    # Construir tabla de la matriz DP (mostrar solo si es manejable)
    rows_s1 = min(len(s1), 15)
    cols_s2 = min(len(s2), 15)
    truncated = len(s1) > 15 or len(s2) > 15

    md = f"""### 📐 Distancia de Levenshtein — Explicación

#### Fundamento matemático

La **distancia de Levenshtein** mide el número mínimo de operaciones para transformar
una cadena en otra. Las operaciones permitidas son:

| Operación | Costo |
|-----------|-------|
| Inserción | 1 |
| Eliminación | 1 |
| Sustitución | 1 |

#### Fórmula de recurrencia

$$
D(i, j) = \\min \\begin{{cases}}
D(i-1, j) + 1 & \\text{{(eliminación)}} \\\\
D(i, j-1) + 1 & \\text{{(inserción)}} \\\\
D(i-1, j-1) + c(i,j) & \\text{{(sustitución)}}
\\end{{cases}}
$$

Donde $c(i,j) = 0$ si $s_1[i] = s_2[j]$, y $c(i,j) = 1$ en caso contrario.

#### Casos base
- $D(i, 0) = i$ (eliminar todos los caracteres de $s_1$)
- $D(0, j) = j$ (insertar todos los caracteres de $s_2$)

#### Resultado para los textos dados
- **Distancia de Levenshtein:** {distance}
- **Longitud máxima:** {max(len(s1), len(s2))}
- **Similitud:** $1 - \\frac{{{distance}}}{{{max(len(s1), len(s2))}}} = {similarity}$

"""

    if truncated:
        md += f"""
> **Nota:** Los textos son largos ({len(s1)} y {len(s2)} caracteres).
> La matriz DP completa tiene {len(s1)+1}×{len(s2)+1} = {(len(s1)+1)*(len(s2)+1)} celdas.
> Se muestra una versión resumida.
"""
    else:
        # Tabla de la matriz
        header = "|   | ε |" + "|".join(f" {c} " for c in s2[:cols_s2]) + "|\n"
        sep = "|---|---|" + "|".join("---" for _ in s2[:cols_s2]) + "|\n"
        rows_text = ""
        for i in range(rows_s1 + 1):
            label = "ε" if i == 0 else s1[i - 1]
            row_vals = "|".join(f" {matrix[i][j]} " for j in range(cols_s2 + 1))
            rows_text += f"| **{label}** | {row_vals} |\n"

        md += f"""
#### Matriz de programación dinámica

{header}{sep}{rows_text}
"""

    md += f"""
#### Complejidad
- **Temporal:** $O(n \\times m)$ donde $n = {len(s1)}$, $m = {len(s2)}$
- **Espacial:** $O(n \\times m)$
"""
    return md


def explain_jaccard(trace: dict) -> str:
    """Explicación paso a paso del algoritmo de Jaccard."""
    return f"""### 📐 Similitud de Jaccard — Explicación

#### Fundamento matemático

La **similitud de Jaccard** mide la proporción de elementos compartidos entre dos conjuntos:

$$
J(A, B) = \\frac{{|A \\cap B|}}{{|A \\cup B|}}
$$

Donde $A$ y $B$ son conjuntos de tokens (palabras) de cada texto.

#### Proceso paso a paso

**1. Tokenización:**
- Texto A → {len(trace['tokens_a'])} tokens únicos
- Texto B → {len(trace['tokens_b'])} tokens únicos

**2. Cálculo de conjuntos:**
- **Intersección** ($A \\cap B$): {trace['intersection_size']} palabras comunes
- **Unión** ($A \\cup B$): {trace['union_size']} palabras totales

**3. Palabras en la intersección:**
> {', '.join(trace['intersection'][:20])}{'...' if len(trace['intersection']) > 20 else ''}

**4. Resultado:**

$$
J(A, B) = \\frac{{{trace['intersection_size']}}}{{{trace['union_size']}}} = {trace['similarity']}
$$

#### Interpretación
- $J = 1.0$: textos idénticos (mismas palabras)
- $J = 0.0$: sin palabras en común
- No considera el orden ni la frecuencia de las palabras

#### Complejidad
- **Temporal:** $O(n + m)$ donde $n$, $m$ son el número de tokens
- **Espacial:** $O(n + m)$
"""


def explain_cosine_tfidf(trace: dict) -> str:
    """Explicación paso a paso del algoritmo Coseno TF-IDF."""
    # Seleccionar top 10 palabras con mayor peso TF-IDF
    vocab = trace["vocabulary"]
    vec_a = trace["tfidf_vector_a"]
    vec_b = trace["tfidf_vector_b"]

    top_words = sorted(
        zip(vocab, vec_a, vec_b),
        key=lambda x: max(x[1], x[2]),
        reverse=True
    )[:10]

    words_table = "| Palabra | TF-IDF (A) | TF-IDF (B) |\n|---------|-----------|------------|\n"
    for word, va, vb in top_words:
        words_table += f"| {word} | {va:.4f} | {vb:.4f} |\n"

    return f"""### 📐 Coseno TF-IDF — Explicación

#### Fundamento matemático

La **similitud del coseno con TF-IDF** combina dos técnicas:

**1. TF (Term Frequency):**
$$
TF(t, d) = \\frac{{\\text{{frecuencia de }} t \\text{{ en }} d}}{{\\text{{total de tokens en }} d}}
$$

**2. IDF (Inverse Document Frequency):**
$$
IDF(t) = \\ln\\left(\\frac{{N + 1}}{{df(t) + 1}}\\right) + 1
$$

**3. TF-IDF:**
$$
TF\\text{{-}}IDF(t, d) = TF(t, d) \\times IDF(t)
$$

**4. Similitud del coseno:**
$$
\\cos(\\theta) = \\frac{{\\vec{{A}} \\cdot \\vec{{B}}}}{{\\|\\vec{{A}}\\| \\times \\|\\vec{{B}}\\|}}
$$

#### Proceso paso a paso

**1. Tokenización:**
- Texto A → {len(trace['tokens_a'])} tokens
- Texto B → {len(trace['tokens_b'])} tokens

**2. Vocabulario conjunto:** {len(vocab)} palabras únicas

**3. Top 10 palabras con mayor peso TF-IDF:**

{words_table}

**4. Producto punto:** $\\vec{{A}} \\cdot \\vec{{B}} = {trace['dot_product']}$

**5. Normas:**
- $\\|\\vec{{A}}\\| = {trace['norm_a']}$
- $\\|\\vec{{B}}\\| = {trace['norm_b']}$

**6. Resultado:**
$$
\\cos(\\theta) = \\frac{{{trace['dot_product']}}}{{{trace['norm_a']} \\times {trace['norm_b']}}} = {trace['similarity']}
$$

#### Complejidad
- **Temporal:** $O(V)$ donde $V$ es el tamaño del vocabulario
- **Espacial:** $O(V)$
"""


def explain_hamming(trace: dict) -> str:
    """Explicación paso a paso del algoritmo de Hamming."""
    # Primeras diferencias
    diffs = [c for c in trace["comparisons_sample"] if not c["match"]]
    diff_examples = diffs[:5]

    diff_table = ""
    if diff_examples:
        diff_table = "| Posición | Char A | Char B |\n|----------|--------|--------|\n"
        for d in diff_examples:
            char_a = repr(d['char_a'])
            char_b = repr(d['char_b'])
            diff_table += f"| {d['pos']} | {char_a} | {char_b} |\n"
        if len(diffs) > 5:
            diff_table += f"| ... | ... | ... |\n"

    return f"""### 📐 Distancia de Hamming — Explicación

#### Fundamento matemático

La **distancia de Hamming** cuenta el número de posiciones donde los caracteres correspondientes difieren.

$$
H(s_1, s_2) = \\sum_{{i=0}}^{{n-1}} \\begin{{cases}} 1 & \\text{{si }} s_1[i] \\neq s_2[i] \\\\ 0 & \\text{{si }} s_1[i] = s_2[i] \\end{{cases}}
$$

Para cadenas de diferente longitud, se rellena la más corta con espacios (padding).

$$
\\text{{Similitud}} = 1 - \\frac{{H(s_1, s_2)}}{{\\max(|s_1|, |s_2|)}}
$$

#### Proceso paso a paso

**1. Preparación:**
- Longitud texto A: {trace['original_len_a']} caracteres
- Longitud texto B: {trace['original_len_b']} caracteres
- Longitud después de padding: {trace['padded_len']} caracteres

**2. Comparación posición a posición:**
- Posiciones iguales: {trace['total_positions'] - trace['distance']}
- Posiciones diferentes: {trace['distance']}

{f"**3. Primeras diferencias encontradas:**" + chr(10) + diff_table if diff_table else ""}

**Resultado:**
$$
\\text{{Similitud}} = 1 - \\frac{{{trace['distance']}}}{{{trace['total_positions']}}} = {trace['similarity']}
$$

#### Limitaciones
- Muy sensible a desplazamientos: si un texto tiene una palabra extra al inicio,
  todas las posiciones siguientes serán diferentes.
- Mejor para textos de longitud similar con cambios puntuales.

#### Complejidad
- **Temporal:** $O(n)$ donde $n = \\max(|s_1|, |s_2|)$
- **Espacial:** $O(n)$ (por el padding)
"""


def explain_sbert(trace: dict) -> str:
    """Explicación paso a paso del algoritmo Sentence-BERT."""
    if "error" in trace:
        return f"⚠️ Error: {trace['error']}"

    emb_a = [f"{v:.4f}" for v in trace["embedding_a_sample"]]
    emb_b = [f"{v:.4f}" for v in trace["embedding_b_sample"]]

    return f"""### 🤖 Sentence-BERT — Explicación

#### Fundamento

**Sentence-BERT (SBERT)** es una modificación de BERT que genera embeddings de oraciones
con significado semántico. Usa una arquitectura siamesa que procesa ambos textos
independientemente y luego compara los embeddings resultantes.

#### Modelo utilizado
- **Nombre:** `{trace['model_name']}`
- **Dimensiones del embedding:** {trace['embedding_dim']}
- **Arquitectura:** Transformer (BERT) con pooling

#### Proceso paso a paso

**1. Tokenización y codificación:**
Cada texto se tokeniza usando WordPiece tokenizer y se pasa por
las capas del transformer (6 capas de atención).

**2. Pooling:**
Los embeddings de todos los tokens se combinan (mean pooling)
para obtener un vector de {trace['embedding_dim']} dimensiones.

**3. Embeddings generados (primeros 10 componentes):**
- Vector A: [{', '.join(emb_a)}...]
- Vector B: [{', '.join(emb_b)}...]

**4. Similitud del coseno:**
$$
\\cos(\\theta) = \\frac{{\\vec{{A}} \\cdot \\vec{{B}}}}{{\\|\\vec{{A}}\\| \\times \\|\\vec{{B}}\\|}} = {trace['similarity']}
$$

#### ¿Por qué usar SBERT?
- Captura **significado semántico**: "coche" y "automóvil" tendrán alta similitud
- Es **contextual**: la misma palabra tiene diferentes embeddings según el contexto
- Entrenado con **NLI** (Natural Language Inference): entiende relaciones lógicas

#### Complejidad
- **Temporal:** $O(n^2)$ por la atención del transformer (n = tokens)
- **Espacial:** $O(n \\times d)$ donde $d = {trace['embedding_dim']}$
"""


def explain_spacy(trace: dict) -> str:
    """Explicación paso a paso del algoritmo spaCy Word Vectors."""
    if "error" in trace:
        return f"⚠️ Error: {trace['error']}"

    vec_a = [f"{v:.4f}" for v in trace["vector_a_sample"]]
    vec_b = [f"{v:.4f}" for v in trace["vector_b_sample"]]

    return f"""### 🤖 spaCy Word Vectors — Explicación

#### Fundamento

**spaCy** usa vectores de palabras pre-entrenados basados en **GloVe**
(Global Vectors for Word Representation). Cada palabra tiene un vector
denso aprendido de co-ocurrencias en un corpus masivo.

#### Modelo utilizado
- **Nombre:** `{trace['model_name']}`
- **Dimensiones del vector:** {trace['vector_dim']}
- **Base:** GloVe (entrenado en Common Crawl)

#### Proceso paso a paso

**1. Tokenización:**
- Texto A → {len(trace['tokens_a'])} tokens
  - Con vector: {len(trace['tokens_with_vectors_a'])} tokens
- Texto B → {len(trace['tokens_b'])} tokens
  - Con vector: {len(trace['tokens_with_vectors_b'])} tokens

**2. Vectorización:**
Para cada token con vector conocido, se recupera su embedding GloVe
de {trace['vector_dim']} dimensiones.

**3. Promedio de vectores:**
El vector del documento se calcula como el promedio de todos los
vectores de sus tokens:

$$
\\vec{{D}} = \\frac{{1}}{{n}} \\sum_{{i=1}}^{{n}} \\vec{{w_i}}
$$

**4. Vectores resultantes (primeros 10 componentes):**
- Vector A: [{', '.join(vec_a)}...]
- Vector B: [{', '.join(vec_b)}...]

**5. Similitud del coseno:**
$$
\\text{{sim}} = \\frac{{\\vec{{D_A}} \\cdot \\vec{{D_B}}}}{{\\|\\vec{{D_A}}\\| \\times \\|\\vec{{D_B}}\\|}} = {trace['similarity']}
$$

#### Diferencia con SBERT
| Aspecto | spaCy (GloVe) | SBERT |
|---------|---------------|-------|
| Tipo | Estático | Contextual |
| Contexto | No | Sí |
| Velocidad | Más rápido | Más lento |
| Semántica | Superficial | Profunda |

#### Complejidad
- **Temporal:** $O(n)$ donde $n$ es el número de tokens
- **Espacial:** $O(n \\times d)$ donde $d = {trace['vector_dim']}$
"""


# Registro de funciones de explicación
EXPLANATION_FUNCS = {
    "levenshtein": explain_levenshtein,
    "jaccard": explain_jaccard,
    "cosine_tfidf": explain_cosine_tfidf,
    "hamming": explain_hamming,
    "sbert": explain_sbert,
    "spacy": explain_spacy,
}
