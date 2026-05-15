"""
Requerimiento 3 — Explicaciones Matemáticas y Algorítmicas
============================================================
Genera explicaciones detalladas en Markdown (con LaTeX para Streamlit)
para los tres algoritmos del Requerimiento 3.
"""

from __future__ import annotations

from typing import Dict, List


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FRECUENCIA DE TÉRMINOS PREDEFINIDOS
# ═══════════════════════════════════════════════════════════════════════════════

def explain_frequency_counting(freq_results: Dict, n_abstracts: int) -> str:
    """
    Explicación matemática y algorítmica del conteo de términos predefinidos.

    Parameters
    ----------
    freq_results  : salida de ``count_category_term_frequencies``
    n_abstracts   : número total de abstracts analizados
    """
    # Top 3 términos para el ejemplo en la explicación
    sorted_terms = sorted(
        freq_results.items(), key=lambda x: -x[1]["doc_frequency"]
    )
    top3 = sorted_terms[:3]
    example_rows = ""
    for term, data in top3:
        example_rows += (
            f"| `{term}` | {data['doc_frequency']} | "
            f"{data['total_occurrences']} | "
            f"{data['doc_frequency_pct']:.1f}% |\n"
        )

    return f"""### 📊 Frecuencia de Términos Predefinidos — Explicación

#### Objetivo
Medir la presencia de los 15 conceptos clave de la categoría
**"Concepts of Generative AI in Education"** en el corpus de {n_abstracts} abstracts.

---

#### Métricas calculadas

Para cada término $t$:

| Métrica | Fórmula | Descripción |
|---------|---------|-------------|
| **Frecuencia de documento** | $df(t)$ | N.° de abstracts que contienen $t$ al menos una vez |
| **Ocurrencias totales** | $\\sum_{{a}} \\text{{count}}(t, a)$ | Suma de todas las apariciones de $t$ en el corpus |
| **Porcentaje** | $df(t) / N \\times 100$ | Cobertura relativa respecto a $N = {n_abstracts}$ abstracts |

---

#### Algoritmo de detección

Cada término se busca mediante una **expresión regular con límite de palabra** (`\\b`),
compilada una sola vez para eficiencia:

```python
# Ejemplo para "fine-tuning"
pattern = re.compile(r"\\b" + "fine" + r"[\\-\\s]?" + "tuning" + r"\\b",
                     re.IGNORECASE)
# → detecta "fine-tuning", "fine tuning" y variaciones de mayúsculas
```

El manejo flexible del guión es crucial: en la literatura científica es común
encontrar el mismo concepto escrito como "fine-tuning" o "fine tuning".

```
Para cada término t:
  doc_freq[t] ← 0
  total_occ[t] ← 0
  Para cada abstract a en el corpus:
    matches ← pattern[t].findall(a)
    Si matches no vacío:
      doc_freq[t] += 1
    total_occ[t] += len(matches)
```

---

#### Diferencia entre las dos métricas

- **Frecuencia de documento** `df(t)`: indica cuántos artículos *abordan* el concepto.
  Es más robusta a textos largos.
- **Ocurrencias totales**: indica con qué *intensidad* se usa el término.
  Un valor alto puede indicar que el concepto es central en el artículo.

---

#### Resultados muestra (top 3 por frecuencia de documento)

| Término | Docs | Ocurrencias | % Cobertura |
|---------|------|-------------|-------------|
{example_rows}

---

#### Complejidad

- **Temporal:** $O(T \\times N \\times L)$ con $T = 15$ términos, $N = {n_abstracts}$ abstracts,
  $L$ = longitud media del abstract. En la práctica se reduce significativamente porque
  los patrones se compilan una sola vez y el motor regex usa autómatas finitos.
- **Espacial:** $O(T)$ — solo almacena los conteos finales.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXTRACCIÓN POR NPMI
# ═══════════════════════════════════════════════════════════════════════════════

def explain_npmi_algorithm(pipeline_result: Dict) -> str:
    """
    Explicación matemática y algorítmica de la extracción de nuevas palabras
    mediante Información Mutua Puntual Normalizada (NPMI).

    Parameters
    ----------
    pipeline_result : salida de ``run_r3_pipeline``
    """
    n = pipeline_result.get("n_abstracts", 0)
    df_cat = pipeline_result.get("df_cat", 0)
    p_cat_str = f"{df_cat / n:.3f}" if n > 0 else "—"

    # Ejemplo con el primer término extraído
    new_terms = pipeline_result.get("new_terms", [])
    example_block = ""
    if new_terms:
        ex = new_terms[0]
        ex_n = n
        ex_dfw = ex.get("doc_frequency", 0)
        ex_dfcat = df_cat
        ex_dfwcat = ex.get("docs_with_category", 0)
        ex_npmi = ex.get("npmi", 0.0)
        ex_term = ex.get("term", "?")

        p_w = ex_dfw / ex_n if ex_n > 0 else 0
        p_cat = ex_dfcat / ex_n if ex_n > 0 else 0
        p_wcat = ex_dfwcat / ex_n if ex_n > 0 else 0

        example_block = f"""---

#### Ejemplo numérico con `"{ex_term}"`

| Variable | Valor |
|----------|-------|
| $N$ (abstracts totales) | {ex_n} |
| $df(w)$ (docs con `{ex_term}`) | {ex_dfw} |
| $df(\\text{{cat}})$ (docs con ≥1 término de categoría) | {ex_dfcat} |
| $df(w, \\text{{cat}})$ (docs con ambos) | {ex_dfwcat} |
| $P(w)$ | {p_w:.4f} |
| $P(\\text{{cat}})$ | {p_cat:.4f} |
| $P(w, \\text{{cat}})$ | {p_wcat:.4f} |

$$
PMI = \\log_2 \\frac{{{p_wcat:.4f}}}{{{p_w:.4f} \\times {p_cat:.4f}}} = \\log_2({(p_wcat / (p_w * p_cat)):.4f}) \\approx {(p_wcat / max(p_w * p_cat, 1e-10)):.4f}
$$

$$
NPMI = \\frac{{PMI}}{{-\\log_2({p_wcat:.4f})}} \\approx \\mathbf{{{ex_npmi:.4f}}}
$$
"""

    return f"""### 🔬 Extracción de Nuevas Palabras por NPMI — Explicación

#### Objetivo
Descubrir algorítmicamente nuevos términos que están estadísticamente
asociados con la categoría, sin haberlos definido a priori.

---

#### ¿Por qué NPMI?

La **Información Mutua Puntual Normalizada** (NPMI, *Normalized Pointwise Mutual
Information*) mide la asociación estadística entre dos eventos. A diferencia de
usar solo la frecuencia bruta, NPMI penaliza términos muy frecuentes en todo el
corpus (que aparecerían por azar junto a cualquier categoría) y premia los que
aparecen *proporcionalmente más* en documentos relevantes.

---

#### Definiciones probabilísticas

Sea $D = \\{{d_1, d_2, \\ldots, d_N\\}}$ el corpus de $N = {n}$ abstracts.

Definimos los eventos a nivel de **documento** (binario: contiene o no contiene):

$$
P(w) = \\frac{{df(w)}}{{N}}, \\quad
P(\\text{{cat}}) = \\frac{{df(\\text{{cat}})}}{{N}} = {p_cat_str}, \\quad
P(w, \\text{{cat}}) = \\frac{{df(w, \\text{{cat}})}}{{N}}
$$

Donde $df(\\text{{cat}}) = {df_cat}$ es el número de documentos que contienen
al menos un término predefinido de la categoría.

---

#### Fórmulas

**Información Mutua Puntual (PMI):**

$$
PMI(w, \\text{{cat}}) = \\log_2 \\frac{{P(w, \\text{{cat}})}}{{P(w) \\cdot P(\\text{{cat}})}}
$$

> PMI > 0 → $w$ aparece *más de lo esperado* junto a la categoría.
> PMI = 0 → independencia estadística.
> PMI < 0 → $w$ aparece *menos de lo esperado*.

**Normalización (NPMI):**

$$
NPMI(w, \\text{{cat}}) = \\frac{{PMI(w, \\text{{cat}})}}{{-\\log_2 P(w, \\text{{cat}})}}
$$

La normalización por $-\\log_2 P(w, \\text{{cat}})$ lleva el rango al intervalo $[-1, 1]$,
lo que permite comparar términos independientemente de su frecuencia absoluta.

| NPMI | Interpretación |
|------|---------------|
| $\\approx 1$ | Asociación perfecta: $w$ aparece solo en docs con la categoría |
| $\\approx 0$ | Independencia estadística |
| $\\approx -1$ | Exclusión mutua: $w$ nunca aparece con la categoría |

---

#### Algoritmo paso a paso

```
ENTRADA: corpus de N abstracts, términos de categoría, max_terms, min_doc_freq

1. Calcular D_cat = documentos con ≥1 término de categoría
   df_cat = |D_cat|

2. Para cada abstract a en el corpus:
   a. Tokenizar → unigramas (≥3 chars) + bigramas (≥2 chars c/u)
   b. Excluir tokens que coincidan con términos predefinidos
   c. Registrar doc_id en term_doc_ids[token]
   d. Incrementar term_count[token]

3. Para cada candidato w con df(w) ≥ min_doc_freq:
   a. df_w_cat = |term_doc_ids[w] ∩ D_cat|
   b. Calcular NPMI(w, cat) con las fórmulas anteriores

4. Filtrar: mantener solo candidatos con NPMI > 0
5. Ordenar: descendente por NPMI (desempate por df)
6. Retornar: top max_terms candidatos
```

{example_block}

---

#### Complejidad

- **Temporal:** $O(N \\times V)$ donde $V$ es el tamaño del vocabulario candidato.
  La búsqueda del conjunto $D_{{cat}}$ es $O(N \\times T)$ con $T=15$ términos.
- **Espacial:** $O(V)$ para almacenar los conjuntos de doc_ids por término.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EVALUACIÓN DE PRECISIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def explain_precision_evaluation(new_terms: List[Dict], df_cat: int, n: int) -> str:
    """
    Explicación de la métrica de precisión por co-ocurrencia.

    Parameters
    ----------
    new_terms : lista de dicts con precisión calculada
    df_cat    : número de documentos con términos de categoría
    n         : total de abstracts
    """
    # Ejemplo con el término de mayor precisión
    if new_terms:
        best = max(new_terms, key=lambda x: x.get("precision", 0))
        ex_term = best.get("term", "?")
        ex_dfw = best.get("doc_frequency", 0)
        ex_dfwcat = best.get("docs_with_category", 0)
        ex_prec = best.get("precision_pct", 0)
        example_str = (
            f"\n#### Ejemplo: `\"{ex_term}\"`\n\n"
            f"$$\\text{{precision}}(\\text{{{ex_term}}}) = "
            f"\\frac{{{ex_dfwcat}}}{{{ex_dfw}}} = {ex_prec:.1f}\\%$$\n\n"
            f"El término `{ex_term}` aparece en {ex_dfw} abstracts, "
            f"de los cuales {ex_dfwcat} también contienen al menos un "
            f"término predefinido de la categoría.\n"
        )
    else:
        example_str = ""

    n_very = sum(1 for t in new_terms if t.get("precision", 0) >= 0.75)
    n_mod = sum(1 for t in new_terms if 0.50 <= t.get("precision", 0) < 0.75)
    n_weak = sum(1 for t in new_terms if 0.25 <= t.get("precision", 0) < 0.50)
    n_low = sum(1 for t in new_terms if t.get("precision", 0) < 0.25)

    return f"""### 🎯 Evaluación de Precisión — Explicación

#### ¿Qué significa "precisar" un término extraído?

En el contexto de recuperación de información, **precisión** mide qué fracción
de los resultados recuperados son realmente relevantes. Aquí la adaptamos a:

> *¿Con qué frecuencia el nuevo término co-ocurre con conceptos de la categoría?*

---

#### Métrica: Precisión de Co-ocurrencia

Para cada nuevo término $t$:

$$
\\text{{precision}}(t) = \\frac{{|D(t) \\cap D_{{\\text{{cat}}}}|}}{{|D(t)|}}
$$

Donde:
- $D(t)$ = conjunto de documentos que contienen el término $t$
- $D_{{\\text{{cat}}}}$ = conjunto de documentos con ≥1 término predefinido ($|D_{{\\text{{cat}}}}| = {df_cat}$ de $N = {n}$ abstracts)

Esta métrica responde: **"¿En qué fracción de los documentos donde aparece el
nuevo término también aparece algún concepto de la categoría?"**

---

#### Escala de interpretación

| Rango de precisión | Etiqueta | Significado |
|--------------------|----------|-------------|
| $[0.75,\\ 1.00]$ | **Muy relevante** | El término está fuertemente asociado a la categoría |
| $[0.50,\\ 0.74]$ | **Moderadamente relevante** | Asociación parcial; aparece en otros contextos también |
| $[0.25,\\ 0.49]$ | **Débilmente relevante** | Poca especificidad para la categoría |
| $[0.00,\\ 0.24]$ | **Poco relevante** | Probable ruido o término demasiado genérico |

{example_str}

---

#### Relación entre NPMI y Precisión

| Métrica | Mide | ¿Para qué sirve? |
|---------|------|-----------------|
| **NPMI** | Asociación estadística (binaria) | *Descubrimiento* de candidatos |
| **Precisión** | Co-ocurrencia proporcional | *Validación* de la relevancia |

Un término puede tener NPMI alto pero precisión moderada (si es frecuente en
el corpus general). La precisión es el criterio definitivo de calidad.

---

#### Distribución de precisión en los términos extraídos

| Nivel | Cantidad |
|-------|---------|
| Muy relevante (≥75%) | {n_very} |
| Moderadamente relevante (50–74%) | {n_mod} |
| Débilmente relevante (25–49%) | {n_weak} |
| Poco relevante (<25%) | {n_low} |

---

#### Ventajas y limitaciones

**Ventajas:**
- Interpretable directamente como proporción (0 a 1)
- Independiente del tamaño del corpus
- Consistente con la cobertura de la categoría

**Limitaciones:**
- No captura relaciones semánticas latentes (solo co-ocurrencia directa)
- Sensible al umbral de `min_doc_freq`
- Para corpora pequeños (< 50 abstracts), los valores pueden ser inestables
"""