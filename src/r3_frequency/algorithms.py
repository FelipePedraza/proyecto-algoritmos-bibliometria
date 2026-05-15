"""
Requerimiento 3 — Frecuencia de Términos y Extracción de Palabras Asociadas
============================================================================
Dada la categoría "Concepts of Generative AI in Education" con sus 15 palabras
predefinidas, este módulo implementa:

  1. Frecuencia de aparición de cada término predefinido en los abstracts
     (frecuencia de documento + total de ocurrencias).
  2. Extracción algorítmica de hasta 15 nuevas palabras asociadas usando
     Información Mutua Puntual Normalizada (NPMI) sobre unigramas y bigramas.
  3. Evaluación de precisión de las nuevas palabras mediante la proporción
     de documentos en que co-ocurren con los términos predefinidos.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# DEFINICIÓN DE LA CATEGORÍA
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORY_NAME: str = "Concepts of Generative AI in Education"

CATEGORY_TERMS: List[str] = [
    "generative models",
    "prompting",
    "machine learning",
    "multimodality",
    "fine-tuning",
    "training data",
    "algorithmic bias",
    "explainability",
    "transparency",
    "ethics",
    "privacy",
    "personalization",
    "human-ai interaction",
    "ai literacy",
    "co-creation",
]

# ═══════════════════════════════════════════════════════════════════════════════
# STOP WORDS (para extracción de nuevos términos)
# ═══════════════════════════════════════════════════════════════════════════════

_STOP_WORDS: Set[str] = {
    # Artículos y determinantes
    "a", "an", "the",
    # Conjunciones y preposiciones
    "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "through", "during", "between",
    "against", "without", "within", "along", "following", "across",
    # Verbos auxiliares
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can",
    "shall", "need", "ought",
    # Pronombres
    "it", "its", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "they", "them", "their",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "our", "us", "my", "me", "him", "her", "his",
    # Adverbios y cuantificadores comunes
    "all", "both", "each", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "not", "no", "nor", "so", "yet",
    "as", "if", "while", "although", "because", "since",
    "after", "before", "also", "only", "over",
    "then", "here", "there", "any", "many", "few", "same",
    "two", "three", "et", "al",
    # Términos bibliométricos genéricos sin contenido informativo
    "paper", "study", "research", "article", "work", "works",
    "results", "result", "approach", "approaches",
    "method", "methods", "technique", "techniques",
    "using", "used", "use", "uses", "based",
    "show", "shows", "shown", "demonstrate", "demonstrates",
    "propose", "proposed", "present", "presents", "presented",
    "analysis", "analyze", "analyses", "evaluated", "evaluation",
    "new", "novel", "different", "various", "several", "specific",
    "however", "thus", "therefore", "hence", "furthermore",
    "first", "second", "third", "finally", "one", "can",
    "provide", "provides", "provided", "general", "important",
    "often", "well", "given", "both", "also",
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE PATRONES Y CONJUNTO DE EXCLUSIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def _term_to_pattern(term: str) -> re.Pattern:
    """
    Crea un patrón regex flexible para un término de la categoría.

    Maneja:
    - Términos multi-palabra: búsqueda de la frase exacta
    - Términos con guión: permite guión o espacio (e.g. "fine-tuning" o "fine tuning")
    - Límite de palabra (\\b) para evitar coincidencias parciales

    Examples
    --------
    "fine-tuning"          → matches "fine-tuning", "fine tuning"
    "human-ai interaction" → matches "human-ai interaction", "human ai interaction"
    "generative models"    → matches "generative models"
    "prompting"            → matches "prompting" (no "re-prompting")
    """
    escaped = re.escape(term.lower())
    # Guión → opcional (guión o espacio, o ninguno)
    flexible = escaped.replace(r"\-", r"[\-\s]?")
    return re.compile(r"\b" + flexible + r"\b", re.IGNORECASE)


# Pre-compilar patrones de los términos de la categoría
_CATEGORY_PATTERNS: Dict[str, re.Pattern] = {
    term: _term_to_pattern(term) for term in CATEGORY_TERMS
}

# Conjunto de formas normalizadas de los términos de la categoría
# (para exclusión durante la extracción de nuevos términos)
_CATEGORY_EXCLUSION: Set[str] = set()
for _t in CATEGORY_TERMS:
    _tl = _t.lower().strip()
    _CATEGORY_EXCLUSION.add(_tl)                          # original
    _CATEGORY_EXCLUSION.add(re.sub(r"\-", " ", _tl).strip())  # guión → espacio
    _CATEGORY_EXCLUSION.add(re.sub(r"\-", "", _tl).strip())   # sin guión


def _is_category_term(candidate: str) -> bool:
    """
    Devuelve True si el candidato coincide exactamente con alguna forma
    normalizada de un término de la categoría predefinida.

    No excluye los componentes individuales de frases multi-palabra
    (e.g., "model" o "learning" son candidatos válidos aunque formen parte
    de "generative models" o "machine learning").
    """
    norm = re.sub(r"\s+", " ", candidate.lower().strip())
    return norm in _CATEGORY_EXCLUSION


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FRECUENCIA DE TÉRMINOS PREDEFINIDOS
# ═══════════════════════════════════════════════════════════════════════════════

def count_category_term_frequencies(
    abstracts: List[str],
) -> Dict[str, Dict]:
    """
    Calcula la frecuencia de aparición de cada uno de los 15 términos
    predefinidos de la categoría en los abstracts del corpus.

    Para cada término retorna:
      - ``doc_frequency``     : número de abstracts que contienen el término
      - ``total_occurrences`` : suma total de ocurrencias en todos los abstracts
      - ``doc_frequency_pct`` : porcentaje de abstracts que contienen el término

    Algoritmo
    ---------
    Para cada término ``t`` y cada abstract ``a``:
      1. Aplicar el patrón regex compilado (con \\b y manejo de guiones).
      2. ``findall`` devuelve la lista de coincidencias → acumular conteos.

    El patrón maneja variaciones tipográficas:
      "fine-tuning" detecta también "fine tuning"
      "human-ai interaction" detecta "human ai interaction"

    Complejidad temporal: O(T × A × L) con T=15 términos, A=abstracts, L=longitud media.
    Complejidad espacial: O(T)

    Parameters
    ----------
    abstracts : lista de textos de abstracts (strings; se ignoran los vacíos)

    Returns
    -------
    dict ``{término: {"doc_frequency": int, "total_occurrences": int,
                      "doc_frequency_pct": float}}``
    """
    valid = [a for a in abstracts if str(a).strip()]
    n = len(valid)
    results: Dict[str, Dict] = {}

    for term in CATEGORY_TERMS:
        pattern = _CATEGORY_PATTERNS[term]
        doc_freq = 0
        total_occ = 0

        for abstract in valid:
            matches = pattern.findall(abstract)
            if matches:
                doc_freq += 1
                total_occ += len(matches)

        results[term] = {
            "doc_frequency": doc_freq,
            "total_occurrences": total_occ,
            "doc_frequency_pct": round(doc_freq / n * 100, 2) if n > 0 else 0.0,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TOKENIZACIÓN PARA EXTRACCIÓN DE NUEVOS TÉRMINOS
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenize_with_ngrams(
    text: str,
    include_bigrams: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Tokeniza un texto y genera unigramas y bigramas candidatos.

    Proceso
    -------
    1. Minúsculas y eliminación de todo lo que no sea letra o espacio.
    2. Tokenización por espacios.
    3. Unigramas: tokens con longitud ≥ 3 que no sean stop words.
    4. Bigramas: pares de tokens consecutivos donde:
       - Cada token tenga longitud ≥ 2.
       - El token más largo tenga longitud ≥ 3.
       - No sean ambos stop words.

    Parameters
    ----------
    text            : texto del abstract (string crudo)
    include_bigrams : si True, también genera bigramas

    Returns
    -------
    (unigrams, bigrams) como listas de strings en minúscula
    """
    text_lower = str(text).lower()
    text_clean = re.sub(r"[^a-z\s]", " ", text_lower)
    text_clean = re.sub(r"\s+", " ", text_clean).strip()
    raw = text_clean.split()

    # Unigramas: longitud ≥ 3 y no stop word
    unigrams = [t for t in raw if len(t) >= 3 and t not in _STOP_WORDS]

    bigrams: List[str] = []
    if include_bigrams:
        for i in range(len(raw) - 1):
            w1, w2 = raw[i], raw[i + 1]
            if (
                len(w1) >= 2
                and len(w2) >= 2
                and max(len(w1), len(w2)) >= 3
                and not (w1 in _STOP_WORDS and w2 in _STOP_WORDS)
            ):
                bigrams.append(f"{w1} {w2}")

    return unigrams, bigrams


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXTRACCIÓN POR NPMI
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_npmi(
    n_docs: int,
    df_w: int,
    df_cat: int,
    df_w_cat: int,
) -> float:
    """
    Calcula la Información Mutua Puntual Normalizada (NPMI).

    Definiciones probabilísticas (a nivel de documento):
      P(w)     = df_w / N
      P(cat)   = df_cat / N
      P(w,cat) = df_w_cat / N

    Cálculo:
      PMI(w, cat)  = log₂[ P(w,cat) / (P(w) · P(cat)) ]
      NPMI(w, cat) = PMI(w, cat) / −log₂[ P(w,cat) ]

    Propiedades:
      Rango: [−1, 1]
       1  → asociación perfecta (w aparece exclusivamente en docs con categoría)
       0  → independencia estadística
      −1  → exclusión mutua perfecta

    Parameters
    ----------
    n_docs    : total de documentos en el corpus  (N)
    df_w      : documentos que contienen w        (df(w))
    df_cat    : documentos con ≥1 término cat.    (df(cat))
    df_w_cat  : documentos con w Y ≥1 término cat.(df(w, cat))

    Returns
    -------
    float en [−1.0, 1.0]; devuelve −1.0 si la probabilidad conjunta es 0
    """
    if df_w_cat == 0 or df_w == 0 or df_cat == 0 or n_docs == 0:
        return -1.0

    p_w = df_w / n_docs
    p_cat = df_cat / n_docs
    p_w_cat = df_w_cat / n_docs

    if p_w_cat <= 0.0 or p_w <= 0.0 or p_cat <= 0.0:
        return -1.0

    pmi = math.log2(p_w_cat / (p_w * p_cat))
    h_joint = -math.log2(p_w_cat)

    if h_joint == 0.0:
        return 1.0  # Cobertura perfecta

    npmi = pmi / h_joint
    return round(max(-1.0, min(1.0, npmi)), 4)


def extract_new_associated_terms(
    abstracts: List[str],
    max_terms: int = 15,
    min_doc_freq: int = 2,
    include_bigrams: bool = True,
) -> List[Dict]:
    """
    Extrae nuevas palabras asociadas a la categoría usando NPMI.

    Algoritmo completo (5 pasos)
    ----------------------------
    **Paso 1 — Identificar documentos relevantes:**
      Un documento es "relevante a la categoría" si contiene al menos
      uno de los 15 términos predefinidos. Sea D_cat ⊆ D el subconjunto.

    **Paso 2 — Tokenizar y recolectar candidatos:**
      Para cada abstract, generar unigramas (≥3 chars, no stop-word)
      y bigramas (tokens ≥2 chars, al menos uno ≥3). Excluir términos
      que coincidan con los predefinidos de la categoría.

    **Paso 3 — Calcular frecuencias de documento:**
      Para cada candidato t:
        df(t)   = documentos que contienen t
        df(t,cat) = documentos que contienen t Y pertenecen a D_cat

    **Paso 4 — Calcular NPMI:**
      NPMI(t, cat) = PMI(t, cat) / −log₂[P(t, cat)]
      Mayor NPMI → mayor asociación estadística con la categoría.

    **Paso 5 — Filtrar y ordenar:**
      Filtrar: df(t) ≥ min_doc_freq, NPMI > 0.
      Ordenar: descendente por NPMI (desempate por df).
      Retornar: top max_terms candidatos.

    Complejidad temporal: O(N × V) donde N = abstracts, V = vocabulario
    Complejidad espacial: O(V)

    Parameters
    ----------
    abstracts      : lista de textos de abstracts
    max_terms      : máximo de nuevos términos a retornar (≤ 15)
    min_doc_freq   : frecuencia mínima de documento para incluir un candidato
    include_bigrams: si True, analiza también bigramas

    Returns
    -------
    Lista de dicts:
      ``{"term", "doc_frequency", "total_occurrences",
         "doc_frequency_pct", "npmi", "docs_with_category"}``
    """
    valid = [str(a) for a in abstracts if str(a).strip()]
    n = len(valid)

    if n == 0:
        return []

    # ── Paso 1: Documentos relevantes a la categoría ──────────────────────────
    cat_doc_ids: Set[int] = set()
    for doc_id, abstract in enumerate(valid):
        for pattern in _CATEGORY_PATTERNS.values():
            if pattern.search(abstract):
                cat_doc_ids.add(doc_id)
                break
    df_cat = len(cat_doc_ids)

    # ── Paso 2: Tokenizar y recolectar candidatos ─────────────────────────────
    # term → set de doc_ids donde aparece
    term_doc_ids: Dict[str, Set[int]] = defaultdict(set)
    # term → total de ocurrencias (suma de todas las apariciones)
    term_count: Counter = Counter()

    for doc_id, abstract in enumerate(valid):
        unigrams, bigrams = _tokenize_with_ngrams(abstract, include_bigrams=include_bigrams)
        for candidate in unigrams + bigrams:
            if _is_category_term(candidate):
                continue
            term_doc_ids[candidate].add(doc_id)
            term_count[candidate] += 1

    # ── Pasos 3–4: Frecuencias y NPMI ────────────────────────────────────────
    results: List[Dict] = []
    for term, doc_ids in term_doc_ids.items():
        df_w = len(doc_ids)
        if df_w < min_doc_freq:
            continue

        df_w_cat = len(doc_ids & cat_doc_ids)
        npmi = _compute_npmi(n, df_w, df_cat, df_w_cat)

        results.append({
            "term": term,
            "doc_frequency": df_w,
            "total_occurrences": term_count[term],
            "doc_frequency_pct": round(df_w / n * 100, 2),
            "npmi": npmi,
            "docs_with_category": df_w_cat,
        })

    # ── Paso 5: Filtrar, ordenar y retornar ──────────────────────────────────
    results = [r for r in results if r["npmi"] > 0]
    results.sort(key=lambda x: (-x["npmi"], -x["doc_frequency"]))
    return results[:max_terms]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EVALUACIÓN DE PRECISIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_precision(
    new_terms: List[Dict],
    abstracts: List[str],
) -> List[Dict]:
    """
    Evalúa la precisión de cada nuevo término extraído respecto a la categoría.

    **Métrica: Precisión de Co-ocurrencia**

    Para cada nuevo término t:

    .. math::
       \\text{precision}(t) = \\frac{|D(t) \\cap D_{\\text{cat}}|}{|D(t)|}

    Donde:
      D(t)     = documentos que contienen el término t
      D_cat    = documentos con ≥ 1 término predefinido de la categoría

    Interpretación:
      "¿En qué fracción de los documentos donde aparece t
      también aparece algún concepto de la categoría?"

    Rangos de interpretación:
      0.75 – 1.00  → Muy relevante para la categoría
      0.50 – 0.74  → Moderadamente relevante
      0.25 – 0.49  → Débilmente relevante
      0.00 – 0.24  → Poco relevante o ruido

    Además se computa:
      - ``npmi_rank``: posición en el ranking por NPMI (antes de re-ordenar)
      - ``precision_pct``: precision en porcentaje

    Parameters
    ----------
    new_terms : lista de dicts de salida de ``extract_new_associated_terms``
    abstracts : lista de textos de abstracts (mismo corpus)

    Returns
    -------
    Lista enriquecida con ``"precision"``, ``"precision_pct"`` y
    ``"precision_label"`` en cada dict, ordenada por precisión descendente.
    """
    valid = [str(a) for a in abstracts if str(a).strip()]
    n = len(valid)

    if n == 0 or not new_terms:
        return new_terms

    # Reconstruir conjunto de documentos con categoría
    cat_doc_ids: Set[int] = set()
    for doc_id, abstract in enumerate(valid):
        for pattern in _CATEGORY_PATTERNS.values():
            if pattern.search(abstract):
                cat_doc_ids.add(doc_id)
                break

    enriched: List[Dict] = []
    for rank_npmi, item in enumerate(new_terms, start=1):
        term = item["term"]
        # Patrón de búsqueda para el nuevo término (exacto, con límite de palabra)
        pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)

        # Documentos que contienen el nuevo término
        t_doc_ids: Set[int] = set()
        for doc_id, abstract in enumerate(valid):
            if pat.search(abstract):
                t_doc_ids.add(doc_id)

        df_w = len(t_doc_ids)
        df_w_cat = len(t_doc_ids & cat_doc_ids)
        precision = df_w_cat / df_w if df_w > 0 else 0.0

        if precision >= 0.75:
            label = "Muy relevante"
        elif precision >= 0.50:
            label = "Moderadamente relevante"
        elif precision >= 0.25:
            label = "Débilmente relevante"
        else:
            label = "Poco relevante"

        enriched.append({
            **item,
            "npmi_rank": rank_npmi,
            "precision": round(precision, 4),
            "precision_pct": round(precision * 100, 1),
            "precision_label": label,
        })

    # Ordenar por precisión descendente (desempate por NPMI)
    enriched.sort(key=lambda x: (-x["precision"], -x["npmi"]))
    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PIPELINE COMPLETO
# ═══════════════════════════════════════════════════════════════════════════════

def run_r3_pipeline(
    abstracts: List[str],
    max_new_terms: int = 15,
    min_doc_freq: int = 2,
    include_bigrams: bool = True,
) -> Dict:
    """
    Ejecuta el pipeline completo del Requerimiento 3.

    Etapas
    ------
    1. ``count_category_term_frequencies`` — frecuencia de los 15 términos predefinidos
    2. ``extract_new_associated_terms``    — extracción NPMI de nuevas palabras
    3. ``evaluate_precision``              — evaluación de precisión por co-ocurrencia

    Returns
    -------
    dict con claves:
      - ``n_abstracts``            : total de abstracts con texto válido
      - ``n_category_docs``        : documentos que contienen ≥1 término de categoría
      - ``category_coverage_pct``  : porcentaje de cobertura de la categoría
      - ``predefined_frequencies`` : dict de R1
      - ``new_terms``              : lista de dicts de R2 + R3 (con precisión)
      - ``df_cat``                 : tamaño de D_cat (para las fórmulas)
    """
    valid = [str(a) for a in abstracts if str(a).strip()]
    n = len(valid)

    # Frecuencias predefinidas
    predefined_freq = count_category_term_frequencies(valid)

    # Extracción de nuevas palabras
    new_terms_raw = extract_new_associated_terms(
        valid,
        max_terms=max_new_terms,
        min_doc_freq=min_doc_freq,
        include_bigrams=include_bigrams,
    )

    # Evaluación de precisión
    new_terms = evaluate_precision(new_terms_raw, valid)

    # Estadísticas de cobertura de la categoría
    cat_doc_ids: Set[int] = set()
    for doc_id, abstract in enumerate(valid):
        for pattern in _CATEGORY_PATTERNS.values():
            if pattern.search(abstract):
                cat_doc_ids.add(doc_id)
                break

    return {
        "n_abstracts": n,
        "n_category_docs": len(cat_doc_ids),
        "category_coverage_pct": round(len(cat_doc_ids) / n * 100, 2) if n > 0 else 0.0,
        "predefined_frequencies": predefined_freq,
        "new_terms": new_terms,
        "df_cat": len(cat_doc_ids),
    }