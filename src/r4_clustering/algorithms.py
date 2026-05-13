"""
Requerimiento 4 — Algoritmos de Clustering Jerárquico
=======================================================
Implementa 3 algoritmos de agrupamiento jerárquico DESDE CERO:

  1. Enlace Completo  (Complete Linkage) — distancia máxima entre clusters
  2. Enlace Promedio  (Average Linkage / UPGMA) — distancia promedio ponderada
  3. Método de Ward   — minimiza el incremento de varianza intra-cluster

Pipeline completo:
  1. Preprocesamiento de texto (minúsculas, limpieza, stop words, stemming)
  2. Vectorización TF-IDF manual
  3. Matriz de distancias coseno (n×n)
  4. Clustering jerárquico con cada uno de los 3 métodos
  5. Evaluación con coeficiente de correlación cofenética

Complejidad de los 3 algoritmos: O(n³) — implementación naive para claridad académica
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple

# ── Stop words en inglés (especializado para abstracts científicos) ──────────

ENGLISH_STOP_WORDS: set[str] = {
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
    "shall", "need", "dare", "ought",
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
    # Palabras comunes en abstracts científicos (no informativas)
    "paper", "study", "research", "article", "work", "works",
    "results", "result", "approach", "approaches",
    "method", "methods", "technique", "techniques",
    "using", "used", "use", "uses", "based", "basis",
    "show", "shows", "shown", "demonstrate", "demonstrates",
    "propose", "proposed", "present", "presents", "presented",
    "analysis", "analyze", "analyses", "evaluated", "evaluation",
    "new", "novel", "different", "various", "several", "specific",
    "model", "models", "system", "systems", "data", "dataset",
    "however", "thus", "therefore", "hence", "furthermore",
    "first", "second", "third", "finally", "one", "can",
}


# ── 1. Preprocesamiento de texto ─────────────────────────────────────────────


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    apply_stemming: bool = True,
    min_token_length: int = 3,
) -> List[str]:
    """
    Pipeline de preprocesamiento de texto para abstracts científicos.

    Pasos:
      1. Convertir a minúsculas
      2. Eliminar caracteres no alfabéticos (números, símbolos, puntuación)
      3. Tokenizar por espacios
      4. Filtrar tokens cortos (< min_token_length)
      5. Eliminar stop words
      6. Stemming simple (sufijos comunes en inglés)

    Args:
        text: texto crudo del abstract
        remove_stopwords: si True, elimina palabras vacías
        apply_stemming: si True, aplica stemming
        min_token_length: longitud mínima de token a conservar

    Returns:
        Lista de tokens preprocesados

    Complejidad temporal: O(L) donde L = longitud del texto
    Complejidad espacial: O(T) donde T = número de tokens
    """
    # Paso 1: Minúsculas
    text = text.lower()

    # Paso 2: Eliminar todo lo que no sea letra o espacio
    text = re.sub(r"[^a-z\s]", " ", text)

    # Paso 3: Tokenizar
    tokens = text.split()

    # Paso 4: Filtrar tokens muy cortos
    tokens = [t for t in tokens if len(t) >= min_token_length]

    # Paso 5: Eliminar stop words
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

    # Paso 6: Stemming
    if apply_stemming:
        tokens = [_simple_stem(t) for t in tokens]

    return tokens


def _simple_stem(word: str) -> str:
    """
    Stemmer simple basado en eliminación de sufijos comunes en inglés.

    Estrategia: quitar el sufijo más largo que cumpla que la raíz
    resultante tenga al menos 3 caracteres.

    Inspirado en el algoritmo de Porter (versión simplificada).

    Ejemplos:
        "learning"   → "learn"
        "proposed"   → "propos"
        "algorithms" → "algorithm"

    Complejidad: O(S) donde S = número de sufijos posibles (constante ~40)
    """
    if len(word) <= 4:
        return word

    # Sufijos ordenados de mayor a menor longitud
    suffixes = [
        "ational", "tional", "ization", "isation",
        "iveness", "fulness", "ousness", "ations",
        "ation", "ments", "ating", "izing", "ising",
        "ical", "ness", "ment", "tion", "able", "ible",
        "ance", "ence", "ator", "ious", "ings",
        "ing", "ely", "ive", "ize", "ise", "ous",
        "ful", "ism", "ist", "ity",
        "al", "ed", "er", "ly", "es", "rs",
    ]

    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]

    return word


# ── 2. Vectorización TF-IDF ──────────────────────────────────────────────────


def compute_tfidf_vectors(
    corpus: List[List[str]],
) -> Tuple[List[List[float]], List[str]]:
    """
    Calcula la matriz TF-IDF para un corpus de documentos tokenizados.

    Fórmulas:
      TF(t, d) = freq(t, d) / |d|
      IDF(t)   = log((N + 1) / (df(t) + 1)) + 1     ← suavizado sklearn
      TF-IDF(t, d) = TF(t, d) × IDF(t)

    Donde:
      N     = número de documentos
      df(t) = documentos que contienen el término t
      |d|   = número de tokens en el documento d

    Args:
        corpus: lista de listas de tokens [doc1_tokens, doc2_tokens, ...]

    Returns:
        (matrix, vocabulary):
        - matrix: lista de N vectores TF-IDF (N × V)
        - vocabulary: lista ordenada de V términos únicos

    Complejidad temporal: O(N × V) donde N = documentos, V = vocabulario
    Complejidad espacial: O(N × V)
    """
    n_docs = len(corpus)
    if n_docs == 0:
        return [], []

    # Vocabulario global (ordenado para reproducibilidad)
    vocab_set: set[str] = set()
    for tokens in corpus:
        vocab_set.update(tokens)
    vocabulary = sorted(vocab_set)
    vocab_index = {word: i for i, word in enumerate(vocabulary)}
    V = len(vocabulary)

    # Term Frequency por documento
    tf_list: List[dict] = []
    for tokens in corpus:
        total = len(tokens) if tokens else 1
        count: dict[str, int] = {}
        for t in tokens:
            count[t] = count.get(t, 0) + 1
        tf = {word: freq / total for word, freq in count.items()}
        tf_list.append(tf)

    # Document Frequency por término
    doc_sets = [set(tokens) for tokens in corpus]
    df: dict[str, int] = {}
    for word in vocabulary:
        df[word] = sum(1 for s in doc_sets if word in s)

    # IDF (suavizado log)
    idf: dict[str, float] = {}
    for word in vocabulary:
        idf[word] = math.log((n_docs + 1) / (df[word] + 1)) + 1.0

    # Construir matriz TF-IDF: N × V
    matrix: List[List[float]] = []
    for tf in tf_list:
        vec = [tf.get(word, 0.0) * idf[word] for word in vocabulary]
        matrix.append(vec)

    return matrix, vocabulary


# ── 3. Matriz de distancias coseno ────────────────────────────────────────────


def _vec_norm(vec: List[float]) -> float:
    """Norma L2 de un vector."""
    return math.sqrt(sum(x * x for x in vec))


def cosine_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcula la distancia coseno entre dos vectores TF-IDF.

    distancia_coseno = 1 - similitud_coseno
                     = 1 - (A · B) / (‖A‖ × ‖B‖)

    Rango: [0, 1]
      0 = vectores idénticos (abstracts casi iguales)
      1 = vectores ortogonales (sin vocabulario en común)

    Complejidad: O(V) donde V = dimensiones del vector
    """
    dot = sum(a * b for a, b in zip(vec1, vec2))
    n1 = _vec_norm(vec1)
    n2 = _vec_norm(vec2)
    if n1 == 0.0 or n2 == 0.0:
        return 1.0  # Abstracto vacío → máxima distancia
    cos_sim = dot / (n1 * n2)
    # Clamp numérico para evitar valores fuera de [-1, 1]
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


def compute_distance_matrix(tfidf_matrix: List[List[float]]) -> List[List[float]]:
    """
    Calcula la matriz de distancias coseno simétrica N×N entre todos los documentos.

    Propiedades:
      - Diagonal = 0 (documento consigo mismo)
      - Simétrica: dist[i][j] == dist[j][i]
      - Valores en [0, 1]

    Complejidad temporal: O(n² × V)
    Complejidad espacial: O(n²)
    """
    n = len(tfidf_matrix)
    dist: List[List[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance(tfidf_matrix[i], tfidf_matrix[j])
            dist[i][j] = d
            dist[j][i] = d  # Simetría

    return dist


# ── 4. Algoritmos de Clustering Jerárquico ────────────────────────────────────
#
# Los tres algoritmos siguen el mismo esquema general (HAC):
#   1. Iniciar: cada documento es su propio cluster
#   2. Repetir n-1 veces:
#       a. Encontrar los dos clusters más cercanos
#       b. Fusionarlos en un nuevo cluster
#       c. Actualizar la matriz de distancias (regla diferente por algoritmo)
#   3. Retornar la lista de pasos de fusión (linkage matrix)
#
# Formato de salida compatible con scipy:
#   [(cluster_a, cluster_b, distancia, tamaño_nuevo_cluster), ...]
#   Los IDs de clusters sintéticos empiezan en n y aumentan de 1 en 1.


def complete_linkage(dist_matrix: List[List[float]]) -> List[Tuple]:
    """
    Clustering jerárquico — Enlace Completo (Complete Linkage).

    Regla de distancia entre clusters:
        d(A∪B, C) = max(d(A, C), d(B, C))

    La distancia entre dos clusters es la MÁXIMA distancia entre cualquier
    par de puntos (uno de cada cluster). También llamado "vecino más lejano".

    Características:
      + Produce clusters compactos y bien separados
      - Sensible a outliers (un punto alejado domina la distancia)
      - Tiende a romper clusters grandes

    Complejidad: O(n³) — n-1 pasos, cada uno con búsqueda O(n²)

    Returns:
        Lista de n-1 tuplas (idx_a, idx_b, distancia, tamaño_nuevo_cluster)
        compatible con scipy dendrogram.
    """
    n = len(dist_matrix)
    if n < 2:
        return []

    # Diccionario de distancias entre pares de clusters activos
    dist_dict: dict[tuple, float] = {}
    for i in range(n):
        for j in range(n):
            dist_dict[(i, j)] = dist_matrix[i][j]

    active_ids: List[int] = list(range(n))         # IDs de clusters activos
    cluster_members: dict[int, List[int]] = {i: [i] for i in range(n)}
    synthetic_id = n                                # Siguiente ID sintético

    linkage_steps: List[Tuple] = []

    for _ in range(n - 1):
        # ── Paso 2a: Encontrar el par más cercano ─────────────────────────
        min_dist = math.inf
        best_i, best_j = -1, -1

        for ai in range(len(active_ids)):
            for aj in range(ai + 1, len(active_ids)):
                ci = active_ids[ai]
                cj = active_ids[aj]
                d = dist_dict.get((ci, cj), math.inf)
                if d < min_dist:
                    min_dist = d
                    best_i, best_j = ci, cj

        # ── Paso 2b: Fusionar best_i y best_j ────────────────────────────
        new_id = synthetic_id
        new_members = cluster_members[best_i] + cluster_members[best_j]
        cluster_members[new_id] = new_members
        new_size = len(new_members)

        linkage_steps.append((best_i, best_j, min_dist, new_size))

        # ── Paso 2c: Actualizar distancias — regla de enlace completo ─────
        # d(A∪B, C) = max(d(A, C), d(B, C))
        for ck in active_ids:
            if ck == best_i or ck == best_j:
                continue
            d_ik = dist_dict.get((best_i, ck), math.inf)
            d_jk = dist_dict.get((best_j, ck), math.inf)
            new_d = max(d_ik, d_jk)          # Enlace completo: máximo
            dist_dict[(new_id, ck)] = new_d
            dist_dict[(ck, new_id)] = new_d

        active_ids.remove(best_i)
        active_ids.remove(best_j)
        active_ids.append(new_id)
        synthetic_id += 1

    return linkage_steps


def average_linkage(dist_matrix: List[List[float]]) -> List[Tuple]:
    """
    Clustering jerárquico — Enlace Promedio (Average Linkage / UPGMA).

    Regla de distancia entre clusters:
        d(A∪B, C) = (|A| × d(A, C) + |B| × d(B, C)) / (|A| + |B|)

    La distancia entre dos clusters es el PROMEDIO PONDERADO de las distancias
    entre sus miembros. UPGMA = Unweighted Pair Group Method with Arithmetic Mean.

    Características:
      + Balance entre enlace completo y simple
      + Menos sensible a outliers que el enlace completo
      + Generalmente produce el mejor coeficiente cofenético
      - Puede producir clusters desiguales en tamaño

    Complejidad: O(n³) — n-1 pasos, cada uno con búsqueda O(n²)

    Returns:
        Lista de n-1 tuplas (idx_a, idx_b, distancia, tamaño_nuevo_cluster)
    """
    n = len(dist_matrix)
    if n < 2:
        return []

    dist_dict: dict[tuple, float] = {}
    for i in range(n):
        for j in range(n):
            dist_dict[(i, j)] = dist_matrix[i][j]

    active_ids: List[int] = list(range(n))
    cluster_sizes: dict[int, int] = {i: 1 for i in range(n)}
    cluster_members: dict[int, List[int]] = {i: [i] for i in range(n)}
    synthetic_id = n

    linkage_steps: List[Tuple] = []

    for _ in range(n - 1):
        # ── Paso 2a: Par más cercano ──────────────────────────────────────
        min_dist = math.inf
        best_i, best_j = -1, -1

        for ai in range(len(active_ids)):
            for aj in range(ai + 1, len(active_ids)):
                ci = active_ids[ai]
                cj = active_ids[aj]
                d = dist_dict.get((ci, cj), math.inf)
                if d < min_dist:
                    min_dist = d
                    best_i, best_j = ci, cj

        # ── Paso 2b: Fusionar ─────────────────────────────────────────────
        size_i = cluster_sizes[best_i]
        size_j = cluster_sizes[best_j]
        new_size = size_i + size_j
        new_id = synthetic_id

        cluster_members[new_id] = cluster_members[best_i] + cluster_members[best_j]
        cluster_sizes[new_id] = new_size

        linkage_steps.append((best_i, best_j, min_dist, new_size))

        # ── Paso 2c: Actualizar distancias — media ponderada ──────────────
        # d(A∪B, C) = (|A| × d(A,C) + |B| × d(B,C)) / (|A| + |B|)
        for ck in active_ids:
            if ck == best_i or ck == best_j:
                continue
            d_ik = dist_dict.get((best_i, ck), math.inf)
            d_jk = dist_dict.get((best_j, ck), math.inf)
            new_d = (size_i * d_ik + size_j * d_jk) / new_size  # Promedio ponderado
            dist_dict[(new_id, ck)] = new_d
            dist_dict[(ck, new_id)] = new_d

        active_ids.remove(best_i)
        active_ids.remove(best_j)
        active_ids.append(new_id)
        synthetic_id += 1

    return linkage_steps


def ward_linkage(dist_matrix: List[List[float]]) -> List[Tuple]:
    """
    Clustering jerárquico — Método de Ward.

    Ward minimiza el incremento de la inercia intra-cluster al fusionar.
    Se usa la fórmula de Lance-Williams:

        d(A∪B, C)² = ((|A|+|C|) × d(A,C)² + (|B|+|C|) × d(B,C)² - |C| × d(A,B)²)
                     ─────────────────────────────────────────────────────────────────
                                        |A| + |B| + |C|

    Internamente opera con distancias al cuadrado para la actualización;
    reporta la raíz cuadrada como distancia de merge en el dendrograma.

    Nota: Ward fue diseñado originalmente para distancias Euclídeas.
    Aquí se aplica la fórmula de Lance-Williams a distancias coseno,
    lo que es válido como criterio de agrupamiento aunque no minimiza
    exactamente la varianza intra-cluster (requeriría los vectores originales).

    Características:
      + Produce clusters de tamaño equilibrado
      + Muy sensible a la estructura real de los datos
      - Más sensible a outliers que el enlace promedio
      - La interpretación de distancias difiere con métricas no Euclídeas

    Complejidad: O(n³) — n-1 pasos, cada uno con búsqueda O(n²)

    Returns:
        Lista de n-1 tuplas (idx_a, idx_b, distancia, tamaño_nuevo_cluster)
    """
    n = len(dist_matrix)
    if n < 2:
        return []

    # Ward opera internamente con distancias al cuadrado
    dist_dict: dict[tuple, float] = {}
    for i in range(n):
        for j in range(n):
            dist_dict[(i, j)] = dist_matrix[i][j] ** 2

    active_ids: List[int] = list(range(n))
    cluster_sizes: dict[int, int] = {i: 1 for i in range(n)}
    cluster_members: dict[int, List[int]] = {i: [i] for i in range(n)}
    synthetic_id = n

    linkage_steps: List[Tuple] = []

    for _ in range(n - 1):
        # ── Paso 2a: Par más cercano (por distancia² Ward) ────────────────
        min_dist_sq = math.inf
        best_i, best_j = -1, -1

        for ai in range(len(active_ids)):
            for aj in range(ai + 1, len(active_ids)):
                ci = active_ids[ai]
                cj = active_ids[aj]
                d_sq = dist_dict.get((ci, cj), math.inf)
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
                    best_i, best_j = ci, cj

        # ── Paso 2b: Fusionar ─────────────────────────────────────────────
        size_i = cluster_sizes[best_i]
        size_j = cluster_sizes[best_j]
        new_size = size_i + size_j
        new_id = synthetic_id

        cluster_members[new_id] = cluster_members[best_i] + cluster_members[best_j]
        cluster_sizes[new_id] = new_size

        # Distancia de merge para el dendrograma = √(distancia²)
        merge_dist = math.sqrt(max(0.0, min_dist_sq))
        linkage_steps.append((best_i, best_j, merge_dist, new_size))

        # ── Paso 2c: Fórmula Lance-Williams para Ward ─────────────────────
        # d(A∪B, C)² = [(nA+nC)×d(A,C)² + (nB+nC)×d(B,C)² - nC×d(A,B)²]
        #              ─────────────────────────────────────────────────────
        #                                nA + nB + nC
        d_ab_sq = min_dist_sq

        for ck in active_ids:
            if ck == best_i or ck == best_j:
                continue
            size_k = cluster_sizes[ck]
            d_ik_sq = dist_dict.get((best_i, ck), math.inf)
            d_jk_sq = dist_dict.get((best_j, ck), math.inf)

            total = size_i + size_j + size_k
            new_d_sq = (
                (size_i + size_k) * d_ik_sq
                + (size_j + size_k) * d_jk_sq
                - size_k * d_ab_sq
            ) / total

            new_d_sq = max(0.0, new_d_sq)  # Protección contra errores numéricos
            dist_dict[(new_id, ck)] = new_d_sq
            dist_dict[(ck, new_id)] = new_d_sq

        active_ids.remove(best_i)
        active_ids.remove(best_j)
        active_ids.append(new_id)
        synthetic_id += 1

    return linkage_steps


# ── 5. Conversión al formato de scipy ─────────────────────────────────────────


def linkage_to_scipy_format(linkage_steps: List[Tuple]) -> "list[list[float]]":
    """
    Convierte la lista de pasos de linkage al formato de scipy.linkage:

    Formato scipy: array Z de forma (n-1) × 4
      Z[k] = [cluster_a, cluster_b, distancia, tamaño]

    Donde:
      - cluster_a, cluster_b < n → puntos originales
      - cluster_a, cluster_b >= n → clusters sintéticos de pasos anteriores
      - El k-ésimo paso crea el cluster con índice (n + k)

    Nuestro algoritmo genera IDs sintéticos en el mismo orden, por lo que
    la conversión es directa: solo transformar tuplas a listas de floats.
    """
    return [[float(a), float(b), float(d), float(s)] for a, b, d, s in linkage_steps]


def ensure_monotone(linkage_steps: List[Tuple]) -> List[Tuple]:
    """
    Garantiza que las distancias de merge sean no decrecientes.

    scipy.cluster.hierarchy.dendrogram requiere monotonicidad estricta.
    Si por errores numéricos alguna distancia disminuye, la igualamos
    a la distancia máxima hasta ese punto.
    """
    result = []
    max_d = 0.0
    for a, b, d, s in linkage_steps:
        d = max(d, max_d + 1e-10)
        max_d = d
        result.append((a, b, d, s))
    return result


# ── 6. Coeficiente de correlación cofenética ──────────────────────────────────


def compute_cophenetic_correlation(
    dist_matrix: List[List[float]],
    linkage_steps: List[Tuple],
    n: int,
) -> float:
    """
    Calcula el coeficiente de correlación cofenética (Sokal & Rohlf, 1962).

    Definición:
      La distancia cofenética entre dos puntos i y j es la altura (distancia
      de merge) en el dendrograma en la que i y j son fusionados por primera vez.

      El coeficiente es la correlación de Pearson entre:
        - Las distancias originales (dist_matrix)
        - Las distancias cofenéticas (del dendrograma)

    Interpretación:
      Cercano a 1.0 → el dendrograma representa fielmente las distancias
      Cercano a 0.0 → el dendrograma distorsiona mucho las distancias

    Referencias típicas:
      > 0.75 → representación aceptable
      > 0.90 → representación excelente

    Complejidad: O(n² × pasos)

    Args:
        dist_matrix: matriz de distancias originales N×N
        linkage_steps: lista de pasos [(a, b, dist, size), ...]
        n: número de documentos originales

    Returns:
        Correlación de Pearson en [-1, 1]
    """
    if len(linkage_steps) < 1:
        return 0.0

    # Reconstruir distancias cofenéticas
    # coph[i][j] = altura del primer merge que conecta los puntos i y j
    coph: List[List[float]] = [[0.0] * n for _ in range(n)]

    # Membresía de cada cluster (ID → set de índices originales)
    members: dict[int, set] = {i: {i} for i in range(n)}

    for step_idx, (a, b, merge_dist, _) in enumerate(linkage_steps):
        new_id = n + step_idx
        # Todos los pares (orig_i de A, orig_j de B) tienen distancia cofenética = merge_dist
        for orig_i in members[a]:
            for orig_j in members[b]:
                coph[orig_i][orig_j] = merge_dist
                coph[orig_j][orig_i] = merge_dist
        members[new_id] = members[a] | members[b]

    # Extraer vectores 1D (triángulo superior, i < j)
    orig_flat: List[float] = []
    coph_flat: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            orig_flat.append(dist_matrix[i][j])
            coph_flat.append(coph[i][j])

    if len(orig_flat) < 2:
        return 1.0

    # Correlación de Pearson
    m = len(orig_flat)
    mean_o = sum(orig_flat) / m
    mean_c = sum(coph_flat) / m

    cov = sum((o - mean_o) * (c - mean_c) for o, c in zip(orig_flat, coph_flat))
    var_o = sum((o - mean_o) ** 2 for o in orig_flat)
    var_c = sum((c - mean_c) ** 2 for c in coph_flat)

    if var_o == 0.0 or var_c == 0.0:
        return 0.0

    return cov / math.sqrt(var_o * var_c)


# ── 7. Pipeline completo ──────────────────────────────────────────────────────

# Registro de los 3 algoritmos de clustering
CLUSTERING_ALGORITHMS = {
    "complete": {
        "name": "Enlace Completo",
        "name_en": "Complete Linkage",
        "formula": r"d(A \cup B,\, C) = \max(d(A, C),\; d(B, C))",
        "func": complete_linkage,
        "color": "#E74C3C",
        "icon": "🔴",
        "description": (
            "Usa la distancia MÁXIMA entre cualquier par de puntos "
            "de los dos clusters. Produce clusters compactos."
        ),
    },
    "average": {
        "name": "Enlace Promedio",
        "name_en": "Average Linkage (UPGMA)",
        "formula": r"d(A \cup B,\, C) = \frac{|A| \cdot d(A,C) + |B| \cdot d(B,C)}{|A| + |B|}",
        "func": average_linkage,
        "color": "#27AE60",
        "icon": "🟢",
        "description": (
            "Usa la distancia PROMEDIO ponderada entre clusters. "
            "Balanceado y menos sensible a outliers."
        ),
    },
    "ward": {
        "name": "Método de Ward",
        "name_en": "Ward's Method",
        "formula": r"d(A \cup B,\, C)^2 = \frac{(n_A+n_C) d(A,C)^2 + (n_B+n_C) d(B,C)^2 - n_C\, d(A,B)^2}{n_A+n_B+n_C}",
        "func": ward_linkage,
        "color": "#2980B9",
        "icon": "🔵",
        "description": (
            "Minimiza el incremento de varianza intra-cluster. "
            "Produce clusters de tamaño equilibrado."
        ),
    },
}


def run_clustering_pipeline(
    abstracts: List[str],
    titles: List[str],
    remove_stopwords: bool = True,
    apply_stemming: bool = True,
) -> dict:
    """
    Ejecuta el pipeline completo de clustering jerárquico.

    Etapas:
      1. Preprocesamiento de cada abstract
      2. Vectorización TF-IDF
      3. Cálculo de matriz de distancias coseno
      4. Aplicación de los 3 algoritmos de clustering
      5. Cálculo de coeficientes cofenéticos

    Args:
        abstracts: lista de textos de abstractos (strings crudos)
        titles: lista de títulos (para etiquetas en el dendrograma)
        remove_stopwords: activar eliminación de stop words
        apply_stemming: activar stemming

    Returns:
        Diccionario con todos los resultados del pipeline
    """
    n = len(abstracts)
    assert n >= 2, "Se necesitan al menos 2 documentos para clustering"
    assert len(titles) == n, "titles y abstracts deben tener el mismo largo"

    # ── Paso 1: Preprocesamiento ──────────────────────────────────────────
    processed_tokens: List[List[str]] = [
        preprocess_text(a, remove_stopwords=remove_stopwords, apply_stemming=apply_stemming)
        for a in abstracts
    ]

    # ── Paso 2: TF-IDF ───────────────────────────────────────────────────
    tfidf_matrix, vocabulary = compute_tfidf_vectors(processed_tokens)

    # ── Paso 3: Distancias coseno ─────────────────────────────────────────
    dist_matrix = compute_distance_matrix(tfidf_matrix)

    # ── Paso 4: Clustering × 3 algoritmos ────────────────────────────────
    linkage_results: dict[str, List[Tuple]] = {}
    coph_scores: dict[str, float] = {}

    for key, algo in CLUSTERING_ALGORITHMS.items():
        steps = algo["func"](dist_matrix)
        steps = ensure_monotone(steps)   # Garantizar monotonicidad para scipy
        linkage_results[key] = steps
        coph_scores[key] = compute_cophenetic_correlation(dist_matrix, steps, n)

    # Determinar el mejor algoritmo según coeficiente cofenético
    best_algo = max(coph_scores, key=lambda k: coph_scores[k])

    return {
        "n": n,
        "abstracts": abstracts,
        "titles": titles,
        "processed_tokens": processed_tokens,
        "vocabulary": vocabulary,
        "vocabulary_size": len(vocabulary),
        "tfidf_matrix": tfidf_matrix,
        "dist_matrix": dist_matrix,
        "linkage": linkage_results,        # {key: [(a, b, d, size), ...]}
        "cophenetic": coph_scores,          # {key: float}
        "best_algorithm": best_algo,
    }
