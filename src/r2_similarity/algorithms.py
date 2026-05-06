"""
Requerimiento 2 — Algoritmos de Similitud Textual
===================================================
Implementa 6 algoritmos de similitud:
  Clásicos:
    1. Distancia de Levenshtein (distancia de edición)
    2. Similitud de Jaccard (conjuntos de tokens)
    3. Coseno con TF-IDF (vectorización estadística)
    4. Distancia de Hamming normalizada (distancia de edición)
  Modelos de IA:
    5. Sentence-BERT (sentence-transformers)
    6. spaCy Word Vectors
"""

from __future__ import annotations

import math
import re
from collections import Counter

# ── 1. Distancia de Levenshtein ──────────────────────────────────────────────


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calcula la similitud entre dos textos usando distancia de Levenshtein.

    La distancia de Levenshtein mide el número mínimo de operaciones
    (inserción, eliminación, sustitución) para transformar s1 en s2.

    Similitud = 1 - (distancia / max(len(s1), len(s2)))

    Complejidad temporal: O(n * m)
    Complejidad espacial: O(n * m)
    """
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Matriz de programación dinámica (len1+1) x (len2+1)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Caso base: transformar cadena vacía
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Llenar la matriz
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # eliminación
                dp[i][j - 1] + 1,        # inserción
                dp[i - 1][j - 1] + cost  # sustitución
            )

    distance = dp[len1][len2]
    return 1.0 - (distance / max(len1, len2))


def levenshtein_step_by_step(s1: str, s2: str) -> dict:
    """
    Retorna la traza completa del algoritmo de Levenshtein:
    la matriz DP, la distancia final y la similitud.
    """
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    distance = dp[len1][len2]
    max_len = max(len1, len2) if max(len1, len2) > 0 else 1
    similarity = 1.0 - (distance / max_len)

    return {
        "matrix": dp,
        "distance": distance,
        "similarity": round(similarity, 4),
        "s1": s1,
        "s2": s2,
    }


# ── 2. Similitud de Jaccard ─────────────────────────────────────────────────


def _tokenize(text: str) -> set[str]:
    """Tokeniza texto en conjunto de palabras en minúscula."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def jaccard_similarity(s1: str, s2: str) -> float:
    """
    Calcula la similitud de Jaccard entre dos textos.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Donde A y B son conjuntos de tokens (palabras) de cada texto.

    Complejidad temporal: O(n + m) donde n, m son tokens
    Complejidad espacial: O(n + m)
    """
    set1 = _tokenize(s1)
    set2 = _tokenize(s2)

    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = set1 & set2
    union = set1 | set2

    return len(intersection) / len(union)


def jaccard_step_by_step(s1: str, s2: str) -> dict:
    """Retorna la traza completa del algoritmo de Jaccard."""
    set1 = _tokenize(s1)
    set2 = _tokenize(s2)
    intersection = set1 & set2
    union = set1 | set2

    sim = len(intersection) / len(union) if union else 1.0

    return {
        "tokens_a": sorted(set1),
        "tokens_b": sorted(set2),
        "intersection": sorted(intersection),
        "union": sorted(union),
        "intersection_size": len(intersection),
        "union_size": len(union),
        "similarity": round(sim, 4),
    }


# ── 3. Coseno con TF-IDF ────────────────────────────────────────────────────


def _compute_tf(tokens: list[str]) -> dict[str, float]:
    """Calcula Term Frequency para una lista de tokens."""
    count = Counter(tokens)
    total = len(tokens)
    return {word: freq / total for word, freq in count.items()}


def _compute_idf(doc_tokens_list: list[list[str]]) -> dict[str, float]:
    """Calcula Inverse Document Frequency."""
    n_docs = len(doc_tokens_list)
    all_words: set[str] = set()
    for tokens in doc_tokens_list:
        all_words.update(tokens)

    idf = {}
    for word in all_words:
        # Número de documentos que contienen la palabra
        df = sum(1 for tokens in doc_tokens_list if word in tokens)
        idf[word] = math.log((n_docs + 1) / (df + 1)) + 1  # suavizado

    return idf


def cosine_tfidf_similarity(s1: str, s2: str) -> float:
    """
    Calcula la similitud del coseno usando representación TF-IDF.

    Proceso:
      1. Tokenizar ambos textos
      2. Calcular TF (Term Frequency) por documento
      3. Calcular IDF (Inverse Document Frequency) sobre ambos
      4. Construir vectores TF-IDF
      5. Calcular similitud del coseno: cos(θ) = (A·B) / (||A|| × ||B||)

    Complejidad temporal: O(n + m) después de tokenizar
    Complejidad espacial: O(V) donde V es el vocabulario
    """
    tokens1 = re.findall(r'\b\w+\b', s1.lower())
    tokens2 = re.findall(r'\b\w+\b', s2.lower())

    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    # TF por documento
    tf1 = _compute_tf(tokens1)
    tf2 = _compute_tf(tokens2)

    # IDF sobre los dos documentos
    idf = _compute_idf([tokens1, tokens2])

    # Vocabulario conjunto
    vocab = sorted(set(tokens1) | set(tokens2))

    # Vectores TF-IDF
    vec1 = [tf1.get(w, 0.0) * idf.get(w, 0.0) for w in vocab]
    vec2 = [tf2.get(w, 0.0) * idf.get(w, 0.0) for w in vocab]

    # Similitud del coseno
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def cosine_tfidf_step_by_step(s1: str, s2: str) -> dict:
    """Retorna la traza completa del algoritmo Coseno TF-IDF."""
    tokens1 = re.findall(r'\b\w+\b', s1.lower())
    tokens2 = re.findall(r'\b\w+\b', s2.lower())

    tf1 = _compute_tf(tokens1)
    tf2 = _compute_tf(tokens2)
    idf = _compute_idf([tokens1, tokens2])

    vocab = sorted(set(tokens1) | set(tokens2))

    vec1 = [round(tf1.get(w, 0.0) * idf.get(w, 0.0), 4) for w in vocab]
    vec2 = [round(tf2.get(w, 0.0) * idf.get(w, 0.0), 4) for w in vocab]

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))

    similarity = dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

    return {
        "tokens_a": tokens1,
        "tokens_b": tokens2,
        "tf_a": {k: round(v, 4) for k, v in tf1.items()},
        "tf_b": {k: round(v, 4) for k, v in tf2.items()},
        "idf": {k: round(v, 4) for k, v in sorted(idf.items())},
        "vocabulary": vocab,
        "tfidf_vector_a": vec1,
        "tfidf_vector_b": vec2,
        "dot_product": round(dot_product, 4),
        "norm_a": round(norm1, 4),
        "norm_b": round(norm2, 4),
        "similarity": round(similarity, 4),
    }


# ── 4. Distancia de Hamming normalizada ─────────────────────────────────────


def hamming_similarity(s1: str, s2: str) -> float:
    """
    Calcula la similitud de Hamming entre dos textos.

    Distancia de Hamming: número de posiciones donde los caracteres difieren.
    Para cadenas de distinta longitud, se rellena la más corta con espacios.

    Similitud = 1 - (distancia / max_longitud)

    Complejidad temporal: O(max(n, m))
    Complejidad espacial: O(max(n, m))
    """
    s1_lower = s1.lower()
    s2_lower = s2.lower()

    if s1_lower == s2_lower:
        return 1.0

    max_len = max(len(s1_lower), len(s2_lower))
    if max_len == 0:
        return 1.0

    # Padding con espacios para igualar longitudes
    s1_padded = s1_lower.ljust(max_len)
    s2_padded = s2_lower.ljust(max_len)

    # Contar diferencias
    distance = sum(c1 != c2 for c1, c2 in zip(s1_padded, s2_padded))

    return 1.0 - (distance / max_len)


def hamming_step_by_step(s1: str, s2: str) -> dict:
    """Retorna la traza completa del algoritmo de Hamming."""
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    max_len = max(len(s1_lower), len(s2_lower))

    s1_padded = s1_lower.ljust(max_len) if max_len > 0 else ""
    s2_padded = s2_lower.ljust(max_len) if max_len > 0 else ""

    comparisons = []
    distance = 0
    for i, (c1, c2) in enumerate(zip(s1_padded, s2_padded)):
        match = c1 == c2
        if not match:
            distance += 1
        comparisons.append({"pos": i, "char_a": c1, "char_b": c2, "match": match})

    similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0

    return {
        "s1_padded": s1_padded,
        "s2_padded": s2_padded,
        "original_len_a": len(s1_lower),
        "original_len_b": len(s2_lower),
        "padded_len": max_len,
        "comparisons_sample": comparisons[:50],  # primeras 50 para UI
        "distance": distance,
        "total_positions": max_len,
        "similarity": round(similarity, 4),
    }


# ── 5. Sentence-BERT (IA) ───────────────────────────────────────────────────

_sbert_model = None


def _get_sbert_model():
    """Carga el modelo Sentence-BERT de forma lazy (singleton)."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def sbert_similarity(s1: str, s2: str) -> float:
    """
    Calcula la similitud semántica usando Sentence-BERT.

    Proceso:
      1. Codifica cada texto como un vector denso de 384 dimensiones
         usando el modelo 'all-MiniLM-L6-v2' (transformers)
      2. Calcula la similitud del coseno entre ambos embeddings

    Este modelo captura significado semántico profundo:
    textos con el mismo significado pero distintas palabras
    tendrán alta similitud.

    Complejidad: O(n) para la inferencia del modelo (dominada por el tamaño del texto)
    """
    if not s1.strip() and not s2.strip():
        return 1.0
    if not s1.strip() or not s2.strip():
        return 0.0

    model = _get_sbert_model()
    embeddings = model.encode([s1, s2], convert_to_numpy=True)

    # Similitud del coseno
    from numpy import dot
    from numpy.linalg import norm
    cos_sim = float(dot(embeddings[0], embeddings[1]) /
                     (norm(embeddings[0]) * norm(embeddings[1])))

    # Normalizar a [0, 1] (coseno puede ser negativo)
    return max(0.0, cos_sim)


def sbert_step_by_step(s1: str, s2: str) -> dict:
    """Retorna información del proceso Sentence-BERT."""
    if not s1.strip() or not s2.strip():
        return {"similarity": 0.0, "error": "Texto vacío"}

    model = _get_sbert_model()
    embeddings = model.encode([s1, s2], convert_to_numpy=True)

    from numpy import dot
    from numpy.linalg import norm

    cos_sim = float(dot(embeddings[0], embeddings[1]) /
                     (norm(embeddings[0]) * norm(embeddings[1])))
    similarity = max(0.0, cos_sim)

    return {
        "model_name": "all-MiniLM-L6-v2",
        "embedding_dim": embeddings[0].shape[0],
        "embedding_a_sample": embeddings[0][:10].tolist(),
        "embedding_b_sample": embeddings[1][:10].tolist(),
        "cosine_raw": round(cos_sim, 6),
        "similarity": round(similarity, 4),
    }


# ── 6. spaCy Word Vectors (IA) ──────────────────────────────────────────────

_spacy_nlp = None


def _get_spacy_model():
    """Carga el modelo spaCy de forma lazy (singleton)."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_md")
        except OSError:
            # Si no está instalado, descargarlo
            import subprocess, sys
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "en_core_web_md"
            ])
            _spacy_nlp = spacy.load("en_core_web_md")
    return _spacy_nlp


def spacy_similarity(s1: str, s2: str) -> float:
    """
    Calcula la similitud semántica usando vectores de palabras de spaCy.

    Proceso:
      1. Procesa cada texto con el modelo en_core_web_md (300 dimensiones, GloVe)
      2. spaCy calcula el vector promedio de todas las palabras del texto
      3. Calcula similitud del coseno entre los vectores promedio

    A diferencia de SBERT, spaCy usa vectores de palabras estáticos (no contextuales),
    lo que lo hace más rápido pero menos preciso para frases complejas.

    Complejidad: O(n) donde n es el número de tokens
    """
    if not s1.strip() and not s2.strip():
        return 1.0
    if not s1.strip() or not s2.strip():
        return 0.0

    nlp = _get_spacy_model()
    doc1 = nlp(s1)
    doc2 = nlp(s2)

    sim = doc1.similarity(doc2)
    return max(0.0, float(sim))


def spacy_step_by_step(s1: str, s2: str) -> dict:
    """Retorna información del proceso spaCy."""
    if not s1.strip() or not s2.strip():
        return {"similarity": 0.0, "error": "Texto vacío"}

    nlp = _get_spacy_model()
    doc1 = nlp(s1)
    doc2 = nlp(s2)

    sim = doc1.similarity(doc2)

    return {
        "model_name": "en_core_web_md",
        "vector_dim": doc1.vector.shape[0] if doc1.has_vector else 0,
        "tokens_a": [t.text for t in doc1],
        "tokens_b": [t.text for t in doc2],
        "tokens_with_vectors_a": [t.text for t in doc1 if t.has_vector],
        "tokens_with_vectors_b": [t.text for t in doc2 if t.has_vector],
        "vector_a_sample": doc1.vector[:10].tolist() if doc1.has_vector else [],
        "vector_b_sample": doc2.vector[:10].tolist() if doc2.has_vector else [],
        "similarity": round(max(0.0, float(sim)), 4),
    }


# ── Registro de algoritmos ──────────────────────────────────────────────────

ALGORITHMS = {
    "levenshtein": {
        "name": "Distancia de Levenshtein",
        "type": "Clásico — Distancia de edición",
        "func": levenshtein_similarity,
        "step_func": levenshtein_step_by_step,
    },
    "jaccard": {
        "name": "Similitud de Jaccard",
        "type": "Clásico — Conjuntos de tokens",
        "func": jaccard_similarity,
        "step_func": jaccard_step_by_step,
    },
    "cosine_tfidf": {
        "name": "Coseno TF-IDF",
        "type": "Clásico — Vectorización estadística",
        "func": cosine_tfidf_similarity,
        "step_func": cosine_tfidf_step_by_step,
    },
    "hamming": {
        "name": "Distancia de Hamming",
        "type": "Clásico — Distancia de edición",
        "func": hamming_similarity,
        "step_func": hamming_step_by_step,
    },
    "sbert": {
        "name": "Sentence-BERT",
        "type": "IA — Transformers (all-MiniLM-L6-v2)",
        "func": sbert_similarity,
        "step_func": sbert_step_by_step,
    },
    "spacy": {
        "name": "spaCy Word Vectors",
        "type": "IA — GloVe (en_core_web_md)",
        "func": spacy_similarity,
        "step_func": spacy_step_by_step,
    },
}
