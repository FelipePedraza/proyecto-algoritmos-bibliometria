"""
Utilidades de texto compartidas para todo el proyecto bibliométrico.
Normalización, limpieza y comparación de cadenas.
"""

import re
import unicodedata


def normalize_title(title: str) -> str:
    """
    Normaliza un título para comparación:
    - Convierte a minúsculas
    - Elimina acentos y caracteres especiales
    - Colapsa espacios múltiples
    - Elimina puntuación no alfanumérica
    """
    if not isinstance(title, str):
        return ""
    # Normalizar unicode → ASCII
    nfkd = unicodedata.normalize("NFKD", title)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    # Minúsculas
    lower = ascii_str.lower()
    # Eliminar todo lo que no sea letra, número o espacio
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lower)
    # Colapsar espacios
    return re.sub(r"\s+", " ", cleaned).strip()


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calcula similitud entre dos cadenas usando distancia de Levenshtein.
    Retorna un valor entre 0.0 (completamente distintos) y 1.0 (idénticos).
    """
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Matriz de programación dinámica
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,       # eliminación
                matrix[i][j - 1] + 1,       # inserción
                matrix[i - 1][j - 1] + cost # sustitución
            )

    distance = matrix[len1][len2]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len)


def titles_are_duplicate(title1: str, title2: str, threshold: float = 0.92) -> bool:
    """
    Determina si dos títulos son duplicados usando similitud de Levenshtein
    sobre títulos normalizados.
    """
    n1 = normalize_title(title1)
    n2 = normalize_title(title2)
    if not n1 or not n2:
        return False
    return levenshtein_similarity(n1, n2) >= threshold