"""
Parser para archivos CSV exportados desde EBSCO
(Business Source Ultimate, Academic Search, EconLit, etc.).

Cadena de búsqueda: "generative artificial intelligence"

Columnas típicas del CSV de EBSCO:
  longDBName, shortDBName, an, title, abstract, publicationDate,
  contributors, docTypes, pubTypes, coverDate, peerReviewed,
  source, subjects, issns, publisherLocations, doi, plink,
  authorLocations, language, publisher, citedByCount, ...

Estrategia para la columna `country`:
  1. Si existe `authorLocations` (exportaciones antiguas de EBSCO), se usa directamente.
  2. Si no existe, se busca el país en la columna `subjects`: EBSCO incluye
     nombres de países como términos de materia (ej: "Pakistan ; China ; ...").
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ── Nombres canónicos de países reconocidos en subjects de EBSCO ─────────────
# Conjunto de nombres en inglés tal como aparecen en los subject terms de EBSCO
_KNOWN_COUNTRIES: set[str] = {
    "Afghanistan", "Albania", "Algeria", "Argentina", "Armenia", "Australia",
    "Austria", "Azerbaijan", "Bangladesh", "Belgium", "Bolivia", "Bosnia",
    "Brazil", "Bulgaria", "Cambodia", "Canada", "Chile", "China", "Colombia",
    "Croatia", "Cuba", "Czech Republic", "Czechia", "Denmark", "Ecuador",
    "Egypt", "Estonia", "Ethiopia", "Finland", "France", "Georgia", "Germany",
    "Ghana", "Greece", "Guatemala", "Hungary", "India", "Indonesia", "Iran",
    "Iraq", "Ireland", "Israel", "Italy", "Japan", "Jordan", "Kazakhstan",
    "Kenya", "Kosovo", "Latvia", "Lebanon", "Lithuania", "Malaysia", "Mexico",
    "Morocco", "Netherlands", "New Zealand", "Nigeria", "North Korea", "Norway",
    "Pakistan", "Palestine", "Panama", "Peru", "Philippines", "Poland",
    "Portugal", "Romania", "Russia", "Saudi Arabia", "Senegal", "Serbia",
    "Singapore", "Slovakia", "Slovenia", "South Africa", "South Korea", "Spain",
    "Sri Lanka", "Sweden", "Switzerland", "Taiwan", "Thailand", "Tunisia",
    "Turkey", "Türkiye", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "Uruguay", "Venezuela", "Vietnam",
    "Zimbabwe",
}

# Alias comunes que EBSCO también usa en subjects
_SUBJECT_ALIASES: dict[str, str] = {
    "USA": "United States",
    "US": "United States",
    "UK": "United Kingdom",
    "England": "United Kingdom",
    "Great Britain": "United Kingdom",
    "Britain": "United Kingdom",
    "People's Republic of China": "China",
    "PR China": "China",
    "PRC": "China",
    "Korea": "South Korea",
    "Republic of Korea": "South Korea",
    "Russian Federation": "Russia",
    "Türkiye": "Turkey",
    "Brasil": "Brazil",
    "Deutschland": "Germany",
    "España": "Spain",
    "México": "Mexico",
    "UAE": "United Arab Emirates",
}

# Adjetivos nacionales → nombre canónico del país (para búsqueda en abstracts)
_NATIONAL_ADJECTIVES: dict[str, str] = {
    "japanese": "Japan", "chinese": "China", "american": "United States",
    "british": "United Kingdom", "german": "Germany", "french": "France",
    "italian": "Italy", "spanish": "Spain", "brazilian": "Brazil",
    "indian": "India", "korean": "South Korea", "australian": "Australia",
    "canadian": "Canada", "dutch": "Netherlands", "swedish": "Sweden",
    "norwegian": "Norway", "danish": "Denmark", "finnish": "Finland",
    "portuguese": "Portugal", "greek": "Greece", "polish": "Poland",
    "russian": "Russia", "turkish": "Turkey", "iranian": "Iran",
    "egyptian": "Egypt", "nigerian": "Nigeria", "kenyan": "Kenya",
    "singaporean": "Singapore", "taiwanese": "Taiwan", "thai": "Thailand",
    "vietnamese": "Vietnam", "indonesian": "Indonesia", "malaysian": "Malaysia",
    "pakistani": "Pakistan", "bangladeshi": "Bangladesh", "saudi": "Saudi Arabia",
    "israeli": "Israel", "ukrainian": "Ukraine", "romanian": "Romania",
    "hungarian": "Hungary", "czech": "Czech Republic", "slovak": "Slovakia",
    "croatian": "Croatia", "serbian": "Serbia", "colombian": "Colombia",
    "mexican": "Mexico", "argentinian": "Argentina", "chilean": "Chile",
    "peruvian": "Peru", "venezuelan": "Venezuela", "cuban": "Cuba",
    "philippine": "Philippines", "filipino": "Philippines",
}

# ── Mapeo columnas EBSCO → esquema canónico del proyecto ────────────────────
EBSCO_COLUMN_MAP: dict[str, str] = {
    # Campos principales
    "title":              "title",
    "abstract":           "abstract",
    "source":             "source",          # nombre de la revista/libro
    "doi":                "doi",
    "contributors":       "authors",
    "docTypes":           "document_type",
    "pubTypes":           "pub_type",        # alias según versión de exportación
    "plink":              "url",
    "issns":              "issn",
    "isbns":              "isbn",            # libros usan isbn, no issn
    "subjects":           "keywords",
    # Geografía — columna central de este parser
    "authorLocations":    "country",
    # Sinónimos observados en distintas configuraciones de EBSCO
    "Author Locations":   "country",
    "authorAffiliations": "country",
    "affiliations":       "country",
    # Campos complementarios (no estándar, se incluyen si existen)
    "language":           "language",
    "publisher":          "publisher",
    "citedByCount":       "cited_by",
    "isOpenAccess":       "open_access",
    "peerReviewed":       "peer_reviewed",
    "coverDate":          "cover_date",
    "publicationDate":    "pub_date_raw",    # se procesa aparte para extraer año
    "volume":             "volume",
    "issue":              "issue",
    "pageStart":          "page_start",
    "pageEnd":            "page_end",
}

# Columnas obligatorias del esquema canónico
_CANONICAL_CORE = [
    "title", "authors", "year", "source", "doi",
    "abstract", "document_type", "url", "issn", "keywords",
    "country",
]


# ── Función pública ──────────────────────────────────────────────────────────

def parse_ebsco_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Lee un archivo CSV de EBSCO y retorna un DataFrame normalizado
    con el esquema canónico del proyecto.

    La columna ``country`` queda poblada con el valor de ``authorLocations``
    (o su sinónimo), que contiene las ubicaciones institucionales de los
    autores separadas por ``";"`` en el formato de EBSCO.

    Parameters
    ----------
    filepath : ruta al archivo .csv exportado desde EBSCO, o un objeto
               file-like (p. ej. ``UploadedFile`` de Streamlit).

    Returns
    -------
    pd.DataFrame con columnas del esquema canónico y columna
    adicional ``source_db`` = ``'EBSCO'``.
    """
    # Aceptar rutas (str/Path) y objetos file-like (Streamlit UploadedFile)
    if isinstance(filepath, (str, Path)):
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Archivo EBSCO no encontrado: {path}")
        source = path
    else:
        source = filepath

    # EBSCO exporta UTF-8; el BOM es frecuente en descargas de Windows
    df = pd.read_csv(source, encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip()

    # Renombrar columnas conocidas
    rename_map = {
        col: EBSCO_COLUMN_MAP[col]
        for col in df.columns
        if col in EBSCO_COLUMN_MAP
    }
    df = df.rename(columns=rename_map)

    # Eliminar columnas duplicadas (cuando dos columnas EBSCO mapean al mismo nombre)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # ── Extraer año ──────────────────────────────────────────────────────────
    if "year" not in df.columns:
        # Primero intentar pub_date_raw (publicationDate: "20260326")
        if "pub_date_raw" in df.columns:
            df["year"] = df["pub_date_raw"].apply(_extract_year)
        # Fallback: cover_date (puede ser "Mar2026" o "2025")
        elif "cover_date" in df.columns:
            df["year"] = df["cover_date"].apply(_extract_year)
        else:
            df["year"] = ""

    # ── Extraer país desde subjects / abstract si authorLocations no existe ─────
    # Estrategia por prioridad:
    #   1. subjects: país como term exacto (más fiable — ej "Pakistan", "China")
    #   2. abstract: adjetivo nacional o nombre de país en el texto (fallback)
    if "country" not in df.columns or df["country"].str.strip().eq("").all():
        kw_col = "keywords" if "keywords" in df.columns else (
                 "subjects" if "subjects" in df.columns else None)
        abs_col = "abstract" if "abstract" in df.columns else None

        def _resolve_country(row):
            # 1. Buscar en subjects
            subj = row.get(kw_col, "") if kw_col else ""
            c = _extract_country_from_subjects(subj)
            if c:
                return c
            # 2. Fallback: buscar en abstract
            if abs_col:
                return _extract_country_from_text(row.get(abs_col, ""))
            return ""

        df["country"] = df.apply(_resolve_country, axis=1)

    # ── Asegurar esquema canónico completo ────────────────────────────────────
    for col in _CANONICAL_CORE:
        if col not in df.columns:
            df[col] = ""

    df = df[_CANONICAL_CORE].copy()
    df["source_db"] = "EBSCO"

    # ── Limpieza básica ──────────────────────────────────────────────────────
    df = df.fillna("").apply(
        lambda col: col.map(
            lambda v: "" if str(v).strip().lower() in ("nan", "none") else str(v).strip()
        )
    )

    # Descartar filas sin título
    df = df[df["title"].str.strip() != ""].reset_index(drop=True)

    return df


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_country_from_subjects(subjects_str: str) -> str:
    """
    Extrae el primer país reconocido de la cadena de subjects de EBSCO.

    EBSCO incluye países como términos de materia dentro de la columna
    `subjects`, separados por ` ; `.  Ejemplos:
      "Generative artificial intelligence ; Pakistan ; Emergency management"
      "Generative artificial intelligence ; China ; Machine learning"

    Estrategia:
      1. Dividir por " ; " y limpiar cada término.
      2. Comparar contra el conjunto _KNOWN_COUNTRIES (case-insensitive).
      3. Si no hay coincidencia exacta, probar _SUBJECT_ALIASES.

    Returns el nombre canónico del país o cadena vacía si no se encuentra.
    """
    if not subjects_str or str(subjects_str).strip().lower() in ("", "nan", "none"):
        return ""

    terms = [t.strip() for t in str(subjects_str).split(";") if t.strip()]
    for term in terms:
        # Coincidencia exacta (case-insensitive) con país conocido
        for country in _KNOWN_COUNTRIES:
            if term.lower() == country.lower():
                return country
        # Coincidencia con alias
        for alias, canonical in _SUBJECT_ALIASES.items():
            if term.lower() == alias.lower():
                return canonical

    return ""


def _extract_country_from_text(text: str) -> str:
    """
    Busca el primer país reconocido en un texto libre (abstract o título).

    Estrategia:
      1. Buscar adjetivos nacionales (Japanese → Japan, Chinese → China, etc.)
      2. Buscar nombres de países directamente (_KNOWN_COUNTRIES)

    Prioriza adjetivos porque aparecen naturalmente en frases como
    "Japanese agricultural markets" o "Chinese manufacturing firms".

    Returns el nombre canónico del país o cadena vacía.
    """
    if not text or str(text).strip().lower() in ("", "nan", "none"):
        return ""

    words = re.findall(r"\b[A-Za-z]+\b", str(text))
    for word in words:
        canon = _NATIONAL_ADJECTIVES.get(word.lower())
        if canon:
            return canon
    # Segunda pasada: nombres exactos de países
    for word in words:
        if word in _KNOWN_COUNTRIES:
            return word
    return ""


def _extract_year(date_str: str) -> str:
    """
    Extrae el año de un campo de fecha en distintos formatos de EBSCO.

    Formatos soportados:
      - ``"20260326"``  → ``"2026"``   (YYYYMMDD numérico)
      - ``"2026"``      → ``"2026"``   (sólo año)
      - ``"Mar2026"``   → ``"2026"``   (abreviatura mes + año)
      - ``"2025-09-15"``→ ``"2025"``   (ISO 8601)
    """
    s = str(date_str).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return ""
    # Busca cuatro dígitos consecutivos que empiecen por 19xx o 20xx
    m = re.search(r"\b((?:19|20)\d{2})\b", s)
    if m:
        return m.group(1)
    # Fallback: cualquier secuencia de 4 dígitos
    m = re.search(r"(\d{4})", s)
    return m.group(1) if m else ""
