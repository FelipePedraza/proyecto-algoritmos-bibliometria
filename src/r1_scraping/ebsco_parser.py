"""
Parser para archivos CSV exportados desde EBSCO
(Business Source Ultimate, Academic Search, EconLit, etc.).

Cadena de búsqueda: "generative artificial intelligence"

Columnas típicas del CSV de EBSCO:
  longDBName, shortDBName, an, title, abstract, publicationDate,
  contributors, docTypes, pubTypes, coverDate, peerReviewed,
  source, subjects, issns, publisherLocations, doi, plink,
  authorLocations, language, publisher, citedByCount, ...

La columna clave para geografía es `authorLocations`, que contiene
las ubicaciones institucionales de los autores separadas por ";".
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ── Mapeo columnas EBSCO → esquema canónico del proyecto ────────────────────
EBSCO_COLUMN_MAP: dict[str, str] = {
    # Campos principales
    "title":              "title",
    "abstract":           "abstract",
    "source":             "source",          # nombre de la revista/libro
    "doi":                "doi",
    "contributors":       "authors",
    "docTypes":           "document_type",
    "pubTypes":           "document_type",   # alias según versión de exportación
    "plink":              "url",
    "issns":              "issn",
    "isbns":              "issn",            # fallback para libros
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
