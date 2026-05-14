"""
Parser para archivos CSV exportados desde ScienceDirect (Elsevier).
Cadena de búsqueda: "generative artificial intelligence"

Columnas tipicas de ScienceDirect CSV:
  Title, Authors, Year, Source title, DOI, Abstract,
  Document Type, Link, ISSN, Author Keywords, Index Keywords
"""

import pandas as pd
from pathlib import Path

# Mapeo de columnas ScienceDirect -> esquema canonico del proyecto
SD_COLUMN_MAP = {
    "Title":            "title",
    "Authors":          "authors",
    "Year":             "year",
    "Source title":     "source",
    "DOI":              "doi",
    "Abstract":         "abstract",
    "Document Type":    "document_type",
    "Link":             "url",
    "ISSN":             "issn",
    "Author Keywords":  "keywords",
    "Index Keywords":   "index_keywords",
    # Sinonimos posibles en distintas versiones de exportacion
    "Author(s)":        "authors",
    "Publication Year": "year",
    "Source":           "source",
}


def parse_sciencedirect_csv(filepath):
    """
    Lee un archivo CSV de ScienceDirect y retorna un DataFrame normalizado
    con el esquema canonico del proyecto.

    Parameters
    ----------
    filepath : ruta al archivo .csv exportado desde ScienceDirect.

    Returns
    -------
    pd.DataFrame con columnas del esquema canonico y columna
    adicional 'source_db' = 'ScienceDirect'.
    """
    if isinstance(filepath, (str, Path)):
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Archivo ScienceDirect no encontrado: {path}")
        source = path
    else:
        source = filepath

    df = pd.read_csv(source, encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip()

    rename_map = {col: SD_COLUMN_MAP[col]
                  for col in df.columns if col in SD_COLUMN_MAP}
    df = df.rename(columns=rename_map)

    if "keywords" in df.columns and "index_keywords" in df.columns:
        df["keywords"] = df.apply(
            lambda row: _merge_keywords(row["keywords"], row["index_keywords"]),
            axis=1
        )
        df.drop(columns=["index_keywords"], inplace=True)
    elif "index_keywords" in df.columns:
        df.rename(columns={"index_keywords": "keywords"}, inplace=True)

    # Asegurar esquema canonico completo (incluye 'country')
    canonical_cols = ["title", "authors", "year", "source", "doi",
                      "abstract", "document_type", "url", "issn", "keywords",
                      "country"]
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[canonical_cols].copy()
    df["source_db"] = "ScienceDirect"

    df = df.fillna("").apply(
        lambda col: col.map(lambda v: "" if str(v).strip().lower() == "nan" else str(v).strip())
    )

    df = df[df["title"].str.strip() != ""].reset_index(drop=True)

    return df


def _merge_keywords(kw1, kw2):
    """Une dos listas de keywords separadas por ; eliminando duplicados."""
    set1 = set()
    for k in str(kw1).split(";"):
        if k.strip():
            set1.add(k.strip().lower())
    set2 = set()
    for k in str(kw2).split(";"):
        if k.strip():
            set2.add(k.strip().lower())
    merged = sorted(set1 | set2)
    return "; ".join(merged) if merged else ""
