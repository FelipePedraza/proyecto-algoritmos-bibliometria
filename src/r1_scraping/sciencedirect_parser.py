"""
Parser para archivos CSV exportados desde ScienceDirect (Elsevier).
Cadena de búsqueda: "generative artificial intelligence"

Columnas típicas de ScienceDirect CSV:
  Title, Authors, Year, Source title, DOI, Abstract,
  Document Type, Link, ISSN, Author Keywords, Index Keywords
"""

import pandas as pd
from pathlib import Path

# Mapeo de columnas ScienceDirect → esquema canónico del proyecto
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
    # Sinónimos posibles en distintas versiones de exportación
    "Author(s)":        "authors",
    "Publication Year": "year",
    "Source":           "source",
}


def parse_sciencedirect_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Lee un archivo CSV de ScienceDirect y retorna un DataFrame normalizado
    con el esquema canónico del proyecto.

    Parameters
    ----------
    filepath : ruta al archivo .csv exportado desde ScienceDirect.

    Returns
    -------
    pd.DataFrame con columnas del esquema canónico y columna
    adicional 'source_db' = 'ScienceDirect'.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Archivo ScienceDirect no encontrado: {path}")

    # ScienceDirect suele exportar en UTF-8
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip()

    # Renombrar columnas conocidas
    rename_map = {col: SD_COLUMN_MAP[col]
                  for col in df.columns if col in SD_COLUMN_MAP}
    df = df.rename(columns=rename_map)

    # Combinar keywords: Author Keywords + Index Keywords si ambas existen
    if "keywords" in df.columns and "index_keywords" in df.columns:
        df["keywords"] = df.apply(
            lambda row: _merge_keywords(row["keywords"], row["index_keywords"]),
            axis=1
        )
        df.drop(columns=["index_keywords"], inplace=True)
    elif "index_keywords" in df.columns:
        df.rename(columns={"index_keywords": "keywords"}, inplace=True)

    # Asegurar esquema canónico completo
    canonical_cols = ["title", "authors", "year", "source", "doi",
                      "abstract", "document_type", "url", "issn", "keywords"]
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[canonical_cols].copy()
    df["source_db"] = "ScienceDirect"

    # Limpieza básica
    df = df.fillna("").apply(
        lambda col: col.map(lambda v: "" if str(v).strip().lower() == "nan" else str(v).strip())
    )

    df = df[df["title"].str.strip() != ""].reset_index(drop=True)

    return df


def _merge_keywords(kw1: str, kw2: str) -> str:
    """Une dos listas de keywords separadas por ';' eliminando duplicados."""
    set1 = {k.strip().lower() for k in str(kw1).split(";") if k.strip()}
    set2 = {k.strip().lower() for k in str(kw2).split(";") if k.strip()}
    merged = sorted(set1 | set2)
    return "; ".join(merged) if merged else ""