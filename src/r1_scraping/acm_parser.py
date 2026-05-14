"""
Parser para archivos CSV exportados desde ACM Digital Library.
Cadena de busqueda: "generative artificial intelligence"

Columnas tipicas de ACM CSV:
  Title, Author, Publication Year, Source, DOI, Abstract,
  Document Type, URL, ISBN/ISSN, Keywords
"""

import pandas as pd
from pathlib import Path

ACM_COLUMN_MAP = {
    "Title":            "title",
    "Author":           "authors",
    "Publication Year": "year",
    "Source":           "source",
    "DOI":              "doi",
    "Abstract":         "abstract",
    "Document Type":    "document_type",
    "URL":              "url",
    "ISBN/ISSN":        "issn",
    "Keywords":         "keywords",
    "Authors":          "authors",
    "Year":             "year",
    "Venue":            "source",
}


def parse_acm_csv(filepath):
    """
    Lee un archivo CSV de ACM y retorna un DataFrame normalizado
    con el esquema canonico del proyecto.
    """
    if isinstance(filepath, (str, Path)):
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Archivo ACM no encontrado: {path}")
        source = path
    else:
        source = filepath

    df = pd.read_csv(source, encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip()

    rename_map = {col: ACM_COLUMN_MAP[col]
                  for col in df.columns if col in ACM_COLUMN_MAP}
    df = df.rename(columns=rename_map)

    # Asegurar que existan todas las columnas canonicas (incluye 'country')
    canonical_cols = ["title", "authors", "year", "source", "doi",
                      "abstract", "document_type", "url", "issn", "keywords",
                      "country"]
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[canonical_cols].copy()
    df["source_db"] = "ACM"

    df = df.fillna("").apply(
        lambda col: col.map(lambda v: "" if str(v).strip().lower() == "nan" else str(v).strip())
    )

    df = df[df["title"].str.strip() != ""].reset_index(drop=True)

    return df
