"""
Parser para archivos CSV exportados desde ACM Digital Library.
Cadena de búsqueda: "generative artificial intelligence"

Columnas típicas de ACM CSV:
  Title, Author, Publication Year, Source, DOI, Abstract, 
  Document Type, URL, ISBN/ISSN, Keywords
"""

import pandas as pd
from pathlib import Path

# Mapeo de columnas ACM → esquema canónico del proyecto
ACM_COLUMN_MAP = {
    "Title":            "title",
    "Author":           "authors",
    "Publication Year": "year",
    "Source":           "source",          # nombre de la revista/conferencia
    "DOI":              "doi",
    "Abstract":         "abstract",
    "Document Type":    "document_type",
    "URL":              "url",
    "ISBN/ISSN":        "issn",
    "Keywords":         "keywords",
    # Sinónimos posibles
    "Authors":          "authors",
    "Year":             "year",
    "Venue":            "source",
}


def parse_acm_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Lee un archivo CSV de ACM y retorna un DataFrame normalizado
    con el esquema canónico del proyecto.

    Parameters
    ----------
    filepath : ruta al archivo .csv exportado desde ACM.

    Returns
    -------
    pd.DataFrame con columnas del esquema canónico y columna
    adicional 'source_db' = 'ACM'.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Archivo ACM no encontrado: {path}")

    # ACM usa encoding UTF-8 con posible BOM
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.strip()

    # Renombrar columnas conocidas
    rename_map = {col: ACM_COLUMN_MAP[col]
                  for col in df.columns if col in ACM_COLUMN_MAP}
    df = df.rename(columns=rename_map)

    # Asegurar que existan todas las columnas canónicas
    canonical_cols = ["title", "authors", "year", "source", "doi",
                      "abstract", "document_type", "url", "issn", "keywords"]
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[canonical_cols].copy()
    df["source_db"] = "ACM"

    # Limpieza básica: quitar espacios extra y valores "nan" literales
    df = df.fillna("").apply(
        lambda col: col.map(lambda v: "" if str(v).strip().lower() == "nan" else str(v).strip())
    )

    # Filtrar filas completamente vacías en título
    df = df[df["title"].str.strip() != ""].reset_index(drop=True)

    return df