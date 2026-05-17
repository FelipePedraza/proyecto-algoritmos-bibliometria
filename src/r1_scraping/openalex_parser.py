"""
Parser para resultados de la API de OpenAlex.

Transforma el JSON devuelto por https://api.openalex.org/works
al esquema canonico de 12 columnas del proyecto:

  title, authors, year, source, doi, abstract,
  document_type, url, issn, keywords, country, source_db

Nota sobre abstracts:
  OpenAlex almacena el abstract como un inverted index
  (diccionario {palabra: [posicion, ...]}).  La funcion
  _reconstruct_abstract() lo convierte a texto plano.

Nota sobre DOIs:
  OpenAlex devuelve el DOI con prefijo completo de URL
  ("https://doi.org/10.xxxx/yyyy"). El parser elimina el prefijo.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_COLS = [
    "title", "authors", "year", "source", "doi",
    "abstract", "document_type", "url", "issn",
    "keywords", "country", "source_db",
]


def parse_openalex_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convierte una lista de objetos Work de OpenAlex a un DataFrame
    con el esquema canonico del proyecto.

    Parameters
    ----------
    records : lista de dicts tal como devuelve la API de OpenAlex
              (campo results de la respuesta paginada).

    Returns
    -------
    DataFrame con las columnas CANONICAL_COLS y source_db == "OpenAlex".
    Filas sin titulo son descartadas.
    """
    rows = [_map_record(r) for r in records]
    df = pd.DataFrame(rows, columns=CANONICAL_COLS)

    df = df.fillna("").apply(
        lambda col: col.map(
            lambda v: "" if str(v).strip().lower() in ("nan", "none") else str(v).strip()
        )
    )

    df = df[df["title"].str.strip() != ""].reset_index(drop=True)

    logger.info("OpenAlex parser: %d registros procesados, %d con titulo valido",
                len(records), len(df))
    return df


def _map_record(rec: dict) -> dict:
    """Extrae y transforma los campos de un objeto Work de OpenAlex."""

    title = rec.get("title") or ""

    authorships = rec.get("authorships") or []
    author_names = [
        a.get("author", {}).get("display_name", "")
        for a in authorships
        if a.get("author", {}).get("display_name")
    ]
    authors = ", ".join(author_names)

    year = str(rec.get("publication_year") or "")

    primary_loc = rec.get("primary_location") or {}
    source_info = primary_loc.get("source") or {}
    source = source_info.get("display_name") or ""

    raw_doi = rec.get("doi") or ""
    doi = raw_doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()

    inverted_index = rec.get("abstract_inverted_index")
    abstract = _reconstruct_abstract(inverted_index) if inverted_index else ""

    document_type = rec.get("type") or ""

    url = primary_loc.get("landing_page_url") or ""
    if not url:
        oa_info = rec.get("open_access") or {}
        url = oa_info.get("oa_url") or ""

    issn = source_info.get("issn_l") or ""
    if not issn:
        issn_list = source_info.get("issn") or []
        issn = issn_list[0] if issn_list else ""

    keyword_objs = rec.get("keywords") or []
    keyword_terms = [k.get("keyword", "") for k in keyword_objs if k.get("keyword")]

    if not keyword_terms:
        topics = rec.get("topics") or []
        keyword_terms = [t.get("display_name", "") for t in topics if t.get("display_name")]

    keywords = ", ".join(keyword_terms)

    country = _extract_country(authorships)

    return {
        "title":         title,
        "authors":       authors,
        "year":          year,
        "source":        source,
        "doi":           doi,
        "abstract":      abstract,
        "document_type": document_type,
        "url":           url,
        "issn":          issn,
        "keywords":      keywords,
        "country":       country,
        "source_db":     "OpenAlex",
    }


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """
    Convierte el abstract_inverted_index de OpenAlex a texto plano.

    OpenAlex almacena el abstract como un diccionario
    { "palabra": [posicion1, posicion2, ...], ... }.
    La funcion reconstruye la secuencia ordenando por posicion.

    Parameters
    ----------
    inverted_index : dict {palabra: [posiciones]}

    Returns
    -------
    Cadena de texto con el abstract reconstruido, o cadena vacia si
    el indice es vacio o invalido.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    position_word: dict[int, str] = {}
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                position_word[pos] = word

    if not position_word:
        return ""

    return " ".join(position_word[i] for i in sorted(position_word))


def _extract_country(authorships: list[dict]) -> str:
    """
    Devuelve el primer codigo de pais encontrado entre las instituciones
    de los autores (e.g. "US", "DE", "CO").

    Retorna cadena vacia si no hay informacion geografica.
    """
    for authorship in (authorships or []):
        institutions = authorship.get("institutions") or []
        for inst in institutions:
            code = inst.get("country_code") or ""
            if code.strip():
                return code.strip().upper()
    return ""
