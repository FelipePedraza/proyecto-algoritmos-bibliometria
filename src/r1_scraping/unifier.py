"""
Motor de unificacion bibliografica para el Requerimiento 1.

Proceso:
  1. Recibe DataFrames (ACM, ScienceDirect, EBSCO, OpenAlex) con esquema canonico.
  2. Concatena todos manteniendo el campo source_db.
  3. Detecta duplicados con similitud de Levenshtein (umbral configurable, 0.92).
  4. Para cada grupo de duplicados conserva el registro mas completo
     segun la jerarquia DB_PRIORITY y luego numero de campos rellenos.
  5. Enriquece abstracts vacios con abstracts de duplicados de OpenAlex.
  6. Exporta unified.csv y duplicates.csv.

Jerarquia: ScienceDirect > EBSCO > ACM > OpenAlex
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.utils.text_utils import normalize_title, levenshtein_similarity

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.92
OUTPUT_UNIFIED    = "data/processed/unified.csv"
OUTPUT_DUPLICATES = "data/duplicates/duplicates.csv"

DB_PRIORITY: dict[str, int] = {
    "ScienceDirect": 3,
    "EBSCO":         2,
    "ACM":           1,
    "OpenAlex":      0,
}
_PRIORITY_WEIGHT = 1000


def unify_databases(df_acm, df_sd, threshold=DEFAULT_THRESHOLD,
                    output_dir=".", extra_sources=None):
    """
    Unifica dos o mas DataFrames bibliograficos eliminando duplicados.

    Parameters
    ----------
    df_acm        : DataFrame de ACM (esquema canonico + source_db).
    df_sd         : DataFrame de ScienceDirect.
    threshold     : Umbral de similitud Levenshtein para declarar duplicado.
    output_dir    : Directorio raiz del proyecto para escribir los CSV.
    extra_sources : DataFrames adicionales (EBSCO, OpenAlex, etc.).
                    Los DataFrames de OpenAlex enriquecen abstracts vacios.

    Returns
    -------
    (df_unified, df_duplicates)
    """
    all_dfs = [df for df in [df_acm, df_sd] if df is not None and len(df) > 0]
    if extra_sources:
        all_dfs.extend([df for df in extra_sources if df is not None and len(df) > 0])

    total = sum(len(d) for d in all_dfs)
    logger.info("Iniciando unificacion: %d fuentes, %d registros totales",
                len(all_dfs), total)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["_norm_title"] = df_all["title"].apply(normalize_title)
    df_all["_record_id"]  = df_all.index

    groups = _find_duplicate_groups(df_all, threshold)

    keep_ids       = []
    discard_ids    = []
    duplicate_meta = []

    for group in groups:
        rep = _choose_representative(df_all, group)
        keep_ids.append(rep)
        for rid in group:
            if rid != rep:
                discard_ids.append(rid)
                duplicate_meta.append({
                    "_record_id":      rid,
                    "_kept_record_id": rep,
                    "_kept_title":     df_all.loc[rep, "title"],
                    "_kept_db":        df_all.loc[rep, "source_db"],
                })

    all_grouped = {rid for g in groups for rid in g}
    for idx in df_all.index:
        if idx not in all_grouped:
            keep_ids.append(idx)

    output_cols = ["title", "authors", "year", "source", "doi",
                   "abstract", "document_type", "url", "issn",
                   "keywords", "country", "source_db"]
    for col in output_cols:
        if col not in df_all.columns:
            df_all[col] = ""

    df_unified = (
        df_all.loc[sorted(set(keep_ids)), output_cols]
        .copy()
        .reset_index(drop=True)
    )

    df_dup_base = (
        df_all.loc[discard_ids, output_cols].copy()
        if discard_ids
        else pd.DataFrame(columns=output_cols)
    )

    if duplicate_meta:
        df_dup_meta  = pd.DataFrame(duplicate_meta).set_index("_record_id")
        df_duplicates = df_dup_base.join(df_dup_meta).reset_index(drop=True)
    else:
        df_duplicates = df_dup_base.reset_index(drop=True)

    logger.info("Resultado: %d unicos, %d duplicados eliminados",
                len(df_unified), len(df_duplicates))

    df_unified = _enrich_abstracts(df_unified, df_all, discard_ids, duplicate_meta)

    _export(df_unified, df_duplicates, output_dir)
    return df_unified, df_duplicates


def _find_duplicate_groups(df, threshold):
    """Agrupa indices con titulos similares usando union-find transitivo."""
    n           = len(df)
    norm_titles = df["_norm_title"].tolist()
    record_ids  = df["_record_id"].tolist()
    parent      = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if levenshtein_similarity(norm_titles[i], norm_titles[j]) >= threshold:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(record_ids[i])

    return [g for g in clusters.values() if len(g) > 1]


def _choose_representative(df, group):
    """
    Elige el registro mas completo de un grupo de duplicados.

    Criterios (peso descendente):
      1. Prioridad de BD (DB_PRIORITY * _PRIORITY_WEIGHT).
      2. Numero de campos no vacios.
      3. Longitud del abstract.
    """
    candidates = df.loc[group].copy()

    def _score(row):
        db_s   = DB_PRIORITY.get(str(row.get("source_db", "")), 0)
        fields = sum(1 for v in row if isinstance(v, str) and v.strip() != "")
        ablen  = len(str(row.get("abstract", "")))
        return db_s * _PRIORITY_WEIGHT + fields + ablen

    scored = candidates.apply(_score, axis=1)
    return int(scored.idxmax())


def _enrich_abstracts(df_unified, df_all, discard_ids, duplicate_meta):
    """
    Completa abstracts vacios en df_unified usando duplicados descartados de OpenAlex.

    Si el representante elegido no tiene abstract y un duplicado descartado
    de OpenAlex si lo tiene, se copia el abstract mas largo al representante.
    """
    if not duplicate_meta:
        return df_unified

    kept_to_oa: dict[int, list[str]] = defaultdict(list)

    for meta in duplicate_meta:
        did = meta["_record_id"]
        kid = meta["_kept_record_id"]
        if did not in df_all.index:
            continue
        row = df_all.loc[did]
        if (str(row.get("source_db", "")) == "OpenAlex"
                and str(row.get("abstract", "")).strip()):
            kept_to_oa[kid].append(str(row["abstract"]).strip())

    if not kept_to_oa:
        return df_unified

    title_to_idx = {
        row["title"].strip().lower(): i
        for i, row in df_unified.iterrows()
        if row["title"].strip()
    }

    enriched = 0
    for kid, abstracts in kept_to_oa.items():
        if kid not in df_all.index:
            continue
        title_key = str(df_all.loc[kid, "title"]).strip().lower()
        uidx = title_to_idx.get(title_key)
        if uidx is None:
            continue
        if not str(df_unified.at[uidx, "abstract"]).strip():
            df_unified.at[uidx, "abstract"] = max(abstracts, key=len)
            enriched += 1

    if enriched:
        logger.info("Abstracts enriquecidos con OpenAlex: %d registros", enriched)

    return df_unified


def _export(df_unified, df_duplicates, output_dir):
    """Escribe los dos CSV de salida creando directorios si no existen."""
    base = Path(output_dir)

    unified_path    = base / OUTPUT_UNIFIED
    duplicates_path = base / OUTPUT_DUPLICATES

    unified_path.parent.mkdir(parents=True, exist_ok=True)
    duplicates_path.parent.mkdir(parents=True, exist_ok=True)

    df_unified.to_csv(unified_path,    index=False, encoding="utf-8-sig")
    df_duplicates.to_csv(duplicates_path, index=False, encoding="utf-8-sig")

    logger.info("Archivos exportados:\n  -> %s\n  -> %s",
                unified_path, duplicates_path)
