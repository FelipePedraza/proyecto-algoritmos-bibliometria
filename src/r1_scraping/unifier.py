"""
Motor de unificacion bibliografica para el Requerimiento 1.

Proceso:
  1. Recibe dos o mas DataFrames (ACM, ScienceDirect, EBSCO) con esquema canonico.
  2. Concatena todos manteniendo el campo 'source_db'.
  3. Detecta duplicados comparando titulos normalizados con similitud
     de Levenshtein (umbral configurable, default 0.92).
  4. Para cada grupo de duplicados conserva el registro mas completo
     (mayor numero de campos no vacios) y prioriza ScienceDirect.
  5. Exporta:
     - unified.csv   -> registros unicos enriquecidos
     - duplicates.csv -> registros descartados con referencia al que se conservo
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.utils.text_utils import normalize_title, levenshtein_similarity

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD  = 0.92
PRIORITY_DB        = "ScienceDirect"
OUTPUT_UNIFIED     = "data/processed/unified.csv"
OUTPUT_DUPLICATES  = "data/duplicates/duplicates.csv"


def unify_databases(
    df_acm,
    df_sd,
    threshold=DEFAULT_THRESHOLD,
    output_dir=".",
    extra_sources=None,
):
    """
    Unifica dos o mas DataFrames bibliograficos eliminando duplicados.

    Parameters
    ----------
    df_acm         : DataFrame de ACM (esquema canonico + source_db).
    df_sd          : DataFrame de ScienceDirect (esquema canonico + source_db).
    threshold      : Umbral de similitud Levenshtein para declarar duplicado.
    output_dir     : Directorio raiz del proyecto para escribir los CSV.
    extra_sources  : DataFrames adicionales (p. ej. EBSCO) con el mismo
                     esquema canonico. Se incluyen en la deduplicacion.

    Returns
    -------
    (df_unified, df_duplicates)
    """
    all_dfs = [df for df in [df_acm, df_sd] if df is not None and len(df) > 0]
    if extra_sources:
        all_dfs.extend([df for df in extra_sources if df is not None and len(df) > 0])

    total = sum(len(d) for d in all_dfs)
    logger.info("Iniciando unificacion: %d fuentes, %d registros totales", len(all_dfs), total)

    # 1. Concatenar
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["_norm_title"] = df_all["title"].apply(normalize_title)
    df_all["_record_id"] = df_all.index

    # 2. Detectar grupos de duplicados
    groups = _find_duplicate_groups(df_all, threshold)

    # 3. Elegir representante por grupo
    keep_ids = []
    discard_ids = []
    duplicate_meta = []

    for group in groups:
        representative_id = _choose_representative(df_all, group)
        keep_ids.append(representative_id)
        for rid in group:
            if rid != representative_id:
                discard_ids.append(rid)
                duplicate_meta.append({
                    "_record_id": rid,
                    "_kept_record_id": representative_id,
                    "_kept_title": df_all.loc[representative_id, "title"],
                    "_kept_db":    df_all.loc[representative_id, "source_db"],
                })

    all_grouped = {rid for g in groups for rid in g}
    for idx in df_all.index:
        if idx not in all_grouped:
            keep_ids.append(idx)

    # 4. Construir DataFrames de salida
    # Incluye 'country' para preservar datos geograficos de EBSCO (authorLocations)
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
        df_dup_meta = pd.DataFrame(duplicate_meta).set_index("_record_id")
        df_duplicates = df_dup_base.join(df_dup_meta).reset_index(drop=True)
    else:
        df_duplicates = df_dup_base.reset_index(drop=True)

    logger.info("Resultado: %d unicos, %d duplicados eliminados",
                len(df_unified), len(df_duplicates))

    # 5. Exportar a CSV
    _export(df_unified, df_duplicates, output_dir)

    return df_unified, df_duplicates


def _find_duplicate_groups(df, threshold):
    """
    Agrupa indices de registros cuyo titulo normalizado supera el umbral
    de similitud. Usa union-find para manejar grupos transitivos.
    """
    n = len(df)
    norm_titles = df["_norm_title"].tolist()
    record_ids  = df["_record_id"].tolist()

    parent = list(range(n))

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
    Criterios: 1) BD prioritaria  2) campos no vacios  3) abstract mas largo.
    """
    candidates = df.loc[group].copy()

    scored = candidates.apply(
        lambda row: (
            int(row["source_db"] == PRIORITY_DB) * 1000
            + sum(1 for v in row if isinstance(v, str) and v.strip() != "")
            + len(str(row.get("abstract", "")))
        ),
        axis=1,
    )
    return int(scored.idxmax())


def _export(df_unified, df_duplicates, output_dir):
    """Escribe los dos CSV de salida creando directorios si no existen."""
    base = Path(output_dir)

    unified_path    = base / OUTPUT_UNIFIED
    duplicates_path = base / OUTPUT_DUPLICATES

    unified_path.parent.mkdir(parents=True, exist_ok=True)
    duplicates_path.parent.mkdir(parents=True, exist_ok=True)

    df_unified.to_csv(unified_path,    index=False, encoding="utf-8-sig")
    df_duplicates.to_csv(duplicates_path, index=False, encoding="utf-8-sig")

    logger.info("Archivos exportados:\n  -> %s\n  -> %s", unified_path, duplicates_path)
