"""
Motor de unificación bibliográfica para el Requerimiento 1.

Proceso:
  1. Recibe dos DataFrames (ACM + ScienceDirect) con esquema canónico.
  2. Concatena ambos manteniendo el campo 'source_db'.
  3. Detecta duplicados comparando títulos normalizados con similitud
     de Levenshtein (umbral configurable, default 0.92).
  4. Para cada grupo de duplicados conserva el registro más completo
     (mayor número de campos no vacíos) y prioriza ScienceDirect
     por tener habitualmente abstracts más completos.
  5. Exporta:
     - unified.csv   → registros únicos enriquecidos
     - duplicates.csv → registros descartados con referencia al que se conservó
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.text_utils import normalize_title, levenshtein_similarity

logger = logging.getLogger(__name__)

# ── Constantes ──────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD   = 0.92   # similitud mínima para considerar duplicado
PRIORITY_DB         = "ScienceDirect"   # BD preferida al elegir representante
OUTPUT_UNIFIED      = "data/processed/unified.csv"
OUTPUT_DUPLICATES   = "data/duplicates/duplicates.csv"


# ── Función principal ────────────────────────────────────────────────────────
def unify_databases(
    df_acm: pd.DataFrame,
    df_sd: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    output_dir: str | Path = ".",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unifica dos DataFrames bibliográficos eliminando duplicados.

    Parameters
    ----------
    df_acm       : DataFrame de ACM (esquema canónico + source_db).
    df_sd        : DataFrame de ScienceDirect (esquema canónico + source_db).
    threshold    : Umbral de similitud Levenshtein para declarar duplicado.
    output_dir   : Directorio raíz del proyecto para escribir los CSV.

    Returns
    -------
    (df_unified, df_duplicates)
    """
    logger.info("Iniciando unificación: ACM=%d, ScienceDirect=%d registros",
                len(df_acm), len(df_sd))

    # 1. Concatenar
    df_all = pd.concat([df_acm, df_sd], ignore_index=True)
    df_all["_norm_title"] = df_all["title"].apply(normalize_title)
    df_all["_record_id"] = df_all.index  # ID interno temporal

    # 2. Detectar grupos de duplicados
    groups = _find_duplicate_groups(df_all, threshold)

    # 3. Para cada grupo, elegir representante y marcar descartados
    keep_ids:    list[int] = []
    discard_ids: list[int] = []
    duplicate_meta: list[dict] = []   # info extra para el archivo duplicates.csv

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

    # Registros que no pertenecen a ningún grupo de duplicados → se conservan
    all_grouped = {rid for g in groups for rid in g}
    for idx in df_all.index:
        if idx not in all_grouped:
            keep_ids.append(idx)

    # 4. Construir DataFrames de salida
    output_cols = ["title", "authors", "year", "source", "doi",
                   "abstract", "document_type", "url", "issn",
                   "keywords", "source_db"]

    df_unified = (
        df_all.loc[sorted(set(keep_ids)), output_cols]
        .copy()
        .reset_index(drop=True)
    )

    df_dup_base = df_all.loc[discard_ids, output_cols].copy() if discard_ids else pd.DataFrame(columns=output_cols)

    if duplicate_meta:
        df_dup_meta = pd.DataFrame(duplicate_meta).set_index("_record_id")
        df_duplicates = df_dup_base.join(df_dup_meta).reset_index(drop=True)
    else:
        df_duplicates = df_dup_base.reset_index(drop=True)

    logger.info("Resultado: %d únicos, %d duplicados eliminados",
                len(df_unified), len(df_duplicates))

    # 5. Exportar a CSV
    _export(df_unified, df_duplicates, output_dir)

    return df_unified, df_duplicates


# ── Helpers internos ─────────────────────────────────────────────────────────
def _find_duplicate_groups(
    df: pd.DataFrame, threshold: float
) -> list[list[int]]:
    """
    Agrupa índices de registros cuyo título normalizado supera el umbral
    de similitud. Usa union-find para manejar grupos transitivos.

    Algoritmo O(n²) sobre títulos normalizados. Aceptable para corpus
    bibliométricos típicos (< 5 000 registros).
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

    # Agrupar por raíz del union-find
    from collections import defaultdict
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(record_ids[i])

    # Solo devolver grupos con 2+ miembros (son los duplicados)
    return [g for g in clusters.values() if len(g) > 1]


def _choose_representative(df: pd.DataFrame, group: list[int]) -> int:
    """
    Elige el registro más completo de un grupo de duplicados.
    Criterios (en orden de prioridad):
      1. Proviene de la BD prioritaria (ScienceDirect).
      2. Mayor número de campos no vacíos.
      3. Abstract más largo (mayor riqueza de información).
    """
    candidates = df.loc[group].copy()

    # Puntaje de completitud
    scored = candidates.apply(
        lambda row: (
            int(row["source_db"] == PRIORITY_DB) * 1000
            + sum(1 for v in row if isinstance(v, str) and v.strip() != "")
            + len(str(row.get("abstract", "")))
        ),
        axis=1,
    )
    return int(scored.idxmax())


def _export(
    df_unified: pd.DataFrame,
    df_duplicates: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Escribe los dos CSV de salida creando directorios si no existen."""
    base = Path(output_dir)

    unified_path    = base / OUTPUT_UNIFIED
    duplicates_path = base / OUTPUT_DUPLICATES

    unified_path.parent.mkdir(parents=True, exist_ok=True)
    duplicates_path.parent.mkdir(parents=True, exist_ok=True)

    df_unified.to_csv(unified_path,    index=False, encoding="utf-8-sig")
    df_duplicates.to_csv(duplicates_path, index=False, encoding="utf-8-sig")

    logger.info("Archivos exportados:\n  → %s\n  → %s",
                unified_path, duplicates_path)