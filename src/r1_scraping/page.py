"""
Requerimiento 1 — Interfaz Streamlit
=====================================
Permite al usuario:
  1. Cargar los archivos CSV exportados desde ACM y ScienceDirect.
  2. Ver una vista previa de cada base de datos.
  3. Ejecutar la unificación automática con detección de duplicados.
  4. Descargar:
       - unified.csv      (registros únicos)
       - duplicates.csv   (registros descartados)
  5. Ver estadísticas del proceso.
"""

import io
import logging

import pandas as pd
import streamlit as st

from src.r1_scraping.acm_parser import parse_acm_csv
from src.r1_scraping.sciencedirect_parser import parse_sciencedirect_csv
from src.r1_scraping.unifier import unify_databases

logger = logging.getLogger(__name__)

# ── Constantes de UI ─────────────────────────────────────────────────────────
PAGE_TITLE = "R1 · Automatización y Unificación Bibliométrica"
THRESHOLD_DEFAULT = 0.92


def render():
    """Renderiza la página completa del Requerimiento 1."""
    st.title("Requerimiento 1: Automatización y Unificación")
    st.markdown(
        """
        **Objetivo:** Cargar los archivos CSV exportados desde **ACM Digital Library**
        y **ScienceDirect** con la cadena de búsqueda `"generative artificial intelligence"`,
        unificarlos en un solo dataset y eliminar duplicados automáticamente.
        """
    )

    # ── Sidebar de configuración ─────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuración")
        threshold = st.slider(
            "Umbral de similitud para duplicados",
            min_value=0.80,
            max_value=1.00,
            value=THRESHOLD_DEFAULT,
            step=0.01,
            help=(
                "Dos títulos se consideran duplicados si su similitud "
                "de Levenshtein ≥ este valor. 0.92 es el valor recomendado."
            ),
        )
        st.caption("Similitud calculada sobre títulos normalizados (sin acentos, minúsculas).")

    # ── Carga de archivos ─────────────────────────────────────────────────────
    st.header("1. Carga de archivos")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ACM Digital Library")
        acm_file = st.file_uploader(
            "Sube el CSV de ACM", type=["csv"], key="acm_upload"
        )
        if acm_file:
            st.success(f"✅ {acm_file.name} cargado")

    with col2:
        st.subheader("ScienceDirect")
        sd_file = st.file_uploader(
            "Sube el CSV de ScienceDirect", type=["csv"], key="sd_upload"
        )
        if sd_file:
            st.success(f"✅ {sd_file.name} cargado")

    # ── Vista previa de los archivos ──────────────────────────────────────────
    if acm_file or sd_file:
        st.header("2. Vista previa de los datos cargados")

    if acm_file:
        with st.expander("Vista previa ACM", expanded=False):
            try:
                df_acm_raw = pd.read_csv(acm_file, encoding="utf-8-sig", dtype=str)
                st.dataframe(df_acm_raw.head(10), use_container_width=True)
                st.caption(f"Total de filas: {len(df_acm_raw)}")
                acm_file.seek(0)  # rebobinar para el parser
            except Exception as e:
                st.error(f"Error al previsualizar ACM: {e}")

    if sd_file:
        with st.expander("Vista previa ScienceDirect", expanded=False):
            try:
                df_sd_raw = pd.read_csv(sd_file, encoding="utf-8-sig", dtype=str)
                st.dataframe(df_sd_raw.head(10), use_container_width=True)
                st.caption(f"Total de filas: {len(df_sd_raw)}")
                sd_file.seek(0)
            except Exception as e:
                st.error(f"Error al previsualizar ScienceDirect: {e}")

    # ── Botón de unificación ──────────────────────────────────────────────────
    st.header("3. Unificación automática")

    can_run = acm_file is not None and sd_file is not None

    if not can_run:
        st.info("Sube ambos archivos CSV para habilitar la unificación.")

    if can_run and st.button("Ejecutar unificación", type="primary"):
        with st.spinner("Procesando y detectando duplicados..."):
            try:
                # Parsear con los módulos especializados
                acm_file.seek(0)
                sd_file.seek(0)

                df_acm = parse_acm_csv(acm_file)
                df_sd  = parse_sciencedirect_csv(sd_file)

                df_unified, df_duplicates = unify_databases(
                    df_acm, df_sd,
                    threshold=threshold,
                    output_dir=".",  # carpeta raíz del proyecto
                )

                # Guardar en session_state para persistencia
                st.session_state["r1_unified"]    = df_unified
                st.session_state["r1_duplicates"] = df_duplicates
                st.session_state["r1_acm_count"]  = len(df_acm)
                st.session_state["r1_sd_count"]   = len(df_sd)

            except Exception as e:
                st.error(f"❌ Error durante la unificación: {e}")
                logger.exception("Unification error")

    # ── Resultados ────────────────────────────────────────────────────────────
    if "r1_unified" in st.session_state:
        df_unified    = st.session_state["r1_unified"]
        df_duplicates = st.session_state["r1_duplicates"]
        acm_count     = st.session_state["r1_acm_count"]
        sd_count      = st.session_state["r1_sd_count"]

        st.header("4. Resultados")

        # Métricas
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Registros ACM",            acm_count)
        col_b.metric("Registros ScienceDirect",  sd_count)
        col_c.metric("Registros únicos",          len(df_unified),
                     delta=f"-{acm_count + sd_count - len(df_unified)} duplicados")
        col_d.metric("Duplicados eliminados",     len(df_duplicates))

        # Tabla de registros unificados
        st.subheader("Registros unificados")
        display_cols = ["title", "authors", "year", "source", "source_db", "doi"]
        st.dataframe(
            df_unified[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=350,
        )

        # Tabla de duplicados
        with st.expander(f"Registros duplicados eliminados ({len(df_duplicates)})", expanded=False):
            dup_display = ["title", "source_db", "_kept_title", "_kept_db"]
            dup_cols_available = [c for c in dup_display if c in df_duplicates.columns]
            st.dataframe(df_duplicates[dup_cols_available], use_container_width=True)

        # ── Descargas ─────────────────────────────────────────────────────────
        st.header("5. Descarga de archivos")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            st.download_button(
                label="Descargar unified.csv",
                data=_to_csv_bytes(df_unified),
                file_name="unified.csv",
                mime="text/csv",
                help="Registros únicos con información completa.",
            )

        with col_dl2:
            st.download_button(
                label="Descargar duplicates.csv",
                data=_to_csv_bytes(df_duplicates),
                file_name="duplicates.csv",
                mime="text/csv",
                help="Registros descartados por aparecer repetidos.",
            )

        # ── Distribución por base de datos ────────────────────────────────────
        st.subheader("Distribución por base de datos (registros únicos)")
        dist = df_unified["source_db"].value_counts().reset_index()
        dist.columns = ["Base de datos", "Registros"]
        st.bar_chart(dist.set_index("Base de datos"))

        # ── Completitud de campos ─────────────────────────────────────────────
        st.subheader("Completitud de campos en el dataset unificado")
        completeness = (
            df_unified.apply(lambda col: (col.str.strip() != "").sum())
            / len(df_unified) * 100
        ).round(1)
        completeness_df = completeness.reset_index()
        completeness_df.columns = ["Campo", "% completitud"]
        st.dataframe(completeness_df, use_container_width=True)


# ── Utilidad ──────────────────────────────────────────────────────────────────
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serializa un DataFrame a bytes UTF-8 con BOM para compatibilidad con Excel."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue().encode("utf-8-sig")