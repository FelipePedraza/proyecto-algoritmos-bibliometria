"""
Requerimiento 1 — Interfaz Streamlit
=====================================
Permite al usuario:
  1. Cargar CSV exportados desde ACM, ScienceDirect y/o EBSCO.
  2. Buscar y descargar artículos automáticamente desde OpenAlex API.
  3. Ver una vista previa de cada fuente de datos.
  4. Ejecutar la unificación automática con detección de duplicados.
  5. Descargar:
       - unified.csv      (registros únicos)
       - duplicates.csv   (registros descartados)
  6. Ver estadísticas del proceso.

EBSCO es la tercera fuente de CSV opcional; aporta la columna ``country``
(mapeada desde ``authorLocations``) que habilita el mapa de calor
geográfico en el Requerimiento 5.

OpenAlex (https://openalex.org) es una base de datos académica abierta
con API gratuita que permite automatizar la búsqueda y descarga de
metadatos sin necesidad de exportar CSV manualmente.
"""

import io
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.r1_scraping.acm_parser import parse_acm_csv
from src.r1_scraping.sciencedirect_parser import parse_sciencedirect_csv
from src.r1_scraping.ebsco_parser import parse_ebsco_csv
from src.r1_scraping.openalex_client import OpenAlexClient, read_jsonl
from src.r1_scraping.openalex_parser import parse_openalex_records
from src.r1_scraping.unifier import unify_databases

logger = logging.getLogger(__name__)

# ── Constantes de UI ─────────────────────────────────────────────────────────
PAGE_TITLE        = "R1 · Automatización y Unificación Bibliométrica"
THRESHOLD_DEFAULT = 0.92
OA_JSONL_PATH     = "data/raw/openalex_raw.jsonl"
OA_DEFAULT_QUERY  = "generative artificial intelligence"
OA_MAX_DEFAULT    = 500


def render():
    """Renderiza la página completa del Requerimiento 1."""
    st.title("Requerimiento 1: Automatización y Unificación")
    st.markdown(
        """
        **Objetivo:** Recopilar artículos sobre *"generative artificial intelligence"*
        desde múltiples fuentes, unificarlos en un solo dataset y eliminar duplicados.

        Usa la pestaña **📂 Carga manual** para subir CSV exportados de ACM, ScienceDirect
        o EBSCO, o la pestaña **🔍 OpenAlex API** para descargar artículos automáticamente
        sin necesidad de exportar archivos.
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

    # ── Pestañas de fuentes ───────────────────────────────────────────────────
    st.header("1. Fuentes de datos")
    tab_csv, tab_oa = st.tabs(["📂 Carga manual (CSV)", "🔍 OpenAlex API (automático)"])

    # ═══════════════════════════════════════════════════════════════════════════
    # Pestaña 1: Carga manual de CSV
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_csv:
        st.markdown(
            "Descarga los CSV de las bases de datos y súbelos aquí.  "
            "ACM y ScienceDirect bloquean scrapers automatizados; la carga manual es intencional."
        )

        col1, col2, col3 = st.columns(3)

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

        with col3:
            st.subheader("EBSCO *(opcional)*")
            ebsco_file = st.file_uploader(
                "Sube el CSV de EBSCO", type=["csv"], key="ebsco_upload",
                help="Business Source Ultimate, Academic Search, etc. "
                     "Aporta datos geográficos (authorLocations) para el mapa en R5.",
            )
            if ebsco_file:
                st.success(f"✅ {ebsco_file.name} cargado")

        # ── Vista previa de CSV cargados ──────────────────────────────────────
        loaded_csv = [f for f in [acm_file, sd_file, ebsco_file] if f is not None]
        if loaded_csv:
            st.markdown("---")
            st.subheader("Vista previa")

        if acm_file:
            with st.expander("Vista previa ACM", expanded=False):
                try:
                    df_acm_raw = pd.read_csv(acm_file, encoding="utf-8-sig", dtype=str)
                    st.dataframe(df_acm_raw.head(10), use_container_width=True)
                    st.caption(f"Total de filas: {len(df_acm_raw)}")
                    acm_file.seek(0)
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

        if ebsco_file:
            with st.expander("Vista previa EBSCO", expanded=False):
                try:
                    df_ebsco_raw = pd.read_csv(ebsco_file, encoding="utf-8-sig", dtype=str)
                    st.dataframe(df_ebsco_raw.head(10), use_container_width=True)
                    st.caption(
                        f"Total de filas: {len(df_ebsco_raw)}  •  "
                        f"Columnas: {', '.join(df_ebsco_raw.columns.tolist())}"
                    )
                    _geo_cols = {"authorLocations", "Author Locations", "subjects"}
                    if "authorLocations" in df_ebsco_raw.columns or "Author Locations" in df_ebsco_raw.columns:
                        st.success("✅ Columna `authorLocations` detectada — el mapa geográfico estará disponible en R5.")
                    elif "subjects" in df_ebsco_raw.columns:
                        st.success("✅ Columna `subjects` detectada — país se extrae de subject terms.")
                    else:
                        st.warning("⚠️ No se encontró columna geográfica (`authorLocations` ni `subjects`).")
                    ebsco_file.seek(0)
                except Exception as e:
                    st.error(f"Error al previsualizar EBSCO: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Pestaña 2: OpenAlex API
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_oa:
        _render_openalex_tab()

    # ── Determinar fuentes disponibles para unificación ───────────────────────
    # Inicializar variables de archivos CSV por si el usuario está en la pestaña OA
    if "acm_upload" not in st.session_state:
        acm_file = None
    if "sd_upload" not in st.session_state:
        sd_file = None
    if "ebsco_upload" not in st.session_state:
        ebsco_file = None

    # ── Botón de unificación ──────────────────────────────────────────────────
    st.header("2. Unificación automática")

    has_csv  = any(f is not None for f in [acm_file, sd_file, ebsco_file])
    has_oa   = bool(st.session_state.get("r1_oa_df") is not None
                    and len(st.session_state["r1_oa_df"]) > 0)
    can_run  = has_csv or has_oa

    if not can_run:
        st.info(
            "Sube al menos un archivo CSV **o** realiza una búsqueda en OpenAlex "
            "para habilitar la unificación."
        )

    if can_run and st.button("Ejecutar unificación", type="primary"):
        with st.spinner("Procesando y detectando duplicados..."):
            try:
                df_acm    = _safe_parse(acm_file,   parse_acm_csv)
                df_sd     = _safe_parse(sd_file,    parse_sciencedirect_csv)
                df_ebsco  = _safe_parse(ebsco_file, parse_ebsco_csv)
                df_oa     = st.session_state.get("r1_oa_df", pd.DataFrame())

                extra = [df for df in [df_ebsco, df_oa]
                         if df is not None and len(df) > 0]

                df_unified, df_duplicates = unify_databases(
                    df_acm, df_sd,
                    threshold=threshold,
                    output_dir=".",
                    extra_sources=extra if extra else None,
                )

                st.session_state["r1_unified"]     = df_unified
                st.session_state["r1_duplicates"]  = df_duplicates
                st.session_state["r1_acm_count"]   = len(df_acm) if df_acm is not None else 0
                st.session_state["r1_sd_count"]    = len(df_sd) if df_sd is not None else 0
                st.session_state["r1_ebsco_count"] = len(df_ebsco) if df_ebsco is not None else 0
                st.session_state["r1_oa_count"]    = len(df_oa) if df_oa is not None else 0
                st.success("✅ Unificación completada.")

            except Exception as e:
                st.error(f"❌ Error durante la unificación: {e}")
                logger.exception("Unification error")

    # ── Resultados ────────────────────────────────────────────────────────────
    if "r1_unified" in st.session_state:
        df_unified    = st.session_state["r1_unified"]
        df_duplicates = st.session_state["r1_duplicates"]
        acm_count     = st.session_state.get("r1_acm_count",   0)
        sd_count      = st.session_state.get("r1_sd_count",    0)
        ebsco_count   = st.session_state.get("r1_ebsco_count", 0)
        oa_count      = st.session_state.get("r1_oa_count",    0)
        total_input   = acm_count + sd_count + ebsco_count + oa_count

        st.header("3. Resultados")

        # Métricas
        mc = st.columns(6)
        mc[0].metric("ACM",           acm_count)
        mc[1].metric("ScienceDirect", sd_count)
        mc[2].metric("EBSCO",         ebsco_count)
        mc[3].metric("OpenAlex",      oa_count)
        mc[4].metric(
            "Registros únicos", len(df_unified),
            delta=f"-{total_input - len(df_unified)} dup.",
        )
        mc[5].metric("Duplicados eliminados", len(df_duplicates))

        # Advertencia si EBSCO no tiene datos geográficos en el resultado
        if ebsco_count > 0:
            has_country = (
                "country" in df_unified.columns
                and df_unified["country"].str.strip().replace("", pd.NA).notna().any()
            )
            if has_country:
                st.success("✅ Datos geográficos (country) presentes — el mapa del R5 estará activo.")
            else:
                st.warning("⚠️ EBSCO procesado pero sin países detectados. El mapa del R5 puede estar vacío.")

        # Tabla de registros unificados
        st.subheader("Registros unificados")
        display_cols = ["title", "authors", "year", "source", "source_db", "doi"]
        if "country" in df_unified.columns:
            display_cols.append("country")
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
        st.header("4. Descarga de archivos")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            st.download_button(
                label="Descargar unified.csv",
                data=_to_csv_bytes(df_unified),
                file_name="unified.csv",
                mime="text/csv",
            )

        with col_dl2:
            st.download_button(
                label="Descargar duplicates.csv",
                data=_to_csv_bytes(df_duplicates),
                file_name="duplicates.csv",
                mime="text/csv",
            )

        # ── Distribución por fuente ───────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════════
# Sección OpenAlex
# ═══════════════════════════════════════════════════════════════════════════════

def _render_openalex_tab():
    """
    Renderiza la pestaña de búsqueda automática en OpenAlex.

    Flujo:
      1. El usuario escribe la query y elige el número máximo de resultados.
      2. Al pulsar "Buscar", el cliente pagina la API con barra de progreso.
      3. Los resultados se almacenan en JSONL y se parsean al esquema canónico.
      4. Se muestra una vista previa; el usuario confirma si los incluye en la
         unificación mediante el botón "Usar estos resultados".
    """
    st.markdown(
        """
        [OpenAlex](https://openalex.org) es una base de datos académica abierta con
        más de 250 millones de artículos indexados.  Su API es **gratuita y sin
        autenticación**, lo que permite automatizar la descarga de metadatos sin
        exportar archivos manualmente.

        Los resultados se guardan en `data/raw/openalex_raw.jsonl` para reutilización.
        """
    )

    # ── Formulario de búsqueda ────────────────────────────────────────────────
    with st.form("openalex_search_form"):
        oa_query = st.text_input(
            "Término de búsqueda",
            value=OA_DEFAULT_QUERY,
            help="Se envía al parámetro ?search= de la API de OpenAlex.",
        )
        oa_max = st.slider(
            "Número máximo de artículos",
            min_value=50,
            max_value=5000,
            value=OA_MAX_DEFAULT,
            step=50,
            help=(
                "OpenAlex entrega hasta 200 registros por página; "
                "el cliente pagina automáticamente hasta alcanzar este límite."
            ),
        )
        oa_overwrite = st.checkbox(
            "Sobreescribir caché JSONL existente",
            value=False,
            help=(
                "Si está desactivado y ya existe data/raw/openalex_raw.jsonl, "
                "los nuevos resultados se añaden al final del archivo."
            ),
        )
        submitted = st.form_submit_button("🔍 Buscar en OpenAlex", type="primary")

    # ── Ejecutar búsqueda ─────────────────────────────────────────────────────
    if submitted and oa_query.strip():
        _run_openalex_search(oa_query.strip(), oa_max, oa_overwrite)

    # ── Mostrar resultados almacenados ────────────────────────────────────────
    _show_openalex_results()


def _run_openalex_search(query: str, max_results: int, overwrite: bool):
    """Ejecuta la búsqueda en OpenAlex con barra de progreso y almacena el JSONL."""
    progress_bar  = st.progress(0.0, text="Iniciando búsqueda en OpenAlex...")
    status_text   = st.empty()
    client        = OpenAlexClient()
    all_records   = []

    try:
        Path("data/raw").mkdir(parents=True, exist_ok=True)

        if overwrite and Path(OA_JSONL_PATH).exists():
            Path(OA_JSONL_PATH).unlink()

        for batch, downloaded, total_estimated in client.fetch_with_progress(query, max_results):
            all_records.extend(batch)

            # Guardar batch incremental en JSONL
            with open(OA_JSONL_PATH, "a", encoding="utf-8") as fh:
                for record in batch:
                    import json
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            pct = min(downloaded / max(total_estimated, 1), 1.0)
            progress_bar.progress(pct, text=f"Descargados {downloaded:,} de ~{total_estimated:,} artículos")
            status_text.text(f"Última página: {len(batch)} registros")

        progress_bar.progress(1.0, text="✅ Descarga completada")
        status_text.empty()

        # Parsear al esquema canónico y guardar en session_state
        df_oa = parse_openalex_records(all_records)
        st.session_state["r1_oa_df"]    = df_oa
        st.session_state["r1_oa_query"] = query
        st.session_state["r1_oa_raw"]   = all_records

        st.success(
            f"✅ **{len(df_oa):,} artículos** descargados de OpenAlex para la query "
            f"*\"{query}\"*.  Guardados en `{OA_JSONL_PATH}`."
        )

    except Exception as exc:
        progress_bar.empty()
        st.error(f"❌ Error al consultar OpenAlex: {exc}")
        logger.exception("OpenAlex search error")


def _show_openalex_results():
    """Muestra la vista previa de los resultados de OpenAlex en session_state."""
    df_oa: pd.DataFrame | None = st.session_state.get("r1_oa_df")

    # Si no hay resultados en session pero existe JSONL en disco, ofrecer cargarlos
    if df_oa is None:
        jsonl_path = Path(OA_JSONL_PATH)
        if jsonl_path.exists():
            n_lines = sum(1 for _ in jsonl_path.open("r", encoding="utf-8") if _.strip())
            st.info(
                f"Se encontró un caché previo con **{n_lines:,} registros** en "
                f"`{OA_JSONL_PATH}`.  Pulsa el botón para cargarlos sin realizar "
                "una nueva búsqueda."
            )
            if st.button("📂 Cargar resultados del caché JSONL"):
                raw = read_jsonl(OA_JSONL_PATH)
                df_oa = parse_openalex_records(raw)
                st.session_state["r1_oa_df"]  = df_oa
                st.session_state["r1_oa_raw"] = raw
                st.rerun()
        return

    if df_oa is None or len(df_oa) == 0:
        return

    query = st.session_state.get("r1_oa_query", "")
    st.markdown(f"---")
    st.subheader(f"Vista previa — {len(df_oa):,} artículos de OpenAlex")
    if query:
        st.caption(f"Query: *\"{query}\"*")

    # Métricas rápidas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total artículos",  len(df_oa))
    m2.metric("Con abstract",     int((df_oa["abstract"].str.strip() != "").sum()))
    m3.metric("Con país",         int((df_oa["country"].str.strip() != "").sum()))
    m4.metric("Con DOI",          int((df_oa["doi"].str.strip() != "").sum()))

    # Tabla de preview (primeras 50 filas)
    preview_cols = ["title", "authors", "year", "source", "doi", "country"]
    st.dataframe(
        df_oa[preview_cols].head(50).reset_index(drop=True),
        use_container_width=True,
        height=320,
    )
    if len(df_oa) > 50:
        st.caption(f"Mostrando 50 de {len(df_oa):,} registros.")

    # Distribución por año
    with st.expander("Distribución por año de publicación", expanded=False):
        year_dist = (
            df_oa[df_oa["year"].str.strip() != ""]["year"]
            .value_counts()
            .sort_index()
        )
        st.bar_chart(year_dist)

    # Opción de limpiar
    if st.button("🗑️ Limpiar resultados de OpenAlex", help="Elimina los resultados de la sesión (no borra el JSONL en disco)."):
        st.session_state.pop("r1_oa_df",    None)
        st.session_state.pop("r1_oa_raw",   None)
        st.session_state.pop("r1_oa_query", None)
        st.rerun()


# ── Utilidades ────────────────────────────────────────────────────────────────

def _safe_parse(file_obj, parser_fn) -> pd.DataFrame:
    """
    Parsea un archivo subido por Streamlit con la función dada.
    Devuelve DataFrame vacío si el archivo es None o falla el parsing.
    """
    if file_obj is None:
        return pd.DataFrame()
    try:
        file_obj.seek(0)
        return parser_fn(file_obj)
    except Exception as exc:
        logger.warning("Error parseando archivo: %s", exc)
        return pd.DataFrame()


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serializa un DataFrame a bytes UTF-8 con BOM para compatibilidad con Excel."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue().encode("utf-8-sig")
