"""
Requerimiento 5 — Interfaz Streamlit: Visualización Científica
=================================================================
Funcionalidades:
  1. Carga/reutilización del dataset unificado (unified.csv del R1).
  2. Mapa de calor con distribución geográfica del primer autor.
  3. Nube de palabras dinámica (abstracts + keywords).
  4. Línea temporal de publicaciones por año y por revista.
  5. Exportación de las tres visualizaciones a PDF.
"""

from __future__ import annotations

import logging
import pandas as pd
import streamlit as st

from src.r5_visualization.algorithms import (
    count_countries,
    build_geo_dataframe,
    compute_word_frequencies,
    compute_timeline,
    get_top_sources,
    make_wordcloud_figure,
    make_timeline_figure,
    generate_pdf_report,
    detect_country_column,
)
from src.r5_visualization.explanations import (
    explain_geo_heatmap,
    explain_wordcloud,
    explain_timeline,
    explain_pdf_export,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de configuración de la UI
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_TOP_WORDS = 100
_DEFAULT_TOP_SOURCES = 8
_DEFAULT_KEYWORD_WEIGHT = 3


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def render() -> None:
    """Renderiza la página completa del Requerimiento 5."""

    st.title("Requerimiento 5: Visualización Científica")
    st.markdown(
        """
        **Objetivo:** Analizar visualmente la producción científica sobre
        *"Generative Artificial Intelligence"* mediante tres visualizaciones:
        mapa de calor geográfico, nube de palabras dinámica y línea temporal
        de publicaciones — todo exportable a PDF.
        """
    )

    # ── Sección 0: Carga de datos ────────────────────────────────────────────
    st.markdown("---")
    df = _load_dataset()
    if df is None or df.empty:
        return

    st.success(f"✅ Dataset cargado: **{len(df)} artículos** disponibles.")
    _show_dataset_summary(df)

    # ── Configuración global ─────────────────────────────────────────────────
    st.markdown("---")
    st.header("⚙️ Configuración")

    with st.expander("Parámetros de visualización", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            top_words = st.slider(
                "Máximo de palabras en la nube",
                min_value=20, max_value=200,
                value=_DEFAULT_TOP_WORDS, step=10,
                key="r5_top_words",
            )
        with col2:
            top_sources = st.slider(
                "Top fuentes en línea temporal",
                min_value=3, max_value=15,
                value=_DEFAULT_TOP_SOURCES,
                key="r5_top_sources",
            )
        with col3:
            kw_weight = st.slider(
                "Peso de keywords vs abstracts",
                min_value=1, max_value=10,
                value=_DEFAULT_KEYWORD_WEIGHT,
                key="r5_kw_weight",
                help="Cada keyword suma este número de veces (vs 1 para palabras de abstracts).",
            )

        col4, col5 = st.columns(2)
        with col4:
            use_abstracts = st.checkbox("Incluir abstracts", value=True, key="r5_use_abs")
        with col5:
            use_keywords = st.checkbox("Incluir keywords", value=True, key="r5_use_kw")

        extra_sw_raw = st.text_input(
            "Stop-words adicionales (separadas por comas)",
            value="",
            key="r5_extra_sw",
            placeholder="p.ej.: generative, artificial, intelligence",
        )
        extra_stopwords = {
            w.strip().lower()
            for w in extra_sw_raw.split(",")
            if w.strip()
        }

    # ── Pre-computar datos ────────────────────────────────────────────────────
    with st.spinner("Procesando datos…"):
        country_counts = count_countries(df)
        freq_dict = compute_word_frequencies(
            df,
            top_n=top_words,
            use_abstracts=use_abstracts,
            use_keywords=use_keywords,
            extra_stopwords=extra_stopwords,
            keyword_weight=kw_weight,
        )
        df_by_year, df_by_year_source = compute_timeline(df)

    # ── Sección 1: Mapa de calor geográfico ──────────────────────────────────
    st.markdown("---")
    st.header("1. 🗺️ Mapa de Calor Geográfico")
    st.markdown(
        "Distribución de publicaciones por país del **primer autor**. "
        "El color indica el volumen de publicaciones: de azul claro (pocas) a rojo oscuro (muchas)."
    )

    _render_geo_section(df, country_counts)

    with st.expander("📖 Explicación matemática y algorítmica", expanded=False):
        st.markdown(explain_geo_heatmap())

    # ── Sección 2: Nube de palabras ───────────────────────────────────────────
    st.markdown("---")
    st.header("2. ☁️ Nube de Palabras")
    st.markdown(
        "Términos más frecuentes en abstracts y keywords del corpus. "
        "El **tamaño** de cada término es proporcional a su frecuencia acumulada. "
        "La nube se actualiza automáticamente al cambiar los parámetros o el dataset."
    )

    _render_wordcloud_section(freq_dict)

    with st.expander("📖 Explicación matemática y algorítmica", expanded=False):
        st.markdown(explain_wordcloud())

    # ── Sección 3: Línea temporal ─────────────────────────────────────────────
    st.markdown("---")
    st.header("3. 📅 Línea Temporal de Publicaciones")
    st.markdown(
        "Evolución de publicaciones por año y distribución entre las "
        f"**top {top_sources}** revistas/conferencias del corpus."
    )

    _render_timeline_section(df_by_year, df_by_year_source, top_sources)

    with st.expander("📖 Explicación matemática y algorítmica", expanded=False):
        st.markdown(explain_timeline())

    # ── Sección 4: Exportar PDF ───────────────────────────────────────────────
    st.markdown("---")
    st.header("4. 📤 Exportar a PDF")
    _render_export_section(
        country_counts, freq_dict, df_by_year, df_by_year_source, len(df)
    )

    with st.expander("📖 Detalles del proceso de exportación", expanded=False):
        st.markdown(explain_pdf_export())


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIONES INTERNAS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_dataset() -> pd.DataFrame | None:
    """Carga el dataset desde la sesión del R1 o desde un archivo CSV."""

    # Opción 1: reutilizar resultado del R1 (ya en session_state)
    if "r1_unified" in st.session_state:
        st.success("✅ Dataset del Requerimiento 1 disponible en sesión.")
        use_r1 = st.checkbox("Usar dataset del R1", value=True, key="r5_use_r1")
        if use_r1:
            return st.session_state["r1_unified"]

    # Opción 2: cargar CSV manualmente
    st.info(
        "Sube el archivo `unified.csv` generado en el Requerimiento 1, "
        "o cualquier CSV con columnas: `title`, `authors`, `year`, `source`, "
        "`abstract`, `keywords`."
    )
    uploaded = st.file_uploader(
        "Sube unified.csv", type=["csv"], key="r5_upload"
    )

    if uploaded is None:
        return None

    try:
        df = pd.read_csv(uploaded, encoding="utf-8-sig", dtype=str).fillna("")
        # Normalizar: quitar "nan" literales
        df = df.apply(
            lambda col: col.map(
                lambda v: "" if str(v).strip().lower() in ("nan", "none") else str(v).strip()
            )
        )
        if "title" not in df.columns:
            st.error("❌ El CSV debe tener al menos la columna `title`.")
            return None
        st.success(f"✅ {uploaded.name} cargado — {len(df)} artículos")
        return df
    except Exception as exc:
        st.error(f"Error al leer el archivo: {exc}")
        return None


def _show_dataset_summary(df: pd.DataFrame) -> None:
    """Muestra métricas rápidas del dataset."""
    col1, col2, col3, col4 = st.columns(4)

    n_with_abstract = int((df.get("abstract", pd.Series(dtype=str)) != "").sum())
    n_with_keywords = int((df.get("keywords", pd.Series(dtype=str)) != "").sum())

    years_valid = pd.to_numeric(df.get("year", pd.Series(dtype=str)), errors="coerce").dropna()
    year_range = (
        f"{int(years_valid.min())} – {int(years_valid.max())}"
        if not years_valid.empty else "—"
    )

    geo_col = detect_country_column(df)
    geo_status = f"✅ Columna: `{geo_col}`" if geo_col else "⚠️ Sin datos de país"

    col1.metric("📄 Artículos", len(df))
    col2.metric("📝 Con abstract", n_with_abstract)
    col3.metric("🔑 Con keywords", n_with_keywords)
    col4.metric("📅 Rango de años", year_range)

    if not geo_col:
        st.warning(
            "⚠️ **Sin datos geográficos:** el dataset no contiene columnas de país ni afiliación. "
            "El mapa de calor mostrará un estado vacío. Para habilitarlo, agrega una columna "
            "`country` o `affiliations` al CSV con el país del primer autor."
        )


def _render_geo_section(df: pd.DataFrame, country_counts: dict[str, int]) -> None:
    """Renderiza el mapa de calor geográfico."""

    if not country_counts:
        st.info(
            "📍 No se encontraron datos de país en el dataset. "
            "Agrega una columna `country` o `affiliations` al CSV "
            "para ver el mapa de calor."
        )
        _show_demo_geo_option()
        return

    # Métricas
    total_with_country = sum(country_counts.values())
    col1, col2, col3 = st.columns(3)
    col1.metric("🌍 Países identificados", len(country_counts))
    col2.metric("📄 Artículos con país", total_with_country)
    col3.metric(
        "🥇 País líder",
        max(country_counts, key=country_counts.get),
        delta=str(max(country_counts.values())),
    )

    # Choropleth interactivo con Plotly
    try:
        import plotly.express as px

        geo_df = build_geo_dataframe(country_counts)
        # Filtrar filas sin código ISO válido para el mapa
        geo_df_valid = geo_df[geo_df["iso_a3"] != ""]
        geo_df_niso = geo_df[geo_df["iso_a3"] == ""]

        fig = px.choropleth(
            geo_df_valid,
            locations="iso_a3",
            color="count",
            hover_name="country",
            hover_data={"count": True, "iso_a3": False},
            color_continuous_scale="YlOrRd",
            labels={"count": "Publicaciones"},
            title="Distribución Geográfica del Primer Autor",
        )
        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True,
                     coastlinecolor="lightgray", projection_type="natural earth"),
            coloraxis_colorbar=dict(title="Publicaciones"),
            margin=dict(l=0, r=0, t=40, b=0),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        if not geo_df_niso.empty:
            st.caption(
                f"⚠️ {len(geo_df_niso)} países sin código ISO-3 no aparecen en el mapa: "
                + ", ".join(geo_df_niso["country"].tolist())
            )

    except ImportError:
        st.warning("Plotly no disponible — mostrando tabla de datos geográficos.")

    # Tabla de países
    with st.expander("📊 Ver tabla de países", expanded=False):
        geo_df_table = build_geo_dataframe(country_counts)
        geo_df_table.index = range(1, len(geo_df_table) + 1)
        geo_df_table.columns = ["País", "ISO-3", "Publicaciones"]
        st.dataframe(geo_df_table, use_container_width=True)


def _show_demo_geo_option() -> None:
    """Muestra un modo demo del mapa con datos de muestra."""
    if st.checkbox("🔬 Ver demostración con datos de ejemplo", key="r5_geo_demo"):
        demo_data = {
            "United States": 42, "China": 38, "Germany": 15,
            "United Kingdom": 12, "India": 11, "South Korea": 9,
            "France": 8, "Canada": 7, "Australia": 6, "Brazil": 5,
            "Japan": 5, "Italy": 4, "Netherlands": 4, "Spain": 3,
            "Singapore": 3,
        }
        st.info("📍 Datos de demostración — no corresponden al dataset cargado.")

        try:
            import plotly.express as px
            from src.r5_visualization.algorithms import build_geo_dataframe

            geo_df = build_geo_dataframe(demo_data)
            geo_df_valid = geo_df[geo_df["iso_a3"] != ""]

            fig = px.choropleth(
                geo_df_valid,
                locations="iso_a3",
                color="count",
                hover_name="country",
                color_continuous_scale="YlOrRd",
                labels={"count": "Publicaciones"},
                title="[DEMO] Distribución Geográfica del Primer Autor",
            )
            fig.update_layout(
                geo=dict(showframe=False, showcoastlines=True,
                         coastlinecolor="lightgray",
                         projection_type="natural earth"),
                margin=dict(l=0, r=0, t=40, b=0),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            import pandas as pd
            df_demo = pd.DataFrame(
                sorted(demo_data.items(), key=lambda x: -x[1]),
                columns=["País", "Publicaciones"],
            )
            st.dataframe(df_demo, use_container_width=True, hide_index=True)


def _render_wordcloud_section(freq_dict: dict[str, int]) -> None:
    """Renderiza la nube de palabras."""

    if not freq_dict:
        st.info("Sin datos de texto suficientes para generar la nube de palabras.")
        return

    # Métricas
    top5 = list(freq_dict.items())[:5]
    col1, col2, col3 = st.columns(3)
    col1.metric("📝 Términos únicos", len(freq_dict))
    col2.metric("🔤 Término más frecuente", top5[0][0], delta=str(top5[0][1]))
    col3.metric(
        "📊 Top 5 frecuencia media",
        f"{sum(v for _, v in top5) / len(top5):.0f}",
    )

    # Nube de palabras
    try:
        from wordcloud import WordCloud  # noqa: F401 — sólo verificar import
        fig = make_wordcloud_figure(
            freq_dict, title="Nube de Palabras — Abstracts y Keywords"
        )
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    except ImportError:
        st.warning(
            "La librería `wordcloud` no está instalada. "
            "Mostrando frecuencias como tabla y gráfico de barras."
        )
        _render_frequency_fallback(freq_dict)

    # Top 30 palabras en tabla expandible
    with st.expander("📊 Ver tabla de frecuencias (top 30)", expanded=False):
        import pandas as pd
        top30 = list(freq_dict.items())[:30]
        df_freq = pd.DataFrame(top30, columns=["Término", "Frecuencia"])
        df_freq.index = range(1, len(df_freq) + 1)
        st.dataframe(df_freq, use_container_width=True)


def _render_frequency_fallback(freq_dict: dict[str, int]) -> None:
    """Fallback: barras de frecuencia cuando wordcloud no está disponible."""
    import matplotlib.pyplot as plt
    import numpy as np

    top = list(freq_dict.items())[:25]
    words, counts = zip(*top)
    cmap = plt.cm.viridis
    colors = [cmap(0.2 + 0.7 * (c / max(counts))) for c in counts]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(len(words)), counts, color=colors, edgecolor="white")
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Frecuencia")
    ax.set_title("Términos más frecuentes en Abstracts y Keywords", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_timeline_section(
    df_by_year: pd.DataFrame,
    df_by_year_source: pd.DataFrame,
    top_sources: int,
) -> None:
    """Renderiza la línea temporal de publicaciones."""

    if df_by_year.empty:
        st.info("Sin datos de año disponibles en el dataset.")
        return

    # Métricas
    total_pubs = int(df_by_year["count"].sum())
    peak_year = int(df_by_year.loc[df_by_year["count"].idxmax(), "year"])
    peak_count = int(df_by_year["count"].max())
    n_years = len(df_by_year)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📚 Total publicaciones", total_pubs)
    col2.metric("📅 Año pico", str(peak_year), delta=f"{peak_count} pubs")
    col3.metric("🗓️ Años cubiertos", n_years)
    if not df_by_year_source.empty:
        n_sources = df_by_year_source["source_short"].nunique()
        col4.metric("📰 Fuentes únicas", n_sources)

    # ── Visualización interactiva con Plotly ──────────────────────────────────
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # Panel 1: total por año
        fig_year = go.Figure()
        fig_year.add_trace(go.Bar(
            x=df_by_year["year"],
            y=df_by_year["count"],
            name="Publicaciones",
            marker_color="#42A5F5",
            opacity=0.7,
        ))
        fig_year.add_trace(go.Scatter(
            x=df_by_year["year"],
            y=df_by_year["count"],
            mode="lines+markers",
            name="Tendencia",
            line=dict(color="#1565C0", width=2.5),
            marker=dict(size=7),
        ))
        fig_year.update_layout(
            title="Publicaciones por Año",
            xaxis_title="Año",
            yaxis_title="Número de publicaciones",
            xaxis=dict(tickmode="linear", dtick=1),
            legend=dict(orientation="h", y=1.08),
            hovermode="x unified",
            height=380,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_year, use_container_width=True)

        # Panel 2: por fuente (si hay datos)
        if not df_by_year_source.empty:
            top_src_names = get_top_sources(df_by_year_source, top_n=top_sources)
            df_filtered = df_by_year_source[
                df_by_year_source["source_short"].isin(top_src_names)
            ]

            fig_src = px.bar(
                df_filtered,
                x="year",
                y="count",
                color="source_short",
                title=f"Publicaciones por Revista/Conferencia (Top {top_sources})",
                labels={
                    "year": "Año",
                    "count": "Publicaciones",
                    "source_short": "Fuente",
                },
                barmode="stack",
                height=420,
            )
            fig_src.update_layout(
                xaxis=dict(tickmode="linear", dtick=1),
                legend=dict(
                    orientation="v", x=1.01, y=0.5,
                    font=dict(size=10),
                ),
                margin=dict(t=60, b=40, r=20),
            )
            st.plotly_chart(fig_src, use_container_width=True)

    except ImportError:
        # Fallback: matplotlib estático
        st.warning("Plotly no disponible — mostrando versión estática.")
        fig = make_timeline_figure(df_by_year, df_by_year_source, top_sources)
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    # Tabla detallada
    with st.expander("📊 Ver datos por año y fuente", expanded=False):
        if not df_by_year_source.empty:
            st.dataframe(df_by_year_source, use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_by_year, use_container_width=True, hide_index=True)


def _render_export_section(
    country_counts: dict[str, int],
    freq_dict: dict[str, int],
    df_by_year: pd.DataFrame,
    df_by_year_source: pd.DataFrame,
    n_articles: int,
) -> None:
    """Renderiza el botón de exportación a PDF."""

    st.markdown(
        "El PDF incluye una **portada** con estadísticas resumen y las "
        "tres visualizaciones en páginas independientes."
    )

    col_btn, col_info = st.columns([1, 3])

    with col_btn:
        generate_btn = st.button(
            "🖨️ Generar PDF",
            type="primary",
            key="r5_gen_pdf",
        )

    with col_info:
        st.caption(
            f"📄 Contenido: portada + 3 páginas de visualizaciones  \n"
            f"🌍 {len(country_counts)} países · "
            f"☁️ {len(freq_dict)} términos · "
            f"📅 {len(df_by_year)} años"
        )

    if generate_btn:
        with st.spinner("Generando PDF…"):
            try:
                pdf_bytes = generate_pdf_report(
                    country_counts=country_counts,
                    freq_dict=freq_dict,
                    df_by_year=df_by_year,
                    df_by_year_source=df_by_year_source,
                    n_articles=n_articles,
                )
                st.session_state["r5_pdf_bytes"] = pdf_bytes
                st.success("✅ PDF generado correctamente.")
            except Exception as exc:
                st.error(f"Error al generar el PDF: {exc}")
                logger.exception("Error generando PDF en R5")

    if "r5_pdf_bytes" in st.session_state:
        st.download_button(
            label="⬇️ Descargar PDF",
            data=st.session_state["r5_pdf_bytes"],
            file_name="r5_visualizacion_bibliometrica.pdf",
            mime="application/pdf",
            key="r5_download_pdf",
        )
        st.caption(
            f"Tamaño: {len(st.session_state['r5_pdf_bytes']) / 1024:.1f} KB"
        )
