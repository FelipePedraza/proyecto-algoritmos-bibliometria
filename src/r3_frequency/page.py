"""
Requerimiento 3 — Interfaz Streamlit: Frecuencia de Términos
=============================================================
Funcionalidades:
  1. Carga o reutilización del dataset unificado (unified.csv del R1).
  2. Frecuencia de los 15 términos predefinidos de la categoría
     "Concepts of Generative AI in Education" en los abstracts.
  3. Extracción algorítmica (NPMI) de hasta 15 nuevos términos asociados.
  4. Evaluación de la precisión de los nuevos términos por co-ocurrencia.
  5. Explicaciones matemáticas y algorítmicas paso a paso.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.r3_frequency.algorithms import (
    CATEGORY_NAME,
    CATEGORY_TERMS,
    run_r3_pipeline,
)
from src.r3_frequency.explanations import (
    explain_frequency_counting,
    explain_npmi_algorithm,
    explain_precision_evaluation,
)

logger = logging.getLogger(__name__)

# ─── Colores para gráficas ────────────────────────────────────────────────────
_COLOR_PREDEFINED = "#4A90D9"   # azul para términos predefinidos
_COLOR_NEW        = "#E67E22"   # naranja para nuevos términos
_COLOR_PRECISION  = "#27AE60"   # verde para precisión


# ═══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

def render() -> None:
    """Renderiza la página completa del Requerimiento 3."""
    st.title("Requerimiento 3: Frecuencia de Terminos")
    st.markdown(
        f"""
        **Categoria analizada:** *{CATEGORY_NAME}*

        **Objetivo:** Calcular la frecuencia de los 15 terminos predefinidos de la
        categoria en los abstracts del corpus, extraer algoritmicamente nuevas palabras
        asociadas usando **Informacion Mutua Puntual Normalizada (NPMI)**, y evaluar la
        precision de los terminos descubiertos.
        """
    )

    # ── Sección 0: Carga de datos ────────────────────────────────────────────
    st.markdown("---")
    df = _load_dataset()
    if df is None or df.empty:
        return

    # Extraer abstracts válidos
    abstracts = [
        str(a).strip()
        for a in df.get("abstract", pd.Series(dtype=str))
        if str(a).strip() and str(a).strip().lower() not in ("nan", "none", "")
    ]

    if not abstracts:
        st.error("No se encontraron abstracts validos en el dataset.")
        return

    st.success(f"**{len(abstracts)}** abstracts disponibles para analisis.")

    # ── Configuracion del analisis ────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Configuracion del analisis")
        st.caption(
            f"Categoria: *{CATEGORY_NAME}* | "
            f"Terminos predefinidos: **{len(CATEGORY_TERMS)}**"
        )

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

        with col_cfg1:
            max_new_terms = st.slider(
                "Maximo de nuevas palabras",
                min_value=5,
                max_value=15,
                value=15,
                key="r3_max_terms",
                help="Numero maximo de nuevas palabras asociadas a extraer (max 15).",
            )

        with col_cfg2:
            min_doc_freq = st.slider(
                "Frecuencia minima de documento",
                min_value=1,
                max_value=max(1, len(abstracts) // 10),
                value=2,
                key="r3_min_df",
                help=(
                    "Un candidato debe aparecer al menos en este numero de abstracts "
                    "para ser considerado. Valores mayores reducen el ruido."
                ),
            )

        with col_cfg3:
            include_bigrams = st.checkbox(
                "Incluir bigramas (pares de palabras)",
                value=True,
                key="r3_bigrams",
                help=(
                    "Si esta activo, analiza tambien pares de palabras consecutivas "
                    "(ej. 'language model', 'text generation')."
                ),
            )

    # ── Botón de ejecución ───────────────────────────────────────────────────
    st.markdown("---")
    run_btn = st.button(
        "Ejecutar analisis de frecuencia",
        type="primary",
        key="r3_run",
    )

    if run_btn:
        with st.spinner("Analizando abstracts..."):
            result = run_r3_pipeline(
                abstracts=abstracts,
                max_new_terms=max_new_terms,
                min_doc_freq=min_doc_freq,
                include_bigrams=include_bigrams,
            )
        st.session_state["r3_result"] = result
        st.session_state["r3_abstracts"] = abstracts

    # ── Mostrar resultados si existen ────────────────────────────────────────
    if "r3_result" not in st.session_state:
        st.info("Pulsa **Ejecutar analisis de frecuencia** para comenzar.")
        return

    result = st.session_state["r3_result"]

    # ── Métricas globales ────────────────────────────────────────────────────
    st.markdown("---")
    _render_global_stats(result)

    # ── Sección 1: Términos predefinidos ─────────────────────────────────────
    st.markdown("---")
    st.header("1. Frecuencia de Terminos Predefinidos")
    st.markdown(
        "Frecuencia de cada uno de los **15 términos** de la categoría "
        f"*{CATEGORY_NAME}* en el corpus de abstracts."
    )
    _render_predefined_frequencies(result)

    with st.expander("Explicacion del algoritmo de conteo", expanded=False):
        st.markdown(
            explain_frequency_counting(
                result["predefined_frequencies"],
                result["n_abstracts"],
            )
        )

    # ── Sección 2: Nuevas palabras extraídas ─────────────────────────────────
    st.markdown("---")
    st.header("2. Extraccion de Nuevas Palabras Asociadas (NPMI)")
    st.markdown(
        "Nuevas palabras descubiertas algorítmicamente que están estadísticamente "
        "asociadas a la categoría. Ordenadas por **NPMI** (mayor = más asociadas)."
    )
    _render_new_terms(result)

    with st.expander("Explicacion del algoritmo NPMI", expanded=False):
        st.markdown(explain_npmi_algorithm(result))

    # ── Sección 3: Evaluación de precisión ───────────────────────────────────
    st.markdown("---")
    st.header("3. Evaluacion de Precision")
    st.markdown(
        "Para cada nuevo término, la **precisión** mide en qué fracción de los "
        "documentos donde aparece también se encuentran conceptos de la categoría."
    )
    _render_precision(result)

    with st.expander("Explicacion de la metrica de precision", expanded=False):
        st.markdown(
            explain_precision_evaluation(
                result["new_terms"],
                result["df_cat"],
                result["n_abstracts"],
            )
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIONES INTERNAS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_dataset() -> pd.DataFrame | None:
    """Carga el dataset desde la sesión del R1 o desde un archivo CSV."""

    if "r1_unified" in st.session_state:
        st.success("Dataset del Requerimiento 1 disponible en sesion.")
        use_r1 = st.checkbox("Usar dataset del R1", value=True, key="r3_use_r1")
        if use_r1:
            df = st.session_state["r1_unified"]
            st.caption(f"{len(df)} articulos disponibles.")
            return df

    st.info(
        "Sube el archivo `unified.csv` generado en el Requerimiento 1, "
        "o cualquier CSV con al menos la columna `abstract`."
    )
    uploaded = st.file_uploader("Sube unified.csv", type=["csv"], key="r3_upload")

    if uploaded is None:
        return None

    try:
        df = pd.read_csv(uploaded, encoding="utf-8-sig", dtype=str).fillna("")
        df = df.apply(
            lambda col: col.map(
                lambda v: "" if str(v).strip().lower() in ("nan", "none") else str(v).strip()
            )
        )
        if "abstract" not in df.columns:
            st.error("El CSV debe contener la columna `abstract`.")
            return None
        st.success(f"{uploaded.name} cargado — {len(df)} articulos.")
        return df
    except Exception as exc:
        st.error(f"Error al leer el archivo: {exc}")
        return None


def _render_global_stats(result: dict) -> None:
    """Muestra las métricas globales del análisis."""
    n = result["n_abstracts"]
    n_cat = result["n_category_docs"]
    pct = result["category_coverage_pct"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Abstracts analizados", n)
    col2.metric(
        "Abstracts con min. 1 termino de categoria",
        n_cat,
        delta=f"{pct:.1f}%",
    )
    col3.metric("Terminos predefinidos", len(CATEGORY_TERMS))
    col4.metric("Nuevas palabras encontradas", len(result["new_terms"]))

    if pct < 10:
        st.warning(
            f"Solo el {pct:.1f}% de los abstracts contienen terminos de la categoria. "
            "El NPMI puede no ser representativo con tan pocos documentos relevantes."
        )
    elif pct > 90:
        st.info(
            f"El {pct:.1f}% de los abstracts contienen terminos de la categoria. "
            "Esto indica un corpus muy específico al dominio. "
            "El NPMI puede converger a valores cercanos a 0 para términos muy frecuentes."
        )


def _render_predefined_frequencies(result: dict) -> None:
    """Visualiza la frecuencia de los términos predefinidos."""
    import matplotlib.pyplot as plt
    import numpy as np

    freq = result["predefined_frequencies"]
    n = result["n_abstracts"]

    # Ordenar por frecuencia de documento (mayor primero)
    sorted_items = sorted(
        freq.items(), key=lambda x: -x[1]["doc_frequency"]
    )
    terms = [item[0] for item in sorted_items]
    doc_freqs = [item[1]["doc_frequency"] for item in sorted_items]
    total_occs = [item[1]["total_occurrences"] for item in sorted_items]
    pcts = [item[1]["doc_frequency_pct"] for item in sorted_items]

    # ── Gráfico de barras horizontales ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, max(6, len(terms) * 0.48)))

    norm_vals = [v / max(doc_freqs) if max(doc_freqs) > 0 else 0 for v in doc_freqs]
    cmap = plt.cm.Blues
    colors = [cmap(0.35 + 0.55 * v) for v in norm_vals]

    bars = ax.barh(
        range(len(terms)),
        doc_freqs,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Frecuencia de documento (n.° de abstracts)", fontsize=10)
    ax.set_title(
        f"Frecuencia de Términos Predefinidos — {CATEGORY_NAME}\n"
        f"(N = {n} abstracts)",
        fontsize=11,
        pad=10,
    )

    # Etiquetas en las barras
    for bar, val, pct in zip(bars, doc_freqs, pcts):
        label = f"{val}  ({pct:.1f}%)"
        ax.text(
            bar.get_width() + max(doc_freqs) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8,
            color="#333333",
        )

    # Línea de referencia: mediana
    median_df = sorted(doc_freqs)[len(doc_freqs) // 2]
    ax.axvline(median_df, color="#E74C3C", linestyle="--", linewidth=1,
               label=f"Mediana: {median_df}")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(doc_freqs) * 1.20)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Métricas rápidas ──────────────────────────────────────────────────────
    present = sum(1 for v in doc_freqs if v > 0)
    most_freq = sorted_items[0][0]
    least_freq = sorted_items[-1][0] if sorted_items[-1][1]["doc_frequency"] == 0 else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Términos presentes en el corpus", f"{present} / {len(CATEGORY_TERMS)}")
    col2.metric("Término más frecuente", most_freq, delta=f"{doc_freqs[0]} docs")
    if least_freq:
        col3.metric("Términos sin apariciones", least_freq)
    else:
        col3.metric(
            "Término menos frecuente",
            sorted_items[-1][0],
            delta=f"{sorted_items[-1][1]['doc_frequency']} docs",
        )

    # ── Tabla detallada ───────────────────────────────────────────────────────
    with st.expander("Tabla detallada de frecuencias", expanded=True):
        df_display = pd.DataFrame(
            [
                {
                    "Término": t,
                    "Frecuencia de documento": data["doc_frequency"],
                    "Ocurrencias totales": data["total_occurrences"],
                    "% Abstracts": f"{data['doc_frequency_pct']:.1f}%",
                    "Estado": (
                        "Presente"
                        if data["doc_frequency"] > 0
                        else "Ausente"
                    ),
                }
                for t, data in sorted_items
            ]
        )

        def _color_status(val):
            if "Presente" in str(val):
                return "color: #27AE60; font-weight: bold"
            return "color: #AAAAAA"

        st.dataframe(
            df_display.style.map(
                _color_status, subset=["Estado"]
            ),
            use_container_width=True,
            hide_index=True,
        )


def _render_new_terms(result: dict) -> None:
    """Visualiza los nuevos términos extraídos con sus métricas NPMI."""
    import matplotlib.pyplot as plt
    import numpy as np

    new_terms = result["new_terms"]

    if not new_terms:
        st.warning(
            "No se encontraron nuevos terminos con los parametros actuales. "
            "Prueba reduciendo la **Frecuencia minima de documento** en la configuracion."
        )
        return

    # Re-ordenar por NPMI para el gráfico (la lista viene ordenada por precision)
    terms_by_npmi = sorted(new_terms, key=lambda x: -x["npmi"])

    terms = [item["term"] for item in terms_by_npmi]
    npmi_vals = [item["npmi"] for item in terms_by_npmi]
    df_vals = [item["doc_frequency"] for item in terms_by_npmi]
    pct_vals = [item["doc_frequency_pct"] for item in terms_by_npmi]

    n_terms = len(terms)
    n = result["n_abstracts"]

    # ── Gráfico NPMI ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, max(5, n_terms * 0.42)),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # Panel izquierdo: barras NPMI
    cmap_orange = plt.cm.YlOrBr
    norm_npmi = [v / max(npmi_vals) if max(npmi_vals) > 0 else 0 for v in npmi_vals]
    colors_npmi = [cmap_orange(0.35 + 0.55 * v) for v in norm_npmi]

    bars1 = ax1.barh(
        range(n_terms),
        npmi_vals,
        color=colors_npmi,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.set_yticks(range(n_terms))
    ax1.set_yticklabels(terms, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("NPMI (Información Mutua Puntual Normalizada)", fontsize=9)
    ax1.set_title("Score NPMI por término extraído", fontsize=10)
    ax1.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    for bar, val in zip(bars1, npmi_vals):
        ax1.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=7.5,
        )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim(0, max(npmi_vals) * 1.22)

    # Panel derecho: frecuencia de documento
    cmap_blue = plt.cm.Blues
    norm_df = [v / max(df_vals) if max(df_vals) > 0 else 0 for v in df_vals]
    colors_df = [cmap_blue(0.3 + 0.6 * v) for v in norm_df]

    bars2 = ax2.barh(
        range(n_terms),
        df_vals,
        color=colors_df,
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.set_yticks(range(n_terms))
    ax2.set_yticklabels([""] * n_terms)
    ax2.invert_yaxis()
    ax2.set_xlabel("Frecuencia de documento", fontsize=9)
    ax2.set_title(f"Docs que contienen\nel término (de {n})", fontsize=10)

    for bar, val, pct in zip(bars2, df_vals, pct_vals):
        ax2.text(
            bar.get_width() + max(df_vals) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val}  ({pct:.1f}%)",
            va="center",
            fontsize=7.5,
        )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlim(0, max(df_vals) * 1.30)

    fig.suptitle(
        f"Nuevas Palabras Asociadas — {CATEGORY_NAME}",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Tabla detallada ───────────────────────────────────────────────────────
    with st.expander("Tabla detallada de nuevas palabras", expanded=True):
        df_display = pd.DataFrame(
            [
                {
                    "Rango NPMI": i + 1,
                    "Término": item["term"],
                    "NPMI": f"{item['npmi']:.4f}",
                    "Freq. documento": item["doc_frequency"],
                    "Ocurrencias": item["total_occurrences"],
                    "% Abstracts": f"{item['doc_frequency_pct']:.1f}%",
                    "Tipo": "Bigrama" if " " in item["term"] else "Unigrama",
                }
                for i, item in enumerate(terms_by_npmi)
            ]
        )
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Distribución unigrama / bigrama
        n_uni = sum(1 for t in new_terms if " " not in t["term"])
        n_bi = len(new_terms) - n_uni
        st.caption(
            f"Composición: **{n_uni} unigramas** · **{n_bi} bigramas**"
        )


def _render_precision(result: dict) -> None:
    """Visualiza la evaluación de precisión de los nuevos términos."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    new_terms = result.get("new_terms", [])

    if not new_terms:
        st.info("Sin términos extraídos para evaluar.")
        return

    # Los terms ya vienen ordenados por precisión desde el pipeline
    terms = [item["term"] for item in new_terms]
    prec_vals = [item["precision_pct"] for item in new_terms]
    npmi_vals = [item["npmi"] for item in new_terms]
    labels = [item["precision_label"] for item in new_terms]

    n_terms = len(terms)

    # Colores por nivel de precisión
    _PREC_COLORS = {
        "Muy relevante":              "#27AE60",
        "Moderadamente relevante":    "#F39C12",
        "Débilmente relevante":       "#E67E22",
        "Poco relevante":             "#E74C3C",
    }
    colors = [_PREC_COLORS.get(lbl, "#AAAAAA") for lbl in labels]

    # ── Gráfico de barras horizontales de precisión ───────────────────────────
    fig, ax = plt.subplots(figsize=(11, max(5, n_terms * 0.45)))

    bars = ax.barh(
        range(n_terms),
        prec_vals,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.65,
    )

    ax.set_yticks(range(n_terms))
    ax.set_yticklabels(terms, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Precisión de co-ocurrencia (%)", fontsize=10)
    ax.set_title(
        "Precisión de los Nuevos Términos Extraídos\n"
        "(fracción de docs donde el término co-ocurre con la categoría)",
        fontsize=10,
        pad=10,
    )
    ax.set_xlim(0, 115)

    # Etiquetas en las barras
    for bar, val, lbl in zip(bars, prec_vals, labels):
        ax.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%  — {lbl}",
            va="center",
            fontsize=8,
        )

    # Líneas de referencia por nivel
    for threshold, color, text in [
        (75, "#27AE60", "Muy relevante (75%)"),
        (50, "#F39C12", "Moderadamente (50%)"),
        (25, "#E74C3C", "Débilmente (25%)"),
    ]:
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1.0,
                   alpha=0.7, label=text)

    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Scatter NPMI vs Precisión ─────────────────────────────────────────────
    with st.expander("Grafico NPMI vs Precision", expanded=False):
        fig2, ax2 = plt.subplots(figsize=(9, 5))

        scatter_colors = [_PREC_COLORS.get(lbl, "#AAAAAA") for lbl in labels]
        sc = ax2.scatter(
            npmi_vals,
            prec_vals,
            c=scatter_colors,
            s=120,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

        # Etiquetas de los puntos (evitar superposición: solo mostrar algunos)
        for i, (term, npmi, prec) in enumerate(zip(terms, npmi_vals, prec_vals)):
            ax2.annotate(
                term,
                (npmi, prec),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7.5,
                color="#333333",
            )

        ax2.axhline(75, color="#27AE60", linestyle="--", linewidth=1, alpha=0.6, label="75% — Muy relevante")
        ax2.axhline(50, color="#F39C12", linestyle="--", linewidth=1, alpha=0.6, label="50% — Moderado")
        ax2.set_xlabel("NPMI", fontsize=10)
        ax2.set_ylabel("Precisión (%)", fontsize=10)
        ax2.set_title("Relación entre NPMI y Precisión de Co-ocurrencia", fontsize=11)
        ax2.legend(fontsize=8, loc="lower right")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_ylim(0, 108)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption(
            "Cada punto es un término extraído. "
            "Idealmente los términos relevantes se ubican en la esquina superior derecha "
            "(alto NPMI y alta precisión)."
        )

    # ── Tabla resumen de precisión ────────────────────────────────────────────
    with st.expander("Tabla completa de resultados", expanded=True):
        df_display = pd.DataFrame(
            [
                {
                    "Rank Precisión": i + 1,
                    "Término": item["term"],
                    "Precisión (%)": f"{item['precision_pct']:.1f}%",
                    "Nivel": item["precision_label"],
                    "NPMI": f"{item['npmi']:.4f}",
                    "Rank NPMI": item.get("npmi_rank", "—"),
                    "Freq. doc.": item["doc_frequency"],
                    "Co-ocurre con cat.": item.get("docs_with_category", "—"),
                }
                for i, item in enumerate(new_terms)
            ]
        )

        def _highlight_precision(val: str) -> str:
            """Colorea las celdas de nivel de precisión."""
            palette = {
                "Muy relevante": "background-color: rgba(39,174,96,0.2); color:#1D8A4E; font-weight:bold",
                "Moderadamente relevante": "background-color: rgba(243,156,18,0.2); color:#9A6000",
                "Débilmente relevante": "background-color: rgba(230,126,34,0.15); color:#7D4E1B",
                "Poco relevante": "background-color: rgba(231,76,60,0.15); color:#922B21",
            }
            return palette.get(val, "")

        st.dataframe(
            df_display.style.map(_highlight_precision, subset=["Nivel"]),
            use_container_width=True,
            hide_index=True,
        )

    # ── Conclusión ────────────────────────────────────────────────────────────
    n_high = sum(1 for t in new_terms if t["precision"] >= 0.75)
    n_total = len(new_terms)
    overall_precision = (
        sum(t["precision"] for t in new_terms) / n_total
        if n_total > 0 else 0.0
    )

    quality_msg = (
        "excelente" if overall_precision >= 0.75
        else "buena" if overall_precision >= 0.50
        else "moderada" if overall_precision >= 0.25
        else "baja"
    )

    st.success(
        f"**Conclusión:** El algoritmo NPMI extrajo **{n_total}** nuevas palabras asociadas. "
        f"**{n_high}** de ellas ({n_high/n_total*100:.0f}%) son *muy relevantes* (precisión ≥ 75%). "
        f"La precisión media global es **{overall_precision*100:.1f}%**, lo que indica "
        f"una calidad de extracción **{quality_msg}**."
    )