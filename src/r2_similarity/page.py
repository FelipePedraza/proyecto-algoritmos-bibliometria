"""
Requerimiento 2 — Interfaz Streamlit
======================================
Permite al usuario:
  1. Cargar o reutilizar el dataset unificado (unified.csv del R1).
  2. Seleccionar dos artículos para comparar sus abstracts.
  3. Elegir algoritmos de similitud a ejecutar.
  4. Ver resultados con explicaciones paso a paso.
  5. Generar una matriz de similitud NxN (opcional).
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.r2_similarity.algorithms import ALGORITHMS
from src.r2_similarity.explanations import EXPLANATION_FUNCS

logger = logging.getLogger(__name__)

# Nombres más amigables para los algoritmos
ALGO_LABELS = {
    "levenshtein": "1. Distancia de Levenshtein (edición)",
    "jaccard": "2. Similitud de Jaccard (conjuntos)",
    "cosine_tfidf": "3. Coseno TF-IDF (vectorización)",
    "hamming": "4. Distancia de Hamming (edición)",
    "sbert": "5. Sentence-BERT (IA — Transformers)",
    "spacy": "6. spaCy Word Vectors (IA — GloVe)",
}

CLASSICAL_KEYS = ["levenshtein", "jaccard", "cosine_tfidf", "hamming"]
AI_KEYS = ["sbert", "spacy"]


def render():
    """Renderiza la página completa del Requerimiento 2."""
    st.title("Requerimiento 2: Similitud Textual")
    st.markdown(
        """
        **Objetivo:** Comparar abstracts de artículos científicos usando
        **4 algoritmos clásicos** y **2 algoritmos con modelos de IA**.
        Cada algoritmo incluye una explicación matemática y algorítmica paso a paso.
        """
    )

    # ── Carga del dataset ────────────────────────────────────────────────────
    df = _load_dataset()
    if df is None or df.empty:
        return

    # ── Selección de artículos ───────────────────────────────────────────────
    st.header("1. Selección de artículos")

    # Crear etiquetas descriptivas para el selector
    df["_label"] = df.apply(
        lambda r: f"[{r.get('source_db', '?')}] {r['title'][:80]}{'…' if len(r['title']) > 80 else ''}",
        axis=1,
    )
    labels = df["_label"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Artículo A")
        idx_a = st.selectbox("Selecciona el primer artículo", range(len(labels)),
                             format_func=lambda i: labels[i], key="sel_a")
    with col2:
        st.subheader("Artículo B")
        default_b = min(1, len(labels) - 1)
        idx_b = st.selectbox("Selecciona el segundo artículo", range(len(labels)),
                             format_func=lambda i: labels[i], key="sel_b",
                             index=default_b)

    if idx_a == idx_b:
        st.warning("⚠️ Has seleccionado el mismo artículo. Los resultados serán 1.0 para todos los algoritmos.")

    # Mostrar abstracts
    abstract_a = str(df.iloc[idx_a].get("abstract", ""))
    abstract_b = str(df.iloc[idx_b].get("abstract", ""))

    with st.expander("📄 Ver abstracts seleccionados", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Artículo A:** {df.iloc[idx_a]['title']}")
            if abstract_a.strip():
                st.text_area("Abstract A", abstract_a, height=200, disabled=True,
                             key="abs_a_display")
            else:
                st.warning("Este artículo no tiene abstract disponible.")
        with c2:
            st.markdown(f"**Artículo B:** {df.iloc[idx_b]['title']}")
            if abstract_b.strip():
                st.text_area("Abstract B", abstract_b, height=200, disabled=True,
                             key="abs_b_display")
            else:
                st.warning("Este artículo no tiene abstract disponible.")

    if not abstract_a.strip() or not abstract_b.strip():
        st.error("❌ Ambos artículos deben tener abstract para ejecutar la comparación.")
        return

    # ── Selección de algoritmos ──────────────────────────────────────────────
    st.header("2. Algoritmos de similitud")

    with st.sidebar:
        st.header("⚙️ Algoritmos")
        st.caption("Selecciona los algoritmos a ejecutar")

        st.markdown("**Clásicos:**")
        selected_classical = []
        for key in CLASSICAL_KEYS:
            if st.checkbox(ALGO_LABELS[key], value=True, key=f"chk_{key}"):
                selected_classical.append(key)

        st.markdown("**Modelos de IA:**")
        selected_ai = []
        for key in AI_KEYS:
            if st.checkbox(ALGO_LABELS[key], value=True, key=f"chk_{key}"):
                selected_ai.append(key)

    selected = selected_classical + selected_ai

    if not selected:
        st.info("Selecciona al menos un algoritmo en la barra lateral.")
        return

    # ── Ejecución ────────────────────────────────────────────────────────────
    if st.button("🔬 Ejecutar análisis de similitud", type="primary"):
        results = {}
        traces = {}

        progress = st.progress(0, text="Iniciando análisis...")

        for i, key in enumerate(selected):
            algo = ALGORITHMS[key]
            progress.progress(
                (i + 1) / len(selected),
                text=f"Ejecutando {algo['name']}..."
            )

            try:
                score = algo["func"](abstract_a, abstract_b)
                trace = algo["step_func"](abstract_a, abstract_b)
                results[key] = round(score, 4)
                traces[key] = trace
            except Exception as e:
                results[key] = None
                traces[key] = {"error": str(e)}
                logger.exception(f"Error in {key}")
                st.error(f"Error en {algo['name']}: {e}")

        progress.empty()

        # Guardar en session_state
        st.session_state["r2_results"] = results
        st.session_state["r2_traces"] = traces
        st.session_state["r2_selected"] = selected

    # ── Resultados ───────────────────────────────────────────────────────────
    if "r2_results" in st.session_state:
        _render_results(
            st.session_state["r2_results"],
            st.session_state["r2_traces"],
            st.session_state["r2_selected"],
        )

    # ── Matriz de similitud (opcional) ───────────────────────────────────────
    st.markdown("---")
    st.header("4. Matriz de similitud (todos los artículos)")
    _render_similarity_matrix(df)


def _load_dataset() -> pd.DataFrame | None:
    """Carga el dataset unificado desde R1 o desde un archivo CSV."""
    st.markdown("---")

    # Opción 1: Reutilizar del R1
    if "r1_unified" in st.session_state:
        st.success("✅ Dataset del Requerimiento 1 disponible en sesión.")
        use_r1 = st.checkbox("Usar dataset del R1", value=True, key="use_r1")
        if use_r1:
            df = st.session_state["r1_unified"]
            st.caption(f"📊 {len(df)} artículos disponibles")
            return df

    # Opción 2: Cargar CSV
    st.info("Carga el archivo `unified.csv` generado en el Requerimiento 1.")
    uploaded = st.file_uploader(
        "Sube unified.csv", type=["csv"], key="r2_upload"
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded, encoding="utf-8-sig", dtype=str).fillna("")
            if "title" not in df.columns or "abstract" not in df.columns:
                st.error("❌ El archivo debe contener las columnas 'title' y 'abstract'.")
                return None
            st.success(f"✅ {uploaded.name} cargado — {len(df)} artículos")
            return df
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return None

    return None


def _render_results(results: dict, traces: dict, selected: list[str]):
    """Muestra los resultados de similitud y las explicaciones."""
    st.header("3. Resultados")

    # ── Tabla resumen ────────────────────────────────────────────────────────
    rows = []
    for key in selected:
        algo = ALGORITHMS[key]
        score = results.get(key)
        rows.append({
            "Algoritmo": algo["name"],
            "Tipo": algo["type"],
            "Similitud": f"{score:.4f}" if score is not None else "Error",
        })

    df_results = pd.DataFrame(rows)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # ── Gráfico de barras ────────────────────────────────────────────────────
    chart_data = {
        ALGORITHMS[k]["name"]: results[k]
        for k in selected
        if results.get(k) is not None
    }

    if chart_data:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams.update({"font.size": 10})

        fig, ax = plt.subplots(figsize=(10, 4))
        names = list(chart_data.keys())
        values = list(chart_data.values())

        colors = []
        for k in selected:
            if results.get(k) is not None:
                if k in CLASSICAL_KEYS:
                    colors.append("#4A90D9")  # azul para clásicos
                else:
                    colors.append("#D94A7A")  # rosa para IA

        bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Similitud")
        ax.set_title("Comparación de Algoritmos de Similitud")

        # Etiquetas en las barras
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)

        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Leyenda
        st.caption("🔵 Algoritmo clásico · 🔴 Modelo de IA")

    # ── Explicaciones paso a paso ────────────────────────────────────────────
    st.subheader("Explicaciones paso a paso")

    for key in selected:
        algo = ALGORITHMS[key]
        trace = traces.get(key, {})

        if "error" in trace:
            with st.expander(f"❌ {algo['name']} — Error"):
                st.error(trace["error"])
            continue

        with st.expander(f"📖 {algo['name']} — Ver explicación detallada"):
            explain_fn = EXPLANATION_FUNCS.get(key)
            if explain_fn:
                explanation = explain_fn(trace)
                st.markdown(explanation)
            else:
                st.info("Explicación no disponible.")


def _render_similarity_matrix(df: pd.DataFrame):
    """Genera una matriz de similitud NxN para un subconjunto de artículos."""
    # Filtrar artículos con abstract
    df_with_abs = df[df["abstract"].str.strip() != ""].reset_index(drop=True)

    if len(df_with_abs) < 2:
        st.warning("Se necesitan al menos 2 artículos con abstract.")
        return

    max_articles = st.slider(
        "Número de artículos a incluir",
        min_value=2,
        max_value=min(20, len(df_with_abs)),
        value=min(5, len(df_with_abs)),
        key="matrix_size"
    )

    algo_key = st.selectbox(
        "Algoritmo para la matriz",
        CLASSICAL_KEYS,
        format_func=lambda k: ALGORITHMS[k]["name"],
        key="matrix_algo",
    )

    if st.button("📊 Generar matriz de similitud", key="gen_matrix"):
        subset = df_with_abs.head(max_articles)
        abstracts = subset["abstract"].tolist()
        titles = [t[:40] + "…" if len(t) > 40 else t for t in subset["title"].tolist()]
        algo_func = ALGORITHMS[algo_key]["func"]

        n = len(abstracts)
        matrix = [[0.0] * n for _ in range(n)]

        progress = st.progress(0, text="Calculando matriz...")
        total = n * (n + 1) // 2
        count = 0

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    score = algo_func(abstracts[i], abstracts[j])
                    matrix[i][j] = round(score, 4)
                    matrix[j][i] = round(score, 4)
                count += 1
                progress.progress(count / total, text=f"Comparando {count}/{total}...")

        progress.empty()

        # Mostrar como DataFrame
        df_matrix = pd.DataFrame(matrix, index=titles, columns=titles)
        st.dataframe(df_matrix.style.background_gradient(cmap="YlOrRd", vmin=0, vmax=1),
                     use_container_width=True)

        # Heatmap con matplotlib
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.6)))
        im = ax.imshow(np.array(matrix), cmap="YlOrRd", vmin=0, vmax=1)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(titles, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(titles, fontsize=8)

        # Valores en las celdas
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if matrix[i][j] > 0.5 else "white")

        fig.colorbar(im, ax=ax, label="Similitud")
        ax.set_title(f"Matriz de Similitud — {ALGORITHMS[algo_key]['name']}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.session_state["r2_matrix"] = df_matrix
