"""
Requerimiento 4 — Interfaz Streamlit
======================================
Permite al usuario:
  1. Cargar o reutilizar el dataset unificado (unified.csv del R1).
  2. Seleccionar cuántos artículos incluir (slider).
  3. Configurar el preprocesamiento.
  4. Visualizar:
       a. Preview de tokens preprocesados
       b. Heatmap de la matriz de distancias coseno
       c. Tres dendrogramas comparados (Complete / Average / Ward)
       d. Tabla de coeficientes cofenéticos
  5. Explorar explicaciones matemáticas paso a paso.
  6. Leer la conclusión sobre cuál algoritmo produce agrupamientos más coherentes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

from src.r4_clustering.algorithms import (
    CLUSTERING_ALGORITHMS,
    run_clustering_pipeline,
    linkage_to_scipy_format,
)
from src.r4_clustering.explanations import (
    ALGO_EXPLANATION_FUNCS,
    explain_preprocessing,
    explain_tfidf,
    explain_cosine_distance,
    explain_cophenetic,
)

logger = logging.getLogger(__name__)

# Nombre legible de cada algoritmo
ALGO_NAMES = {k: v["name"] for k, v in CLUSTERING_ALGORITHMS.items()}
ALGO_COLORS = {k: v["color"] for k, v in CLUSTERING_ALGORITHMS.items()}
ALGO_ICONS = {k: v["icon"] for k, v in CLUSTERING_ALGORITHMS.items()}

MAX_ARTICLES = 40   # Límite para O(n³) en tiempo razonable
DEFAULT_N = 15


def render():
    """Renderiza la página completa del Requerimiento 4."""
    st.title("Requerimiento 4: Agrupamiento Jerárquico")
    st.markdown(
        """
        **Objetivo:** Construir un dendrograma que represente la similitud entre
        abstracts científicos, usando **3 algoritmos de clustering jerárquico** implementados
        desde cero sobre una representación TF-IDF con distancia coseno.

        **Pipeline:** Preprocesamiento → TF-IDF → Distancias coseno → Clustering × 3 → Evaluación
        """
    )

    # ── Sección 0: Carga de datos ────────────────────────────────────────────
    st.markdown("---")
    df = _load_dataset()
    if df is None or df.empty:
        return

    # Filtrar artículos que tienen abstract
    df_valid = df[df["abstract"].str.strip() != ""].reset_index(drop=True)

    if len(df_valid) < 2:
        st.error("❌ Se necesitan al menos 2 artículos con abstract.")
        return

    st.success(f"✅ {len(df_valid)} artículos con abstract disponibles.")

    # ── Sección 1: Configuración ──────────────────────────────────────────────
    st.markdown("---")
    st.header("1. Configuración")

    with st.expander("⚙️ Parámetros del pipeline", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            n_articles = st.slider(
                "Número de artículos",
                min_value=5,
                max_value=min(MAX_ARTICLES, len(df_valid)),
                value=min(DEFAULT_N, len(df_valid)),
                key="r4_n_articles",
                help=f"Máximo {MAX_ARTICLES} artículos para mantener tiempos razonables (O(n³)).",
            )

        with col2:
            sort_option = st.selectbox(
                "Seleccionar artículos",
                ["Primeros N", "Aleatorio"],
                key="r4_sort",
            )

        col3, col4 = st.columns(2)
        with col3:
            remove_sw = st.checkbox("Eliminar stop words", value=True, key="r4_sw")
        with col4:
            apply_stem = st.checkbox("Aplicar stemming", value=True, key="r4_stem")

    # Seleccionar subset de artículos
    if sort_option == "Aleatorio":
        subset = df_valid.sample(n=n_articles, random_state=42).reset_index(drop=True)
    else:
        subset = df_valid.head(n_articles).reset_index(drop=True)

    abstracts = subset["abstract"].tolist()

    # Etiquetas cortas para dendrograma
    if "title" in subset.columns:
        titles_full = subset["title"].tolist()
        labels = [f"[{i}] {t[:45]}…" if len(t) > 45 else f"[{i}] {t}"
                  for i, t in enumerate(titles_full)]
    else:
        labels = [f"Doc {i}" for i in range(n_articles)]
        titles_full = labels

    st.caption(
        f"Se procesarán **{n_articles} artículos** | "
        f"Stop words: {'✅' if remove_sw else '❌'} | "
        f"Stemming: {'✅' if apply_stem else '❌'}"
    )

    # ── Sección 2: Ejecutar pipeline ──────────────────────────────────────────
    st.markdown("---")
    run_btn = st.button(
        "🚀 Ejecutar clustering jerárquico",
        type="primary",
        key="r4_run",
    )

    if run_btn:
        with st.spinner("Ejecutando pipeline completo..."):
            progress = st.progress(0, text="Preprocesando textos...")

            result = run_clustering_pipeline(
                abstracts=abstracts,
                titles=titles_full,
                remove_stopwords=remove_sw,
                apply_stemming=apply_stem,
            )

            progress.progress(100, text="¡Completado!")
            progress.empty()

        st.session_state["r4_result"] = result
        st.session_state["r4_labels"] = labels
        st.session_state["r4_titles_full"] = titles_full
        st.success(
            f"✅ Pipeline ejecutado — {n_articles} documentos × "
            f"{result['vocabulary_size']} términos en vocabulario"
        )

    # ── Mostrar resultados si existen ─────────────────────────────────────────
    if "r4_result" not in st.session_state:
        st.info("Pulsa **Ejecutar clustering jerárquico** para comenzar.")
        return

    result = st.session_state["r4_result"]
    labels = st.session_state["r4_labels"]
    titles_full = st.session_state["r4_titles_full"]

    n = result["n"]
    dist_matrix = result["dist_matrix"]
    vocab = result["vocabulary"]
    processed_tokens = result["processed_tokens"]
    linkage_data = result["linkage"]
    coph_scores = result["cophenetic"]
    best_algo = result["best_algorithm"]

    # ── Sección 2: Preprocesamiento ───────────────────────────────────────────
    st.markdown("---")
    st.header("2. Preprocesamiento y Vectorización")

    with st.expander("🔧 Pipeline de preprocesamiento (ver detalle)", expanded=False):
        st.markdown(explain_preprocessing(processed_tokens, titles_full))

    # Tabla de tokens por documento
    token_data = []
    for i, (toks, title) in enumerate(zip(processed_tokens, titles_full)):
        token_data.append({
            "N°": i,
            "Artículo": title[:55] + "…" if len(title) > 55 else title,
            "Tokens": len(toks),
            "Muestra": ", ".join(toks[:8]) + ("…" if len(toks) > 8 else ""),
        })

    df_tokens = pd.DataFrame(token_data)
    st.dataframe(df_tokens, use_container_width=True, hide_index=True)

    with st.expander("📊 Vectorización TF-IDF y distancia coseno", expanded=False):
        st.markdown(explain_tfidf(result["vocabulary_size"], n, vocab[:30]))
        st.markdown("---")
        st.markdown(explain_cosine_distance())

    # ── Sección 3: Matriz de distancias ──────────────────────────────────────
    st.markdown("---")
    st.header("3. Matriz de Distancias Coseno")

    _render_distance_heatmap(dist_matrix, labels)

    # ── Sección 4: Dendrogramas ───────────────────────────────────────────────
    st.markdown("---")
    st.header("4. Dendrogramas — Los 3 Algoritmos")

    st.markdown(
        "Cada columna muestra el dendrograma de un algoritmo diferente. "
        "La altura de cada nodo en el árbol representa la distancia a la que "
        "se fusionaron los clusters."
    )

    p_threshold = st.slider(
        "Truncar dendrograma a últimos P merges (0 = completo)",
        min_value=0,
        max_value=n - 1,
        value=min(20, n - 1),
        key="r4_p",
        help="Muestra solo las últimas P fusiones. Útil cuando hay muchos documentos.",
    )

    _render_dendrograms(linkage_data, labels, n, p_threshold, coph_scores)

    # ── Sección 5: Métricas de calidad ────────────────────────────────────────
    st.markdown("---")
    st.header("5. Evaluación — Coeficiente Cofenético")

    _render_quality_metrics(coph_scores, best_algo, result)

    # ── Sección 6: Explicaciones matemáticas ─────────────────────────────────
    st.markdown("---")
    st.header("6. Explicaciones Matemáticas Paso a Paso")

    for key, algo in CLUSTERING_ALGORITHMS.items():
        icon = ALGO_ICONS[key]
        name = ALGO_NAMES[key]
        with st.expander(f"{icon} {name} — Ver explicación detallada"):
            explain_fn = ALGO_EXPLANATION_FUNCS.get(key)
            if explain_fn:
                st.markdown(explain_fn(result))

    with st.expander("📏 Coeficiente de Correlación Cofenética — Ver explicación"):
        st.markdown(
            explain_cophenetic(coph_scores, best_algo, ALGO_NAMES)
        )


# ── Funciones auxiliares de renderizado ──────────────────────────────────────


def _load_dataset() -> pd.DataFrame | None:
    """Carga el dataset unificado desde sesión (R1) o desde archivo CSV."""
    # Opción 1: Reutilizar del R1
    if "r1_unified" in st.session_state:
        st.success("✅ Dataset del Requerimiento 1 disponible en sesión.")
        use_r1 = st.checkbox("Usar dataset del R1", value=True, key="r4_use_r1")
        if use_r1:
            df = st.session_state["r1_unified"]
            st.caption(f"📊 {len(df)} artículos en el dataset")
            return df

    # Opción 2: Cargar CSV
    st.info("Sube el archivo `unified.csv` generado en el Requerimiento 1.")
    uploaded = st.file_uploader("Sube unified.csv", type=["csv"], key="r4_upload")

    if uploaded:
        try:
            df = pd.read_csv(uploaded, encoding="utf-8-sig", dtype=str).fillna("")
            if "abstract" not in df.columns:
                st.error("❌ El archivo debe contener la columna 'abstract'.")
                return None
            if "title" not in df.columns:
                df["title"] = [f"Artículo {i}" for i in range(len(df))]
            st.success(f"✅ {uploaded.name} cargado — {len(df)} artículos")
            return df
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return None

    return None


def _render_distance_heatmap(dist_matrix: list, labels: list):
    """Renderiza el heatmap de la matriz de distancias coseno."""
    import matplotlib.pyplot as plt
    import matplotlib

    n = len(dist_matrix)
    mat = np.array(dist_matrix)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(6, n * 0.45)))

    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    short_labels = [f"[{i}]" for i in range(n)]
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)

    # Mostrar valores en celdas si n es pequeño
    if n <= 20:
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                color = "white" if val > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="Distancia coseno [0=idéntico, 1=ortogonal]",
                 fraction=0.046, pad=0.04)
    ax.set_title("Matriz de Distancias Coseno entre Abstracts", fontsize=11, pad=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Estadísticas
    upper_tri = mat[np.triu_indices(n, k=1)]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Distancia mínima", f"{upper_tri.min():.3f}")
    col2.metric("Distancia máxima", f"{upper_tri.max():.3f}")
    col3.metric("Distancia media", f"{upper_tri.mean():.3f}")
    col4.metric("Desviación estándar", f"{upper_tri.std():.3f}")


def _render_dendrograms(
    linkage_data: dict,
    labels: list,
    n: int,
    p_threshold: int,
    coph_scores: dict,
):
    """Renderiza los tres dendrogramas en columnas."""
    try:
        from scipy.cluster.hierarchy import dendrogram
    except ImportError:
        st.error(
            "❌ scipy no está instalado. Ejecuta: `pip install scipy`"
        )
        return

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    keys = list(CLUSTERING_ALGORITHMS.keys())
    cols = st.columns(3)

    for col, key in zip(cols, keys):
        algo = CLUSTERING_ALGORITHMS[key]
        steps = linkage_data.get(key, [])
        if not steps:
            col.warning(f"Sin datos para {algo['name']}")
            continue

        Z = np.array(linkage_to_scipy_format(steps))

        # Altura de la figura: escalar con n para que se lean las etiquetas
        fig_h = max(7, n * 0.38)
        fig, ax = plt.subplots(figsize=(5, fig_h))

        color_hex = algo["color"]

        ddata = dendrogram(
            Z,
            ax=ax,
            labels=labels,
            orientation="left",
            p=p_threshold if p_threshold > 0 else n,
            truncate_mode="lastp" if p_threshold > 0 else None,
            color_threshold=0.7 * Z[:, 2].max(),
            above_threshold_color="#AAAAAA",
            leaf_font_size=7,
            show_contracted=True,
        )

        coph = coph_scores.get(key, 0.0)
        quality = "⭐" if coph == max(coph_scores.values()) else ""

        ax.set_title(
            f"{algo['icon']} {algo['name']}\n"
            f"Cofenético: {coph:.4f} {quality}",
            fontsize=10,
            pad=8,
        )
        ax.set_xlabel("Distancia coseno")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        col.pyplot(fig)
        plt.close(fig)

        # Descripción breve
        col.caption(algo["description"])


def _render_quality_metrics(coph_scores: dict, best_algo: str, result: dict):
    """Renderiza la tabla de métricas de calidad y la conclusión."""
    import matplotlib.pyplot as plt

    # Tabla
    rows = []
    for key, algo in CLUSTERING_ALGORITHMS.items():
        coph = coph_scores.get(key, 0.0)
        is_best = key == best_algo
        rows.append({
            "Algoritmo": f"{algo['icon']} {algo['name']}",
            "Coeficiente Cofenético": round(coph, 4),
            "Calidad": _quality_label(coph),
            "Mejor": "⭐ Sí" if is_best else "",
        })

    df_quality = pd.DataFrame(rows)

    def _highlight_best(s):
        """Resalta el valor máximo con fondo y texto legibles."""
        is_max = s == s.max()
        return [
            "background-color: rgba(39, 174, 96, 0.25); color: #2ecc71; font-weight: bold"
            if v else ""
            for v in is_max
        ]

    st.dataframe(
        df_quality.style.apply(
            _highlight_best,
            subset=["Coeficiente Cofenético"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Gráfico de barras comparativo
    fig, ax = plt.subplots(figsize=(8, 3))

    algo_names = [CLUSTERING_ALGORITHMS[k]["name"] for k in coph_scores]
    coph_vals = [coph_scores[k] for k in coph_scores]
    colors = [ALGO_COLORS[k] for k in coph_scores]
    alphas = [1.0 if k == best_algo else 0.6 for k in coph_scores]

    bars = ax.barh(
        algo_names,
        coph_vals,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Coeficiente Cofenético")
    ax.set_title("Comparación de Algoritmos — Calidad del Dendrograma", fontsize=10)

    # Referencia: umbral de calidad aceptable
    ax.axvline(0.75, color="#F39C12", linestyle="--", linewidth=1, label="Aceptable (0.75)")
    ax.axvline(0.90, color="#27AE60", linestyle="--", linewidth=1, label="Excelente (0.90)")

    for bar, val in zip(bars, coph_vals):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9,
        )

    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Conclusión
    best_name = CLUSTERING_ALGORITHMS[best_algo]["name"]
    best_coph = coph_scores[best_algo]
    best_icon = ALGO_ICONS[best_algo]

    st.success(
        f"**Conclusión:** El algoritmo con agrupamientos más coherentes es "
        f"{best_icon} **{best_name}** con un coeficiente cofenético de **{best_coph:.4f}** "
        f"({_quality_label(best_coph)}). "
        f"Esto significa que su dendrograma preserva mejor la estructura de distancias "
        f"original entre los abstracts."
    )

    # Explicación adicional de por qué es el mejor
    _render_algorithm_comparison(coph_scores, best_algo)


def _render_algorithm_comparison(coph_scores: dict, best_algo: str):
    """Muestra una comparación narrativa de los algoritmos."""
    with st.expander("📖 ¿Por qué este algoritmo es el mejor?"):
        sorted_algos = sorted(coph_scores.items(), key=lambda x: -x[1])
        ranking_text = "\n".join(
            f"{i+1}. **{CLUSTERING_ALGORITHMS[k]['name']}** ({v:.4f}) — "
            f"{CLUSTERING_ALGORITHMS[k]['description']}"
            for i, (k, v) in enumerate(sorted_algos)
        )

        st.markdown(f"""
#### Ranking de algoritmos por coherencia

{ranking_text}

#### ¿Cómo interpretar los dendrogramas?

- **Ramas largas**: documentos muy diferentes entre sí
- **Ramas cortas**: documentos muy similares, agrupados temprano
- **Altura de corte**: eligiendo un umbral horizontal se obtienen los clusters
- **Grupos naturales**: cuando hay "saltos" grandes de altura, indican grupos bien definidos

#### Criterio de evaluación

El **coeficiente cofenético** es el criterio estándar para comparar
la calidad de dendrogramas. Valores altos indican que:

1. Los documentos que son similares (distancia baja) también se agrupan
   en pasos tempranos del dendrograma (altura baja)
2. Los documentos disímiles se agrupan tarde (altura alta)
3. El árbol refleja fielmente la geometría del espacio de distancias

Un dendrograma con coeficiente cofenético bajo "miente":
agrupa documentos similares tarde o documentos distintos pronto.
""")


def _quality_label(score: float) -> str:
    """Devuelve una etiqueta de calidad para el coeficiente cofenético."""
    if score >= 0.90:
        return "Excelente (> 0.90)"
    elif score >= 0.75:
        return "Buena (0.75 – 0.90)"
    elif score >= 0.60:
        return "Aceptable (0.60 – 0.75)"
    else:
        return "Baja (< 0.60)"
