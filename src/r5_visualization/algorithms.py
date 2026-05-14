"""
Requerimiento 5 — Algoritmos de Visualización Científica
=========================================================
Funciones de cómputo puro para:
  1. Distribución geográfica (mapa de calor por primer autor)
  2. Nube de palabras dinámica (abstracts + keywords)
  3. Línea temporal de publicaciones por año y revista
  4. Exportación a PDF (tres visualizaciones + portada)

Todas las funciones son stateless y operan sobre DataFrames.
"""

from __future__ import annotations

import io
import re
import logging
import datetime
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # backend sin pantalla, seguro en Streamlit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES COMPARTIDAS
# ═══════════════════════════════════════════════════════════════════════════════

# Stop-words en inglés + español + términos bibliométricos genéricos
_STOPWORDS: set[str] = {
    # Inglés — artículos, preposiciones, pronombres
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "must", "can", "could",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "few", "more", "most", "other", "some", "such", "than", "too", "very",
    "just", "also", "i", "we", "you", "he", "she", "it", "they", "this",
    "that", "these", "those", "our", "their", "its", "my", "your", "his",
    "her", "which", "who", "whom", "when", "where", "how", "what", "why",
    "if", "then", "because", "while", "although", "though", "even",
    "however", "therefore", "thus", "hence", "moreover", "furthermore",
    # Español
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "de",
    "del", "en", "con", "por", "para", "que", "se", "al", "es", "son",
    # Términos bibliométricos sin contenido informativo
    "paper", "study", "research", "work", "approach", "method", "methods",
    "based", "using", "show", "shows", "result", "results", "propose",
    "proposed", "present", "presented", "demonstrate", "demonstrates",
    "analysis", "use", "used", "new", "two", "three", "four", "five",
    "first", "second", "different", "various", "including", "among",
    "between", "within", "across", "without", "through", "provide",
    "provides", "also", "well", "one", "can", "could", "may", "might",
    "given", "while", "often", "many", "both", "used", "further",
    "important", "significantly", "significant", "however", "therefore",
    "order", "terms", "term", "high", "higher", "low", "lower", "large",
    "larger", "small", "smaller", "specific", "general",
}

# Alias de nombres/abreviaturas → nombre canónico del país
_COUNTRY_ALIASES: dict[str, str] = {
    # Estados Unidos
    "usa": "United States", "us": "United States", "u.s.a": "United States",
    "u.s": "United States", "united states of america": "United States",
    "united states": "United States",
    # Reino Unido
    "uk": "United Kingdom", "u.k": "United Kingdom",
    "united kingdom": "United Kingdom", "england": "United Kingdom",
    "great britain": "United Kingdom", "britain": "United Kingdom",
    # China
    "china": "China", "p.r. china": "China", "pr china": "China",
    "people's republic of china": "China", "prc": "China",
    # Europa
    "germany": "Germany", "deutschland": "Germany",
    "france": "France", "italia": "Italy", "italy": "Italy",
    "spain": "Spain", "españa": "Spain",
    "netherlands": "Netherlands", "the netherlands": "Netherlands",
    "switzerland": "Switzerland", "sweden": "Sweden",
    "norway": "Norway", "denmark": "Denmark", "finland": "Finland",
    "portugal": "Portugal", "belgium": "Belgium", "austria": "Austria",
    "poland": "Poland", "greece": "Greece",
    "czech republic": "Czech Republic", "czechia": "Czech Republic",
    "hungary": "Hungary", "romania": "Romania", "ukraine": "Ukraine",
    "croatia": "Croatia", "serbia": "Serbia", "slovakia": "Slovakia",
    # Otros
    "canada": "Canada", "australia": "Australia",
    "japan": "Japan", "south korea": "South Korea", "korea": "South Korea",
    "brazil": "Brazil", "brasil": "Brazil",
    "india": "India",
    "russia": "Russia", "russian federation": "Russia",
    "turkey": "Turkey", "türkiye": "Turkey",
    "colombia": "Colombia", "mexico": "Mexico", "méxico": "Mexico",
    "argentina": "Argentina", "chile": "Chile",
    "singapore": "Singapore", "taiwan": "Taiwan",
    "hong kong": "Hong Kong", "new zealand": "New Zealand",
    "saudi arabia": "Saudi Arabia", "israel": "Israel",
    "iran": "Iran", "egypt": "Egypt", "south africa": "South Africa",
    "nigeria": "Nigeria", "kenya": "Kenya",
    "indonesia": "Indonesia", "malaysia": "Malaysia",
    "thailand": "Thailand", "vietnam": "Vietnam",
    "philippines": "Philippines", "pakistan": "Pakistan",
    "bangladesh": "Bangladesh",
}

# Código ISO Alpha-3 para cada país canónico (para choropleth de plotly)
_COUNTRY_ISO3: dict[str, str] = {
    "United States": "USA", "United Kingdom": "GBR", "China": "CHN",
    "Germany": "DEU", "France": "FRA", "Italy": "ITA", "Spain": "ESP",
    "Canada": "CAN", "Australia": "AUS", "Japan": "JPN",
    "South Korea": "KOR", "Brazil": "BRA", "India": "IND",
    "Netherlands": "NLD", "Switzerland": "CHE", "Sweden": "SWE",
    "Norway": "NOR", "Denmark": "DNK", "Finland": "FIN",
    "Portugal": "PRT", "Belgium": "BEL", "Austria": "AUT",
    "Poland": "POL", "Russia": "RUS", "Turkey": "TUR",
    "Colombia": "COL", "Mexico": "MEX", "Argentina": "ARG",
    "Chile": "CHL", "Singapore": "SGP", "Taiwan": "TWN",
    "Hong Kong": "HKG", "New Zealand": "NZL",
    "Saudi Arabia": "SAU", "Israel": "ISR", "Iran": "IRN",
    "Egypt": "EGY", "South Africa": "ZAF", "Nigeria": "NGA",
    "Kenya": "KEN", "Indonesia": "IDN", "Malaysia": "MYS",
    "Thailand": "THA", "Vietnam": "VNM", "Philippines": "PHL",
    "Pakistan": "PAK", "Bangladesh": "BGD", "Greece": "GRC",
    "Czech Republic": "CZE", "Hungary": "HUN", "Romania": "ROU",
    "Ukraine": "UKR", "Croatia": "HRV", "Serbia": "SRB",
    "Slovakia": "SVK",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DISTRIBUCIÓN GEOGRÁFICA
# ═══════════════════════════════════════════════════════════════════════════════

def extract_country_from_text(text: str) -> Optional[str]:
    """
    Intenta extraer el nombre canónico de un país desde una cadena de texto.

    Estrategia:
      1. Normaliza a minúsculas y elimina puntuación.
      2. Busca coincidencias con los alias del diccionario _COUNTRY_ALIASES,
         comenzando por los más largos (evita falsos positivos con "us" vs
         "united states").

    Parameters
    ----------
    text : cadena libre, por ejemplo la afiliación de un autor.

    Returns
    -------
    Nombre canónico del país (str) o None si no se identifica ninguno.
    """
    if not text or not str(text).strip():
        return None

    text_lower = str(text).lower()
    # Quitar puntuación común que dificulta la búsqueda de tokens
    text_clean = re.sub(r"[,;.\(\)\[\]]", " ", text_lower)
    text_clean = re.sub(r"\s+", " ", text_clean).strip()

    # Iterar de alias más largo a más corto para evitar subcoincidencias
    for alias in sorted(_COUNTRY_ALIASES.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text_clean):
            return _COUNTRY_ALIASES[alias]

    return None


def detect_country_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detecta automáticamente columnas que contengan información geográfica.

    Busca columnas cuyos nombres incluyan 'country', 'affili', 'institution'
    o 'organization' (case-insensitive).

    Returns
    -------
    Nombre de la primera columna candidata, o None.
    """
    priority = ["country", "affili", "institution", "organization", "address"]
    for keyword in priority:
        for col in df.columns:
            if keyword in col.lower():
                return col
    return None


def count_countries(df: pd.DataFrame) -> dict[str, int]:
    """
    Cuenta publicaciones por país del **primer autor**.

    Estrategia:
      1. Busca columnas con info geográfica (país / afiliación).
      2. Para cada fila toma la primera parte (primer ";") como datos del
         primer autor y extrae el país con ``extract_country_from_text``.
      3. Si no hay columna con país, retorna dict vacío.

    Parameters
    ----------
    df : DataFrame con esquema canónico del proyecto.

    Returns
    -------
    dict {nombre_país: conteo}
    """
    geo_col = detect_country_column(df)
    if geo_col is None:
        logger.info("No se encontró columna de país/afiliación en el dataset.")
        return {}

    counter: Counter = Counter()
    for val in df[geo_col]:
        raw = str(val).strip()
        if not raw or raw.lower() in ("", "nan", "none"):
            continue
        # Primer autor = primera entrada separada por ";"
        first_affil = raw.split(";")[0]
        country = extract_country_from_text(first_affil)
        if country:
            counter[country] += 1

    return dict(counter)


def build_geo_dataframe(country_counts: dict[str, int]) -> pd.DataFrame:
    """
    Construye un DataFrame {country, iso_a3, count} para el choropleth
    de plotly.

    Parameters
    ----------
    country_counts : salida de ``count_countries``.

    Returns
    -------
    pd.DataFrame ordenado de mayor a menor conteo.
    """
    rows = [
        {
            "country": country,
            "iso_a3": _COUNTRY_ISO3.get(country, ""),
            "count": count,
        }
        for country, count in country_counts.items()
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("count", ascending=False).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NUBE DE PALABRAS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_word_frequencies(
    df: pd.DataFrame,
    top_n: int = 150,
    use_abstracts: bool = True,
    use_keywords: bool = True,
    extra_stopwords: Optional[set] = None,
    keyword_weight: int = 3,
) -> dict[str, int]:
    """
    Calcula las frecuencias de palabras en abstracts y/o keywords.

    Las keywords reciben un peso mayor (``keyword_weight``) porque son términos
    seleccionados explícitamente por los autores para representar el trabajo.

    Algoritmo:
      1. Tokeniza: minúsculas → elimina URLs y números → elimina puntuación.
      2. Filtra: longitud ≥ 3 caracteres, no pertenece a stop-words.
      3. Acumula frecuencias; keywords suman ``keyword_weight`` por ocurrencia.
      4. Retorna el top-N de mayor a menor frecuencia.

    Parameters
    ----------
    df              : DataFrame con columnas 'abstract' y/o 'keywords'.
    top_n           : máximo de palabras únicas a retornar.
    use_abstracts   : incluir términos de los abstracts.
    use_keywords    : incluir términos de las keywords.
    extra_stopwords : palabras adicionales a ignorar.
    keyword_weight  : cuántas veces se suma cada keyword encontrada.

    Returns
    -------
    dict {palabra: frecuencia_acumulada}
    """
    stopwords = _STOPWORDS.copy()
    if extra_stopwords:
        stopwords |= {w.lower().strip() for w in extra_stopwords if w.strip()}

    counter: Counter = Counter()

    def _tokenize(text: str) -> list[str]:
        """Normaliza y tokeniza un texto libre."""
        text = str(text).lower()
        text = re.sub(r"https?://\S+", "", text)          # quitar URLs
        text = re.sub(r"\b\d+\b", "", text)               # quitar números solos
        text = re.sub(r"[^a-z\s\-]", " ", text)           # quitar puntuación
        text = re.sub(r"\s+", " ", text).strip()
        return [
            t.strip("-")
            for t in text.split()
            if len(t.strip("-")) >= 3 and t.strip("-") not in stopwords
        ]

    if use_abstracts and "abstract" in df.columns:
        for text in df["abstract"]:
            if text and str(text).strip() not in ("", "nan", "none"):
                counter.update(_tokenize(text))

    if use_keywords and "keywords" in df.columns:
        for kw_str in df["keywords"]:
            if not kw_str or str(kw_str).strip().lower() in ("", "nan", "none"):
                continue
            # Keywords separadas por ";" o ","
            kws = re.split(r"[;,]", str(kw_str))
            for kw in kws:
                kw_clean = kw.strip().lower()
                if kw_clean and len(kw_clean) >= 3 and kw_clean not in stopwords:
                    counter[kw_clean] += keyword_weight

    return dict(counter.most_common(top_n))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LÍNEA TEMPORAL
# ═══════════════════════════════════════════════════════════════════════════════

def compute_timeline(
    df: pd.DataFrame,
    min_year: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula la distribución temporal de publicaciones.

    Parameters
    ----------
    df       : DataFrame con columnas 'year' y opcionalmente 'source'.
    min_year : filtrar años anteriores a este valor.

    Returns
    -------
    (df_by_year, df_by_year_source)

    - **df_by_year**: columnas [year, count] — total de publicaciones por año.
    - **df_by_year_source**: columnas [year, source_short, count] — detalle por
      año y revista/conferencia. Vacío si no hay columna 'source'.
    """
    if "year" not in df.columns:
        return pd.DataFrame(columns=["year", "count"]), pd.DataFrame()

    df_work = df.copy()
    df_work["year"] = pd.to_numeric(df_work["year"], errors="coerce")
    df_work = df_work.dropna(subset=["year"])
    df_work["year"] = df_work["year"].astype(int)
    df_work = df_work[df_work["year"] >= min_year].copy()

    if df_work.empty:
        return pd.DataFrame(columns=["year", "count"]), pd.DataFrame()

    # — Por año —
    df_by_year = (
        df_work.groupby("year")
        .size()
        .reset_index(name="count")
        .sort_values("year")
    )

    # — Por año y fuente —
    if "source" in df_work.columns:
        df_work["source_short"] = df_work["source"].apply(
            lambda s: (str(s)[:45] + "…") if len(str(s)) > 45 else str(s)
        )
        df_by_year_source = (
            df_work.groupby(["year", "source_short"])
            .size()
            .reset_index(name="count")
            .sort_values(["year", "count"], ascending=[True, False])
        )
    else:
        df_by_year_source = pd.DataFrame()

    return df_by_year, df_by_year_source


def get_top_sources(df_by_year_source: pd.DataFrame, top_n: int = 8) -> list[str]:
    """
    Retorna las top_n fuentes/revistas por volumen total de publicaciones.
    """
    if df_by_year_source.empty or "source_short" not in df_by_year_source.columns:
        return []
    totals = (
        df_by_year_source.groupby("source_short")["count"]
        .sum()
        .sort_values(ascending=False)
    )
    return totals.head(top_n).index.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GENERACIÓN DE FIGURAS MATPLOTLIB (para PDF y fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def make_geo_figure_mpl(country_counts: dict[str, int]) -> plt.Figure:
    """
    Genera una figura matplotlib de distribución geográfica como barras
    horizontales (top 20 países), con gradiente de color según frecuencia.

    Se usa como versión estática para el PDF; en Streamlit se prefiere el
    choropleth interactivo de plotly.
    """
    if not country_counts:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(
            0.5, 0.5,
            "Sin datos de país disponibles en el dataset.\n"
            "Agrega una columna 'country' o 'affiliations' al CSV.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="#666666", wrap=True,
        )
        ax.axis("off")
        ax.set_title("Distribución Geográfica del Primer Autor", fontsize=13)
        return fig

    top = sorted(country_counts.items(), key=lambda x: -x[1])[:20]
    countries, counts = zip(*top)
    n = len(countries)

    cmap = plt.cm.YlOrRd
    norm_vals = np.array(counts) / max(counts)
    colors = [cmap(0.25 + 0.65 * v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.45)))
    bars = ax.barh(range(n), counts, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels(countries, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Número de publicaciones", fontsize=10)
    ax.set_title(
        "Distribución Geográfica del Primer Autor\n(Top 20 países)",
        fontsize=12, pad=10,
    )

    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=8,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def make_wordcloud_figure(
    freq_dict: dict[str, int],
    title: str = "Nube de Palabras — Abstracts y Keywords",
) -> plt.Figure:
    """
    Genera una figura matplotlib con la nube de palabras.

    Usa la librería ``wordcloud`` si está disponible; de lo contrario genera
    un gráfico de barras de frecuencia como fallback.

    Parameters
    ----------
    freq_dict : {palabra: frecuencia} — salida de ``compute_word_frequencies``.
    title     : título de la figura.
    """
    if not freq_dict:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(
            0.5, 0.5, "Sin datos de texto disponibles.",
            ha="center", va="center", transform=ax.transAxes, fontsize=13,
        )
        ax.axis("off")
        ax.set_title(title)
        return fig

    try:
        from wordcloud import WordCloud as _WC

        wc = _WC(
            width=1400,
            height=700,
            background_color="white",
            max_words=120,
            colormap="viridis",
            collocations=False,
            prefer_horizontal=0.75,
            min_font_size=8,
            max_font_size=120,
            relative_scaling=0.5,
        )
        wc.generate_from_frequencies(freq_dict)

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, pad=15, fontweight="bold")
        fig.tight_layout(pad=0.3)

    except ImportError:
        logger.warning("wordcloud no instalado; usando fallback de barras.")
        fig = _make_frequency_bars(freq_dict, title)

    return fig


def _make_frequency_bars(freq_dict: dict[str, int], title: str) -> plt.Figure:
    """Fallback: barras de frecuencia cuando wordcloud no está disponible."""
    top = sorted(freq_dict.items(), key=lambda x: -x[1])[:30]
    if not top:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    words, counts = zip(*top)
    cmap = plt.cm.viridis
    norm = np.array(counts) / max(counts)
    colors = [cmap(0.2 + 0.7 * v) for v in norm]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(len(words)), counts, color=colors, edgecolor="white")
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frecuencia")
    ax.set_title(title, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def make_timeline_figure(
    df_by_year: pd.DataFrame,
    df_by_year_source: pd.DataFrame,
    top_sources: int = 8,
) -> plt.Figure:
    """
    Genera una figura matplotlib de línea temporal con dos paneles:

    - Panel superior: barras + línea de total de publicaciones por año.
    - Panel inferior: barras apiladas por revista/conferencia (top_sources).

    Parameters
    ----------
    df_by_year        : salida de ``compute_timeline`` (total por año).
    df_by_year_source : salida de ``compute_timeline`` (por año y fuente).
    top_sources       : cuántas fuentes mostrar en el panel inferior.
    """
    if df_by_year.empty:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.text(
            0.5, 0.5, "Sin datos de año disponibles.",
            ha="center", va="center", transform=ax.transAxes, fontsize=13,
        )
        ax.axis("off")
        ax.set_title("Línea Temporal de Publicaciones")
        return fig

    years = df_by_year["year"].tolist()
    counts = df_by_year["count"].tolist()
    has_sources = not df_by_year_source.empty

    if has_sources:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(13, 11),
            gridspec_kw={"hspace": 0.45},
            constrained_layout=False,
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(13, 5))
        ax2 = None

    # ── Panel 1: total por año ──────────────────────────────────────────────
    color_bar = "#42A5F5"
    color_line = "#1565C0"

    ax1.bar(years, counts, color=color_bar, alpha=0.65, label="Publicaciones/año",
            zorder=2)
    ax1.plot(years, counts, "o-", color=color_line, linewidth=2,
             markersize=6, zorder=3, label="Tendencia")

    for y, c in zip(years, counts):
        ax1.annotate(
            str(c), (y, c), textcoords="offset points",
            xytext=(0, 7), ha="center", fontsize=8, color=color_line,
        )

    ax1.set_xlabel("Año", fontsize=10)
    ax1.set_ylabel("Número de publicaciones", fontsize=10)
    ax1.set_title("Línea Temporal de Publicaciones por Año", fontsize=12)
    ax1.set_xticks(years)
    ax1.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: stacked bar por fuente ────────────────────────────────────
    if ax2 is not None:
        top_src_names = get_top_sources(df_by_year_source, top_n=top_sources)

        df_filtered = df_by_year_source[
            df_by_year_source["source_short"].isin(top_src_names)
        ]
        df_pivot = df_filtered.pivot_table(
            index="year", columns="source_short",
            values="count", fill_value=0,
        ).reindex(years, fill_value=0)

        cmap_tab = plt.cm.tab20
        src_colors = [cmap_tab(i / max(len(df_pivot.columns), 1))
                      for i in range(len(df_pivot.columns))]

        bottom = np.zeros(len(years))
        for col, color in zip(df_pivot.columns, src_colors):
            vals = df_pivot[col].values.astype(float)
            ax2.bar(years, vals, bottom=bottom, color=color,
                    label=col[:38], alpha=0.88, edgecolor="white", linewidth=0.3)
            bottom += vals

        ax2.set_xlabel("Año", fontsize=10)
        ax2.set_ylabel("Publicaciones", fontsize=10)
        ax2.set_title(
            f"Distribución por Revista/Conferencia (Top {top_sources})",
            fontsize=12,
        )
        ax2.set_xticks(years)
        ax2.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
        ax2.legend(
            loc="upper left", fontsize=6, ncol=2,
            framealpha=0.8, edgecolor="none",
        )
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.1, hspace=0.45)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GENERACIÓN DE PDF
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(
    country_counts: dict[str, int],
    freq_dict: dict[str, int],
    df_by_year: pd.DataFrame,
    df_by_year_source: pd.DataFrame,
    n_articles: int = 0,
) -> bytes:
    """
    Genera un informe PDF de 4 páginas:
      - Portada con resumen estadístico
      - Página 1: Distribución geográfica (barras horizontales)
      - Página 2: Nube de palabras
      - Página 3: Línea temporal de publicaciones

    Parameters
    ----------
    country_counts    : dict {país: conteo}.
    freq_dict         : dict {palabra: frecuencia}.
    df_by_year        : DataFrame [year, count].
    df_by_year_source : DataFrame [year, source_short, count].
    n_articles        : total de artículos en el dataset (para portada).

    Returns
    -------
    bytes con el contenido del PDF listo para ``st.download_button``.
    """
    buf = io.BytesIO()

    with PdfPages(buf) as pdf:

        # ── Portada ────────────────────────────────────────────────────────────
        fig_cover, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        fig_cover.patch.set_facecolor("#FAFAFA")

        # Franja de color superior
        ax.add_patch(plt.Rectangle(
            (0, 0.85), 1, 0.15,
            transform=ax.transAxes, color="#1565C0", zorder=0,
        ))

        ax.text(0.5, 0.915, "Análisis Bibliométrico",
                ha="center", va="center", fontsize=26, fontweight="bold",
                transform=ax.transAxes, color="white", zorder=1)
        ax.text(0.5, 0.875, "Inteligencia Artificial Generativa",
                ha="center", va="center", fontsize=13,
                transform=ax.transAxes, color="#BBDEFB", zorder=1)

        ax.text(0.5, 0.78, "Requerimiento 5 — Visualización Científica",
                ha="center", va="center", fontsize=15, fontweight="bold",
                transform=ax.transAxes, color="#1565C0")

        # Separador
        ax.plot([0.08, 0.92], [0.745, 0.745], transform=ax.transAxes,
                color="#1565C0", linewidth=1.5)

        # Estadísticas (sin emojis para compatibilidad con fuentes de PDF)
        stats = [
            ("[1]", "Articulos analizados", str(n_articles) if n_articles else "—"),
            ("[2]", "Paises identificados", str(len(country_counts))),
            ("[3]", "Terminos unicos", str(len(freq_dict))),
            (
                "[4]",
                "Rango temporal",
                (
                    f"{int(df_by_year['year'].min())} - {int(df_by_year['year'].max())}"
                    if not df_by_year.empty else "—"
                ),
            ),
        ]

        y0 = 0.685
        for badge, label, value in stats:
            ax.text(0.18, y0, badge, ha="center", va="center",
                    fontsize=11, fontweight="bold", transform=ax.transAxes,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1565C0",
                              edgecolor="none"))
            ax.text(0.38, y0 + 0.016, label, ha="left", va="center",
                    fontsize=11, fontweight="bold", transform=ax.transAxes,
                    color="#1565C0")
            ax.text(0.38, y0 - 0.016, value, ha="left", va="center",
                    fontsize=10, transform=ax.transAxes, color="#424242")
            y0 -= 0.10

        # Contenidos
        ax.text(0.5, 0.26, "Contenido del informe:",
                ha="center", va="center", fontsize=11, fontweight="bold",
                transform=ax.transAxes, color="#424242")

        contents = [
            "1. Mapa de calor — Distribución geográfica del primer autor",
            "2. Nube de palabras — Términos más frecuentes en abstracts y keywords",
            "3. Línea temporal — Publicaciones por año y por revista",
        ]
        y_c = 0.225
        for line in contents:
            ax.text(0.5, y_c, line, ha="center", va="center", fontsize=9,
                    transform=ax.transAxes, color="#616161")
            y_c -= 0.04

        # Footer
        ax.add_patch(plt.Rectangle(
            (0, 0), 1, 0.08,
            transform=ax.transAxes, color="#E3F2FD", zorder=0,
        ))
        ax.text(0.5, 0.05,
                "Universidad del Quindío · Ingeniería de Sistemas y Computación\n"
                "Análisis de Algoritmos · Cadena de búsqueda: \"generative artificial intelligence\"",
                ha="center", va="center", fontsize=8,
                transform=ax.transAxes, color="#424242", linespacing=1.5)
        ax.text(0.5, 0.015,
                f"Generado el {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}",
                ha="center", va="center", fontsize=7,
                transform=ax.transAxes, color="#9E9E9E")

        pdf.savefig(fig_cover, bbox_inches="tight", facecolor=fig_cover.get_facecolor())
        plt.close(fig_cover)

        # ── Página 1: Mapa geográfico ──────────────────────────────────────────
        fig1 = make_geo_figure_mpl(country_counts)
        _add_page_footer(fig1, "Requerimiento 5 — Visualización 1: Distribución Geográfica")
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # ── Página 2: Nube de palabras ─────────────────────────────────────────
        fig2 = make_wordcloud_figure(
            freq_dict,
            title="Nube de Palabras — Abstracts y Keywords",
        )
        _add_page_footer(fig2, "Requerimiento 5 — Visualización 2: Nube de Palabras")
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # ── Página 3: Línea temporal ───────────────────────────────────────────
        fig3 = make_timeline_figure(df_by_year, df_by_year_source)
        _add_page_footer(fig3, "Requerimiento 5 — Visualización 3: Línea Temporal")
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

    buf.seek(0)
    return buf.read()


def _add_page_footer(fig: plt.Figure, text: str) -> None:
    """Agrega pie de pagina sutil a una figura matplotlib."""
    fig.text(
        0.5, 0.005, text,
        ha="center", va="bottom", fontsize=6.5,
        color="#BDBDBD", style="italic",
    )
