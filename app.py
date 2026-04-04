"""
app.py — Punto de entrada principal de la aplicación Streamlit.
Proyecto: Análisis de Algoritmos en Bibliometría
Universidad del Quindío — Ingeniería de Sistemas y Computación
Cadena de búsqueda: "generative artificial intelligence"
"""

import logging
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Configuración global de la página ────────────────────────────────────────
st.set_page_config(
    page_title="Bibliometría · IA Generativa",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navegación lateral ────────────────────────────────────────────────────────
st.sidebar.title("Bibliometría · IA Generativa")
st.sidebar.caption("Universidad del Quindío")
st.sidebar.markdown("---")

PAGES = {
    "R1 · Automatización y Unificación":     "r1",
    "R2 · Similitud Textual":                "r2",
    "R3 · Frecuencia de Términos":           "r3",
    "R4 · Agrupamiento Jerárquico":          "r4",
    "R5 · Visualización Científica":         "r5",
}

selection = st.sidebar.radio("Selecciona un requerimiento", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("Cadena de búsqueda: `\"generative artificial intelligence\"`")

# ── Enrutamiento ──────────────────────────────────────────────────────────────
page_key = PAGES[selection]

if page_key == "r1":
    from src.r1_scraping.page import render
    render()

elif page_key == "r2":
    st.title("R2 · Similitud Textual")
    st.info("EN CURSO Requerimiento 2 en desarrollo.")

elif page_key == "r3":
    st.title("R3 · Frecuencia de Términos")
    st.info("EN CURSO Requerimiento 3 en desarrollo.")

elif page_key == "r4":
    st.title("R4 · Agrupamiento Jerárquico")
    st.info("EN CURSO Requerimiento 4 en desarrollo.")

elif page_key == "r5":
    st.title("R5 · Visualización Científica")
    st.info("EN CURSO Requerimiento 5 en desarrollo.")