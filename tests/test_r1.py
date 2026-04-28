"""
Tests unitarios para el Requerimiento 1.
Cubre: normalización de títulos, detección de duplicados, elección de representante.
"""

import pandas as pd
import pytest
import sys, os

# Asegura que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.text_utils import (
    normalize_title,
    levenshtein_similarity,
    titles_are_duplicate,
)
from src.r1_scraping.unifier import unify_databases


# ── normalize_title ───────────────────────────────────────────────────────────
class TestNormalizeTitle:
    def test_lowercase(self):
        assert normalize_title("Hello World") == "hello world"

    def test_removes_accents(self):
        assert normalize_title("Análisis de Algoritmos") == "analisis de algoritmos"

    def test_removes_punctuation(self):
        # normalize_title colapsa espacios extra → no hay espacios dobles en la salida
        result = normalize_title("A.I.: The Future?")
        assert "a" in result and "future" in result
        assert result == result.strip()
        # Sin puntuación original
        assert "." not in result and ":" not in result and "?" not in result

    def test_collapses_spaces(self):
        assert normalize_title("  hello   world  ") == "hello world"

    def test_empty_string(self):
        assert normalize_title("") == ""

    def test_non_string(self):
        assert normalize_title(None) == ""
        assert normalize_title(123) == ""


# ── levenshtein_similarity ────────────────────────────────────────────────────
class TestLevenshteinSimilarity:
    def test_identical_strings(self):
        assert levenshtein_similarity("abc", "abc") == 1.0

    def test_completely_different(self):
        sim = levenshtein_similarity("abc", "xyz")
        assert sim < 0.5

    def test_one_character_difference(self):
        sim = levenshtein_similarity("hello", "helo")
        assert sim >= 0.8  # "helo" vs "hello" → 1 - 1/5 = 0.80

    def test_empty_strings(self):
        assert levenshtein_similarity("", "") == 1.0
        assert levenshtein_similarity("abc", "") == 0.0

    def test_near_duplicate_title(self):
        t1 = "generative artificial intelligence in education"
        t2 = "generative artificial intelligence in education."
        # Diferencia mínima → muy alta similitud
        assert levenshtein_similarity(t1, t2) > 0.95


# ── titles_are_duplicate ──────────────────────────────────────────────────────
class TestTitlesAreDuplicate:
    def test_exact_match(self):
        assert titles_are_duplicate("AI in Education", "AI in Education") is True

    def test_case_insensitive(self):
        assert titles_are_duplicate("AI in Education", "ai in education") is True

    def test_accent_insensitive(self):
        assert titles_are_duplicate(
            "Análisis de Algoritmos", "Analisis de Algoritmos"
        ) is True

    def test_clearly_different(self):
        assert titles_are_duplicate(
            "Machine Learning Basics", "Quantum Computing Overview"
        ) is False

    def test_minor_typo(self):
        # Un carácter diferente en título largo → sigue siendo duplicado
        assert titles_are_duplicate(
            "A Survey of Generative Artificial Intelligence Applications",
            "A Survey of Generative Artificial Intelligence Application",
        ) is True


# ── unify_databases ───────────────────────────────────────────────────────────
def _make_df(records: list[dict], source_db: str) -> pd.DataFrame:
    """Helper: crea un DataFrame canónico mínimo para tests."""
    cols = ["title", "authors", "year", "source", "doi",
            "abstract", "document_type", "url", "issn", "keywords"]
    rows = []
    for r in records:
        row = {c: r.get(c, "") for c in cols}
        row["source_db"] = source_db
        rows.append(row)
    return pd.DataFrame(rows)


class TestUnifyDatabases:
    def test_no_duplicates(self, tmp_path):
        df_acm = _make_df([
            {"title": "Article Alpha", "abstract": "Alpha abstract."},
        ], "ACM")
        df_sd = _make_df([
            {"title": "Article Beta", "abstract": "Beta abstract."},
        ], "ScienceDirect")

        unified, dups = unify_databases(df_acm, df_sd, output_dir=tmp_path)
        assert len(unified) == 2
        assert len(dups) == 0

    def test_exact_duplicate_removed(self, tmp_path):
        title = "Generative AI in Higher Education"
        df_acm = _make_df([{"title": title, "abstract": "Short."}], "ACM")
        df_sd  = _make_df([{"title": title, "abstract": "Longer and more complete abstract."}], "ScienceDirect")

        unified, dups = unify_databases(df_acm, df_sd, output_dir=tmp_path)
        assert len(unified) == 1
        assert len(dups) == 1

    def test_representative_is_most_complete(self, tmp_path):
        title = "Generative AI in Higher Education"
        df_acm = _make_df([{"title": title, "abstract": ""}], "ACM")
        df_sd  = _make_df([{"title": title, "abstract": "Full abstract here."}], "ScienceDirect")

        unified, _ = unify_databases(df_acm, df_sd, output_dir=tmp_path)
        # El representante debe ser el de ScienceDirect (más completo + BD preferida)
        assert unified.iloc[0]["source_db"] == "ScienceDirect"

    def test_output_files_created(self, tmp_path):
        df_acm = _make_df([{"title": "A"}], "ACM")
        df_sd  = _make_df([{"title": "B"}], "ScienceDirect")

        unify_databases(df_acm, df_sd, output_dir=tmp_path)

        assert (tmp_path / "data" / "processed" / "unified.csv").exists()
        assert (tmp_path / "data" / "duplicates" / "duplicates.csv").exists()

    def test_fuzzy_duplicate_detected(self, tmp_path):
        df_acm = _make_df(
            [{"title": "A Survey on Generative Artificial Intelligence"}], "ACM"
        )
        df_sd = _make_df(
            [{"title": "A Survey on Generative Artificial Intelligence."}], "ScienceDirect"
        )
        unified, dups = unify_databases(df_acm, df_sd, output_dir=tmp_path)
        assert len(unified) == 1
        assert len(dups) == 1