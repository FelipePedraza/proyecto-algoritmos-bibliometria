"""
Tests para el módulo OpenAlex (parser + cliente).

Cubre:
  - Reconstrucción del abstract desde inverted index.
  - Extracción de país desde authorships.
  - Mapeo completo de un registro Work al esquema canónico.
  - Limpieza de DOI (eliminación de prefijo URL).
  - Comportamiento con campos nulos o ausentes.
  - Integración real con la API (5 resultados), verificando esquema canónico.
  - Lectura y escritura de archivos JSONL.
  - Enriquecimiento de abstracts en unifier cuando hay duplicados con OpenAlex.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Asegurar que el raíz del proyecto esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.r1_scraping.openalex_parser import (
    _reconstruct_abstract,
    _extract_country,
    _map_record,
    parse_openalex_records,
    CANONICAL_COLS,
)
from src.r1_scraping.openalex_client import OpenAlexClient, read_jsonl, read_jsonl_slice
from src.r1_scraping.unifier import unify_databases


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_inverted_index():
    """Inverted index de ejemplo que corresponde a 'Hello world from OpenAlex'."""
    return {
        "Hello":     [0],
        "world":     [1],
        "from":      [2],
        "OpenAlex":  [3],
    }


@pytest.fixture
def sample_inverted_index_gaps():
    """Inverted index con una palabra repetida en varias posiciones."""
    return {
        "AI":   [0, 4],
        "is":   [1],
        "great": [2],
        "and":  [3],
        "awesome": [5],
    }


@pytest.fixture
def sample_work():
    """Objeto Work de OpenAlex con todos los campos populados."""
    return {
        "title": "Generative AI in Education: A Systematic Review",
        "authorships": [
            {
                "author": {"display_name": "Jane Doe"},
                "institutions": [{"country_code": "US", "display_name": "MIT"}],
            },
            {
                "author": {"display_name": "Juan García"},
                "institutions": [{"country_code": "ES", "display_name": "UPM"}],
            },
        ],
        "publication_year": 2024,
        "primary_location": {
            "source": {
                "display_name": "Journal of Educational Technology",
                "issn_l": "1234-5678",
                "issn": ["1234-5678", "8765-4321"],
            },
            "landing_page_url": "https://example.com/paper",
        },
        "doi": "https://doi.org/10.1234/example.2024",
        "abstract_inverted_index": {
            "This": [0],
            "paper": [1],
            "reviews": [2],
            "generative": [3],
            "AI": [4],
        },
        "type": "article",
        "open_access": {"oa_url": "https://oa.example.com/paper"},
        "keywords": [
            {"keyword": "generative AI"},
            {"keyword": "education"},
        ],
        "topics": [],
    }


@pytest.fixture
def minimal_work():
    """Objeto Work con sólo el título (mínimo válido)."""
    return {"title": "Minimal Paper Title"}


@pytest.fixture
def null_work():
    """Objeto Work con todos los campos opcionales como None o ausentes."""
    return {
        "title": "Paper With Nulls",
        "authorships": None,
        "publication_year": None,
        "primary_location": None,
        "doi": None,
        "abstract_inverted_index": None,
        "type": None,
        "keywords": None,
        "topics": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: _reconstruct_abstract
# ═══════════════════════════════════════════════════════════════════════════════

class TestReconstructAbstract:
    def test_basic_reconstruction(self, sample_inverted_index):
        result = _reconstruct_abstract(sample_inverted_index)
        assert result == "Hello world from OpenAlex"

    def test_repeated_word(self, sample_inverted_index_gaps):
        result = _reconstruct_abstract(sample_inverted_index_gaps)
        words = result.split()
        assert words[0] == "AI"
        assert words[4] == "AI"
        assert words[5] == "awesome"
        assert len(words) == 6

    def test_empty_dict(self):
        assert _reconstruct_abstract({}) == ""

    def test_none_input(self):
        assert _reconstruct_abstract(None) == ""

    def test_wrong_type(self):
        assert _reconstruct_abstract("not a dict") == ""

    def test_single_word(self):
        result = _reconstruct_abstract({"hello": [0]})
        assert result == "hello"

    def test_preserves_order(self):
        # Palabras en orden inverso en el dict, deben salir ordenadas por posición
        idx = {"third": [2], "first": [0], "second": [1]}
        assert _reconstruct_abstract(idx) == "first second third"

    def test_non_integer_positions_skipped(self):
        # Posiciones no enteras se ignoran silenciosamente
        idx = {"word": ["bad_position"]}
        result = _reconstruct_abstract(idx)
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: _extract_country
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractCountry:
    def test_extracts_first_country(self):
        authorships = [
            {"author": {"display_name": "A"}, "institutions": [{"country_code": "DE"}]},
            {"author": {"display_name": "B"}, "institutions": [{"country_code": "FR"}]},
        ]
        assert _extract_country(authorships) == "DE"

    def test_uppercases_code(self):
        authorships = [{"institutions": [{"country_code": "co"}]}]
        assert _extract_country(authorships) == "CO"

    def test_skips_empty_code(self):
        authorships = [
            {"institutions": [{"country_code": ""}]},
            {"institutions": [{"country_code": "BR"}]},
        ]
        assert _extract_country(authorships) == "BR"

    def test_empty_authorships(self):
        assert _extract_country([]) == ""

    def test_none_authorships(self):
        assert _extract_country(None) == ""

    def test_no_institutions(self):
        authorships = [{"author": {"display_name": "A"}, "institutions": []}]
        assert _extract_country(authorships) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: _map_record (mapeo completo)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMapRecord:
    def test_all_fields_populated(self, sample_work):
        row = _map_record(sample_work)

        assert row["title"] == "Generative AI in Education: A Systematic Review"
        assert "Jane Doe" in row["authors"]
        assert "Juan García" in row["authors"]
        assert row["year"] == "2024"
        assert row["source"] == "Journal of Educational Technology"
        assert row["doi"] == "10.1234/example.2024"          # sin prefijo URL
        assert "This paper reviews generative AI" in row["abstract"]
        assert row["document_type"] == "article"
        assert row["url"] == "https://example.com/paper"
        assert row["issn"] == "1234-5678"
        assert "generative AI" in row["keywords"]
        assert row["country"] == "US"
        assert row["source_db"] == "OpenAlex"

    def test_doi_prefix_stripped(self, sample_work):
        row = _map_record(sample_work)
        assert not row["doi"].startswith("https://")
        assert not row["doi"].startswith("http://")

    def test_doi_http_prefix_stripped(self):
        work = {"title": "T", "doi": "http://doi.org/10.9999/test"}
        row = _map_record(work)
        assert row["doi"] == "10.9999/test"

    def test_minimal_work(self, minimal_work):
        row = _map_record(minimal_work)
        assert row["title"] == "Minimal Paper Title"
        assert row["source_db"] == "OpenAlex"
        # Todos los demás campos deben existir (pueden ser vacíos)
        for col in CANONICAL_COLS:
            assert col in row

    def test_null_fields(self, null_work):
        row = _map_record(null_work)
        assert row["title"] == "Paper With Nulls"
        assert row["authors"] == ""
        assert row["year"] == ""
        assert row["abstract"] == ""

    def test_url_fallback_to_oa_url(self):
        work = {
            "title": "OA Paper",
            "primary_location": {"landing_page_url": None, "source": None},
            "open_access": {"oa_url": "https://oa.example.com/fallback"},
        }
        row = _map_record(work)
        assert row["url"] == "https://oa.example.com/fallback"

    def test_issn_fallback_to_list(self):
        work = {
            "title": "ISSN Fallback",
            "primary_location": {
                "source": {"issn_l": None, "issn": ["0000-0001", "0000-0002"]},
                "landing_page_url": None,
            },
        }
        row = _map_record(work)
        assert row["issn"] == "0000-0001"

    def test_keywords_fallback_to_topics(self):
        work = {
            "title": "Topic Paper",
            "keywords": [],
            "topics": [{"display_name": "Machine Learning"}, {"display_name": "NLP"}],
        }
        row = _map_record(work)
        assert "Machine Learning" in row["keywords"]
        assert "NLP" in row["keywords"]

    def test_authors_joined_with_comma(self, sample_work):
        row = _map_record(sample_work)
        authors_list = [a.strip() for a in row["authors"].split(",")]
        assert "Jane Doe" in authors_list
        assert "Juan García" in authors_list


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: parse_openalex_records (función pública)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseOpenAlexRecords:
    def test_returns_dataframe(self, sample_work):
        df = parse_openalex_records([sample_work])
        assert isinstance(df, pd.DataFrame)

    def test_has_all_canonical_columns(self, sample_work):
        df = parse_openalex_records([sample_work])
        for col in CANONICAL_COLS:
            assert col in df.columns, f"Columna faltante: {col}"

    def test_filters_empty_titles(self):
        records = [
            {"title": "Valid Title"},
            {"title": ""},
            {"title": None},
            {},
        ]
        df = parse_openalex_records(records)
        assert len(df) == 1
        assert df.iloc[0]["title"] == "Valid Title"

    def test_all_values_are_strings(self, sample_work):
        df = parse_openalex_records([sample_work])
        for col in CANONICAL_COLS:
            assert df[col].dtype == object  # pandas object = string column

    def test_no_nan_values(self, sample_work, null_work, minimal_work):
        df = parse_openalex_records([sample_work, null_work, minimal_work])
        assert not df.isnull().any().any(), "El DataFrame no debe tener NaN"

    def test_empty_input(self):
        df = parse_openalex_records([])
        assert len(df) == 0
        for col in CANONICAL_COLS:
            assert col in df.columns


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: JSONL utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestJsonlUtils:
    def test_read_write_roundtrip(self):
        records = [{"id": 1, "title": "A"}, {"id": 2, "title": "B"}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
            tmp_path = fh.name

        try:
            result = read_jsonl(tmp_path)
            assert result == records
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_read_nonexistent_file(self):
        result = read_jsonl("/tmp/this_file_does_not_exist_xyz.jsonl")
        assert result == []

    def test_read_jsonl_slice(self):
        records = [{"i": i} for i in range(10)]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
            tmp_path = fh.name

        try:
            sliced, total = read_jsonl_slice(tmp_path, offset=2, limit=3)
            assert total == 10
            assert len(sliced) == 3
            assert sliced[0]["i"] == 2
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_skips_empty_lines(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write('{"a": 1}\n')
            fh.write('\n')          # línea vacía
            fh.write('{"a": 2}\n')
            tmp_path = fh.name

        try:
            result = read_jsonl(tmp_path)
            assert len(result) == 2
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: integración real con la API de OpenAlex
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestOpenAlexClientIntegration:
    """
    Tests que realizan peticiones reales a la API de OpenAlex.
    Se marcan con @pytest.mark.integration para poder excluirlos
    en entornos sin acceso a internet:

        pytest tests/test_openalex.py -v -m "not integration"
    """

    def test_fetch_5_results(self):
        client = OpenAlexClient()
        records = client.fetch_all("generative artificial intelligence", max_results=5)
        assert len(records) >= 1
        assert len(records) <= 5

    def test_records_have_required_keys(self):
        client = OpenAlexClient()
        records = client.fetch_all("generative artificial intelligence", max_results=5)
        for rec in records:
            # Campos mínimos que OpenAlex garantiza en cada Work
            assert "id" in rec
            assert "title" in rec or rec.get("title") is None  # puede ser null

    def test_parsed_schema_is_canonical(self):
        client = OpenAlexClient()
        records = client.fetch_all("generative artificial intelligence", max_results=5)
        df = parse_openalex_records(records)
        assert len(df) >= 1
        for col in CANONICAL_COLS:
            assert col in df.columns, f"Columna canónica faltante: {col}"
        assert (df["source_db"] == "OpenAlex").all()

    def test_no_nan_in_real_results(self):
        client = OpenAlexClient()
        records = client.fetch_all("generative artificial intelligence", max_results=5)
        df = parse_openalex_records(records)
        assert not df.isnull().any().any(), "Resultado real no debe tener NaN"

    def test_fetch_and_save_creates_jsonl(self):
        client = OpenAlexClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.jsonl"
            n = client.fetch_and_save(
                "generative artificial intelligence",
                output_path=output_path,
                max_results=5,
                overwrite=True,
            )
            assert n >= 1
            assert output_path.exists()
            saved = read_jsonl(output_path)
            assert len(saved) == n


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: enriquecimiento de abstracts en unifier
# ═══════════════════════════════════════════════════════════════════════════════

class TestAbstractEnrichment:
    """
    Verifica que unify_databases enriquece abstracts vacíos usando
    el duplicado descartado de OpenAlex cuando existe.
    """

    def _make_df(self, title, abstract, source_db):
        return pd.DataFrame([{
            "title":         title,
            "authors":       "Autor Test",
            "year":          "2024",
            "source":        "Test Journal",
            "doi":           "10.0000/test",
            "abstract":      abstract,
            "document_type": "article",
            "url":           "",
            "issn":          "",
            "keywords":      "",
            "country":       "",
            "source_db":     source_db,
        }])

    def test_abstract_enriched_from_openalex_duplicate(self, tmp_path):
        # ACM tiene el artículo sin abstract
        df_acm = self._make_df(
            "Generative AI Survey",
            "",                  # sin abstract
            "ACM",
        )
        # OpenAlex tiene el mismo artículo con abstract
        df_oa = self._make_df(
            "Generative AI Survey",
            "This is the abstract from OpenAlex.",
            "OpenAlex",
        )

        df_unified, df_dups = unify_databases(
            df_acm, pd.DataFrame(),
            threshold=0.92,
            output_dir=str(tmp_path),
            extra_sources=[df_oa],
        )

        # Sólo debe quedar 1 registro (ACM tiene prioridad sobre OpenAlex)
        assert len(df_unified) == 1
        assert df_unified.iloc[0]["source_db"] == "ACM"
        # El abstract debe haberse enriquecido con el de OpenAlex
        assert df_unified.iloc[0]["abstract"] == "This is the abstract from OpenAlex."

    def test_existing_abstract_not_overwritten(self, tmp_path):
        # ACM ya tiene abstract propio
        df_acm = self._make_df(
            "AI Paper With Abstract",
            "Original ACM abstract.",
            "ACM",
        )
        df_oa = self._make_df(
            "AI Paper With Abstract",
            "OpenAlex abstract (should not overwrite).",
            "OpenAlex",
        )

        df_unified, _ = unify_databases(
            df_acm, pd.DataFrame(),
            threshold=0.92,
            output_dir=str(tmp_path),
            extra_sources=[df_oa],
        )

        assert len(df_unified) == 1
        assert df_unified.iloc[0]["abstract"] == "Original ACM abstract."

    def test_sciencedirect_wins_over_openalex(self, tmp_path):
        # Mismo artículo en ScienceDirect y OpenAlex
        df_sd = self._make_df(
            "Deep Learning Advances",
            "ScienceDirect abstract.",
            "ScienceDirect",
        )
        df_oa = self._make_df(
            "Deep Learning Advances",
            "OpenAlex abstract.",
            "OpenAlex",
        )

        df_unified, _ = unify_databases(
            pd.DataFrame(), df_sd,
            threshold=0.92,
            output_dir=str(tmp_path),
            extra_sources=[df_oa],
        )

        assert len(df_unified) == 1
        assert df_unified.iloc[0]["source_db"] == "ScienceDirect"
        # ScienceDirect ya tiene abstract, no debe ser reemplazado
        assert df_unified.iloc[0]["abstract"] == "ScienceDirect abstract."
