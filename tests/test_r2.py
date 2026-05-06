"""
Tests unitarios para el Requerimiento 2 — Similitud Textual.
Cubre los 4 algoritmos clásicos. Los algoritmos de IA se prueban solo
si los modelos están disponibles.
"""

import pytest
import sys
import os

# Asegura que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.r2_similarity.algorithms import (
    levenshtein_similarity,
    levenshtein_step_by_step,
    jaccard_similarity,
    jaccard_step_by_step,
    cosine_tfidf_similarity,
    cosine_tfidf_step_by_step,
    hamming_similarity,
    hamming_step_by_step,
)


# ── Levenshtein ──────────────────────────────────────────────────────────────
class TestLevenshtein:
    def test_identical(self):
        assert levenshtein_similarity("hello", "hello") == 1.0

    def test_empty_strings(self):
        assert levenshtein_similarity("", "") == 1.0
        assert levenshtein_similarity("abc", "") == 0.0
        assert levenshtein_similarity("", "abc") == 0.0

    def test_one_char_diff(self):
        # "hello" vs "helo" → distance=1, max_len=5 → sim=0.8
        sim = levenshtein_similarity("hello", "helo")
        assert sim == pytest.approx(0.8, abs=0.01)

    def test_completely_different(self):
        sim = levenshtein_similarity("abc", "xyz")
        assert sim < 0.5

    def test_range_0_to_1(self):
        sim = levenshtein_similarity("generative ai", "artificial intelligence")
        assert 0.0 <= sim <= 1.0

    def test_step_by_step_returns_dict(self):
        trace = levenshtein_step_by_step("abc", "abd")
        assert "matrix" in trace
        assert "distance" in trace
        assert "similarity" in trace
        assert trace["distance"] == 1
        assert trace["similarity"] == pytest.approx(1 - 1/3, abs=0.01)

    def test_symmetric(self):
        s1 = "machine learning"
        s2 = "deep learning"
        assert levenshtein_similarity(s1, s2) == levenshtein_similarity(s2, s1)


# ── Jaccard ──────────────────────────────────────────────────────────────────
class TestJaccard:
    def test_identical(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert jaccard_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        # "hello world" ∩ "hello foo" = {"hello"}, union = {"hello", "world", "foo"}
        sim = jaccard_similarity("hello world", "hello foo")
        assert sim == pytest.approx(1/3, abs=0.01)

    def test_empty_strings(self):
        assert jaccard_similarity("", "") == 1.0
        assert jaccard_similarity("hello", "") == 0.0

    def test_case_insensitive(self):
        assert jaccard_similarity("Hello World", "hello world") == 1.0

    def test_step_by_step_structure(self):
        trace = jaccard_step_by_step("hello world", "hello foo")
        assert "tokens_a" in trace
        assert "tokens_b" in trace
        assert "intersection" in trace
        assert "union" in trace
        assert trace["intersection_size"] == 1
        assert trace["union_size"] == 3

    def test_range_0_to_1(self):
        sim = jaccard_similarity("the quick brown fox", "the lazy brown dog")
        assert 0.0 <= sim <= 1.0


# ── Coseno TF-IDF ────────────────────────────────────────────────────────────
class TestCosineTfidf:
    def test_identical(self):
        text = "generative artificial intelligence applications"
        sim = cosine_tfidf_similarity(text, text)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        sim = cosine_tfidf_similarity("hello world", "foo bar baz")
        assert sim == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        sim = cosine_tfidf_similarity(
            "machine learning for education",
            "deep learning in education"
        )
        assert 0.0 < sim < 1.0

    def test_empty_strings(self):
        assert cosine_tfidf_similarity("", "") == 1.0
        assert cosine_tfidf_similarity("hello", "") == 0.0

    def test_step_by_step_structure(self):
        trace = cosine_tfidf_step_by_step("hello world", "hello foo")
        assert "vocabulary" in trace
        assert "tfidf_vector_a" in trace
        assert "tfidf_vector_b" in trace
        assert "dot_product" in trace
        assert "similarity" in trace

    def test_range_0_to_1(self):
        sim = cosine_tfidf_similarity(
            "a comprehensive survey on generative ai",
            "review of artificial intelligence applications"
        )
        assert 0.0 <= sim <= 1.0


# ── Hamming ──────────────────────────────────────────────────────────────────
class TestHamming:
    def test_identical(self):
        assert hamming_similarity("hello", "hello") == 1.0

    def test_one_char_diff(self):
        # "hello" vs "hallo" → 1 diff / 5 = 0.2 → sim = 0.8
        sim = hamming_similarity("hello", "hallo")
        assert sim == pytest.approx(0.8, abs=0.01)

    def test_completely_different(self):
        sim = hamming_similarity("aaa", "zzz")
        assert sim == pytest.approx(0.0, abs=0.01)

    def test_different_lengths(self):
        # "abc" padded to "abc " vs "abcd" → 1 diff at pos 3
        sim = hamming_similarity("abc", "abcd")
        assert sim == pytest.approx(0.75, abs=0.01)

    def test_empty_strings(self):
        assert hamming_similarity("", "") == 1.0

    def test_step_by_step_structure(self):
        trace = hamming_step_by_step("hello", "hallo")
        assert "distance" in trace
        assert "total_positions" in trace
        assert "comparisons_sample" in trace
        assert trace["distance"] == 1

    def test_range_0_to_1(self):
        sim = hamming_similarity("machine learning", "deep learning!!!")
        assert 0.0 <= sim <= 1.0


# ── Tests de integración (todos los algoritmos clásicos) ─────────────────────
class TestIntegration:
    """Verifica que todos los algoritmos producen resultados coherentes."""

    ABSTRACT_A = (
        "This paper presents a comprehensive survey of generative artificial "
        "intelligence applications in higher education, focusing on the use of "
        "large language models for personalized learning and assessment."
    )
    ABSTRACT_B = (
        "We review the impact of generative AI technologies in educational "
        "settings, examining how large language models can enhance student "
        "engagement and facilitate personalized instruction."
    )

    def test_all_classical_return_valid_range(self):
        algos = [
            levenshtein_similarity,
            jaccard_similarity,
            cosine_tfidf_similarity,
            hamming_similarity,
        ]
        for algo in algos:
            score = algo(self.ABSTRACT_A, self.ABSTRACT_B)
            assert 0.0 <= score <= 1.0, f"{algo.__name__} returned {score}"

    def test_identical_texts_all_return_high(self):
        for algo in [levenshtein_similarity, jaccard_similarity,
                     cosine_tfidf_similarity, hamming_similarity]:
            score = algo(self.ABSTRACT_A, self.ABSTRACT_A)
            assert score >= 0.99, f"{algo.__name__} on identical text returned {score}"

    def test_related_texts_have_some_similarity(self):
        """Textos sobre el mismo tema deben tener similitud > 0."""
        for algo in [levenshtein_similarity, jaccard_similarity,
                     cosine_tfidf_similarity]:
            score = algo(self.ABSTRACT_A, self.ABSTRACT_B)
            assert score > 0.0, f"{algo.__name__} returned 0 for related texts"
