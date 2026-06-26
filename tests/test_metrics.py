import numpy as np

from src.common.metrics import (
    calculate_jaccard_similarity,
    calculate_ngram_similarity,
    calculate_stylometric_similarity,
    extract_stylometric_features,
)


def test_calculate_ngram_similarity_returns_overlap_ratio():
    essay = ["the", "quick", "brown", "fox"]
    reference = ["the", "quick", "brown", "cat"]

    assert calculate_ngram_similarity(essay, reference, n=2) == 2 / 3


def test_calculate_ngram_similarity_handles_short_or_empty_inputs():
    assert calculate_ngram_similarity([], ["a", "b"], n=2) == 0.0
    assert calculate_ngram_similarity(["a"], ["a", "b"], n=2) == 0.0


def test_calculate_jaccard_similarity():
    assert calculate_jaccard_similarity(["a", "b", "c"], ["b", "c", "d"]) == 0.5
    assert calculate_jaccard_similarity([], ["b"]) == 0.0


def test_extract_stylometric_features_shape_and_nonzero_values():
    features = extract_stylometric_features(
        "This is a simple sentence. This second sentence has more words."
    )

    assert features.shape == (7,)
    assert np.all(np.isfinite(features))
    assert features.sum() > 0


def test_calculate_stylometric_similarity_bounds_and_zero_vectors():
    features = extract_stylometric_features("One clear sentence with several useful words.")

    assert calculate_stylometric_similarity(features, features) == 1.0
    assert calculate_stylometric_similarity(features, np.zeros(7)) == 0.0
