"""End-to-end smoke tests for the Python bindings."""

import os
import tempfile

import numpy as np
import pytest

from rotorvec import IdMapIndex, RotorQuantIndex


ALL_ROTATIONS = ["planar2", "rotor3", "iso4"]


def random_vectors(n, dim, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, dim)).astype(np.float32)


@pytest.mark.parametrize("rotation", ALL_ROTATIONS)
def test_build_search_roundtrip(rotation):
    dim, n, bits = 128, 200, 4
    vectors = random_vectors(n, dim, 1)
    queries = random_vectors(3, dim, 2)

    idx = RotorQuantIndex(dim, bits, rotation=rotation)
    idx.add(vectors)
    assert len(idx) == n
    assert idx.rotation == rotation
    assert idx.dim == dim
    assert idx.bits == bits

    scores, indices = idx.search(queries, k=5)
    assert scores.shape == (3, 5)
    assert indices.shape == (3, 5)
    # Sorted descending per row.
    assert np.all(np.diff(scores, axis=1) <= 1e-5)
    # All indices in range.
    assert np.all((indices >= 0) & (indices < n))


@pytest.mark.parametrize("rotation", ALL_ROTATIONS)
def test_recall_top1(rotation):
    dim, n, bits, nq = 256, 500, 4, 50
    vectors = random_vectors(n, dim, 7)
    queries = random_vectors(nq, dim, 8)

    idx = RotorQuantIndex(dim, bits, rotation=rotation)
    idx.add(vectors)
    _, indices = idx.search(queries, k=1)

    # Brute-force ground truth top-1.
    sims = queries @ vectors.T
    truth = sims.argmax(axis=1)
    hits = int((indices[:, 0] == truth).sum())
    assert hits >= nq // 2, f"{rotation}: recall@1 = {hits}/{nq}"


@pytest.mark.parametrize("rotation", ALL_ROTATIONS)
def test_save_load(rotation):
    dim, n, bits = 64, 30, 4
    vectors = random_vectors(n, dim, 11)
    queries = random_vectors(2, dim, 12)

    idx = RotorQuantIndex(dim, bits, rotation=rotation)
    idx.add(vectors)
    s_before, i_before = idx.search(queries, k=5)

    with tempfile.NamedTemporaryFile(suffix=".rv", delete=False) as f:
        path = f.name
    try:
        idx.write(path)
        loaded = RotorQuantIndex.load(path)
        assert loaded.rotation == rotation
        s_after, i_after = loaded.search(queries, k=5)
        np.testing.assert_array_equal(s_before, s_after)
        np.testing.assert_array_equal(i_before, i_after)
    finally:
        os.unlink(path)


def test_id_map():
    dim, bits = 64, 4
    vectors = random_vectors(5, dim, 21)
    ids = np.array([100, 200, 300, 400, 500], dtype=np.uint64)

    idx = IdMapIndex(dim, bits, rotation="planar2")
    idx.add_with_ids(vectors, ids)
    assert len(idx) == 5
    assert 300 in idx
    assert idx.contains(300)

    assert idx.remove(300) is True
    assert 300 not in idx
    assert len(idx) == 4

    q = vectors[:1]
    _, returned_ids = idx.search(q, k=4)
    flat = set(returned_ids.flatten().tolist())
    assert 100 in flat
    assert 300 not in flat


def test_default_rotation_is_rotor3():
    idx = RotorQuantIndex(dim=64, bits=4)
    assert idx.rotation == "rotor3"


def test_invalid_rotation_raises():
    with pytest.raises(ValueError, match="unknown rotation"):
        RotorQuantIndex(dim=64, bits=4, rotation="hadamard")


def test_repr():
    idx = RotorQuantIndex(dim=64, bits=4, rotation="iso4")
    r = repr(idx)
    assert "iso4" in r
    assert "dim=64" in r
