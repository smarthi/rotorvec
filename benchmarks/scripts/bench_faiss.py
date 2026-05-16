"""FAISS-PQ baseline that emits results in the same format as rotorvec-bench.

Loads a prepped dataset (run `prep_datasets.py` first), normalizes both
train and queries to unit length, recomputes ground truth on the
normalized vectors, then runs FAISS `IndexPQ` at parameters that match
rotorvec's 4-bit compression rate.

For rotorvec at d=128 with 4-bit codes, total bits per vector = 128 × 4 = 512.
FAISS PQ with m sub-quantizers at nbits=8 gives 8m bits per vector, so the
matched configuration is m = 64 (= d × bits / 8). At d=104 (padded GloVe-100)
that's m = 52.

Usage:

    uv run python benchmarks/scripts/bench_faiss.py glove-100 --k 10

Outputs:

    benchmarks/results/<dataset>_faiss.{md,json}

# /// script
# requires-python = ">=3.10"
# dependencies = ["faiss-cpu", "numpy", "tqdm"]
# ///
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import faiss  # type: ignore
import numpy as np  # type: ignore


def cache_root() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "rotorvec" / "datasets"


def read_fvecs(path: Path) -> np.ndarray:
    """Read INRIA-style .fvecs into a 2D float32 array."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    n = raw.size // (1 + dim)
    return raw.reshape(n, 1 + dim)[:, 1:].copy().view(np.float32)


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (arr / norms).astype(np.float32, copy=False)


def brute_force_topk(queries: np.ndarray, train: np.ndarray, k: int) -> np.ndarray:
    """Compute top-k by inner product (assumes inputs are unit-normalized)."""
    # IndexFlatIP gives exact maximum-inner-product top-k.
    idx = faiss.IndexFlatIP(train.shape[1])
    idx.add(np.ascontiguousarray(train))
    _, neighbors = idx.search(np.ascontiguousarray(queries), k)
    return neighbors.astype(np.int32)


def recall_at_k(approx: np.ndarray, truth: np.ndarray) -> float:
    """approx, truth: (n_queries, k) int arrays. Returns avg recall."""
    n_queries, k = approx.shape
    hits = 0
    for q in range(n_queries):
        truth_set = set(truth[q].tolist())
        hits += sum(1 for x in approx[q] if int(x) in truth_set)
    return hits / float(n_queries * k)


def bench(name: str, k: int, m: int | None = None, out_dir: Path | None = None) -> dict:
    src = cache_root() / name
    if not src.exists():
        raise SystemExit(
            f"Dataset '{name}' not prepped. Run:\n"
            f"    uv run python benchmarks/scripts/prep_datasets.py {name}\n"
        )
    print(f"Loading {name} ...")
    train_raw = read_fvecs(src / "base.fvecs")
    query_raw = read_fvecs(src / "query.fvecs")
    n_train, dim = train_raw.shape
    n_queries = query_raw.shape[0]
    print(f"  n_train={n_train} n_queries={n_queries} dim={dim}")

    # Match rotorvec's bench: normalize, then recompute ground truth on
    # normalized vectors so all methods are scored against the same target.
    print("Normalizing + recomputing ground truth ...")
    train = normalize_rows(train_raw)
    queries = normalize_rows(query_raw)
    truth = brute_force_topk(queries, train, k)

    # Match rotorvec's 4-bit compression rate.
    # rotorvec: dim * 4 bits/vec. FAISS PQ: m * nbits bits/vec.
    # For 4-bit rotorvec, m = dim * 4 / nbits. At nbits=4 (LUT256-style)
    # that's m = dim. At nbits=8 (standard PQ), m = dim/2.
    nbits = 4
    pq_m = m if m is not None else dim  # m = dim at nbits=4 → matches bit rate
    if dim % pq_m != 0:
        # PQ requires dim divisible by m.
        # Fall back to nbits=8 with m = dim/2 (less common, but works).
        nbits = 8
        pq_m = dim // 2
        print(f"  falling back to nbits=8 m={pq_m} (dim not divisible by m={dim})")

    # Build BOTH variants:
    # 1. IndexPQ at nbits=4 — uses plain ADC, slow but good recall
    # 2. IndexPQFastScan at nbits=4 — SIMD scan, fast but slightly lower recall
    results: list[dict] = []

    for variant_name, builder in [
        ("IndexPQ", lambda: faiss.IndexPQ(dim, pq_m, nbits)),
        ("IndexPQFastScan", lambda: faiss.IndexPQFastScan(dim, pq_m, nbits)),
    ]:
        print(f"\nBuilding FAISS {variant_name}(d={dim}, m={pq_m}, nbits={nbits}) ...")
        t0 = time.perf_counter()
        index = builder()
        index.train(train)
        index.add(train)
        build_secs = time.perf_counter() - t0
        print(f"  built in {build_secs:.2f}s")

        print(f"Searching k={k} ...")
        t1 = time.perf_counter()
        _, approx = index.search(np.ascontiguousarray(queries), k)
        search_secs = time.perf_counter() - t1
        qps = n_queries / search_secs if search_secs > 0 else float("inf")
        recall = recall_at_k(approx.astype(np.int32), truth)
        print(f"  recall@{k}={recall:.3f} search={search_secs:.2f}s qps={qps:.0f}")

        results.append({
            "method": f"faiss_{variant_name.lower()}",
            "variant": variant_name,
            "config": f"m={pq_m},nbits={nbits}",
            "build_secs": round(build_secs, 4),
            "search_secs": round(search_secs, 4),
            "queries": n_queries,
            "qps": round(qps, 1),
            "recall_at_k": round(recall, 6),
            "k": k,
            "dataset": name,
            "dim": dim,
            "n_train": n_train,
        })

    # Persist both variants.
    if out_dir is None:
        out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}_faiss.json"
    md_path = out_dir / f"{name}_faiss.md"
    with open(json_path, "w") as f:
        json.dump({"variants": results}, f, indent=2)
    with open(md_path, "w") as f:
        f.write(
            f"## FAISS PQ baseline — {name}\n\n"
            f"Config: `m={pq_m}, nbits={nbits}` — matches rotorvec's 4-bit "
            f"compression rate at {pq_m * nbits} bits/vector. FAISS trains "
            f"its centroids on the data (data-aware), while turbovec/rotorvec "
            f"are data-oblivious — that's the recall vs no-training trade.\n\n"
            f"| Method | Recall@{k} | Build (s) | Search (s) | QPS |\n"
            f"|---|---:|---:|---:|---:|\n"
        )
        for r in results:
            f.write(
                f"| faiss {r['variant']} | {r['recall_at_k']:.3f} | "
                f"{r['build_secs']:.2f} | {r['search_secs']:.2f} | "
                f"{r['qps']:.0f} |\n"
            )
    print(f"\nWrote {json_path}\nWrote {md_path}")
    return {"variants": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("datasets", nargs="+", help="prepped dataset name(s)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="number of PQ sub-quantizers (default: dim, matching 4-bit rate)",
    )
    args = parser.parse_args()
    for d in args.datasets:
        print(f"\n=== {d} ===")
        bench(d, args.k, args.m)


if __name__ == "__main__":
    main()
