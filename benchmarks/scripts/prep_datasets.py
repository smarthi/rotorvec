"""Download ann-benchmarks HDF5 datasets and convert to .fvecs / .ivecs.

The Rust benchmark crate reads INRIA's TexMex .fvecs / .ivecs binary format
because parsing it needs no extra deps (no libhdf5). This script bridges the
gap: ann-benchmarks hosts standard ANN datasets as HDF5 with precomputed
ground truth; we download, decompress, and re-emit in the simpler format.

Usage:

    uv run python benchmarks/scripts/prep_datasets.py glove-100 sift-128

Outputs:

    ~/Library/Caches/rotorvec/datasets/<name>/{base,query}.fvecs
    ~/Library/Caches/rotorvec/datasets/<name>/groundtruth.ivecs

Requires:  h5py  numpy  requests  tqdm   (uv-managed via the inline deps).

# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py", "numpy", "requests", "tqdm"]
# ///
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import h5py  # type: ignore
import numpy as np  # type: ignore
import requests  # type: ignore
from tqdm import tqdm  # type: ignore


# Canonical ann-benchmarks dataset ids → URL.
# Full list: https://github.com/erikbern/ann-benchmarks#data-sets
DATASETS = {
    "glove-25":  ("http://ann-benchmarks.com/glove-25-angular.hdf5",  "angular"),
    "glove-50":  ("http://ann-benchmarks.com/glove-50-angular.hdf5",  "angular"),
    "glove-100": ("http://ann-benchmarks.com/glove-100-angular.hdf5", "angular"),
    "glove-200": ("http://ann-benchmarks.com/glove-200-angular.hdf5", "angular"),
    "sift-128":  ("http://ann-benchmarks.com/sift-128-euclidean.hdf5", "euclidean"),
    "nytimes":   ("http://ann-benchmarks.com/nytimes-256-angular.hdf5", "angular"),
    "gist":      ("http://ann-benchmarks.com/gist-960-euclidean.hdf5", "euclidean"),
}


def cache_root() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    root = base / "rotorvec" / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already downloaded → {dest}")
        return
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)


def write_fvecs(path: Path, arr: np.ndarray) -> None:
    """Write a 2D float32 array in INRIA .fvecs format.

    Each record: [int32 dim] [float32 x dim]
    """
    n, dim = arr.shape
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        header = struct.pack("<i", dim)
        for i in range(n):
            f.write(header)
            arr[i].tofile(f)


def write_ivecs(path: Path, arr: np.ndarray) -> None:
    """Write a 2D int32 array in .ivecs format.

    Each record: [int32 k] [int32 x k]
    """
    n, k = arr.shape
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    with open(path, "wb") as f:
        header = struct.pack("<i", k)
        for i in range(n):
            f.write(header)
            arr[i].tofile(f)


def prep_one(name: str) -> None:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name}. choices: {list(DATASETS)}")
    url, distance = DATASETS[name]
    out_dir = cache_root() / name
    out_dir.mkdir(parents=True, exist_ok=True)

    hdf5_path = out_dir / Path(url).name
    download(url, hdf5_path)

    base_path = out_dir / "base.fvecs"
    query_path = out_dir / "query.fvecs"
    gt_path = out_dir / "groundtruth.ivecs"
    meta_path = out_dir / "meta.txt"

    if base_path.exists() and query_path.exists() and gt_path.exists():
        print(f"  {name}: already converted, skipping")
        return

    print(f"  converting {hdf5_path.name} → .fvecs / .ivecs")
    with h5py.File(hdf5_path, "r") as h5:
        train = np.asarray(h5["train"]).astype(np.float32)
        test = np.asarray(h5["test"]).astype(np.float32)
        neighbors = np.asarray(h5["neighbors"]).astype(np.int32)

        # ann-benchmarks 'angular' datasets store unit-normalized vectors so
        # IP-style search works directly. 'euclidean' datasets don't — they're
        # stored as-is and the recall is for L2 distance. For rotorvec /
        # turbovec (which only do IP / cosine), the bench harness normalizes
        # both train and test, then recomputes ground truth on the normalized
        # vectors. We just emit the raw arrays here.
        n_train, dim = train.shape
        n_test = test.shape[0]
        gt_k = neighbors.shape[1]

    write_fvecs(base_path, train)
    write_fvecs(query_path, test)
    write_ivecs(gt_path, neighbors)
    with open(meta_path, "w") as f:
        f.write(
            f"name={name}\n"
            f"distance={distance}\n"
            f"n_train={n_train}\n"
            f"n_queries={n_test}\n"
            f"dim={dim}\n"
            f"gt_k={gt_k}\n"
        )
    print(
        f"  done: n_train={n_train} n_queries={n_test} dim={dim} gt_k={gt_k} "
        f"→ {out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "datasets",
        nargs="+",
        help=f"one or more of: {', '.join(DATASETS)}",
    )
    args = parser.parse_args()
    for name in args.datasets:
        print(f"\n[{name}]")
        prep_one(name)
    print("\nAll done.")


if __name__ == "__main__":
    main()
