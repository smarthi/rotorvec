# rotorvec

[![CI](https://github.com/smarthi/rotorvec/actions/workflows/ci.yml/badge.svg)](https://github.com/smarthi/rotorvec/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rotorvec?label=pypi&color=blue)](https://pypi.org/project/rotorvec/)
[![crates.io](https://img.shields.io/crates/v/rotorvec?label=crates.io&color=blue)](https://crates.io/crates/rotorvec)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A vector index built on **block-diagonal rotor quantization** — a family of
data-oblivious quantizers that swap TurboQuant's dense d×d random orthogonal
matrix for small per-block rotations from Clifford algebra. Inspired by
[turbovec](https://github.com/RyanCodrai/turbovec) (TurboQuant) and the
[RotorQuant paper](https://www.scrya.com/rotorquant.pdf).

> *"Replace the d×d random orthogonal matrix with Clifford rotors... exploiting
> algebraic sparsity"* — RotorQuant paper

## Four rotation variants

| Variant | Block | Algebra | Cost / vector | When to pick it |
|---|---:|---|---|---|
| `Planar2` | 2 | 2D Givens (SO(2)) | O(d) | Cheapest. Per the RotorQuant paper, **best PPL** for KV-cache decorrelation. |
| `Rotor3` | 3 | Cl(3,0) sandwich `R v R̃` | O(d) | Default. Richer algebraic structure than Givens, modest cost. |
| `Iso4` | 4 | Quaternion left-iso (SO(4)) | O(d) | Most decorrelation per group; slightly more compute than `Planar2`. |
| `WalshRotor3` ✨ | 3 + global | signed FWHT then Cl(3,0) sandwich | O(d log d) | **v0.2.3 default for power-of-two dims.** Closes the recall gap on data with strong cross-coordinate correlations (image features, SIFT). Requires `dim ∈ {128, 256, 512, 1024, 2048, 4096}`. |

For comparison, TurboQuant uses **16,384 parameters** (a 128×128 matrix) and
O(d²) FMAs per vector. The first three rotorvec variants are O(d) per vector;
`WalshRotor3` is O(d log d) — sitting between block-only and dense-GEMM in
both cost and decorrelation power.

### Picking the right variant

| Your data has... | Pick |
|---|---|
| `dim` is a power of 2 (SIFT-128, CLIP-512, BGE-1024) | `WalshRotor3` — strongest recall |
| `dim` not a power of 2 (GloVe-100, MiniLM-384, CLIP-768, OpenAI-1536) | `Rotor3` (default), `Planar2`, or `Iso4` |
| Want the cheapest per-vector cost | `Planar2` |

## Quick start

### Python

Build the extension locally with [uv](https://github.com/astral-sh/uv)
(a published wheel is on the v0.2 roadmap). uv handles the venv, pulls
in `maturin` as the PEP 517 build backend automatically, and produces
an editable install:

```bash
cd rotorvec-python
uv venv --python 3.12
uv pip install -e .
```

Then either activate the venv (`source .venv/bin/activate`) or prefix
your commands with `uv run` (e.g. `uv run pytest tests/`).

Then:

```python
import numpy as np
from rotorvec import RotorQuantIndex, IdMapIndex

# Default: Cl(3,0) rotors, 4-bit codes
index = RotorQuantIndex(dim=1536, bits=4)

# Or pick a variant explicitly
planar = RotorQuantIndex(dim=1536, bits=4, rotation="planar2")
iso    = RotorQuantIndex(dim=1536, bits=4, rotation="iso4")

# Power-of-two dim → use WalshRotor3 for best recall on hard data
walsh  = RotorQuantIndex(dim=1024, bits=4, rotation="walshrotor3")

vectors = np.random.randn(10_000, 1536).astype(np.float32)
queries = np.random.randn(8, 1536).astype(np.float32)

index.add(vectors)
scores, indices = index.search(queries, k=10)   # (8, 10) float32 / int64

index.write("index.rv")
loaded = RotorQuantIndex.load("index.rv")
```

Stable external ids:

```python
ids = np.array([1001, 1002, 1003], dtype=np.uint64)
m = IdMapIndex(dim=1536, bits=4, rotation="planar2")
m.add_with_ids(vectors[:3], ids)
scores, returned_ids = m.search(queries, k=3)   # ids are uint64
m.remove(1002)
```

Accepted `rotation` values: `"planar2"`, `"rotor3"` (default), `"iso4"`,
`"walshrotor3"` (aliases: `"walsh"`, `"walsh-rotor3"`).

### Rust

```rust
use rotorvec::{RotorQuantIndex, Rotation};

let mut index = RotorQuantIndex::new(1536, 4);                       // Rotor3 default
let mut planar = RotorQuantIndex::with_rotation(1536, 4, Rotation::Planar2);
let mut iso    = RotorQuantIndex::with_rotation(1536, 4, Rotation::Iso4);
let mut walsh  = RotorQuantIndex::with_rotation(1024, 4, Rotation::WalshRotor3); // dim must be pow2

index.add(&vectors);
let results = index.search(&queries, 10);
index.write("index.rv").unwrap();
let loaded = RotorQuantIndex::load("index.rv").unwrap();
```

For stable external ids:

```rust
use rotorvec::{IdMapIndex, Rotation};

let mut index = IdMapIndex::with_rotation(1536, 4, Rotation::Planar2);
index.add_with_ids(&vectors, &[1001, 1002, 1003]);
let (scores, ids) = index.search(&queries, 10);
index.remove(1002);
```

## How it works

The pipeline per vector:

1. **Normalize** to unit length, store the norm separately.
2. **(WalshRotor3 only)** Sign-flip by a deterministic ±1 vector, then
   apply the fast Walsh-Hadamard transform, then rescale by `1/√d`. This
   spreads each coordinate's energy across all `d` outputs in `O(d log d)`
   work — the cross-block mixing the block-only variants lack.
3. **Generate per-block rotation matrices** from a deterministic ChaCha8
   seed:
   * `Planar2`: one angle θ → 2×2 Givens matrix per pair.
   * `Rotor3` / `WalshRotor3`: a unit Cl(3,0) rotor `R = (s, b₁₂, b₁₃, b₂₃)`
     → 3×3 SO(3) matrix (the sandwich `R v R̃` reduces algebraically to a
     3×3 rotation for grade-1 input).
   * `Iso4`: a unit quaternion `q = (w, x, y, z)` → 4×4 SO(4) matrix
     (left-isoclinic action `M v ↔ q · v`).
4. **Block-rotate** — block-diagonal matrix multiply, no inter-block
   dependencies.
5. **Quantize** each rotated coordinate to a Lloyd-Max centroid (2/3/4 bit).
6. **Bit-pack** into bit-plane format.

Search reverses the process: apply the same WHT pre-pass (if any) and
block rotation to the query, accumulate per-coordinate inner products
against each stored vector's quantized codes, multiply by stored norms,
return top-k.

The trade-off across block-only variants (`Planar2`, `Rotor3`, `Iso4`):
they only decorrelate within blocks, not across them. That's fine for
data with weak cross-coordinate correlations (typical text embeddings)
but loses recall on data with strong cross-coord structure (image
features like SIFT). `WalshRotor3` exists specifically to close that
gap on power-of-2 dimensions.

## What's different from turbovec

| | turbovec (TurboQuant) | rotorvec (block-only) | rotorvec (`WalshRotor3`) |
|---|---|---|---|
| Decorrelation | Dense d×d random orthogonal matrix | Block-diagonal small rotations | Signed FWHT + block rotation |
| Rotation cost | O(d²) per vector (BLAS GEMM) | O(d) per vector | O(d log d) per vector |
| Parameters | d² floats | ~d floats | ~d floats + d sign bits |
| BLAS dependency | Yes (Accelerate / OpenBLAS) | None | None |
| Decorrelation scope | All d coordinates | Within 2/3/4-element blocks only | All d coordinates |
| Dim constraint | `dim % 8 == 0` | `dim % 8 == 0` | `dim` power of 2 |

## Repo layout

This is a Cargo workspace with two members:

* [`rotorvec/`](rotorvec) — the Rust library (`use rotorvec::...`).
* [`rotorvec-python/`](rotorvec-python) — PyO3 bindings (`import rotorvec`).

Run the Rust test suite with `cargo test -p rotorvec --release`. Run the
Python suite with `cd rotorvec-python && uv run pytest tests/` (after
`uv pip install -e .`).

## CI / Release

Three GitHub Actions workflows live in [`.github/workflows/`](.github/workflows):

| Workflow | Trigger | What it does |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | every push to `main` and every PR | `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`, `pytest` across Python 3.10 / 3.12 / 3.13 on Linux + macOS |
| [`release-pypi.yml`](.github/workflows/release-pypi.yml) | tag `py-vX.Y.Z` | Builds wheels for Linux x86_64/aarch64, macOS x86_64/aarch64, Windows x64; builds sdist; publishes to PyPI via OIDC trusted publishing |
| [`release-crates.yml`](.github/workflows/release-crates.yml) | tag `vX.Y.Z` | `cargo publish -p rotorvec` to crates.io via OIDC trusted publishing |

Tagging convention:

```bash
# Rust crate release
git tag v0.2.0 && git push origin v0.2.0

# Python wheel release (independent versioning)
git tag py-v0.2.0 && git push origin py-v0.2.0
```

**One-time PyPI / crates.io setup** (see each registry's docs for the
exact UI):

1. **PyPI**: create the project on [pypi.org](https://pypi.org/manage/account/publishing/),
   add a "pending publisher" pointing at `smarthi/rotorvec`, workflow
   `release-pypi.yml`, environment `pypi`. Then create a GitHub
   environment named `pypi` on the repo (Settings → Environments).
2. **crates.io**: configure OIDC trusted publishing in your crates.io
   account settings, scoped to the same repo + workflow + environment
   (`crates-io`). Create the matching GitHub environment.

Once configured, neither workflow needs API tokens stored as secrets.

## Status

**v0.2.3 — current release.** Published on [crates.io](https://crates.io/crates/rotorvec)
and [PyPI](https://pypi.org/project/rotorvec/). 26 tests passing (16 unit +
9 integration + 1 doctest). CI green on Linux + macOS, Python 3.10/3.12/3.13.

### Release history

| Version | Headline |
|---|---|
| **v0.2.3** | `WalshRotor3` variant — beats turbovec on SIFT-1M recall (0.496 vs 0.487) at 1.8× faster build |
| v0.2.2 | Real-dataset benchmarks (GloVe-100, SIFT-1M) + FAISS baseline + apples-to-apples audit |
| v0.2.1 | NEON 4-bit search kernel — 130× faster search on aarch64, 96% of turbovec QPS at d=128 |
| v0.2.0 | Benchmark harness + Rotor3 trailing-padding fix (recall@10 at d=128: 0.78 → 0.83) |
| v0.1.x | Initial release — three rotation variants, scalar search, file format, CI/CD |

### Roadmap (v0.2.4+)

- **v0.2.4** — Generalized cross-block mixing for non-power-of-2 dim (covers 384, 768, 1536)
- **v0.2.5** — x86 AVX2 / AVX-512 search kernels (cloud x86 coverage)
- **v0.2.6** — NEON for 2-bit / 3-bit code widths
- **v0.3.0** — HNSW wrapper for production-scale corpora (>10M vectors)
- LangChain / LlamaIndex / Haystack integrations once HNSW lands

## File format

The on-disk format encodes which rotation variant the index uses, so loading
a `.rv` or `.rvim` file picks the right block size automatically. See the
header doc-comment in [`src/io.rs`](src/io.rs) for the byte layout.

## Attribution

- **TurboQuant** — Google Research, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **turbovec** — Ryan Codrai, [github.com/RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec) — architecture and bit-packing layout adapted here
- **RotorQuant** — John Pope, [scrya.com/rotorquant.pdf](https://www.scrya.com/rotorquant.pdf) — Clifford rotor (`Rotor3`) decorrelation algorithm
- **PlanarQuant / IsoQuant** — [ParaMind2025](https://github.com/ParaMind2025/isoquant) — 2D Givens (`Planar2`) and 4D quaternion (`Iso4`) variants

## License

MIT
