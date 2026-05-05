# rotorvec

A vector index built on **block-diagonal rotor quantization** — a family of
data-oblivious quantizers that swap TurboQuant's dense d×d random orthogonal
matrix for small per-block rotations from Clifford algebra. Inspired by
[turbovec](https://github.com/RyanCodrai/turbovec) (TurboQuant) and the
[RotorQuant paper](https://www.scrya.com/rotorquant.pdf).

> *"Replace the d×d random orthogonal matrix with Clifford rotors... exploiting
> algebraic sparsity"* — RotorQuant paper

## Three rotation variants

| Variant | Block | Algebra | Params (d=128) | When to pick it |
|---|---:|---|---:|---|
| `Planar2` | 2 | 2D Givens (SO(2)) | 128 | Cheapest. Per the RotorQuant paper, **best PPL** for KV-cache decorrelation. |
| `Rotor3` | 3 | Cl(3,0) sandwich `R v R̃` | 172 | Default. Richer algebraic structure than Givens, modest cost. |
| `Iso4` | 4 | Quaternion left-iso (SO(4)) | 128 | Most decorrelation per group; slightly more compute than `Planar2`. |

For comparison, TurboQuant uses **16,384 parameters** (a 128×128 matrix) and
O(d²) FMAs per vector. All three rotorvec variants are O(d) per vector.

## Quick start

### Python

Build the extension locally with [maturin](https://github.com/PyO3/maturin)
(a published wheel is on the v0.2 roadmap):

```bash
cd rotorvec-python
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release
```

Then:

```python
import numpy as np
from rotorvec import RotorQuantIndex, IdMapIndex

# Default: Cl(3,0) rotors, 4-bit codes
index = RotorQuantIndex(dim=1536, bits=4)

# Or pick a variant explicitly
planar = RotorQuantIndex(dim=1536, bits=4, rotation="planar2")
iso    = RotorQuantIndex(dim=1536, bits=4, rotation="iso4")

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

Accepted `rotation` values: `"planar2"`, `"rotor3"` (default), `"iso4"`.

### Rust

```rust
use rotorvec::{RotorQuantIndex, Rotation};

let mut index = RotorQuantIndex::new(1536, 4);                       // Rotor3 default
let mut planar = RotorQuantIndex::with_rotation(1536, 4, Rotation::Planar2);
let mut iso    = RotorQuantIndex::with_rotation(1536, 4, Rotation::Iso4);

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

For each variant the pipeline is the same — only the rotation changes:

1. **Normalize** each vector to unit length, store the norm separately.
2. **Generate per-block rotation matrices** from a deterministic ChaCha8
   seed:
   * `Planar2`: sample one angle θ → 2×2 Givens matrix.
   * `Rotor3`: sample a unit Cl(3,0) rotor `R = (s, b₁₂, b₁₃, b₂₃)` →
     3×3 SO(3) matrix (the sandwich `R v R̃` reduces algebraically to a
     3×3 rotation for grade-1 input).
   * `Iso4`: sample a unit quaternion `q = (w, x, y, z)` → 4×4 SO(4) matrix
     (left-isoclinic action `M v ↔ q · v`).
3. **Block-rotate** the unit vector — block-diagonal matrix multiply, no
   inter-block dependencies.
4. **Quantize** each rotated coordinate to a Lloyd-Max centroid (2/3/4 bit).
5. **Bit-pack** into bit-plane format.

Search reverses the process: rotate the query, accumulate per-coordinate
inner products against each stored vector's quantized codes, multiply by
stored norms, return top-k.

The trade-off across all three variants: block-diagonal rotation only
decorrelates within blocks, not across them. Per the RotorQuant paper this
is sufficient for KV cache vectors (low-rank manifolds). For general
embedding search, recall vs turbovec is an empirical question — benchmark
before committing.

## What's different from turbovec

| | turbovec (TurboQuant) | rotorvec (any variant) |
|---|---|---|
| Decorrelation | Dense d×d random orthogonal matrix | Block-diagonal small rotations |
| Rotation cost | O(d²) per vector (BLAS GEMM) | O(d) per vector |
| Parameters | d² floats | ~d floats |
| BLAS dependency | Yes (Accelerate / OpenBLAS) | None |
| Decorrelation scope | All d coordinates | Within 2/3/4-element blocks only |

## Repo layout

This is a Cargo workspace with two members:

* [`rotorvec/`](rotorvec) — the Rust library (`use rotorvec::...`).
* [`rotorvec-python/`](rotorvec-python) — PyO3 bindings (`import rotorvec`).

Run the Rust test suite with `cargo test -p rotorvec --release`. Run the
Python suite with `cd rotorvec-python && pytest tests/` after `maturin
develop`.

## Status

**v0.1 — proof of concept**
- All three variants implemented and tested
  - 13 Rust tests (unit + integration + doctest)
  - 13 Python tests (parametrized across all three variants)
- Pure Rust, no SIMD intrinsics
- Will be slower than turbovec's hand-tuned NEON/AVX-512 search by 3–5×
- Use for: experimentation, recall benchmarks, baseline implementations

**Roadmap (v0.2+)**
- Published wheels on PyPI
- NEON / AVX-512 search kernels (port turbovec's blocked layout)
- Recall benchmarks across all three variants vs turbovec on standard
  datasets (GloVe, SIFT, GIST)
- LangChain / LlamaIndex / Haystack integrations

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
