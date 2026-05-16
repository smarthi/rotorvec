# rotorvec

Vector index using **block-diagonal Clifford-rotor quantization** with
optional **Walsh-Hadamard cross-block mixing**. Four rotation variants,
one API:

- `"planar2"` — 2D Givens, O(d), cheapest
- `"rotor3"` — Cl(3,0) sandwich, O(d), default
- `"iso4"` — quaternion left-iso, O(d)
- `"walshrotor3"` ✨ — signed FWHT then Cl(3,0) sandwich, O(d log d).
  Closes the recall gap on data with strong cross-coordinate
  correlations (image features, SIFT). Requires
  `dim ∈ {128, 256, 512, 1024, 2048, 4096}`.

A block-diagonal cousin of [turbovec](https://github.com/RyanCodrai/turbovec)
(TurboQuant): same compression and bit-plane layout, but the dense d×d
random rotation is replaced by small per-block rotors. Beats turbovec on
SIFT-1M recall (0.496 vs 0.487) at 1.8× faster build time when
`"walshrotor3"` is used.

See the [main repo README](https://github.com/smarthi/rotorvec) for
the full algorithm description, attribution, and benchmarks.

## Install

```bash
pip install rotorvec
```

Wheels are built for Linux (x86_64, aarch64), macOS (x86_64, aarch64),
and Windows (x64), abi3-py310 (one binary per platform covers Python
3.10–3.13).

## Usage

```python
import numpy as np
from rotorvec import RotorQuantIndex, IdMapIndex

# Default: Cl(3,0) rotors, 4-bit codes
index = RotorQuantIndex(dim=1536, bits=4)

# Or pick a variant explicitly
planar = RotorQuantIndex(dim=1536, bits=4, rotation="planar2")
iso    = RotorQuantIndex(dim=1536, bits=4, rotation="iso4")

# Power-of-two dim → use walshrotor3 for the strongest recall
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
`"walshrotor3"` (case-insensitive; aliases `"givens"`, `"clifford"`,
`"quaternion"`, `"walsh"`, `"walsh-rotor3"`, `"rotor3+walsh"` also work).

## Development

```bash
git clone https://github.com/smarthi/rotorvec
cd rotorvec/rotorvec-python
uv venv --python 3.10
uv pip install -e .
uv run pytest tests/
```

## License

MIT — see [LICENSE](https://github.com/smarthi/rotorvec/blob/main/LICENSE).
