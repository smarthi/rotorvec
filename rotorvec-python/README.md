# rotorvec

Vector index using **block-diagonal Clifford-rotor quantization**. Three
rotation variants, one API: `Planar2` (2D Givens), `Rotor3` (Cl(3,0)
sandwich, default), `Iso4` (4D quaternion).

A block-diagonal cousin of [turbovec](https://github.com/RyanCodrai/turbovec)
(TurboQuant): same compression and bit-plane layout, but the dense d×d
random rotation is replaced by small per-block rotors — O(d) work, ~d
parameters, no BLAS dependency.

See the [main repo README](https://github.com/suneelmarthi/rotorvec) for
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

Accepted `rotation` values: `"planar2"`, `"rotor3"` (default), `"iso4"`
(case-insensitive; aliases `"givens"`, `"clifford"`, `"quaternion"` also
work).

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
