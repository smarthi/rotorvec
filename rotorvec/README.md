# rotorvec

Vector index using **block-diagonal Clifford-rotor quantization** with
optional **Walsh-Hadamard cross-block mixing**. Four rotation variants,
one API:

- `Planar2` — 2D Givens, O(d), cheapest
- `Rotor3` — Cl(3,0) sandwich, O(d), default
- `Iso4` — quaternion left-iso, O(d)
- `WalshRotor3` ✨ — signed FWHT then Cl(3,0) sandwich, O(d log d),
  closes the recall gap on data with strong cross-coordinate correlations
  (image features, SIFT). Requires `dim ∈ {128, 256, 512, 1024, 2048, 4096}`.

A block-diagonal cousin of [turbovec](https://github.com/RyanCodrai/turbovec)
(TurboQuant): same compression and bit-plane layout, but the dense d×d
random rotation is replaced by small per-block rotors. Beats turbovec on
SIFT-1M recall (0.496 vs 0.487) at 1.8× faster build time when
`WalshRotor3` is used.

See the [main repo README](https://github.com/smarthi/rotorvec) for
the full algorithm description, attribution, and benchmarks.

## Install

```toml
[dependencies]
rotorvec = "0.2"
```

## Usage

```rust
use rotorvec::{RotorQuantIndex, Rotation};

// Default: Cl(3,0) rotors, 4-bit codes
let mut index = RotorQuantIndex::new(1536, 4);

// Or pick a variant explicitly
let mut planar = RotorQuantIndex::with_rotation(1536, 4, Rotation::Planar2);
let mut iso    = RotorQuantIndex::with_rotation(1536, 4, Rotation::Iso4);

// Power-of-two dim → use WalshRotor3 for the strongest recall
let mut walsh  = RotorQuantIndex::with_rotation(1024, 4, Rotation::WalshRotor3);

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

## Python bindings

A Python package with the same API is published as
[`rotorvec`](https://pypi.org/project/rotorvec) on PyPI:

```bash
pip install rotorvec
```

## License

MIT — see [LICENSE](https://github.com/smarthi/rotorvec/blob/main/LICENSE).
