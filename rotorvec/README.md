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

```toml
[dependencies]
rotorvec = "0.1"
```

## Usage

```rust
use rotorvec::{RotorQuantIndex, Rotation};

// Default: Cl(3,0) rotors, 4-bit codes
let mut index = RotorQuantIndex::new(1536, 4);

// Or pick a variant explicitly
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

## Python bindings

A Python package with the same API is published as
[`rotorvec`](https://pypi.org/project/rotorvec) on PyPI:

```bash
pip install rotorvec
```

## License

MIT — see [LICENSE](https://github.com/suneelmarthi/rotorvec/blob/main/LICENSE).
