# rotorvec

A vector index built on **RotorQuant** — Clifford algebra Cl(3,0) rotors for
data-oblivious vector quantization. Block-diagonal cousin of
[turbovec](https://github.com/RyanCodrai/turbovec)'s TurboQuant.

> *"Replace the d×d random orthogonal matrix with Clifford rotors... exploiting
> algebraic sparsity"* — [RotorQuant paper](https://www.scrya.com/rotorquant.pdf)

## What's different from turbovec

| | turbovec (TurboQuant) | **rotorvec (RotorQuant)** |
|---|---|---|
| Decorrelation | Dense d×d random orthogonal matrix | Block-diagonal SO(3) rotors |
| Rotation cost | O(d²) per vector (BLAS GEMM) | O(d) per vector |
| Parameters | d² floats | 4 × (d/3) floats |
| BLAS dependency | Yes (Accelerate / OpenBLAS) | None |
| Decorrelation scope | All d coordinates | Within 3-element blocks only |

The trade-off: block-diagonal rotation is dramatically cheaper but only
decorrelates within 3-element blocks. Per the RotorQuant paper, this is
sufficient for KV cache vectors (low-rank manifolds). For general embedding
search, recall vs turbovec is an empirical question — benchmark before
committing.

## Quick start

```rust
use rotorvec::RotorQuantIndex;

let mut index = RotorQuantIndex::new(1536, 4);
index.add(&vectors);
let results = index.search(&queries, 10);
index.write("index.rv").unwrap();
let loaded = RotorQuantIndex::load("index.rv").unwrap();
```

For stable external ids:

```rust
use rotorvec::IdMapIndex;

let mut index = IdMapIndex::new(1536, 4);
index.add_with_ids(&vectors, &[1001, 1002, 1003]);
let (scores, ids) = index.search(&queries, 10);
index.remove(1002);
```

## How it works

1. **Normalize** each vector to unit length, store norm separately.
2. **Generate rotors** — one Cl(3,0) rotor R per 3-element block,
   deterministically seeded. Each rotor R = (cos θ/2, sin θ/2 · b̂) where b̂
   is a unit bivector.
3. **Block-rotate** — apply the rotor sandwich `R v R̃` to each 3-block.
   For grade-1 input vectors, this reduces algebraically to a 3×3 SO(3)
   rotation (we precompute the matrix at index init time).
4. **Quantize** each rotated coordinate to a Lloyd-Max centroid (2/3/4 bit).
5. **Bit-pack** into bit-plane format.

Search reverses the process: rotate the query, accumulate per-coordinate
inner products against each stored vector's quantized codes, multiply by
stored norms, return top-k.

## Status

**v0.1 — proof of concept**
- Correct algorithmically
- Pure Rust, no SIMD intrinsics
- Will be slower than turbovec's hand-tuned NEON/AVX-512 search by 3–5×
- Use for: experimentation, recall benchmarks, baseline implementations

**Roadmap (v0.2+)**
- NEON / AVX-512 search kernels (port turbovec's blocked layout)
- PlanarQuant (2D Givens) and IsoQuant (4D quaternion) variants
- Python bindings via PyO3
- Recall benchmarks vs turbovec on standard datasets (GloVe, SIFT, GIST)

## Attribution

- **TurboQuant** — Google Research, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **turbovec** — Ryan Codrai, [github.com/RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec) — architecture and bit-packing layout ported here
- **RotorQuant** — John Pope, [scrya.com/rotorquant.pdf](https://www.scrya.com/rotorquant.pdf) — Clifford rotor decorrelation algorithm

## License

MIT
