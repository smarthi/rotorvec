# v0.2.4 — Walsh-Hadamard cross-block mixing closes the SIFT recall gap

v0.2.2's real-dataset benchmarks surfaced a sharp finding: on data with strong cross-coordinate correlations (SIFT-1M image features), rotorvec's block-diagonal rotation lagged turbovec on recall by 10–15 percentage points. Block-only mixing wasn't enough.

v0.2.3 adds **`Rotation::WalshRotor3`** — a signed Walsh-Hadamard transform applied *before* the existing Cl(3,0) block-diagonal rotation. Cost: `O(d log d)` per vector, sitting between the `O(d)` block-only and `O(d²)` dense-GEMM ends of the spectrum.

The result: rotorvec now beats turbovec on the recall benchmark that was previously its weakest case.

No on-disk format changes. v0.2.2 indices load and search unchanged under v0.2.3. The new variant is opt-in via `Rotation::WalshRotor3`.

Available now on [crates.io](https://crates.io/crates/rotorvec/0.2.3) and [PyPI](https://pypi.org/project/rotorvec/0.2.3/).

## Headline result

**SIFT-1M, n=1M, d=128, 4-bit, k=10:**

| Method | Recall@10 | Build (s) | QPS |
|---|---:|---:|---:|
| FAISS IndexPQFastScan (data-aware baseline) | 0.818 | 2.86 | 6,042 |
| turbovec | 0.487 | 5.45 | 4,652 |
| rotorvec rotor3 (v0.2.2, block-only) | 0.383 | 2.52 | 3,258 |
| **rotorvec walshrotor3 (v0.2.3, +WHT)** | **0.496** | **3.04** | 3,212 |

WalshRotor3 beats turbovec on recall (+0.9pp) while keeping rotorvec's 1.8× build-time advantage (3.04s vs 5.45s). Search QPS is essentially unchanged from `Rotor3` — the WHT pass runs once per query at LUT-build time, not per database vector.

## No regression on data that wasn't the problem

**Synthetic Gaussian, d=128, n=50K, 4-bit, k=10:**

| Method               | Recall@10 | QPS |
|----------------------|---:|---:|
| rotor3 (v0.2.2)      | 0.818 | 77,577 |
| walshrotor3 (v0.2.4) | 0.817 | 75,470 |

**Synthetic Gaussian, d=512, n=50K, 4-bit, k=10:**

| Method               | Recall@10 | QPS |
|----------------------|---:|---:|
| rotor3 (v0.2.2)      | 0.827 | 24,831 |
| walshrotor3 (v0.2.4) | 0.825 | 23,729 |

Within noise. The WHT pass is invisible on data where it isn't needed.

## How it works

The pipeline per vector:

1. **Normalize** to unit length, store norm separately *(existing)*
2. **Sign-flip** by a deterministic ±1 vector seeded from the index's RNG stream
3. **Fast Walsh-Hadamard transform** (in-place butterfly, `O(d log d)`)
4. **Rescale** by `1/sqrt(d)` to keep the unit-norm invariant
5. **Block-diagonal rotation** (Cl(3,0) rotors per 3-block) *(existing)*
6. **Quantize** via Lloyd-Max *(existing)*

Steps 2–4 are new and only run when `Rotation::WalshRotor3` is selected. The signed-WHT pre-pass is structurally similar to the random projections in FastFood / SRHT — spreads each coordinate's energy across all `d` outputs cheaply, breaking up cross-coordinate correlations that block-diagonal rotation alone preserves.

## Constraint: `dim` must be a power of two

The FWHT butterfly only handles power-of-two lengths. `Rotation::WalshRotor3` therefore covers:

- ✅ `dim ∈ {128, 256, 512, 1024, 2048, 4096}` — works directly
- ❌ `dim ∈ {100, 384, 768, 1536}` — use one of the other variants

`with_options()` panics with a clear message if dim isn't a power of two when `WalshRotor3` is selected:

```
Rotation::WalshRotor3 requires dim to be a power of 2 (got 192)
```

Generalizing the cross-block mixing scheme to arbitrary `dim` is a v0.2.4 candidate.

## Picking the right variant

| Your data has... | Pick |
|---|---|
| `dim` is a power of 2 (SIFT-128, CLIP-512, BGE-1024) | `WalshRotor3` — default unless you have a reason not to |
| `dim` not a power of 2 (GloVe-100, MiniLM-384, CLIP-768, ada-1536) | `Rotor3` (default) or `Planar2` / `Iso4` |
| Want the smallest cost | `Planar2` — 2D Givens, fewest FMAs |

The bench harness includes WalshRotor3 in the `all` method set; you can also select it explicitly via `--methods walshrotor3`.

## Install / upgrade

```bash
# Rust
cargo update -p rotorvec

# Python
pip install -U rotorvec
```

To opt into the new variant:

```rust
use rotorvec::{RotorQuantIndex, Rotation};
let mut idx = RotorQuantIndex::with_rotation(128, 4, Rotation::WalshRotor3);
```

```python
from rotorvec import RotorQuantIndex
idx = RotorQuantIndex(dim=128, bits=4, rotation="walshrotor3")
```

Existing code that uses `Rotor3` / `Planar2` / `Iso4` keeps working unchanged.

## New tests (3, total 26)

- `walsh_rotor3_roundtrip_pow2_dim` — save/load preserves the variant and produces bit-identical search results
- `walsh_rotor3_rejects_non_pow2_dim` — panics with a clear message on incompatible dim
- `walsh_rotor3_top1_recall_beats_random` — recall sanity floor at d=256

Plus 5 unit tests in `wht::tests` covering FWHT involution, norm scaling, signed inverse, and sign-vector determinism.

## CI hardening

Beyond v0.2.4 itself, this release also pins the CI Rust toolchain to `1.95.0` and adds an explicit `rustfmt.toml`. Stable rustfmt has been drifting between Rust releases (we hit two formatting-only CI failures in v0.2.0 → v0.2.3), and the pin makes `cargo fmt --check` reproducible across machines.

## Full changelog

| Commit | Summary |
|---|---|
| [`b30098c`](https://github.com/smarthi/rotorvec/commit/b30098c) | Add Walsh-Hadamard cross-block mixing variant — closes SIFT recall gap |
| [`dd185ef`](https://github.com/smarthi/rotorvec/commit/dd185ef) | ci: fix rustfmt — collapse walshrotor3 match arm to one line |
| [`90fec59`](https://github.com/smarthi/rotorvec/commit/90fec59) | ci: pin rustfmt config + Rust 1.95.0 toolchain |

## What's next (v0.2.4+)

| Milestone | Focus |
|---|---|
| **v0.2.4** | Generalized cross-block mixing for non-power-of-2 dim (covers 384, 768, 1536) |
| **v0.2.5** | x86 AVX2 / AVX-512 search kernels (cloud x86 coverage) |
| **v0.2.6** | NEON for 2-bit / 3-bit code widths |
| **v0.3.0** | HNSW wrapper for production-scale corpora (>10M vectors) |

## Attribution

The signed-WHT-then-quantize idea has deep roots — see also FastFood (Le et al. 2013), SRHT (Tropp 2011), and QJL (Zandieh et al. 2024) for related structured random projection schemes. The specific composition with Cl(3,0) block rotors is new to rotorvec; the underlying TurboQuant algorithm and bit-plane layout are unchanged from [turbovec](https://github.com/RyanCodrai/turbovec) (Ryan Codrai) and the original [TurboQuant paper](https://arxiv.org/abs/2504.19874) (Google Research).
