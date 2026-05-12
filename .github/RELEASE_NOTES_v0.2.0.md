# v0.2.0 — Validated algorithm + benchmark harness + Rotor3 padding fix

This release moves rotorvec from "pure proof of concept" to "validated against the turbovec baseline." Recall is competitive, the algorithm works, and we now have a benchmark harness to drive future optimization.

Available now on [crates.io](https://crates.io/crates/rotorvec/0.2.0) and [PyPI](https://pypi.org/project/rotorvec/0.2.0/).

## Highlights

### 🎯 Recall is competitive with turbovec

Block-diagonal rotation gives recall on par with turbovec's full d×d random orthogonal matrix on synthetic Gaussian unit-vector data — the v0.1 hypothesis is supported.

**Synthetic Gaussian, n=100K, d=768, 4-bit, k=10:**

| Method | Recall@10 | Build (s) | QPS |
|---|---:|---:|---:|
| turbovec | 0.814 | 3.43 | 10,715 |
| rotorvec planar2 | **0.833** | **1.56** | 59 |
| rotorvec rotor3 | 0.829 | 1.57 | 56 |
| rotorvec iso4 | 0.829 | 1.48 | 57 |

All three rotorvec variants slightly *exceed* turbovec on recall while building 2× faster. The 100× search-QPS gap is entirely turbovec's hand-tuned NEON kernels vs our scalar Rust + rayon — closing that gap is the v0.2.2 target.

### 🐛 Rotor3 trailing-padding fix

When `dim % 3 != 0` (e.g. d=128, the most common embedding dimension), the v0.1 Rotor3 generated a random SO(3) rotation for every 3-block including the trailing partial one. The trailing block was `[v_{dim-2}, v_{dim-1}, 0]` — random rotation mixed real coordinates into the zero-padded slot we then dropped before quantization, throwing away signal.

**v0.2.0 uses an identity matrix for the trailing partial block.** Real coords pass through untouched.

| | v0.1 | v0.2.0 |
|---|---:|---:|
| Rotor3 recall@10 at d=128 | 0.782 | **0.827** |
| Rotor3 recall@10 at d=768 | 0.829 | unchanged |

Only Rotor3 is affected — Planar2 and Iso4 always divide cleanly when `dim % 8 == 0`.

### 📊 New `rotorvec-bench` crate

A new workspace member at `benchmarks/`. Compares all three rotorvec variants against turbovec on recall@k and search QPS, writes markdown + JSON reports.

```bash
cargo build -p rotorvec-bench --release

# Synthetic dataset (no network)
./target/release/rotorvec-bench --dataset random:100000:768 --bits 4 --k 10

# SIFT-1M from INRIA (downloads ~160 MB on first run)
./target/release/rotorvec-bench --dataset sift-1m --bits 4 --k 10
```

See [`benchmarks/README.md`](https://github.com/smarthi/rotorvec/blob/main/benchmarks/README.md) for full results and methodology.

## ⚠️ Breaking change

**Indices saved with v0.1.x using `Rotation::Rotor3` and `dim % 3 != 0` won't decode correctly under v0.2.0.** The rotation matrices changed (the trailing block is now identity instead of random), so packed codes from older versions decode against the wrong inverse.

**Affected configurations:** Rotor3 with `dim ∈ {128, 256, 512, 1024, ...}` (anything not divisible by 3).

**Unaffected:**
- Any Planar2 or Iso4 index
- Rotor3 indices with `dim ∈ {192, 384, 768, 1536, ...}` (divisible by 3) — bit-identical between versions

**Migration:** re-encode affected indices on upgrade (call `add()` again with your original float vectors).

## Install / upgrade

```bash
# Rust
cargo add rotorvec        # new project
cargo update -p rotorvec  # existing project

# Python
pip install -U rotorvec
```

## Full changelog

| Commit | Summary |
|---|---|
| [`07bbe73`](https://github.com/smarthi/rotorvec/commit/07bbe73) | Add v0.2.0 benchmark harness — recall + throughput vs turbovec |
| [`f477959`](https://github.com/smarthi/rotorvec/commit/f477959) | Fix Rotor3 padding: use identity for trailing partial block |
| [`c4cec90`](https://github.com/smarthi/rotorvec/commit/c4cec90) | Bump rotorvec + rotorvec-python to 0.2.0; fix README owner reference |
| [`683a28d`](https://github.com/smarthi/rotorvec/commit/683a28d) | ci: apply cargo fmt across workspace; sync rotorvec-python crate version |

## What's next (v0.2.x roadmap)

| Milestone | Focus | Why |
|---|---|---|
| **v0.2.1** | Real datasets (HDF5 ann-benchmarks loader); pre-commit fmt hook | Confirm recall holds outside synthetic Gaussian |
| **v0.2.2** | NEON SIMD search kernel for aarch64 | Close the 100× search QPS gap on Apple Silicon + Graviton |
| **v0.2.3** | AVX-512 / AVX2 search kernels for x86 | Cloud x86 coverage |
| **v0.2.4** | Python benchmark suite + FAISS comparison | Broader baseline for the next LinkedIn post |

## Attribution

- **TurboQuant** — Google Research, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **turbovec** — Ryan Codrai, [github.com/RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec) — architecture and bit-packing layout adapted here
- **RotorQuant** — John Pope, [scrya.com/rotorquant.pdf](https://www.scrya.com/rotorquant.pdf) — Clifford rotor decorrelation algorithm
- **PlanarQuant / IsoQuant** — [github.com/ParaMind2025/isoquant](https://github.com/ParaMind2025/isoquant) — block-diagonal Givens / quaternion variants
