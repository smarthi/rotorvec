# rotorvec-bench

Recall + throughput benchmarks for rotorvec and turbovec.

## Quick start

```bash
# Build
cargo build -p rotorvec-bench --release

# Synthetic Gaussian unit vectors — n=100k, d=128 (no network needed)
./target/release/rotorvec-bench --dataset random:100000:128 --bits 4 --k 10

# Larger / sentence-transformer dim
./target/release/rotorvec-bench --dataset random:100000:768 --bits 4 --k 10

# SIFT-1M (downloads ~160 MB on first run, may need a working mirror)
./target/release/rotorvec-bench --dataset sift-1m --bits 4 --k 10
```

Results land in `benchmarks/results/` as both `.md` (human-readable) and
`.json` (machine-readable).

## CLI

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `sift-1m` | `random[:N[:D]]` or `sift-1m` |
| `--bits` | `4` | quantization bits per coordinate (2/3/4) |
| `--k` | `10` | top-k |
| `--methods` | `all` | comma-separated: `turbovec,planar2,rotor3,iso4` |
| `--max-train` | (full) | cap training set (smoke test) |
| `--max-queries` | (full) | cap query set |
| `--output` | auto | path for the `.md` report |

## Datasets

| Name | Source | Size | Distance | Notes |
|---|---|---|---|---|
| `random:N:D` | synthetic Gaussian | configurable | inner product | seeded, reproducible. Each coord ~ N(0,1), then unit-normalized |
| `sift-1m` | INRIA TexMex | 1M × 128 | L2 → IP after normalize | ~160 MB download. INRIA's HTTP mirror is intermittently unavailable |

Cached under `~/Library/Caches/rotorvec/` (macOS) or `~/.cache/rotorvec/`
(Linux). Ground truth is recomputed per `(dataset, n_train, n_queries)`
and disk-cached separately (see `ground_truth.rs`).

## Initial findings (v0.2.0)

### Synthetic Gaussian, n=100k, d=768, 4-bit, k=10

| Method | Recall@10 | Build (s) | QPS |
|---|---:|---:|---:|
| turbovec        | 0.814 | 3.43 | 10,715 |
| rotorvec planar2 | **0.833** | **1.56** | 59 |
| rotorvec rotor3  | 0.829 | 1.57 | 56 |
| rotorvec iso4    | 0.829 | 1.48 | 57 |

### Synthetic Gaussian, n=50k, d=128, 4-bit, k=10

| Method | Recall@10 | Build (s) | QPS |
|---|---:|---:|---:|
| turbovec        | 0.820 | 0.41 | 89,370 |
| rotorvec planar2 | **0.828** | **0.26** | 663 |
| rotorvec rotor3  | 0.782 | 0.27 | 653 |
| rotorvec iso4    | 0.826 | 0.25 | 662 |

### What this tells us

**Recall: rotorvec is competitive.** Block-diagonal rotation gives recall
that matches turbovec's full d×d random rotation on synthetic Gaussian
unit-vector data. Two of the three variants (Planar2, Iso4) consistently
match or slightly exceed turbovec at both d=128 and d=768. The "rotorvec
algorithm is sound for general embeddings" hypothesis from v0.1's README
is supported by this initial data.

**Rotor3 underperforms at d=128.** The Cl(3,0) variant lags by ~5
percentage points at d=128. Most likely cause: 128 isn't divisible by 3,
so the last 3-block is `[v_127, 0, 0]` zero-padded. Block rotation mixes
real data into padded coordinates that we then drop. At d=768
(divisible by 3) the gap closes — supporting that diagnosis.
**Action item:** either avoid Rotor3 when `dim % 3 != 0`, or implement
"residual-aware" rotation that doesn't waste bits on padded coords.

**Build: rotorvec is 2× faster.** No BLAS GEMM, just block-diagonal 3×3
(or smaller) matmuls. Expected.

**Search: turbovec is 100×+ faster.** This is *entirely* about turbovec's
hand-written NEON kernels vs our scalar Rust + rayon. Closing this gap
is the primary v0.2.2 work — porting the SIMD-blocked layout and writing
NEON intrinsics for the per-coordinate dot-product accumulation.

### v0.2.x roadmap implication

The benchmark answers the v0.2.0 gating question: **"is the algorithm
worth optimizing for general embeddings?"** Yes — recall is on par with
turbovec at d=128 and d=768. Investing in NEON/AVX-512 kernels is
justified.

Next milestones in priority order:

1. **v0.2.1** — Investigate Rotor3 padding loss; explore real datasets
   (HDF5 ann-benchmarks data via Python prep step), confirm the
   recall story holds outside synthetic Gaussian
2. **v0.2.2** — NEON SIMD kernel for aarch64
3. **v0.2.3** — AVX-512 / AVX2 kernels for x86

## Caveats

- **Synthetic ≠ real.** Random Gaussian unit vectors are an idealized
  case. Embeddings from real models (CLIP, sentence-transformers, GloVe)
  have hubness, clusters, and low-rank structure that change recall
  behavior. The current numbers should be read as "block-diagonal
  rotation isn't broken on idealized data," not "rotorvec ≥ turbovec on
  your dataset."
- **Search QPS is not apples-to-apples.** turbovec uses NEON; rotorvec
  uses scalar Rust. The search-throughput gap will narrow drastically
  in v0.2.2.
- **Single-core variance.** rayon parallelizes across queries. If you
  set `RAYON_NUM_THREADS=1` you get a true per-query measurement.
