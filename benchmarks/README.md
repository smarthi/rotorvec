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

## v0.2.1 findings — NEON search kernel landed

### Synthetic Gaussian, n=100K, d=768, 4-bit, k=10

| Method | Recall@10 | Build (s) | QPS |
|---|---:|---:|---:|
| turbovec        | 0.814 | 3.39 | **11,123** |
| rotorvec planar2 | 0.822 | **1.64** | 7,638 |
| rotorvec rotor3  | 0.817 | 1.64 | 7,383 |
| rotorvec iso4    | 0.812 | **1.56** | 7,922 |

### Synthetic Gaussian, n=50K, d=128, 4-bit, k=10

| Method | Recall@10 | Build (s) | QPS |
|---|---:|---:|---:|
| turbovec        | 0.820 | 0.41 | **87,040** |
| rotorvec planar2 | 0.817 | **0.26** | 83,588 |
| rotorvec rotor3  | 0.818 | 0.27 | 84,531 |
| rotorvec iso4    | 0.817 | **0.26** | 81,240 |

### v0.2.0 → v0.2.1 speedup

| | v0.2.0 (scalar) | v0.2.1 (NEON) | Speedup |
|---|---:|---:|---:|
| rotorvec, d=128 | ~660 QPS | ~83,000 QPS | ~126× |
| rotorvec, d=768 | ~58 QPS | ~7,650 QPS | ~132× |

vs turbovec on aarch64 (Apple Silicon): **96% parity at d=128**, **~70% at d=768**. Build remains 2× faster than turbovec across both dims thanks to block-diagonal rotation.

### What this tells us

**Recall: rotorvec stays competitive after SIMD.** Moving from scalar
f32 accumulation to NEON's u8 quantized lookup tables introduces a small
amount of quantization noise — recall drops ~1 percentage point compared
to the v0.2.0 scalar numbers. That matches turbovec's exact same trade-off
(same u8 LUT design) and lands all four methods within ~2pp of each other.

**Throughput is now ~70–96% of turbovec.** At d=128 we're at parity
(83K vs 87K QPS = 96%); at d=768 there's a ~30% gap likely from
turbovec's deeper inner-loop unrolling and FAISS-style register
scheduling. Stretch goal "beat turbovec" wasn't met, but v0.2.1
establishes the platform — further inner-loop tuning is a small follow-up,
not a re-architecture.

**Rotor3 padding fix landed in v0.2.0.** Earlier benchmarks showed
Rotor3 trailing Planar2/Iso4 by ~5 percentage points at d=128 because
128 isn't divisible by 3 — the last 3-block was `[v_127, 0, 0]` and
rotation mixed real data into the padded coordinate we then dropped.
v0.1.2 uses an identity matrix for the trailing partial block. Recall
went 0.782 → 0.827 at d=128.

**Build: rotorvec is still 2× faster.** No BLAS GEMM, just
block-diagonal 3×3 (or smaller) matmuls. Holds across both dims.

### v0.2.x roadmap implication

The benchmark answered the v0.2.0 gating question ("is the algorithm
worth optimizing?") with a yes. v0.2.1 then closed the search-throughput
gap by ~130× via the NEON kernel.

What's left in v0.2.x:

1. **v0.2.2** — Inner-loop tuning to close the remaining ~30% gap at
   d=768 (8-way unroll, prefetch hints, possible micro-arch-specific
   tuning for Apple Silicon vs Graviton)
2. **v0.2.3** — AVX-512 / AVX2 kernels for x86 (cloud x86 coverage)
3. **v0.2.4** — Real datasets (HDF5 ann-benchmarks loader)
4. **v0.2.5** — Python benchmark suite + FAISS comparison

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
