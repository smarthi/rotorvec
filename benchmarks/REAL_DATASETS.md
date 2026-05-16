# Real-dataset benchmarks (v0.2.2)

First measurement of rotorvec on standard ANN datasets — **GloVe-100** (gentle
cross-coordinate correlations, typical of word embeddings) and **SIFT-1M**
(strong cross-coordinate correlations from image features). Compared against
turbovec (the rotorvec sibling) and **FAISS IndexPQ + IndexPQFastScan** (the
incumbent in the data-aware family).

All numbers are recall@10 on the precomputed ann-benchmarks ground truth,
recomputed on unit-normalized vectors so all methods are scored against the
same inner-product target. Hardware: M-series Apple Silicon, single-machine
single-process, rayon-parallel where applicable.

## Reproduce

```bash
# One-time: download + convert
uv run python benchmarks/scripts/prep_datasets.py glove-100 sift-128

# rotorvec + turbovec
cargo build -p rotorvec-bench --release
./target/release/rotorvec-bench --dataset glove-100 --bits 4 --k 10
./target/release/rotorvec-bench --dataset sift-1m   --bits 4 --k 10

# FAISS (data-aware baseline, both ADC and FastScan variants)
uv run python benchmarks/scripts/bench_faiss.py glove-100 sift-128 --k 10
```

## Results

### GloVe-100-angular (n=1,183,514, d=100 padded to 104, k=10)

GloVe word vectors trained on 6B-token corpus. Gentle correlations between
coords. The most representative "embedding-like" benchmark in this set.

| Method | Recall@10 | Build (s) | QPS |
|---|---:|---:|---:|
| FAISS IndexPQFastScan (m=100, nbits=4) | **0.806** | 2.94 | **5,941** |
| FAISS IndexPQ (m=100, nbits=4) | 0.810 | 2.67 | 133 |
| turbovec 4-bit | 0.786 | 5.47 | 4,411 |
| rotorvec planar2 4-bit | 0.768 | **2.54** | 3,248 |
| rotorvec rotor3 4-bit | 0.772 | 2.62 | 3,247 |
| rotorvec iso4 4-bit | 0.772 | 2.42 | 3,301 |

### SIFT-1M (n=1,000,000, d=128, k=10)

Image-feature vectors with strong cross-coordinate correlations. A harder
benchmark for data-oblivious methods.

| Method | Recall@10 | Build (s) | QPS |
|---|---:|---:|---:|
| FAISS IndexPQFastScan (m=128, nbits=4) | **0.818** | 2.86 | **6,042** |
| FAISS IndexPQ (m=128, nbits=4) | 0.820 | 2.81 | 132 |
| turbovec 4-bit | 0.487 | 5.44 | 4,702 |
| rotorvec planar2 4-bit | 0.336 | **2.39** | 3,192 |
| rotorvec rotor3 4-bit | 0.383 | 2.53 | 3,339 |
| rotorvec iso4 4-bit | 0.351 | 2.37 | 3,350 |

## What this tells us

### 1. FAISS leads recall — that's the cost of being data-oblivious

FAISS trains k-means centroids per sub-quantizer on the actual data. turbovec
and rotorvec don't train at all — Lloyd-Max centroids derived from a
theoretical Beta distribution applied uniformly across all coordinates.

That's the entire point of the TurboQuant paper: zero training, zero data
passes, no rebuilds when the corpus grows. On natural-looking data
(GloVe-100), the recall gap to data-aware FAISS is **2–4 percentage points**.
That's the tax for never needing to retrain.

If absolute recall matters more than streaming-friendly or privacy-preserving
indexing, FAISS is the right answer. If you can't or won't retrain when your
data shifts (RAG corpora that grow daily, on-device personalized vectors,
multi-tenant pipelines where data can't cross trust boundaries), the
data-oblivious family is the right answer — and within it, you pick by
secondary criteria like build time and platform support.

### 2. rotorvec lags turbovec on hard distributions

On GloVe-100, rotorvec is **1–2pp behind** turbovec on recall. On SIFT-1M
that gap widens to **10–15pp**.

Diagnosis: SIFT-1M coordinates are strongly correlated (image gradients
share structure between adjacent dimensions). turbovec's dense d×d random
orthogonal rotation breaks those correlations globally — every output coord
mixes every input coord. rotorvec's block-diagonal rotation only mixes
within 2/3/4-element blocks, so cross-block correlations survive the
rotation and degrade scalar quantization.

This is the predicted failure mode the v0.1 README flagged ("block-diagonal
only decorrelates within blocks"). v0.2.0 confirmed the algorithm is sound
on idealized data; v0.2.2 maps where it actually stops being sound.

### 3. rotorvec build is still fastest

Across both datasets, rotorvec builds in 2.4–2.6 seconds vs turbovec's
5.5 seconds. The 2× advantage from block-diagonal rotation (no BLAS GEMM)
holds even when search recall lags. For workflows that re-index frequently
(streaming ingest, incremental document corpora, hyperparameter sweeps),
this is a real win that compounds across builds.

### 4. QPS rankings hold across data

| | GloVe-100 | SIFT-1M |
|---|---:|---:|
| FAISS FastScan | 5,941 | 6,042 |
| turbovec | 4,411 | 4,702 |
| rotorvec (NEON) | ~3,300 | ~3,300 |

rotorvec at ~70% of turbovec and ~55% of FAISS FastScan QPS. The remaining
gap is inner-loop tuning (8-way unroll, prefetch hints — currently 4-way
unrolled) and is a v0.2.3 target.

## When to pick rotorvec, today

**Use rotorvec when:**
- You won't or can't retrain centroids on your data
- Your embeddings come from modern models (CLIP, sentence-transformers, BGE,
  text-embedding-3) where cross-coordinate correlations are mild
- Build time matters (frequent re-indexing, sharding)
- You want no BLAS dependency or no Python interop

**Use turbovec when:**
- Same data-oblivious requirement as rotorvec, but recall matters more than
  build time, and your data may have stronger cross-coord correlations
- You're on x86 — turbovec has AVX-512 kernels; rotorvec's x86 SIMD is v0.2.3

**Use FAISS PQ when:**
- You can retrain centroids on your corpus and recall is paramount
- Your data has SIFT-like cross-coordinate correlations
- You're already in the FAISS ecosystem

## Caveats

- **Single-hardware measurement.** Numbers are from one Apple M-series chip,
  thread count = physical cores. Cloud aarch64 (Graviton 3/4) and x86
  numbers will look different.
- **No HNSW/IVF layering.** All four methods are flat scans. Realistic
  production systems wrap PQ in an HNSW graph or IVF index. Throughput
  rankings can change once the wrapper dominates.
- **4-bit only.** 2-bit and 3-bit comparisons are deferred — v0.2.x road map
  targets 4-bit kernels first since it's the sweet spot for most RAG-style
  workloads.
- **GloVe-100 padding.** rotorvec processes GloVe-100 at d=104 (4 trailing
  zero coords) to satisfy `dim % 8 == 0`. That's 16 extra bits per vector
  (~4%) above FAISS's d=100 representation. Negligible but worth noting.

## What changes in v0.2.3+

Two paths under consideration:

1. **Close the recall gap** — add a cheap cross-block permutation pass
   between block-diagonal rotation and quantization. Mixes coordinates across
   blocks without going to a full d×d rotation. Cost: O(d) shuffle. If it
   recovers 5–10pp recall on SIFT, that's a real algorithmic contribution.
2. **Close the QPS gap** — 8-way unroll in the NEON kernel, prefetch hints,
   x86 AVX kernels. Mechanical perf work.

Both are credible; recall recovery is the bigger story if it works.
