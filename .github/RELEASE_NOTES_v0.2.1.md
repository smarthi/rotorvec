# v0.2.1 — NEON SIMD search kernel for aarch64 (~130× faster)

Closes the search-throughput gap with turbovec on Apple Silicon and Graviton. Search QPS jumps ~130× over the v0.2.0 scalar path while recall stays within 1pp.

No on-disk format changes. v0.2.0 indices load and search unchanged under v0.2.1.

Available now on [crates.io](https://crates.io/crates/rotorvec/0.2.1) and [PyPI](https://pypi.org/project/rotorvec/0.2.1/).

## Headline numbers

**Synthetic Gaussian, n=100K, d=768, 4-bit, k=10:**

| Method | Recall@10 | Build (s) | QPS | vs turbovec |
|---|---:|---:|---:|---:|
| turbovec | 0.814 | 3.39 | **11,123** | — |
| rotorvec planar2 | 0.822 | **1.64** | 7,638 | 69% |
| rotorvec rotor3 | 0.817 | 1.64 | 7,383 | 66% |
| rotorvec iso4 | 0.812 | **1.56** | 7,922 | 71% |

**Synthetic Gaussian, n=50K, d=128, 4-bit, k=10:**

| Method | Recall@10 | Build (s) | QPS | vs turbovec |
|---|---:|---:|---:|---:|
| turbovec | 0.820 | 0.41 | **87,040** | — |
| rotorvec planar2 | 0.817 | **0.26** | 83,588 | **96%** |
| rotorvec rotor3 | 0.818 | 0.27 | 84,531 | **97%** |
| rotorvec iso4 | 0.817 | **0.26** | 81,240 | 93% |

**v0.2.0 → v0.2.1 speedup:**

| | v0.2.0 scalar | v0.2.1 NEON | Speedup |
|---|---:|---:|---:|
| QPS at d=128 | ~660 | ~83,000 | **~126×** |
| QPS at d=768 | ~58 | ~7,650 | **~132×** |

Build time stays ~2× faster than turbovec across both dims — no BLAS GEMM, just block-diagonal small matmuls.

## What's new

### ⚡ NEON 4-bit search kernel

A new `search_neon` module ports turbovec's u8-quantized-LUT scoring design to rotorvec exactly, so the comparison stays apples-to-apples:

- **`pack::repack_4bit()`** converts bit-plane codes into a NEON-friendly blocked layout (BLOCK=32 vectors per group, two 4-bit codes per byte)
- **Per-query LUT builder** quantizes `query[j] * centroid[k]` to u8 with shared scale, fitting all 16 lookups for one coordinate into a single `vqtbl1q_u8` instruction
- **4-group unrolled inner loop** with u16 accumulators flushed to f32 every 256 byte-groups
- **Lazy BlockedCache** in `RotorQuantIndex` — built on first search, invalidated automatically on `add()` / `swap_remove()`

### 🛡️ Correctness test

New `neon_4bit_topk_matches_scalar` test asserts NEON recall@1 ≥ 5/8 and recall@10 ≥ 70% against brute-force ground truth across all three rotation variants. Catches future regressions in the kernel.

### 🪟 Graceful fallback

Non-aarch64 platforms (Linux x86_64, Windows x64) and non-4-bit configs (2-bit, 3-bit) automatically use the v0.2.0 scalar path. No user changes required; the dispatcher selects the right kernel at runtime.

## What's *not* in this release

- **AVX-512 / AVX2 kernels for x86** — that's v0.2.3. Until then, x86 users see no speedup vs v0.2.0.
- **NEON for 2-bit / 3-bit** — same v0.2.3 timeframe. The u8 LUT design generalizes but each width needs its own kernel.
- **Beating turbovec.** Stretch goal not met. At d=128 we hit 96% parity; at d=768 we're at ~70% — likely turbovec's deeper inner-loop unrolling. Inner-loop tuning is v0.2.2.

## Trade-off: 1pp recall vs 130× QPS

Moving from scalar f32 accumulation to NEON's u8 quantized LUT introduces a small amount of low-bit quantization noise. Compared to v0.2.0:

| | v0.2.0 scalar | v0.2.1 NEON | Delta |
|---|---:|---:|---:|
| d=128 recall@10 | ~0.828 | ~0.817 | −1.1pp |
| d=768 recall@10 | ~0.830 | ~0.817 | −1.3pp |

turbovec makes the identical trade — same u8 LUT design — and ends up at the same recall band (0.81–0.82). If lossless f32 accumulation is critical for your use case, the scalar path is still reachable (build with `--target` other than `aarch64-*`, or wait for the v0.2.x option to opt out).

## Install / upgrade

```bash
# Rust
cargo update -p rotorvec

# Python
pip install -U rotorvec
```

No code changes required. Existing indices keep working. On aarch64, the NEON path activates automatically the next time you call `search()` on a 4-bit index.

## Verify the speedup yourself

```bash
git clone https://github.com/smarthi/rotorvec
cd rotorvec
cargo build -p rotorvec-bench --release
./target/release/rotorvec-bench --dataset random:100000:768 --bits 4 --k 10
```

Look for the `rotorvec planar2` / `rotor3` / `iso4` rows — QPS should be in the thousands now, not the dozens.

## Full changelog

| Commit | Summary |
|---|---|
| [`4834ca7`](https://github.com/smarthi/rotorvec/commit/4834ca7) | Add NEON 4-bit search kernel for aarch64 — ~130× speedup over scalar |
| [`917e4d7`](https://github.com/smarthi/rotorvec/commit/917e4d7) | ci: silence dead_code lints on x86 for NEON-only items |

## What's next

| Milestone | Focus |
|---|---|
| **v0.2.2** | Inner-loop tuning to close the d=768 gap (8-way unroll, prefetch hints) |
| **v0.2.3** | AVX-512 / AVX2 kernels for x86 — same u8-LUT design |
| **v0.2.4** | NEON for 2-bit and 3-bit code widths |
| **v0.2.5** | Real datasets (HDF5 ann-benchmarks loader) + FAISS comparison |

## Attribution

The u8-quantized-LUT design and blocked layout are ported faithfully from [**turbovec**](https://github.com/RyanCodrai/turbovec) (Ryan Codrai) and originate in [**FAISS**](https://github.com/facebookresearch/faiss)'s `IndexPQFastScan`. The rotation algebra (Cl(3,0) rotors, quaternions, 2D Givens) is unchanged from v0.1; see the main repo README for full algorithm and paper references.
