//! NEON-accelerated 4-bit search kernel for aarch64.
//!
//! Mirrors turbovec's NEON scoring strategy so the two crates can be
//! compared apples-to-apples:
//!
//! 1. Per-query, build two 16-entry sub-tables per byte-group from
//!    `q_rot[2g] * centroid[k]` and `q_rot[2g+1] * centroid[k]`. Find the
//!    per-sub-table min and the global max range; uniformly quantize all
//!    sub-tables to `u8` so a single `vqtbl1q_u8` instruction does 16
//!    parallel lookups.
//! 2. For each block of 32 vectors, accumulate the u8 partial scores into
//!    u16 lanes (one per vector). Periodically flush to f32 to avoid u16
//!    overflow — at most 256 groups between flushes per turbovec.
//! 3. Multiply the per-vector f32 accumulator by the stored norm to get
//!    the final approximate inner product, push into a per-query heap.
//!
//! Layout of `blocked_codes` matches `pack::repack_4bit`:
//!
//! ```text
//! blocked[block * n_byte_groups * BLOCK + g * BLOCK + lane]
//! = (code[2*g] << 4) | code[2*g + 1]      for vector (block*BLOCK + lane)
//! ```

#![cfg(target_arch = "aarch64")]

use crate::rotation::rotate_batch;
use crate::{BLOCK, FLUSH_EVERY};
use rayon::prelude::*;
use std::cmp::Ordering;

/// Per-query precomputed lookup table.
///
/// `uint8_luts` is `n_byte_groups * 32` bytes. For each byte-group `g`:
/// * `uint8_luts[g*32 + 0 .. g*32 + 16]` — sub-table for coord `2g`
///   (queried by `code >> 4`, the *high* nibble of the packed code byte).
/// * `uint8_luts[g*32 + 16 .. g*32 + 32]` — sub-table for coord `2g+1`
///   (queried by `code & 0x0F`, the *low* nibble).
///
/// Each entry encodes `q_rot[coord] * centroid[k]` via a uniform quantizer:
/// `entry = round((value - sub_table_min) / scale)` clamped to `[0, 127]`.
/// To reconstruct the partial score: `value = sub_table_min + scale * entry`.
///
/// `bias` holds the sum of all `sub_table_min` values; the kernel adds it
/// once before flushing accumulators.
struct QueryLut {
    uint8_luts: Vec<u8>,
    scale: f32,
    bias: f32,
}

fn build_query_lut_4bit(q_rot_row: &[f32], centroids: &[f32], dim: usize) -> QueryLut {
    debug_assert_eq!(centroids.len(), 16);
    debug_assert_eq!(q_rot_row.len(), dim);
    debug_assert_eq!(dim % 2, 0);

    let n_byte_groups = dim / 2;
    let n_subs = n_byte_groups * 2;

    let mut uint8_luts = vec![0u8; n_byte_groups * 32];
    let mut float_vals = vec![0.0f32; n_byte_groups * 32];
    let mut mins = vec![0.0f32; n_subs];
    let mut max_span = 0.0f32;
    let mut bias = 0.0f32;

    for g in 0..n_byte_groups {
        // High-nibble sub-table = coord (2g).
        let q_hi = q_rot_row[2 * g];
        let mut hi_min = f32::INFINITY;
        let mut hi_max = f32::NEG_INFINITY;
        for k in 0..16 {
            let v = q_hi * centroids[k];
            float_vals[g * 32 + k] = v;
            if v < hi_min {
                hi_min = v;
            }
            if v > hi_max {
                hi_max = v;
            }
        }

        // Low-nibble sub-table = coord (2g+1).
        let q_lo = q_rot_row[2 * g + 1];
        let mut lo_min = f32::INFINITY;
        let mut lo_max = f32::NEG_INFINITY;
        for k in 0..16 {
            let v = q_lo * centroids[k];
            float_vals[g * 32 + 16 + k] = v;
            if v < lo_min {
                lo_min = v;
            }
            if v > lo_max {
                lo_max = v;
            }
        }

        mins[g * 2] = hi_min;
        mins[g * 2 + 1] = lo_min;
        bias += hi_min + lo_min;

        let hi_span = hi_max - hi_min;
        let lo_span = lo_max - lo_min;
        if hi_span > max_span {
            max_span = hi_span;
        }
        if lo_span > max_span {
            max_span = lo_span;
        }
    }

    // Quantize. Cap at 127 so u8+u8 add stays in u8 range across two
    // sub-tables per byte-group (max 254).
    let max_lut = 127.0f32;
    let scale = if max_span > 1e-10 {
        max_span / max_lut
    } else {
        1.0
    };
    let inv_scale = 1.0 / scale;

    for g in 0..n_byte_groups {
        let hi_min = mins[g * 2];
        let lo_min = mins[g * 2 + 1];
        for k in 0..16 {
            let j_hi = g * 32 + k;
            let j_lo = g * 32 + 16 + k;
            uint8_luts[j_hi] = ((float_vals[j_hi] - hi_min) * inv_scale)
                .round()
                .clamp(0.0, max_lut) as u8;
            uint8_luts[j_lo] = ((float_vals[j_lo] - lo_min) * inv_scale)
                .round()
                .clamp(0.0, max_lut) as u8;
        }
    }

    QueryLut {
        uint8_luts,
        scale,
        bias,
    }
}

/// Score one block of 32 contiguous database vectors against one query LUT.
/// Writes 32 f32 partial scores (pre-norm multiplication is *included* via
/// the `norms` slice) to `out`.
///
/// # Safety
/// * `blocked_codes` must contain at least `block_offset + n_byte_groups * BLOCK` bytes.
/// * `uint8_luts` must contain at least `n_byte_groups * 32` bytes.
/// * `norms` must contain at least `base_vec + 32` entries when
///   `base_vec + 32 <= n_vectors`; otherwise the partial-block branch is taken
///   and out-of-range lanes are filled with `f32::NEG_INFINITY`.
#[target_feature(enable = "neon")]
unsafe fn score_4bit_block_neon(
    blocked_codes: &[u8],
    uint8_luts: &[u8],
    block_offset: usize,
    n_byte_groups: usize,
    scale: f32,
    bias: f32,
    norms: &[f32],
    base_vec: usize,
    n_vectors: usize,
    out: &mut [f32; BLOCK],
) {
    use std::arch::aarch64::*;

    let mask = vdupq_n_u8(0x0F);
    let v_scale = vdupq_n_f32(scale);
    let n_batches = n_byte_groups.div_ceil(FLUSH_EVERY);

    // 8 × f32x4 = 32 float accumulators, initialised to `bias`.
    let mut fa = [vdupq_n_f32(bias); 8];

    let codes_base = blocked_codes.as_ptr().add(block_offset);
    let luts_base = uint8_luts.as_ptr();

    for batch in 0..n_batches {
        let g_start = batch * FLUSH_EVERY;
        let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);

        // 4 × u16x8 = 32 u16 lanes, fresh each batch.
        let mut accum = [vdupq_n_u16(0); 4];

        // 4-group unrolled inner loop hides `vqtbl1q_u8` latency.
        let mut g = g_start;
        while g + 3 < g_end {
            for off in 0..4 {
                let gg = g + off;
                let lp = luts_base.add(gg * 32);
                let cp = codes_base.add(gg * BLOCK);

                let lut_hi_nib = vld1q_u8(lp); // sub-table for coord 2g
                let lut_lo_nib = vld1q_u8(lp.add(16)); // sub-table for coord 2g+1

                // 32 code bytes for this group across 32 vectors.
                let c0 = vld1q_u8(cp);
                let c1 = vld1q_u8(cp.add(16));

                // High nibble = code >> 4 = coord-2g index → look up in hi sub-table.
                // Low  nibble = code & 0x0F = coord-(2g+1) index → look up in lo sub-table.
                let s0 = vaddq_u8(
                    vqtbl1q_u8(lut_hi_nib, vshrq_n_u8(c0, 4)),
                    vqtbl1q_u8(lut_lo_nib, vandq_u8(c0, mask)),
                );
                let s1 = vaddq_u8(
                    vqtbl1q_u8(lut_hi_nib, vshrq_n_u8(c1, 4)),
                    vqtbl1q_u8(lut_lo_nib, vandq_u8(c1, mask)),
                );

                accum[0] = vaddw_u8(accum[0], vget_low_u8(s0));
                accum[1] = vaddw_u8(accum[1], vget_high_u8(s0));
                accum[2] = vaddw_u8(accum[2], vget_low_u8(s1));
                accum[3] = vaddw_u8(accum[3], vget_high_u8(s1));
            }
            g += 4;
        }

        // Tail.
        while g < g_end {
            let lp = luts_base.add(g * 32);
            let cp = codes_base.add(g * BLOCK);
            let lut_hi_nib = vld1q_u8(lp);
            let lut_lo_nib = vld1q_u8(lp.add(16));
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));
            let s0 = vaddq_u8(
                vqtbl1q_u8(lut_hi_nib, vshrq_n_u8(c0, 4)),
                vqtbl1q_u8(lut_lo_nib, vandq_u8(c0, mask)),
            );
            let s1 = vaddq_u8(
                vqtbl1q_u8(lut_hi_nib, vshrq_n_u8(c1, 4)),
                vqtbl1q_u8(lut_lo_nib, vandq_u8(c1, mask)),
            );
            accum[0] = vaddw_u8(accum[0], vget_low_u8(s0));
            accum[1] = vaddw_u8(accum[1], vget_high_u8(s0));
            accum[2] = vaddw_u8(accum[2], vget_low_u8(s1));
            accum[3] = vaddw_u8(accum[3], vget_high_u8(s1));
            g += 1;
        }

        // Flush u16 → f32 with FMA: fa[i] += scale * accum[i].
        for i in 0..4 {
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(accum[i])));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(accum[i])));
            fa[i * 2] = vfmaq_f32(fa[i * 2], v_scale, lo);
            fa[i * 2 + 1] = vfmaq_f32(fa[i * 2 + 1], v_scale, hi);
        }
    }

    // Multiply by norms and write 32 final scores.
    let end = (base_vec + BLOCK).min(n_vectors);
    let out_ptr = out.as_mut_ptr();
    let norms_ptr = norms.as_ptr().add(base_vec);

    if end - base_vec == BLOCK {
        // Full block — straight FMA into output.
        for i in 0..8 {
            let n = vld1q_f32(norms_ptr.add(i * 4));
            vst1q_f32(out_ptr.add(i * 4), vmulq_f32(fa[i], n));
        }
    } else {
        // Partial last block — write through a scratch and mask out-of-range lanes.
        let mut scratch = [0.0f32; BLOCK];
        for i in 0..8 {
            vst1q_f32(scratch.as_mut_ptr().add(i * 4), fa[i]);
        }
        let valid = end - base_vec;
        for lane in 0..BLOCK {
            *out_ptr.add(lane) = if lane < valid {
                scratch[lane] * *norms_ptr.add(lane)
            } else {
                f32::NEG_INFINITY
            };
        }
    }
}

/// Top-level NEON search entry point. Rotates queries, builds LUTs, and
/// runs the per-block kernel with a small per-query top-k heap.
#[allow(clippy::too_many_arguments)]
pub fn search_4bit_neon(
    queries: &[f32],
    nq: usize,
    dim: usize,
    matrices: &[f32],
    block_size: usize,
    padded_dim: usize,
    blocked_codes: &[u8],
    n_blocks: usize,
    centroids: &[f32],
    norms: &[f32],
    n_vectors: usize,
    k: usize,
) -> (Vec<f32>, Vec<i64>) {
    debug_assert_eq!(centroids.len(), 16);
    debug_assert_eq!(dim % 2, 0);
    let n_byte_groups = dim / 2;

    // Rotate queries (same block-diag rotation as the scalar path).
    let mut q_padded = vec![0.0f32; nq * padded_dim];
    for i in 0..nq {
        let src = &queries[i * dim..(i + 1) * dim];
        let dst = &mut q_padded[i * padded_dim..i * padded_dim + dim];
        dst.copy_from_slice(src);
    }
    let q_rot = rotate_batch(matrices, &q_padded, nq, padded_dim, block_size);

    let k = k.max(1);
    let mut all_scores = vec![f32::NEG_INFINITY; nq * k];
    let mut all_indices = vec![-1i64; nq * k];

    all_scores
        .par_chunks_mut(k)
        .zip(all_indices.par_chunks_mut(k))
        .enumerate()
        .for_each(|(qi, (out_scores, out_idx))| {
            let q = &q_rot[qi * padded_dim..qi * padded_dim + dim];
            let lut = build_query_lut_4bit(q, centroids, dim);

            // Ascending-sorted heap of length k.
            let mut top: Vec<(f32, usize)> = Vec::with_capacity(k);
            let mut block_scores = [0.0f32; BLOCK];

            for b in 0..n_blocks {
                let block_offset = b * n_byte_groups * BLOCK;
                let base_vec = b * BLOCK;

                // SAFETY: lengths satisfied by pack::repack_4bit and the LUT builder.
                unsafe {
                    score_4bit_block_neon(
                        blocked_codes,
                        &lut.uint8_luts,
                        block_offset,
                        n_byte_groups,
                        lut.scale,
                        lut.bias,
                        norms,
                        base_vec,
                        n_vectors,
                        &mut block_scores,
                    );
                }

                for lane in 0..BLOCK {
                    let vi = base_vec + lane;
                    if vi >= n_vectors {
                        break;
                    }
                    let s = block_scores[lane];
                    if top.len() < k {
                        top.push((s, vi));
                        if top.len() == k {
                            top.sort_unstable_by(|a, b| {
                                a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
                            });
                        }
                    } else if s > top[0].0 {
                        top[0] = (s, vi);
                        let mut i = 0;
                        while i + 1 < k && top[i].0 > top[i + 1].0 {
                            top.swap(i, i + 1);
                            i += 1;
                        }
                    }
                }
            }

            // Descending output.
            top.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            for (slot, (s, vi)) in top.into_iter().enumerate() {
                out_scores[slot] = s;
                out_idx[slot] = vi as i64;
            }
        });

    (all_scores, all_indices)
}
