//! Plain (non-SIMD) top-k search.
//!
//! Scoring strategy: for each query, rotate it the same way the database was
//! rotated, then compute the inner product against every stored vector by
//! looking up each coordinate's centroid and accumulating. Multiply by the
//! stored norm to recover an approximation of the original `<q, x>`.
//!
//! v0.1 deliberately avoids SIMD intrinsics — correctness first. Per-query
//! work is parallelized across queries via rayon.

use crate::pack::unpack_vector;
use crate::rotation::rotate_batch;
use rayon::prelude::*;
use std::cmp::Ordering;

#[allow(clippy::too_many_arguments)]
pub fn search(
    queries: &[f32],
    nq: usize,
    dim: usize,
    block_matrices: &[f32],
    packed_codes: &[u8],
    centroids: &[f32],
    norms: &[f32],
    bits: usize,
    n_vectors: usize,
    k: usize,
) -> (Vec<f32>, Vec<i64>) {
    let padded_dim = dim.div_ceil(3) * 3;

    // Rotate queries through the same block-diagonal SO(3). Pad to multiple
    // of 3 first.
    let mut q_padded = vec![0.0f32; nq * padded_dim];
    for i in 0..nq {
        let src = &queries[i * dim..(i + 1) * dim];
        let dst = &mut q_padded[i * padded_dim..i * padded_dim + dim];
        dst.copy_from_slice(src);
    }
    let q_rot = rotate_batch(block_matrices, &q_padded, nq, padded_dim);

    let mut all_scores = vec![0.0f32; nq * k];
    let mut all_indices = vec![-1i64; nq * k];

    all_scores
        .par_chunks_mut(k)
        .zip(all_indices.par_chunks_mut(k))
        .enumerate()
        .for_each(|(qi, (out_scores, out_idx))| {
            let q = &q_rot[qi * padded_dim..qi * padded_dim + dim];
            let topk = score_one_query(q, dim, packed_codes, centroids, norms, bits, n_vectors, k);
            for (slot, (score, idx)) in topk.into_iter().enumerate() {
                out_scores[slot] = score;
                out_idx[slot] = idx as i64;
            }
        });

    (all_scores, all_indices)
}

#[allow(clippy::too_many_arguments)]
fn score_one_query(
    q: &[f32],
    dim: usize,
    packed_codes: &[u8],
    centroids: &[f32],
    norms: &[f32],
    bits: usize,
    n_vectors: usize,
    k: usize,
) -> Vec<(f32, usize)> {
    let mut heap: Vec<(f32, usize)> = Vec::with_capacity(k);

    for vi in 0..n_vectors {
        let codes = unpack_vector(packed_codes, vi, dim, bits);
        let mut dot = 0.0f32;
        for j in 0..dim {
            dot += q[j] * centroids[codes[j] as usize];
        }
        let score = dot * norms[vi];

        if heap.len() < k {
            heap.push((score, vi));
            if heap.len() == k {
                heap.sort_unstable_by(|a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
                });
            }
        } else if score > heap[0].0 {
            heap[0] = (score, vi);
            // Bubble down — heap is sorted ascending, so resort the small
            // prefix. For small k this is cheaper than a real heap.
            let mut i = 0;
            while i + 1 < k && heap[i].0 > heap[i + 1].0 {
                heap.swap(i, i + 1);
                i += 1;
            }
        }
    }

    if heap.len() < k {
        // Pad with sentinel for callers that expect exactly k slots.
        while heap.len() < k {
            heap.push((f32::NEG_INFINITY, usize::MAX));
        }
    }
    // Return descending by score.
    heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    heap
}
