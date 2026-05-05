//! Plain (non-SIMD) top-k search. Per-query work parallelized via rayon.

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
    block_size: usize,
    padded_dim: usize,
    packed_codes: &[u8],
    centroids: &[f32],
    norms: &[f32],
    bits: usize,
    n_vectors: usize,
    k: usize,
) -> (Vec<f32>, Vec<i64>) {
    let mut q_padded = vec![0.0f32; nq * padded_dim];
    for i in 0..nq {
        let src = &queries[i * dim..(i + 1) * dim];
        let dst = &mut q_padded[i * padded_dim..i * padded_dim + dim];
        dst.copy_from_slice(src);
    }
    let q_rot = rotate_batch(block_matrices, &q_padded, nq, padded_dim, block_size);

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
            let mut i = 0;
            while i + 1 < k && heap[i].0 > heap[i + 1].0 {
                heap.swap(i, i + 1);
                i += 1;
            }
        }
    }

    while heap.len() < k {
        heap.push((f32::NEG_INFINITY, usize::MAX));
    }
    heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    heap
}
