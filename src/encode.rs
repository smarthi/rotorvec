//! Encode pipeline: normalize → block-rotate → quantize → bit-pack.

use crate::pack::pack_codes;
use crate::rotation::rotate_batch;

/// Encode `n` vectors of dimension `dim` into bit-packed codes plus per-vector
/// norms. Vectors are zero-padded to `padded_dim = ceil(dim / block_size) *
/// block_size` internally; only the first `dim` rotated coordinates are
/// quantized and stored, matching what `search` consumes on the query side.
///
/// `block_matrices`: precomputed, length `n_groups * block_size²`.
/// `boundaries`: `2^bits - 1` Lloyd-Max boundary values (sorted).
pub fn encode(
    vectors: &[f32],
    n: usize,
    dim: usize,
    block_matrices: &[f32],
    block_size: usize,
    padded_dim: usize,
    boundaries: &[f32],
    bits: usize,
) -> (Vec<u8>, Vec<f32>) {
    debug_assert_eq!(vectors.len(), n * dim);
    debug_assert_eq!(dim % 8, 0);
    debug_assert!(padded_dim >= dim);

    let mut norms = vec![0.0f32; n];
    let mut unit_padded = vec![0.0f32; n * padded_dim];
    for i in 0..n {
        let row = &vectors[i * dim..(i + 1) * dim];
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        norms[i] = norm;
        let inv_norm = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
        let dst = &mut unit_padded[i * padded_dim..i * padded_dim + dim];
        for j in 0..dim {
            dst[j] = row[j] * inv_norm;
        }
    }

    let rotated_padded = rotate_batch(block_matrices, &unit_padded, n, padded_dim, block_size);

    let mut codes = vec![0u8; n * dim];
    for i in 0..n {
        let src = &rotated_padded[i * padded_dim..i * padded_dim + dim];
        let dst = &mut codes[i * dim..(i + 1) * dim];
        for (out, &x) in dst.iter_mut().zip(src.iter()) {
            let mut c = 0u8;
            for &b in boundaries {
                if x > b {
                    c += 1;
                }
            }
            *out = c;
        }
    }

    let packed = pack_codes(&codes, n, dim, bits);
    (packed, norms)
}
