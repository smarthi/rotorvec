//! Encode pipeline: normalize → block-rotate → quantize → bit-pack.

use crate::pack::pack_codes;
use crate::rotation::rotate_batch;

/// Encode `n` vectors of dimension `dim` into bit-packed codes plus per-vector
/// norms. Vectors are zero-padded to the next multiple of 3 internally; on
/// disk we still store `dim` (not padded) coordinates because `dim` is
/// constrained to a multiple of 8 and small unused tail is harmless.
///
/// `block_matrices`: precomputed 3×3 block-diagonal SO(3) matrices,
/// length `n_groups * 9` where `n_groups = ceil(dim / 3)`.
/// `boundaries`: `2^bits - 1` Lloyd-Max boundary values (must be sorted).
///
/// Returns `(packed_codes, norms)`.
pub fn encode(
    vectors: &[f32],
    n: usize,
    dim: usize,
    block_matrices: &[f32],
    boundaries: &[f32],
    bits: usize,
) -> (Vec<u8>, Vec<f32>) {
    debug_assert_eq!(vectors.len(), n * dim);
    debug_assert_eq!(dim % 8, 0);

    let padded_dim = dim.div_ceil(3) * 3;

    // 1. Normalize and pad.
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
        // Tail coordinates (padded_dim - dim) stay at zero from vec init.
    }

    // 2. Block-diagonal rotation in place (cheap — O(d) per vector).
    let rotated_padded = rotate_batch(block_matrices, &unit_padded, n, padded_dim);

    // 3. Quantize: drop padding tail, then per-coordinate threshold against
    //    boundaries. Code for coordinate x is the count of boundaries it
    //    exceeds (so values in `0..n_levels`).
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

    // 4. Bit-pack.
    let packed = pack_codes(&codes, n, dim, bits);
    (packed, norms)
}
