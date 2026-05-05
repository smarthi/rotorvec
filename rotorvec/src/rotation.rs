//! Block-diagonal rotation: apply a precomputed `bs × bs` orthogonal matrix
//! to each consecutive `bs`-element block of a vector. `bs ∈ {2, 3, 4}`
//! depending on the rotation kind.
//!
//! For dimensions not divisible by `bs`, the input must be zero-padded up to
//! `n_groups * bs` by the caller.

const MAX_BLOCK: usize = 4;

/// Apply block-diagonal rotation to a single padded vector in place.
///
/// `matrices`: flat `n_groups * block_size²` row-major matrices.
/// `block_size`: 2, 3, or 4.
/// `v`: length `n_groups * block_size`.
pub fn rotate_inplace(matrices: &[f32], block_size: usize, v: &mut [f32]) {
    debug_assert!(block_size <= MAX_BLOCK);
    debug_assert_eq!(v.len() % block_size, 0);
    let bs = block_size;
    let stride = bs * bs;
    debug_assert_eq!(matrices.len(), (v.len() / bs) * stride);

    let mut tmp = [0.0f32; MAX_BLOCK];
    for (block, mat) in v.chunks_exact_mut(bs).zip(matrices.chunks_exact(stride)) {
        for i in 0..bs {
            let row = &mat[i * bs..(i + 1) * bs];
            let mut acc = 0.0f32;
            for j in 0..bs {
                acc += row[j] * block[j];
            }
            tmp[i] = acc;
        }
        block.copy_from_slice(&tmp[..bs]);
    }
}

/// Apply block-diagonal rotation to a batch (`n × padded_dim`, row-major).
pub fn rotate_batch(
    matrices: &[f32],
    input: &[f32],
    n: usize,
    padded_dim: usize,
    block_size: usize,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), n * padded_dim);
    let mut out = input.to_vec();
    for row in out.chunks_exact_mut(padded_dim) {
        rotate_inplace(matrices, block_size, row);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotor::{precompute_block_matrices, Rotation};

    #[test]
    fn all_variants_preserve_blockwise_norm() {
        for &kind in &[Rotation::Planar2, Rotation::Rotor3, Rotation::Iso4] {
            let bs = kind.block_size();
            let dim = bs * 4;
            let (mats, _, padded) = precompute_block_matrices(dim, kind, 11);
            assert_eq!(padded, dim);
            let mut v: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.13 - 0.5).collect();
            let v_orig = v.clone();
            rotate_inplace(&mats, bs, &mut v);

            for (orig, rot) in v_orig.chunks_exact(bs).zip(v.chunks_exact(bs)) {
                let n_orig: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
                let n_rot: f32 = rot.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    (n_orig - n_rot).abs() < 1e-5,
                    "kind={kind:?}: |orig|={n_orig}, |rot|={n_rot}"
                );
            }
        }
    }
}
