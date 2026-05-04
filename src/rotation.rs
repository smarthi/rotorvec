//! Block-diagonal rotation: apply a precomputed 3×3 SO(3) matrix to each
//! consecutive 3-element block of a vector.
//!
//! For dimensions not divisible by 3, the input is zero-padded up to the
//! next multiple of 3 (handled by the caller via `padded_dim`).

/// Apply block-diagonal rotation to a single d-dim vector in place.
///
/// `matrices`: flat `[f32; n_groups * 9]`, row-major 3×3 per group.
/// `v`: length `padded_dim = n_groups * 3` (caller pads with zeros).
pub fn rotate_inplace(matrices: &[f32], v: &mut [f32]) {
    debug_assert_eq!(v.len() % 3, 0);
    debug_assert_eq!(matrices.len(), (v.len() / 3) * 9);

    for (block, mat) in v.chunks_exact_mut(3).zip(matrices.chunks_exact(9)) {
        let v0 = block[0];
        let v1 = block[1];
        let v2 = block[2];
        block[0] = mat[0] * v0 + mat[1] * v1 + mat[2] * v2;
        block[1] = mat[3] * v0 + mat[4] * v1 + mat[5] * v2;
        block[2] = mat[6] * v0 + mat[7] * v1 + mat[8] * v2;
    }
}

/// Apply block-diagonal rotation to a batch of vectors. Output is freshly
/// allocated; input is read row-major (`n × padded_dim`).
pub fn rotate_batch(matrices: &[f32], input: &[f32], n: usize, padded_dim: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), n * padded_dim);
    let mut out = input.to_vec();
    for row in out.chunks_exact_mut(padded_dim) {
        rotate_inplace(matrices, row);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotor::precompute_block_matrices;

    #[test]
    fn rotation_preserves_blockwise_norm() {
        let dim = 12;
        let (mats, _) = precompute_block_matrices(dim, 42);
        let mut v: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let v_orig = v.clone();
        rotate_inplace(&mats, &mut v);

        // Each 3-block's norm should be preserved.
        for (orig_block, rot_block) in v_orig.chunks_exact(3).zip(v.chunks_exact(3)) {
            let n_orig: f32 = orig_block.iter().map(|x| x * x).sum::<f32>().sqrt();
            let n_rot: f32 = rot_block.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n_orig - n_rot).abs() < 1e-5);
        }

        // Therefore total norm is preserved.
        let n_orig: f32 = v_orig.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n_rot: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n_orig - n_rot).abs() < 1e-4);
    }
}
