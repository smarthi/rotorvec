//! Clifford algebra Cl(3,0) rotors and their SO(3) matrix representation.
//!
//! A rotor in Cl(3,0) is the even-grade subalgebra: `R = s + b12·e12 + b13·e13
//! + b23·e23`. Stored compactly as `[s, b12, b13, b23]`. Rotation acts via the
//! sandwich product `R v R̃`. For a grade-1 vector input, the result is again
//! grade-1, and the action is equivalent to a 3×3 SO(3) rotation. We
//! precompute that matrix at index init time and apply it as plain matrix
//! multiplication at runtime — no Clifford machinery on the hot path.
//!
//! Rotors are generated deterministically from a seed (ChaCha8) so two indices
//! built with the same `(dim, seed)` are bit-identical.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

/// Compact rotor representation: `[s, b12, b13, b23]` (4 nonzero components
/// of the 8-component Cl(3,0) multivector). Always normalized: s² + b12² +
/// b13² + b23² = 1.
pub type Rotor = [f32; 4];

/// Generate a random unit rotor. Uniform over SO(3) (Shoemake's method via
/// Gaussian sampling + normalization — equivalent to uniform on the unit
/// 3-sphere, which double-covers SO(3)).
pub fn random_rotor(rng: &mut ChaCha8Rng) -> Rotor {
    let s: f32 = rng.sample::<f64, _>(StandardNormal) as f32;
    let b12: f32 = rng.sample::<f64, _>(StandardNormal) as f32;
    let b13: f32 = rng.sample::<f64, _>(StandardNormal) as f32;
    let b23: f32 = rng.sample::<f64, _>(StandardNormal) as f32;
    let n = (s * s + b12 * b12 + b13 * b13 + b23 * b23).sqrt().max(1e-30);
    [s / n, b12 / n, b13 / n, b23 / n]
}

/// Generate `n_groups` deterministic rotors from a seed.
pub fn make_rotors(n_groups: usize, seed: u64) -> Vec<Rotor> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n_groups).map(|_| random_rotor(&mut rng)).collect()
}

/// Convert a rotor to its equivalent 3×3 SO(3) rotation matrix (row-major).
///
/// The columns of `M` are `R e_j R̃` for `j = 1, 2, 3`, derived by direct
/// expansion of the Cl(3,0) geometric product. The cross terms differ from
/// the standard quaternion formula in sign conventions because the sandwich
/// rotates in the opposite handedness from the (w, x, y, z) quaternion form,
/// but the result is still a proper rotation (det = +1).
pub fn rotor_to_so3(r: &Rotor) -> [f32; 9] {
    let s = r[0];
    let b12 = r[1];
    let b13 = r[2];
    let b23 = r[3];

    let s2 = s * s;
    let b12_2 = b12 * b12;
    let b13_2 = b13 * b13;
    let b23_2 = b23 * b23;

    let m00 = s2 + b23_2 - b12_2 - b13_2;
    let m01 = 2.0 * (s * b12 - b13 * b23);
    let m02 = 2.0 * (s * b13 + b12 * b23);

    let m10 = -2.0 * (s * b12 + b13 * b23);
    let m11 = s2 + b13_2 - b12_2 - b23_2;
    let m12 = 2.0 * (s * b23 - b12 * b13);

    let m20 = 2.0 * (b12 * b23 - s * b13);
    let m21 = -2.0 * (s * b23 + b12 * b13);
    let m22 = s2 + b12_2 - b13_2 - b23_2;

    [m00, m01, m02, m10, m11, m12, m20, m21, m22]
}

/// Precompute the block-diagonal SO(3) rotation matrices for an entire index.
/// Returns a flat `Vec<f32>` of length `n_groups * 9` where each chunk of 9 is
/// a row-major 3×3 matrix.
pub fn precompute_block_matrices(dim: usize, seed: u64) -> (Vec<f32>, usize) {
    let n_groups = dim.div_ceil(3);
    let rotors = make_rotors(n_groups, seed);
    let mut matrices = Vec::with_capacity(n_groups * 9);
    for r in &rotors {
        matrices.extend_from_slice(&rotor_to_so3(r));
    }
    (matrices, n_groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul3(m: &[f32; 9], v: &[f32; 3]) -> [f32; 3] {
        [
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
        ]
    }

    #[test]
    fn rotor_is_unit_length() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..100 {
            let r = random_rotor(&mut rng);
            let n = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
            assert!((n - 1.0).abs() < 1e-5, "rotor not unit: |R|² = {n}");
        }
    }

    #[test]
    fn so3_matrix_is_orthogonal() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..100 {
            let r = random_rotor(&mut rng);
            let m = rotor_to_so3(&r);

            // Check M Mᵀ = I (rows orthonormal).
            for i in 0..3 {
                for j in 0..3 {
                    let dot = (0..3).map(|k| m[i * 3 + k] * m[j * 3 + k]).sum::<f32>();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() < 1e-4,
                        "M Mᵀ[{i},{j}] = {dot}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn so3_preserves_norm() {
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let r = random_rotor(&mut rng);
        let m = rotor_to_so3(&r);
        let v = [0.3f32, -0.7, 0.5];
        let v_rot = matmul3(&m, &v);
        let n_in = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let n_out = (v_rot[0] * v_rot[0] + v_rot[1] * v_rot[1] + v_rot[2] * v_rot[2]).sqrt();
        assert!((n_in - n_out).abs() < 1e-5);
    }

    #[test]
    fn so3_has_unit_determinant() {
        // Proper rotation: det = +1 (not -1 reflection).
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        for _ in 0..50 {
            let r = random_rotor(&mut rng);
            let m = rotor_to_so3(&r);
            let det = m[0] * (m[4] * m[8] - m[5] * m[7])
                - m[1] * (m[3] * m[8] - m[5] * m[6])
                + m[2] * (m[3] * m[7] - m[4] * m[6]);
            assert!((det - 1.0).abs() < 1e-4, "det = {det}, expected +1");
        }
    }

    #[test]
    fn deterministic_seed() {
        let a = make_rotors(10, 42);
        let b = make_rotors(10, 42);
        assert_eq!(a, b);
        let c = make_rotors(10, 43);
        assert_ne!(a, c);
    }
}
