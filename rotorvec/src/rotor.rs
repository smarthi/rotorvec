//! Block-rotation parameterizations for rotorvec.
//!
//! Three variants:
//! * [`Rotation::Planar2`] — 2D Givens rotation (one angle per pair).
//!   Cheapest, and per the RotorQuant paper actually the strongest on PPL
//!   for KV cache decorrelation.
//! * [`Rotation::Rotor3`] — Cl(3,0) rotor sandwich `R v R̃` per 3-block.
//!   Equivalent to a 3×3 SO(3) rotation; we precompute the matrix.
//! * [`Rotation::Iso4`] — left-isoclinic 4×4 rotation from a unit quaternion
//!   per 4-block. Sits in the SO(4) subgroup `SU(2)_L`.
//!
//! All three generate matrices deterministically from a ChaCha8 seed so two
//! indices built with the same `(dim, kind, seed)` are bit-identical.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use std::f32::consts::TAU;

/// Which block-diagonal rotation family the index uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotation {
    /// 2D Givens (PlanarQuant). Block size 2, 1 angle per group.
    Planar2,
    /// Cl(3,0) Clifford rotor (RotorQuant). Block size 3, 4 params per group.
    Rotor3,
    /// 4D quaternion left-iso rotation (IsoQuant). Block size 4, 4 params per group.
    Iso4,
}

impl Rotation {
    pub const fn block_size(self) -> usize {
        match self {
            Rotation::Planar2 => 2,
            Rotation::Rotor3 => 3,
            Rotation::Iso4 => 4,
        }
    }

    /// Stable on-disk tag. Used by `io.rs`.
    pub const fn as_byte(self) -> u8 {
        match self {
            Rotation::Planar2 => 2,
            Rotation::Rotor3 => 3,
            Rotation::Iso4 => 4,
        }
    }

    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            2 => Some(Rotation::Planar2),
            3 => Some(Rotation::Rotor3),
            4 => Some(Rotation::Iso4),
            _ => None,
        }
    }
}

/// Compact Cl(3,0) rotor: `[s, b12, b13, b23]`, normalized.
pub type Rotor = [f32; 4];

/// Compact unit quaternion: `[w, x, y, z]`, normalized.
pub type Quat = [f32; 4];

fn gauss(rng: &mut ChaCha8Rng) -> f32 {
    rng.sample::<f64, _>(StandardNormal) as f32
}

/// Random unit Cl(3,0) rotor, uniform over SO(3).
pub fn random_rotor(rng: &mut ChaCha8Rng) -> Rotor {
    let s = gauss(rng);
    let b12 = gauss(rng);
    let b13 = gauss(rng);
    let b23 = gauss(rng);
    let n = (s * s + b12 * b12 + b13 * b13 + b23 * b23).sqrt().max(1e-30);
    [s / n, b12 / n, b13 / n, b23 / n]
}

/// Random unit quaternion, uniform on S³ (Shoemake's method).
pub fn random_quaternion(rng: &mut ChaCha8Rng) -> Quat {
    let w = gauss(rng);
    let x = gauss(rng);
    let y = gauss(rng);
    let z = gauss(rng);
    let n = (w * w + x * x + y * y + z * z).sqrt().max(1e-30);
    [w / n, x / n, y / n, z / n]
}

/// Cl(3,0) rotor → 3×3 SO(3) matrix (row-major).
///
/// Derived by direct expansion of the sandwich product `R e_j R̃`.
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

/// 2D Givens matrix from one angle: `[[c, -s], [s, c]]` row-major.
pub fn givens_to_so2(theta: f32) -> [f32; 4] {
    let (s, c) = theta.sin_cos();
    [c, -s, s, c]
}

/// Unit quaternion → 4×4 left-isoclinic SO(4) matrix (row-major).
///
/// `M v` is the matrix form of left quaternion multiplication `q · v` when
/// both `q` and `v` are quaternions. `M` is orthogonal with det = +1.
pub fn quaternion_to_so4(q: &Quat) -> [f32; 16] {
    let w = q[0];
    let x = q[1];
    let y = q[2];
    let z = q[3];
    [
        w, -x, -y, -z,
        x,  w, -z,  y,
        y,  z,  w, -x,
        z, -y,  x,  w,
    ]
}

/// Generate a deterministic block-diagonal rotation field for an index.
///
/// Returns `(matrices, n_groups, padded_dim)`:
/// * `matrices` — flat row-major, length `n_groups * block_size²`.
/// * `n_groups` — `ceil(dim / block_size)`.
/// * `padded_dim` — `n_groups * block_size`. The caller should zero-pad input
///   vectors out to this length before applying the rotation.
pub fn precompute_block_matrices(
    dim: usize,
    kind: Rotation,
    seed: u64,
) -> (Vec<f32>, usize, usize) {
    let bs = kind.block_size();
    let n_groups = dim.div_ceil(bs);
    let padded_dim = n_groups * bs;
    let mat_stride = bs * bs;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut matrices = vec![0.0f32; n_groups * mat_stride];

    for g in 0..n_groups {
        let m = &mut matrices[g * mat_stride..(g + 1) * mat_stride];
        match kind {
            Rotation::Planar2 => {
                let theta = rng.gen::<f32>() * TAU;
                m.copy_from_slice(&givens_to_so2(theta));
            }
            Rotation::Rotor3 => {
                let r = random_rotor(&mut rng);
                m.copy_from_slice(&rotor_to_so3(&r));
            }
            Rotation::Iso4 => {
                let q = random_quaternion(&mut rng);
                m.copy_from_slice(&quaternion_to_so4(&q));
            }
        }
    }

    (matrices, n_groups, padded_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_orthogonal(m: &[f32], n: usize) {
        for i in 0..n {
            for j in 0..n {
                let dot: f32 = (0..n).map(|k| m[i * n + k] * m[j * n + k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "M Mᵀ[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    fn check_det_plus_one(m: &[f32], n: usize) {
        // Cofactor expansion for n ∈ {2, 3, 4}.
        let det: f32 = match n {
            2 => m[0] * m[3] - m[1] * m[2],
            3 => {
                m[0] * (m[4] * m[8] - m[5] * m[7])
                    - m[1] * (m[3] * m[8] - m[5] * m[6])
                    + m[2] * (m[3] * m[7] - m[4] * m[6])
            }
            4 => {
                // Laplace expansion along the first row using 3×3 minors.
                let mut det = 0.0f32;
                for col in 0..4 {
                    let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
                    let mut minor = [0.0f32; 9];
                    let mut idx = 0;
                    for r in 1..4 {
                        for c in 0..4 {
                            if c != col {
                                minor[idx] = m[r * 4 + c];
                                idx += 1;
                            }
                        }
                    }
                    let m3 = minor;
                    let d3 = m3[0] * (m3[4] * m3[8] - m3[5] * m3[7])
                        - m3[1] * (m3[3] * m3[8] - m3[5] * m3[6])
                        + m3[2] * (m3[3] * m3[7] - m3[4] * m3[6]);
                    det += sign * m[col] * d3;
                }
                det
            }
            _ => unreachable!(),
        };
        assert!((det - 1.0).abs() < 1e-3, "det = {det}, expected +1");
    }

    #[test]
    fn all_variants_orthogonal_and_proper() {
        for &kind in &[Rotation::Planar2, Rotation::Rotor3, Rotation::Iso4] {
            let bs = kind.block_size();
            let (mats, n_groups, padded) =
                precompute_block_matrices(bs * 5, kind, 42);
            assert_eq!(n_groups, 5);
            assert_eq!(padded, bs * 5);
            for g in 0..n_groups {
                let m = &mats[g * bs * bs..(g + 1) * bs * bs];
                check_orthogonal(m, bs);
                check_det_plus_one(m, bs);
            }
        }
    }

    #[test]
    fn deterministic_seed() {
        for &kind in &[Rotation::Planar2, Rotation::Rotor3, Rotation::Iso4] {
            let (a, _, _) = precompute_block_matrices(60, kind, 42);
            let (b, _, _) = precompute_block_matrices(60, kind, 42);
            assert_eq!(a, b);
            let (c, _, _) = precompute_block_matrices(60, kind, 43);
            assert_ne!(a, c);
        }
    }

    #[test]
    fn rotation_byte_roundtrip() {
        for &kind in &[Rotation::Planar2, Rotation::Rotor3, Rotation::Iso4] {
            assert_eq!(Rotation::from_byte(kind.as_byte()), Some(kind));
        }
        assert_eq!(Rotation::from_byte(99), None);
    }
}
