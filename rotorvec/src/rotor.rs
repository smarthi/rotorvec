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
    /// **Walsh-Hadamard + Cl(3,0) rotor.** Applies `H · diag(s)` (signed
    /// fast Walsh-Hadamard) as a cross-block mixing pass, then `Rotor3`
    /// block-diagonal rotation. Cost is `O(d log d)` — between pure block-
    /// diagonal `O(d)` and dense d×d `O(d²)`. Designed to close the recall
    /// gap on data with strong cross-coordinate correlations (e.g. SIFT)
    /// while still avoiding BLAS GEMM.
    ///
    /// **Constraint:** `dim` must be a power of two (the FWHT butterfly
    /// network only handles power-of-two lengths). This rules out
    /// `dim = 100, 384, 768, 1536`; covers `128, 256, 512, 1024, 2048,
    /// 4096`. Use one of the other variants for non-power-of-two dims.
    WalshRotor3,
}

impl Rotation {
    pub const fn block_size(self) -> usize {
        match self {
            Rotation::Planar2 => 2,
            Rotation::Rotor3 => 3,
            Rotation::Iso4 => 4,
            Rotation::WalshRotor3 => 3,
        }
    }

    /// True iff this rotation kind applies a signed Walsh-Hadamard pre-pass
    /// (cross-block mixing) before the block-diagonal rotation.
    pub const fn uses_walsh(self) -> bool {
        matches!(self, Rotation::WalshRotor3)
    }

    /// Stable on-disk tag. Used by `io.rs`.
    pub const fn as_byte(self) -> u8 {
        match self {
            Rotation::Planar2 => 2,
            Rotation::Rotor3 => 3,
            Rotation::Iso4 => 4,
            Rotation::WalshRotor3 => 5,
        }
    }

    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            2 => Some(Rotation::Planar2),
            3 => Some(Rotation::Rotor3),
            4 => Some(Rotation::Iso4),
            5 => Some(Rotation::WalshRotor3),
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
    let n = (s * s + b12 * b12 + b13 * b13 + b23 * b23)
        .sqrt()
        .max(1e-30);
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
    [w, -x, -y, -z, x, w, -z, y, y, z, w, -x, z, -y, x, w]
}

/// Generate a deterministic block-diagonal rotation field for an index.
///
/// Returns `(matrices, n_groups, padded_dim)`:
/// * `matrices` — flat row-major, length `n_groups * block_size²`.
/// * `n_groups` — `ceil(dim / block_size)`.
/// * `padded_dim` — `n_groups * block_size`. The caller should zero-pad input
///   vectors out to this length before applying the rotation.
///
/// **Trailing partial block:** if `dim` is not a multiple of `block_size`
/// (only possible for [`Rotation::Rotor3`] under our `dim % 8 == 0`
/// constraint, since 2 and 4 both divide 8 but 3 does not), the last block
/// gets an identity matrix instead of a random rotation. Without this, the
/// last block rotates real coordinates into the zero-padded tail — when we
/// drop the tail before quantization, that rotation throws away signal.
/// Identity preserves the real coordinates untouched at the cost of not
/// decorrelating them within the trailing block (small effect, since at
/// most `block_size - 1` coords are involved).
pub fn precompute_block_matrices(
    dim: usize,
    kind: Rotation,
    seed: u64,
) -> (Vec<f32>, usize, usize) {
    let bs = kind.block_size();
    let n_groups = dim.div_ceil(bs);
    let padded_dim = n_groups * bs;
    let mat_stride = bs * bs;
    let has_partial_last = dim % bs != 0;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut matrices = vec![0.0f32; n_groups * mat_stride];

    for g in 0..n_groups {
        let m = &mut matrices[g * mat_stride..(g + 1) * mat_stride];

        if has_partial_last && g == n_groups - 1 {
            // Identity for the partial trailing block — see doc comment.
            // Note: we deliberately *don't* draw from `rng` here, so an
            // index built with `dim` evenly divisible by `bs` and one
            // built with the same seed but a smaller `dim` produce
            // bit-identical rotations on their shared full blocks.
            for i in 0..bs {
                m[i * bs + i] = 1.0;
            }
            continue;
        }

        // `WalshRotor3` reuses Rotor3's block-rotation generation; the WHT
        // pre-pass is handled separately via `precompute_walsh_signs`.
        let block_kind = match kind {
            Rotation::WalshRotor3 => Rotation::Rotor3,
            other => other,
        };

        match block_kind {
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
            Rotation::WalshRotor3 => unreachable!("normalized to Rotor3 above"),
        }
    }

    (matrices, n_groups, padded_dim)
}

/// Generate the deterministic ±1 sign vector for rotation kinds that apply
/// a signed Walsh-Hadamard pre-pass. Returns `None` for variants that don't
/// use WHT. Uses a seed derived from `seed` (offset to avoid colliding with
/// the block-rotation RNG stream and stay reproducible).
pub fn precompute_walsh_signs(dim: usize, kind: Rotation, seed: u64) -> Option<Vec<f32>> {
    if !kind.uses_walsh() {
        return None;
    }
    debug_assert!(
        crate::wht::is_pow2(dim),
        "WalshRotor3 requires dim to be a power of 2"
    );
    // Wrapping-add offset puts the WHT sign RNG on its own stream so changes
    // to one cache don't shift the other.
    Some(crate::wht::make_sign_vector(
        dim,
        seed.wrapping_add(0x57_4854_u64),
    ))
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
                m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
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
            let (mats, n_groups, padded) = precompute_block_matrices(bs * 5, kind, 42);
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
    fn rotor3_trailing_partial_block_is_identity() {
        // dim=128 isn't divisible by 3 — last block (g=42) should be I.
        let (mats, n_groups, padded) = precompute_block_matrices(128, Rotation::Rotor3, 42);
        assert_eq!(n_groups, 43);
        assert_eq!(padded, 129);

        let last = &mats[(n_groups - 1) * 9..n_groups * 9];
        let identity_3x3: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(last, &identity_3x3, "trailing partial block should be I");

        // Earlier blocks should NOT be identity (vanishingly unlikely from RNG).
        let first = &mats[0..9];
        assert_ne!(
            first, &identity_3x3,
            "first block should be a real rotation"
        );
    }

    #[test]
    fn rotor3_full_block_dim_unchanged() {
        // dim divisible by 3 — every block is a real rotation.
        let (mats, n_groups, padded) = precompute_block_matrices(126, Rotation::Rotor3, 42);
        assert_eq!(n_groups, 42);
        assert_eq!(padded, 126);

        let identity_3x3: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for g in 0..n_groups {
            let m = &mats[g * 9..(g + 1) * 9];
            check_orthogonal(m, 3);
            check_det_plus_one(m, 3);
            assert_ne!(m, &identity_3x3, "block {g} should be a real rotation");
        }
    }

    #[test]
    fn rotor3_partial_block_preserves_real_coords() {
        // After block rotation, the trailing real coords should be exactly
        // the input values — no contamination from the zero-padded tail.
        use crate::rotation::rotate_inplace;

        let dim = 128;
        let (mats, _, padded) = precompute_block_matrices(dim, Rotation::Rotor3, 42);
        // Zero-padded input: last real coord is index 127, padded index 128 = 0.
        let mut v = vec![0.0f32; padded];
        for j in 0..dim {
            v[j] = (j as f32) * 0.01 + 0.5;
        }
        let v_in = v.clone();
        rotate_inplace(&mats, 3, &mut v);

        // Indices 126 and 127 (the real coords inside the trailing partial
        // block) must survive untouched. Index 128 (padded) stays zero.
        assert!(
            (v[126] - v_in[126]).abs() < 1e-6,
            "v[126] changed: {} -> {}",
            v_in[126],
            v[126]
        );
        assert!(
            (v[127] - v_in[127]).abs() < 1e-6,
            "v[127] changed: {} -> {}",
            v_in[127],
            v[127]
        );
        assert!(v[128].abs() < 1e-6, "padded coord should stay zero");
    }

    #[test]
    fn rotation_byte_roundtrip() {
        for &kind in &[Rotation::Planar2, Rotation::Rotor3, Rotation::Iso4] {
            assert_eq!(Rotation::from_byte(kind.as_byte()), Some(kind));
        }
        assert_eq!(Rotation::from_byte(99), None);
    }
}
