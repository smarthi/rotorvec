//! Fast Walsh-Hadamard transform — cross-block mixing for the `WalshRotor3`
//! rotation variant.
//!
//! Block-diagonal rotation alone only decorrelates *within* small blocks. On
//! data with cross-coordinate correlations (image features, SIFT), this
//! costs measurable recall vs a full d×d random rotation (see
//! `benchmarks/REAL_DATASETS.md`). Walsh-Hadamard is the missing piece: it
//! spreads each coordinate's energy across all `d` outputs in O(d log d)
//! work and ~d random bits of state, sitting between the O(d) block-only
//! and O(d²) dense-GEMM ends of the cost spectrum.
//!
//! The transform we apply is `H · diag(s) · v`, where:
//! * `s ∈ {-1, +1}^d` is a deterministic random sign vector (seeded ChaCha8)
//! * `H` is the unnormalized order-`d` Hadamard matrix; we apply the
//!   Sylvester construction in place via the standard butterfly network.
//!
//! `H · H = d · I`, so the inverse transform is `(1/d) · diag(s) · H · w`.
//! Search rotates queries and database vectors symmetrically, so the
//! normalization factor cancels — we apply the unnormalized transform here
//! and let inner products absorb the constant.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// True iff `n` is a power of two (and nonzero).
pub fn is_pow2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// In-place fast Walsh-Hadamard transform, unnormalized.
///
/// `x.len()` must be a power of two. Applies the order-`x.len()` Hadamard
/// matrix `H` to `x` via the butterfly network in `O(n log n)` work.
///
/// After this call, `x_out[i] = sum_j H[i,j] * x_in[j]`. The transform is
/// involutive up to a scalar: applying `fwht` twice multiplies the input by
/// `n`.
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(is_pow2(n), "fwht requires len to be a power of 2, got {n}");

    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            // Butterfly across pairs (i+j, i+j+h) for j in 0..h.
            for j in 0..h {
                let a = x[i + j];
                let b = x[i + j + h];
                x[i + j] = a + b;
                x[i + j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// Apply `H · diag(s)` to `x` in place: element-wise sign-flip then FWHT.
pub fn apply_signed_wht(x: &mut [f32], signs: &[f32]) {
    debug_assert_eq!(x.len(), signs.len());
    for (xi, &si) in x.iter_mut().zip(signs.iter()) {
        *xi *= si;
    }
    fwht(x);
}

/// Generate a deterministic ±1 sign vector of length `dim` from a seed.
/// Returns floats so callers can multiply directly without a branch.
pub fn make_sign_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..dim)
        .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_pow2_basic() {
        for &n in &[1usize, 2, 4, 8, 16, 64, 128, 1024, 4096] {
            assert!(is_pow2(n), "{n} should be pow2");
        }
        for &n in &[0usize, 3, 5, 6, 100, 384, 1536] {
            assert!(!is_pow2(n), "{n} should not be pow2");
        }
    }

    #[test]
    fn fwht_double_application_scales_by_n() {
        // Apply FWHT twice → output should be n × input (H · H = n · I).
        for &n in &[2, 4, 8, 16, 64] {
            let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 0.5).collect();
            let mut x = input.clone();
            fwht(&mut x);
            fwht(&mut x);
            for i in 0..n {
                let expected = input[i] * (n as f32);
                assert!(
                    (x[i] - expected).abs() < 1e-3,
                    "n={n} i={i}: got {}, expected {expected}",
                    x[i]
                );
            }
        }
    }

    #[test]
    fn fwht_preserves_l2_norm_up_to_sqrt_n() {
        // |H · x| = sqrt(n) · |x| for the unnormalized Hadamard.
        let n = 64;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.07 - 0.3).collect();
        let in_norm = (input.iter().map(|x| x * x).sum::<f32>()).sqrt();

        let mut x = input.clone();
        fwht(&mut x);
        let out_norm = (x.iter().map(|x| x * x).sum::<f32>()).sqrt();

        let expected = in_norm * (n as f32).sqrt();
        assert!(
            (out_norm - expected).abs() / expected < 1e-4,
            "got {out_norm}, expected {expected}"
        );
    }

    #[test]
    fn signed_wht_inverse_recovers_input() {
        // T(x) = H · diag(s) · x  is the forward transform.
        // Its inverse is `T⁻¹(y) = diag(s) · (1/n) · H · y` (since
        // H · H = n · I and diag(s)² = I). Applying T then T⁻¹ in that
        // exact order must recover x.
        let n = 32;
        let signs = make_sign_vector(n, 7);
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.13 - 0.5).collect();
        let mut x = input.clone();

        // Forward: sign-flip then FWHT.
        apply_signed_wht(&mut x, &signs);

        // Inverse: FWHT then sign-flip then divide by n.
        fwht(&mut x);
        for (xi, &si) in x.iter_mut().zip(signs.iter()) {
            *xi *= si;
        }
        let inv_n = 1.0 / (n as f32);
        for xi in x.iter_mut() {
            *xi *= inv_n;
        }

        for i in 0..n {
            assert!(
                (x[i] - input[i]).abs() < 1e-4,
                "i={i}: got {}, expected {}",
                x[i],
                input[i]
            );
        }
    }

    #[test]
    fn sign_vector_deterministic() {
        let a = make_sign_vector(64, 42);
        let b = make_sign_vector(64, 42);
        assert_eq!(a, b);
        let c = make_sign_vector(64, 43);
        assert_ne!(a, c);
        for &x in &a {
            assert!(x == 1.0 || x == -1.0);
        }
    }
}
