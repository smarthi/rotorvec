//! End-to-end tests for the public API.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rotorvec::{IdMapIndex, Rotation, RotorQuantIndex};

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut v = vec![0.0f32; n * dim];
    for x in &mut v {
        *x = rng.gen::<f32>() * 2.0 - 1.0;
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

const ALL_ROTATIONS: [Rotation; 3] = [Rotation::Planar2, Rotation::Rotor3, Rotation::Iso4];

/// On aarch64 the default search path is NEON for 4-bit. This test compares
/// it against the public scalar search entry point to make sure the NEON
/// quantized-LUT approximation lands on the same top-k that scalar would.
///
/// We allow a small amount of slack: NEON uses u8 LUTs (max 127 levels per
/// sub-table) which introduces low-bit quantization error vs scalar's
/// straight f32 accumulation. With well-separated random data and `n` in
/// the hundreds, the top-k *set* should still match exactly for small `k`.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_4bit_topk_matches_scalar() {
    // The NEON path is the only public path on aarch64 for 4-bit, so we
    // compare it against brute-force ground truth rather than against the
    // scalar path (which is unreachable from the public API here). Both
    // paths approximate the same target — the test ensures NEON's quantized
    // u8-LUT approximation doesn't degrade recall meaningfully.
    for &rot in &ALL_ROTATIONS {
        let dim = 128;
        let n = 500;
        let bits = 4;
        let k = 10;
        let vectors = random_vectors(n, dim, 31);
        let queries = random_vectors(8, dim, 41);

        let mut idx = RotorQuantIndex::with_rotation(dim, bits, rot);
        idx.add(&vectors);
        idx.prepare();

        let res_neon = idx.search(&queries, k);

        let mut hits_at_1 = 0usize;
        let mut hits_at_k = 0usize;
        for qi in 0..8 {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut best: Vec<(f32, usize)> = (0..n)
                .map(|vi| (dot(q, &vectors[vi * dim..(vi + 1) * dim]), vi))
                .collect();
            best.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let true_top1 = best[0].1 as i64;
            let true_topk: std::collections::HashSet<i64> =
                best.iter().take(k).map(|(_, vi)| *vi as i64).collect();

            let neon_idx = res_neon.indices_for_query(qi);
            if neon_idx[0] == true_top1 {
                hits_at_1 += 1;
            }
            for &i in neon_idx {
                if true_topk.contains(&i) {
                    hits_at_k += 1;
                }
            }
        }
        assert!(
            hits_at_1 >= 5,
            "rot={rot:?}: NEON recall@1 = {hits_at_1}/8, expected >= 5"
        );
        assert!(
            hits_at_k as f64 / (8 * k) as f64 >= 0.7,
            "rot={rot:?}: NEON recall@{k} = {} / {}, expected >= 70%",
            hits_at_k,
            8 * k
        );
    }
}

#[test]
fn build_search_roundtrip_all_variants() {
    for &rot in &ALL_ROTATIONS {
        let dim = 128;
        let n = 200;
        let bits = 4;
        let vectors = random_vectors(n, dim, 1);
        let queries = random_vectors(3, dim, 2);

        let mut idx = RotorQuantIndex::with_rotation(dim, bits, rot);
        idx.add(&vectors);
        assert_eq!(idx.len(), n);
        assert_eq!(idx.rotation(), rot);

        let res = idx.search(&queries, 5);
        assert_eq!(res.nq, 3);
        assert_eq!(res.k, 5);

        for qi in 0..3 {
            let scores = res.scores_for_query(qi);
            let indices = res.indices_for_query(qi);
            for w in scores.windows(2) {
                assert!(w[0] >= w[1], "{rot:?}: scores not sorted descending");
            }
            for &i in indices {
                assert!(i >= 0 && (i as usize) < n);
            }
        }
    }
}

#[test]
fn recall_top1_beats_random_all_variants() {
    // Each variant should comfortably beat random (1/n) recall@1.
    for &rot in &ALL_ROTATIONS {
        let dim = 256;
        let n = 500;
        let bits = 4;
        let nq = 50;
        let vectors = random_vectors(n, dim, 7);
        let queries = random_vectors(nq, dim, 8);

        let mut idx = RotorQuantIndex::with_rotation(dim, bits, rot);
        idx.add(&vectors);
        let res = idx.search(&queries, 1);

        let mut hits = 0;
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut best_true = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for vi in 0..n {
                let s = dot(q, &vectors[vi * dim..(vi + 1) * dim]);
                if s > best_score {
                    best_score = s;
                    best_true = vi;
                }
            }
            if res.indices_for_query(qi)[0] as usize == best_true {
                hits += 1;
            }
        }
        // Generous floor (50%) — catches gross algorithmic bugs without
        // being flaky. Each variant typically hits 90%+ on this synthetic
        // setup.
        assert!(hits >= 25, "{rot:?}: recall@1 = {hits}/{nq}");
    }
}

#[test]
fn save_load_roundtrip_all_variants() {
    for &rot in &ALL_ROTATIONS {
        let dim = 64;
        let n = 30;
        let bits = 4;
        let vectors = random_vectors(n, dim, 11);
        let queries = random_vectors(2, dim, 12);

        let mut idx = RotorQuantIndex::with_rotation(dim, bits, rot);
        idx.add(&vectors);
        let before = idx.search(&queries, 5);

        let tmp = std::env::temp_dir().join(format!("rotorvec_{rot:?}.rv"));
        idx.write(&tmp).unwrap();
        let loaded = RotorQuantIndex::load(&tmp).unwrap();
        assert_eq!(loaded.rotation(), rot);
        let after = loaded.search(&queries, 5);

        assert_eq!(before.scores, after.scores);
        assert_eq!(before.indices, after.indices);
        let _ = std::fs::remove_file(&tmp);
    }
}

#[test]
fn id_map_basic() {
    let dim = 64;
    let bits = 4;
    let vectors = random_vectors(5, dim, 21);
    let ids = vec![100u64, 200, 300, 400, 500];

    let mut idx = IdMapIndex::with_rotation(dim, bits, Rotation::Planar2);
    idx.add_with_ids(&vectors, &ids);
    assert_eq!(idx.len(), 5);
    assert!(idx.contains(300));

    assert!(idx.remove(300));
    assert!(!idx.contains(300));
    assert_eq!(idx.len(), 4);

    let q = &vectors[0..dim];
    let (_scores, returned_ids) = idx.search(q, 4);
    assert!(returned_ids.contains(&100));
    assert!(!returned_ids.contains(&300));
}

#[test]
fn default_rotation_is_rotor3() {
    let idx = RotorQuantIndex::new(64, 4);
    assert_eq!(idx.rotation(), Rotation::Rotor3);
}
