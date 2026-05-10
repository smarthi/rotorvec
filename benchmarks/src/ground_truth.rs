//! Brute-force ground-truth top-k computation, with a disk cache.
//!
//! For an n=1M, n_q=10K, d=128 dataset that's 10B inner products
//! (~1.3T FMAs) — about 30s on a modern multi-core machine via rayon. We
//! cache the result keyed on `(dataset_name, k, normalized_yes_no)` so
//! subsequent runs are instant.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use crate::datasets::cache_dir;

pub struct GroundTruth {
    pub indices: Vec<i32>, // n_queries * k
    #[allow(dead_code)]
    pub k: usize,
}

/// Compute top-k by inner product (assumes vectors are already normalized
/// for cosine-style search). Cached to disk under
/// `<cache>/ground_truth/<key>.bin`.
pub fn compute_or_load(
    cache_key: &str,
    train: &[f32],
    n_train: usize,
    queries: &[f32],
    n_queries: usize,
    dim: usize,
    k: usize,
) -> Result<GroundTruth> {
    let path = cache_path(cache_key, k)?;
    if path.exists() {
        return load(&path, k);
    }

    eprintln!(
        "Computing ground truth: n_train={n_train} n_queries={n_queries} d={dim} k={k}"
    );
    let pb = ProgressBar::new(n_queries as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner} {wide_bar} {pos}/{len} queries ({eta})")
            .unwrap(),
    );

    let mut indices = vec![-1i32; n_queries * k];
    let pb_ref = &pb;
    indices
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(qi, out)| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let topk = brute_force_top_k(q, train, n_train, dim, k);
            for (slot, (_, idx)) in topk.into_iter().enumerate() {
                out[slot] = idx as i32;
            }
            pb_ref.inc(1);
        });
    pb.finish_with_message("done");

    save(&path, &indices, k)?;
    Ok(GroundTruth { indices, k })
}

fn brute_force_top_k(
    q: &[f32],
    train: &[f32],
    n_train: usize,
    dim: usize,
    k: usize,
) -> Vec<(f32, usize)> {
    // Maintain a small ascending-sorted buffer of length k.
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k);

    for vi in 0..n_train {
        let v = &train[vi * dim..(vi + 1) * dim];
        let mut s = 0.0f32;
        for j in 0..dim {
            s += q[j] * v[j];
        }

        if top.len() < k {
            top.push((s, vi));
            if top.len() == k {
                top.sort_unstable_by(|a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
                });
            }
        } else if s > top[0].0 {
            top[0] = (s, vi);
            let mut i = 0;
            while i + 1 < k && top[i].0 > top[i + 1].0 {
                top.swap(i, i + 1);
                i += 1;
            }
        }
    }

    // Return descending so slot 0 is the best.
    top.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    top
}

fn cache_path(key: &str, k: usize) -> Result<PathBuf> {
    let dir = cache_dir()?.join("ground_truth");
    fs::create_dir_all(&dir)?;
    Ok(dir.join(format!("{key}_k{k}.bin")))
}

fn save(path: &PathBuf, indices: &[i32], k: usize) -> Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    let n = (indices.len() / k) as u32;
    let k_u32 = k as u32;
    f.write_all(&n.to_le_bytes())?;
    f.write_all(&k_u32.to_le_bytes())?;
    let bytes: &[u8] = bytemuck_safe(indices);
    f.write_all(bytes)?;
    Ok(())
}

fn load(path: &PathBuf, expected_k: usize) -> Result<GroundTruth> {
    let mut f = BufReader::new(File::open(path).with_context(|| format!("open {path:?}"))?);
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?;
    let n = u32::from_le_bytes(buf4) as usize;
    f.read_exact(&mut buf4)?;
    let k = u32::from_le_bytes(buf4) as usize;
    if k != expected_k {
        return Err(anyhow::anyhow!(
            "cached ground truth has k={k} but caller asked for k={expected_k}"
        ));
    }
    let mut bytes = vec![0u8; n * k * 4];
    f.read_exact(&mut bytes)?;
    let indices: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(GroundTruth { indices, k })
}

/// Small wrapper that reinterprets `&[i32]` as `&[u8]` without pulling in
/// `bytemuck` as a dep. Safe because i32 has no padding and any byte pattern
/// is a valid i32.
fn bytemuck_safe(slice: &[i32]) -> &[u8] {
    let len = std::mem::size_of_val(slice);
    // SAFETY: i32 is plain old data with no padding; len is exactly the
    // byte length of the slice.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, len) }
}

/// recall@k: fraction of approximate top-k results that overlap with the
/// ground-truth top-k (set intersection / k), averaged over all queries.
pub fn recall_at_k(approx: &[i64], truth: &[i32], k: usize) -> f64 {
    let n_queries = approx.len() / k;
    debug_assert_eq!(truth.len() / k, n_queries);

    let total: usize = (0..n_queries)
        .map(|qi| {
            let a = &approx[qi * k..(qi + 1) * k];
            let t = &truth[qi * k..(qi + 1) * k];
            let truth_set: std::collections::HashSet<i32> = t.iter().copied().collect();
            a.iter().filter(|&&x| truth_set.contains(&(x as i32))).count()
        })
        .sum();

    total as f64 / (n_queries * k) as f64
}
