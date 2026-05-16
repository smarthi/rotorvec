//! Dataset loaders for ANN benchmarks.
//!
//! v0.2.2 ships **SIFT-1M** and **GloVe-100** from ann-benchmarks.com,
//! prepped via `benchmarks/scripts/prep_datasets.py`. Run that script
//! once to download the HDF5 source and convert it to the .fvecs/.ivecs
//! binary format this loader reads. Cached under `dirs::cache_dir()`
//! (typically `~/Library/Caches/rotorvec/datasets/` on macOS,
//! `~/.cache/rotorvec/datasets/` on Linux).
//!
//! File format: `.fvecs` for float vectors, `.ivecs` for int vectors.
//! Each record = `[i32 dim, T dim ... T dim]` little-endian. We trust
//! `dim` from the first record and skip the per-record dim header on
//! subsequent reads.

use anyhow::{anyhow, Context, Result};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    /// SIFT-1M from INRIA's TexMex. 1M × 128 vectors, L2 distance natively;
    /// we normalize and recompute ground truth for inner-product search.
    /// Requires `prep_datasets.py sift-128` to have been run first.
    Sift1M,
    /// GloVe word vectors trained on 6B-token corpus. 1.18M × 100 vectors,
    /// cosine/angular distance. dim=100 is padded with zeros to dim=104
    /// internally to satisfy rotorvec's `dim % 8 == 0` constraint. Requires
    /// `prep_datasets.py glove-100` to have been run first.
    GloVe100,
    /// Synthetic standard-Gaussian unit vectors. Useful for rapid
    /// iteration and for isolating algorithm behavior from
    /// dataset-specific quirks (clusters, hubness, low-rank manifolds).
    Random { n: usize, dim: usize },
}

impl Dataset {
    pub fn name(&self) -> String {
        match self {
            Dataset::Sift1M => "sift-1m".to_string(),
            Dataset::GloVe100 => "glove-100".to_string(),
            Dataset::Random { n, dim } => format!("random-n{n}-d{dim}"),
        }
    }
    #[allow(dead_code)]
    pub fn dim(&self) -> usize {
        match self {
            Dataset::Sift1M => 128,
            Dataset::GloVe100 => 104, // padded
            Dataset::Random { dim, .. } => *dim,
        }
    }
}

pub struct Loaded {
    pub train: Vec<f32>, // n_train * dim
    pub n_train: usize,
    pub queries: Vec<f32>, // n_queries * dim
    pub n_queries: usize,
    pub dim: usize,
    /// Optional precomputed ground-truth top-k indices, `n_queries * k_gt`.
    /// `None` means caller must compute it. Kept for future datasets that
    /// ship their own ground truth (e.g. ann-benchmarks HDF5 files).
    #[allow(dead_code)]
    pub ground_truth: Option<(Vec<i32>, usize)>,
}

pub fn cache_dir() -> Result<PathBuf> {
    let base = dirs::cache_dir().ok_or_else(|| anyhow!("could not locate user cache dir"))?;
    let dir = base.join("rotorvec").join("datasets");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn load(dataset: Dataset) -> Result<Loaded> {
    match dataset {
        Dataset::Sift1M => load_prepped("sift-128", 128, None),
        Dataset::GloVe100 => load_prepped("glove-100", 100, Some(104)),
        Dataset::Random { n, dim } => Ok(load_random(n, dim, 1000)),
    }
}

/// Load a dataset prepped by `benchmarks/scripts/prep_datasets.py`.
///
/// Expects three files under `<cache>/datasets/<name>/`:
/// * `base.fvecs` — `n_train × native_dim` train vectors
/// * `query.fvecs` — `n_queries × native_dim` query vectors
/// * `groundtruth.ivecs` — `n_queries × gt_k` precomputed nearest neighbor indices
///
/// If `pad_to_dim` is provided and larger than `native_dim`, each vector is
/// zero-padded to that width. This is how we satisfy rotorvec's
/// `dim % 8 == 0` constraint on datasets like GloVe-100 (native dim=100,
/// padded to dim=104). The padding zeros don't affect inner-product search
/// because zero × anything = zero on both sides.
fn load_prepped(name: &str, native_dim: usize, pad_to_dim: Option<usize>) -> Result<Loaded> {
    let dir = cache_dir()?.parent().unwrap().join("datasets").join(name);
    if !dir.exists() {
        anyhow::bail!(
            "Dataset '{name}' not prepped. Run:\n\
             \n\
             \x20    uv run python benchmarks/scripts/prep_datasets.py {name}\n",
        );
    }
    let base = read_fvecs(&dir.join("base.fvecs"))?;
    let queries = read_fvecs(&dir.join("query.fvecs"))?;
    let (gt_data, gt_k) = read_ivecs(&dir.join("groundtruth.ivecs"))?;
    let dim = native_dim;
    let n_train = base.len() / dim;
    let n_queries = queries.len() / dim;

    let (train_out, queries_out, dim_out) = match pad_to_dim {
        Some(target) if target > dim => {
            let mut t = vec![0.0f32; n_train * target];
            for i in 0..n_train {
                t[i * target..i * target + dim].copy_from_slice(&base[i * dim..(i + 1) * dim]);
            }
            let mut q = vec![0.0f32; n_queries * target];
            for i in 0..n_queries {
                q[i * target..i * target + dim].copy_from_slice(&queries[i * dim..(i + 1) * dim]);
            }
            (t, q, target)
        }
        _ => (base, queries, dim),
    };

    Ok(Loaded {
        train: train_out,
        n_train,
        queries: queries_out,
        n_queries,
        dim: dim_out,
        ground_truth: Some((gt_data, gt_k)),
    })
}

/// Synthetic dataset: `n` train + `n_queries` query vectors drawn from
/// standard Gaussian (each coordinate ~ N(0, 1)). Deterministically seeded
/// so two runs with the same `(n, dim, n_queries)` are bit-identical.
fn load_random(n: usize, dim: usize, n_queries: usize) -> Loaded {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::StandardNormal;

    let mut rng = ChaCha8Rng::seed_from_u64(0xDA7A_5EED_u64);
    let total_train = n * dim;
    let total_q = n_queries * dim;

    let mut train = Vec::with_capacity(total_train);
    for _ in 0..total_train {
        train.push(rng.sample::<f64, _>(StandardNormal) as f32);
    }
    let mut queries = Vec::with_capacity(total_q);
    for _ in 0..total_q {
        queries.push(rng.sample::<f64, _>(StandardNormal) as f32);
    }

    Loaded {
        train,
        n_train: n,
        queries,
        n_queries,
        dim,
        ground_truth: None,
    }
}

// SIFT-1M and other real datasets are loaded via `load_prepped` above. The
// previous INRIA-mirror downloader was removed when the mirror went offline;
// use `benchmarks/scripts/prep_datasets.py` which pulls from
// ann-benchmarks.com instead (more reliable, ships precomputed ground truth).

/// Read an `.fvecs` file as a flat `Vec<f32>` of length `n * dim`.
pub fn read_fvecs(path: &Path) -> Result<Vec<f32>> {
    let mut f = BufReader::new(File::open(path).with_context(|| format!("open {path:?}"))?);
    let mut header = [0u8; 4];

    // Peek dim from first record.
    f.read_exact(&mut header)?;
    let dim = i32::from_le_bytes(header) as usize;
    if dim == 0 || dim > 1_000_000 {
        return Err(anyhow!("implausible dim {dim} in {path:?}"));
    }

    let total_bytes = fs::metadata(path)?.len() as usize;
    let record_size = 4 + dim * 4;
    if total_bytes % record_size != 0 {
        return Err(anyhow!(
            "{path:?} length {total_bytes} not a multiple of record size {record_size} (dim={dim})"
        ));
    }
    let n = total_bytes / record_size;
    let mut out = Vec::with_capacity(n * dim);

    // First record: read dim floats (header already consumed).
    let mut buf = vec![0u8; dim * 4];
    f.read_exact(&mut buf)?;
    out.extend(
        buf.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
    );

    // Remaining records: skip dim header, read floats.
    for _ in 1..n {
        f.read_exact(&mut header)?;
        f.read_exact(&mut buf)?;
        out.extend(
            buf.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
    }

    Ok(out)
}

/// Read an `.ivecs` file as `(flat Vec<i32> of length n * k, k)`.
pub fn read_ivecs(path: &Path) -> Result<(Vec<i32>, usize)> {
    let mut f = BufReader::new(File::open(path).with_context(|| format!("open {path:?}"))?);
    let mut header = [0u8; 4];
    f.read_exact(&mut header)?;
    let k = i32::from_le_bytes(header) as usize;
    if k == 0 || k > 100_000 {
        return Err(anyhow!("implausible k {k} in {path:?}"));
    }

    let total_bytes = fs::metadata(path)?.len() as usize;
    let record_size = 4 + k * 4;
    if total_bytes % record_size != 0 {
        return Err(anyhow!("ivecs size mismatch in {path:?}"));
    }
    let n = total_bytes / record_size;
    let mut out = Vec::with_capacity(n * k);

    let mut buf = vec![0u8; k * 4];
    f.read_exact(&mut buf)?;
    out.extend(
        buf.chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
    );

    for _ in 1..n {
        f.read_exact(&mut header)?;
        f.read_exact(&mut buf)?;
        out.extend(
            buf.chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
    }

    Ok((out, k))
}

/// Normalize each row to unit L2 norm in place. SIFT-1M is L2-distance
/// natively; for an inner-product index we need unit vectors so that
/// `argmax <q, x>` corresponds to `argmin ||q - x||²` *after* normalization.
/// Note: this changes the ground-truth topology — the precomputed ground
/// truth from INRIA is for *unnormalized* L2. After normalization the
/// rankings can differ. For benchmarks we recompute ground truth on the
/// normalized vectors; see [`crate::ground_truth`].
pub fn normalize_inplace(vectors: &mut [f32], dim: usize) {
    use rayon::prelude::*;
    vectors.par_chunks_mut(dim).for_each(|row| {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let inv = 1.0 / norm;
        for x in row.iter_mut() {
            *x *= inv;
        }
    });
}
