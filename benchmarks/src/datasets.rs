//! Dataset loaders for ANN benchmarks.
//!
//! v0.2.0 ships SIFT-1M from INRIA's TexMex collection. The raw archive is
//! ~160 MB compressed, ~520 MB extracted. Cached under `dirs::cache_dir()`
//! (typically `~/Library/Caches/rotorvec` on macOS, `~/.cache/rotorvec` on
//! Linux).
//!
//! File format: `.fvecs` for float vectors, `.ivecs` for int vectors.
//! Each record = `[i32 dim, T dim ... T dim]` little-endian. We trust `dim`
//! from the first record and skip the per-record dim header on subsequent
//! reads.

use anyhow::{anyhow, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};

const SIFT_URL: &str = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz";
const SIFT_URL_HTTP: &str = "http://corpus-texmex.irisa.fr/sift.tar.gz";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dataset {
    Sift1M,
    /// Synthetic standard-Gaussian unit vectors. Useful for rapid
    /// iteration and for isolating algorithm behavior from
    /// dataset-specific quirks (clusters, hubness, low-rank manifolds).
    Random {
        n: usize,
        dim: usize,
    },
}

impl Dataset {
    pub fn name(&self) -> String {
        match self {
            Dataset::Sift1M => "sift-1m".to_string(),
            Dataset::Random { n, dim } => format!("random-n{n}-d{dim}"),
        }
    }
    #[allow(dead_code)]
    pub fn dim(&self) -> usize {
        match self {
            Dataset::Sift1M => 128,
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
        Dataset::Sift1M => load_sift_1m(),
        Dataset::Random { n, dim } => Ok(load_random(n, dim, 1000)),
    }
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

fn load_sift_1m() -> Result<Loaded> {
    let dir = cache_dir()?.join("sift");
    if !dir.exists() {
        download_and_extract_sift(&dir)?;
    }

    let train = read_fvecs(&dir.join("sift_base.fvecs"))?;
    let queries = read_fvecs(&dir.join("sift_query.fvecs"))?;
    let (gt_data, gt_k) = read_ivecs(&dir.join("sift_groundtruth.ivecs"))?;

    let dim = 128;
    let n_train = train.len() / dim;
    let n_queries = queries.len() / dim;

    Ok(Loaded {
        train,
        n_train,
        queries,
        n_queries,
        dim,
        ground_truth: Some((gt_data, gt_k)),
    })
}

fn download_and_extract_sift(dest: &Path) -> Result<()> {
    let parent = dest.parent().unwrap();
    fs::create_dir_all(parent)?;
    let archive = parent.join("sift.tar.gz");

    if !archive.exists() {
        eprintln!("Downloading SIFT-1M (~160 MB) ...");
        // Try HTTP mirror first (FTP is blocked in many networks).
        let resp = ureq::get(SIFT_URL_HTTP)
            .timeout(std::time::Duration::from_secs(300))
            .call()
            .with_context(|| format!("download from {SIFT_URL_HTTP}"))?;

        let len = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok());
        let pb = match len {
            Some(n) => ProgressBar::new(n),
            None => ProgressBar::new_spinner(),
        };
        if len.is_some() {
            pb.set_style(
                ProgressStyle::with_template("{spinner} {wide_bar} {bytes}/{total_bytes} ({eta})")
                    .unwrap(),
            );
        }

        let mut reader = pb.wrap_read(resp.into_reader());
        let mut writer = BufWriter::new(File::create(&archive)?);
        std::io::copy(&mut reader, &mut writer)?;
        pb.finish_with_message("downloaded");
        eprintln!("\nFallback URL if this failed: {SIFT_URL}");
    }

    eprintln!("Extracting to {} ...", dest.display());
    let tar_gz = File::open(&archive)?;
    let tar = flate2::read::GzDecoder::new(tar_gz);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(parent)?;

    Ok(())
}

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
