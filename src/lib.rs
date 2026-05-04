//! RotorQuant vector index — block-diagonal Clifford-rotor quantization.
//!
//! ```no_run
//! use rotorvec::RotorQuantIndex;
//!
//! let mut index = RotorQuantIndex::new(1536, 4);
//! let vectors: Vec<f32> = vec![0.0; 1536 * 10];
//! let queries: Vec<f32> = vec![0.0; 1536 * 2];
//! index.add(&vectors);
//! let results = index.search(&queries, 10);
//! index.write("index.rv").unwrap();
//! let loaded = RotorQuantIndex::load("index.rv").unwrap();
//! ```
//!
//! See [`RotorQuantIndex`] for the main API and [`IdMapIndex`] for stable
//! external ids.

pub mod codebook;
pub mod encode;
pub mod id_map;
pub mod io;
pub mod pack;
pub mod rotation;
pub mod rotor;
pub mod search;

pub use id_map::IdMapIndex;

use std::path::Path;
use std::sync::OnceLock;

/// Default rotor seed. Two indices built with the same `(dim, seed)` produce
/// bit-identical block-rotation matrices.
pub const DEFAULT_ROTOR_SEED: u64 = 0x0707_04EC_u64;

/// Top-k search results, laid out as `nq` consecutive rows of length `k`.
pub struct SearchResults {
    pub scores: Vec<f32>,
    pub indices: Vec<i64>,
    pub nq: usize,
    pub k: usize,
}

impl SearchResults {
    pub fn scores_for_query(&self, qi: usize) -> &[f32] {
        &self.scores[qi * self.k..(qi + 1) * self.k]
    }
    pub fn indices_for_query(&self, qi: usize) -> &[i64] {
        &self.indices[qi * self.k..(qi + 1) * self.k]
    }
}

pub struct RotorQuantIndex {
    dim: usize,
    bits: usize,
    n_vectors: usize,
    seed: u64,
    packed_codes: Vec<u8>,
    norms: Vec<f32>,

    block_matrices: OnceLock<Vec<f32>>,
    centroids: OnceLock<Vec<f32>>,
}

impl RotorQuantIndex {
    /// Create an empty index. `dim` must be a multiple of 8; `bits` must be
    /// 2, 3, or 4.
    pub fn new(dim: usize, bits: usize) -> Self {
        Self::with_seed(dim, bits, DEFAULT_ROTOR_SEED)
    }

    pub fn with_seed(dim: usize, bits: usize, seed: u64) -> Self {
        assert!((2..=4).contains(&bits), "bits must be 2, 3, or 4");
        assert!(dim % 8 == 0, "dim must be a multiple of 8");
        Self {
            dim,
            bits,
            n_vectors: 0,
            seed,
            packed_codes: Vec::new(),
            norms: Vec::new(),
            block_matrices: OnceLock::new(),
            centroids: OnceLock::new(),
        }
    }

    pub fn add(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        assert_eq!(vectors.len(), n * self.dim, "vectors length must be a multiple of dim");
        if n == 0 {
            return;
        }

        let block_matrices = self.ensure_block_matrices().clone();
        let (boundaries, _) = codebook::codebook(self.bits, self.dim);
        let (packed, norms) =
            encode::encode(vectors, n, self.dim, &block_matrices, &boundaries, self.bits);

        self.packed_codes.extend_from_slice(&packed);
        self.norms.extend_from_slice(&norms);
        self.n_vectors += n;
    }

    pub fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);

        let block_matrices = self.ensure_block_matrices();
        let centroids = self.ensure_centroids();
        let k = k.min(self.n_vectors).max(1);

        let (scores, indices) = if self.n_vectors == 0 {
            (vec![f32::NEG_INFINITY; nq * k], vec![-1i64; nq * k])
        } else {
            search::search(
                queries,
                nq,
                self.dim,
                block_matrices,
                &self.packed_codes,
                centroids,
                &self.norms,
                self.bits,
                self.n_vectors,
                k,
            )
        };

        SearchResults { scores, indices, nq, k }
    }

    pub fn prepare(&self) {
        self.ensure_block_matrices();
        self.ensure_centroids();
    }

    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        io::write(
            path,
            self.bits,
            self.dim,
            self.n_vectors,
            self.seed,
            &self.packed_codes,
            &self.norms,
        )
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let (bits, dim, n_vectors, seed, packed_codes, norms) = io::load(path)?;
        Ok(Self::from_parts(dim, bits, n_vectors, seed, packed_codes, norms))
    }

    pub(crate) fn from_parts(
        dim: usize,
        bits: usize,
        n_vectors: usize,
        seed: u64,
        packed_codes: Vec<u8>,
        norms: Vec<f32>,
    ) -> Self {
        Self {
            dim,
            bits,
            n_vectors,
            seed,
            packed_codes,
            norms,
            block_matrices: OnceLock::new(),
            centroids: OnceLock::new(),
        }
    }

    pub(crate) fn packed_codes(&self) -> &[u8] {
        &self.packed_codes
    }
    pub(crate) fn norms(&self) -> &[f32] {
        &self.norms
    }
    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }

    fn ensure_block_matrices(&self) -> &Vec<f32> {
        self.block_matrices
            .get_or_init(|| rotor::precompute_block_matrices(self.dim, self.seed).0)
    }

    fn ensure_centroids(&self) -> &Vec<f32> {
        self.centroids.get_or_init(|| {
            let (_, c) = codebook::codebook(self.bits, self.dim);
            c
        })
    }

    /// Remove the vector at `idx` in O(1) by swap-with-last. Order is **not**
    /// preserved. Returns the old index of the moved vector.
    pub fn swap_remove(&mut self, idx: usize) -> usize {
        assert!(idx < self.n_vectors, "index {idx} out of bounds");
        let bytes_per_vec = self.dim * self.bits / 8;
        let last = self.n_vectors - 1;
        if idx != last {
            let src = last * bytes_per_vec;
            let dst = idx * bytes_per_vec;
            self.packed_codes.copy_within(src..src + bytes_per_vec, dst);
            self.norms[idx] = self.norms[last];
        }
        self.packed_codes.truncate(last * bytes_per_vec);
        self.norms.truncate(last);
        self.n_vectors -= 1;
        last
    }

    pub fn len(&self) -> usize {
        self.n_vectors
    }
    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn bits(&self) -> usize {
        self.bits
    }
}
