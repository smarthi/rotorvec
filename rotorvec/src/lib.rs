// Stylistic lints that don't reflect real bugs in this codebase.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_range_loop
)]

//! RotorQuant vector index — block-diagonal Clifford-rotor quantization.
//!
//! Three rotation variants:
//! * [`Rotation::Planar2`] — 2D Givens (PlanarQuant). Cheapest, often best
//!   on KV-cache PPL per the RotorQuant paper.
//! * [`Rotation::Rotor3`] — Cl(3,0) rotor sandwich (RotorQuant). The
//!   default; balances algebraic richness and cost.
//! * [`Rotation::Iso4`] — quaternion left-iso (IsoQuant). Most decorrelation
//!   per group; slightly more compute than Planar2.
//!
//! ```no_run
//! use rotorvec::{RotorQuantIndex, Rotation};
//!
//! // Default (Rotor3, Cl(3,0))
//! let mut idx = RotorQuantIndex::new(1536, 4);
//!
//! // Or pick a variant explicitly
//! let mut idx2 = RotorQuantIndex::with_rotation(1536, 4, Rotation::Planar2);
//! # let _ = (idx, idx2);
//! ```

pub mod codebook;
pub mod encode;
pub mod id_map;
pub mod io;
pub mod pack;
pub mod rotation;
pub mod rotor;
pub mod search;

pub use id_map::IdMapIndex;
pub use rotor::Rotation;

use std::path::Path;
use std::sync::OnceLock;

/// Default rotor seed. Two indices with the same `(dim, rotation, seed)`
/// produce bit-identical block matrices.
pub const DEFAULT_ROTOR_SEED: u64 = 0x0707_04EC_u64;

/// Default rotation kind for [`RotorQuantIndex::new`]: Cl(3,0) rotors.
pub const DEFAULT_ROTATION: Rotation = Rotation::Rotor3;

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
    rotation: Rotation,
    n_vectors: usize,
    seed: u64,
    packed_codes: Vec<u8>,
    norms: Vec<f32>,

    block_matrices: OnceLock<(Vec<f32>, usize)>, // (matrices, padded_dim)
    centroids: OnceLock<Vec<f32>>,
}

impl RotorQuantIndex {
    /// Empty index with default [`Rotation::Rotor3`] and default seed.
    pub fn new(dim: usize, bits: usize) -> Self {
        Self::with_options(dim, bits, DEFAULT_ROTATION, DEFAULT_ROTOR_SEED)
    }

    /// Empty index with explicit rotation kind, default seed.
    pub fn with_rotation(dim: usize, bits: usize, rotation: Rotation) -> Self {
        Self::with_options(dim, bits, rotation, DEFAULT_ROTOR_SEED)
    }

    /// Empty index with explicit rotation kind and seed.
    pub fn with_options(dim: usize, bits: usize, rotation: Rotation, seed: u64) -> Self {
        assert!((2..=4).contains(&bits), "bits must be 2, 3, or 4");
        assert!(dim % 8 == 0, "dim must be a multiple of 8");
        Self {
            dim,
            bits,
            rotation,
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
        assert_eq!(
            vectors.len(),
            n * self.dim,
            "vectors length not a multiple of dim"
        );
        if n == 0 {
            return;
        }

        let (matrices, padded_dim) = self.ensure_block_matrices().clone();
        let (boundaries, _) = codebook::codebook(self.bits, self.dim);
        let (packed, norms) = encode::encode(
            vectors,
            n,
            self.dim,
            &matrices,
            self.rotation.block_size(),
            padded_dim,
            &boundaries,
            self.bits,
        );

        self.packed_codes.extend_from_slice(&packed);
        self.norms.extend_from_slice(&norms);
        self.n_vectors += n;
    }

    pub fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);

        let (matrices, padded_dim) = self.ensure_block_matrices();
        let centroids = self.ensure_centroids();
        let k = k.min(self.n_vectors).max(1);

        let (scores, indices) = if self.n_vectors == 0 {
            (vec![f32::NEG_INFINITY; nq * k], vec![-1i64; nq * k])
        } else {
            search::search(
                queries,
                nq,
                self.dim,
                matrices,
                self.rotation.block_size(),
                *padded_dim,
                &self.packed_codes,
                centroids,
                &self.norms,
                self.bits,
                self.n_vectors,
                k,
            )
        };

        SearchResults {
            scores,
            indices,
            nq,
            k,
        }
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
            self.rotation,
            self.seed,
            &self.packed_codes,
            &self.norms,
        )
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let (bits, dim, n_vectors, rotation, seed, packed_codes, norms) = io::load(path)?;
        Ok(Self::from_parts(
            dim,
            bits,
            rotation,
            n_vectors,
            seed,
            packed_codes,
            norms,
        ))
    }

    pub(crate) fn from_parts(
        dim: usize,
        bits: usize,
        rotation: Rotation,
        n_vectors: usize,
        seed: u64,
        packed_codes: Vec<u8>,
        norms: Vec<f32>,
    ) -> Self {
        Self {
            dim,
            bits,
            rotation,
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
    pub(crate) fn rotation_kind(&self) -> Rotation {
        self.rotation
    }

    fn ensure_block_matrices(&self) -> &(Vec<f32>, usize) {
        self.block_matrices.get_or_init(|| {
            let (mats, _, padded) =
                rotor::precompute_block_matrices(self.dim, self.rotation, self.seed);
            (mats, padded)
        })
    }

    fn ensure_centroids(&self) -> &Vec<f32> {
        self.centroids.get_or_init(|| {
            let (_, c) = codebook::codebook(self.bits, self.dim);
            c
        })
    }

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
    pub fn rotation(&self) -> Rotation {
        self.rotation
    }
}
