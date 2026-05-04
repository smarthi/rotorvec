//! Stable external IDs on top of [`RotorQuantIndex`].
//!
//! [`RotorQuantIndex`] stores vectors positionally: `swap_remove` invalidates
//! external references because the previously-last vector moves into the
//! deleted slot. `IdMapIndex` wraps it with a bidirectional `id ↔ slot`
//! mapping so callers can identify vectors by a stable `u64` ID.
//!
//! Roughly analogous to FAISS's `IndexIDMap2`. The wrapper delegates all
//! vector storage and search to the inner index and only owns the ID table.

use std::collections::HashMap;
use std::path::Path;

use crate::io;
use crate::RotorQuantIndex;

pub struct IdMapIndex {
    inner: RotorQuantIndex,
    slot_to_id: Vec<u64>,
    id_to_slot: HashMap<u64, usize>,
}

impl IdMapIndex {
    pub fn new(dim: usize, bits: usize) -> Self {
        Self {
            inner: RotorQuantIndex::new(dim, bits),
            slot_to_id: Vec::new(),
            id_to_slot: HashMap::new(),
        }
    }

    pub fn with_seed(dim: usize, bits: usize, seed: u64) -> Self {
        Self {
            inner: RotorQuantIndex::with_seed(dim, bits, seed),
            slot_to_id: Vec::new(),
            id_to_slot: HashMap::new(),
        }
    }

    /// Add `n = vectors.len() / dim` vectors with the given external ids.
    /// Panics if `ids.len() != n`, if any id is already present, or if `ids`
    /// contains duplicates within this call.
    pub fn add_with_ids(&mut self, vectors: &[f32], ids: &[u64]) {
        let dim = self.inner.dim();
        let n = vectors.len() / dim;
        assert_eq!(vectors.len(), n * dim, "vector buffer not a multiple of dim");
        assert_eq!(ids.len(), n, "expected {n} ids, got {}", ids.len());

        self.id_to_slot.reserve(n);
        self.slot_to_id.reserve(n);

        let base_slot = self.inner.len();
        for (i, &id) in ids.iter().enumerate() {
            let slot = base_slot + i;
            if self.id_to_slot.insert(id, slot).is_some() {
                panic!("id {id} already present in index");
            }
        }
        self.slot_to_id.extend_from_slice(ids);
        self.inner.add(vectors);
    }

    /// Remove the vector with the given id. O(1).
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(slot) = self.id_to_slot.remove(&id) else {
            return false;
        };
        let last = self.slot_to_id.len() - 1;
        let moved_from = self.inner.swap_remove(slot);
        debug_assert_eq!(moved_from, last);
        if slot != last {
            let moved_id = self.slot_to_id[last];
            self.slot_to_id[slot] = moved_id;
            self.id_to_slot.insert(moved_id, slot);
        }
        self.slot_to_id.pop();
        true
    }

    /// Top-`k` search returning external ids. Layout matches
    /// [`crate::SearchResults`]: row `qi` occupies indices
    /// `qi * k .. (qi + 1) * k`.
    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        let res = self.inner.search(queries, k);
        let mut ids = Vec::with_capacity(res.indices.len());
        for &slot in &res.indices {
            // Sentinel `-1` (or `usize::MAX` cast) appears only when the
            // index has fewer than `k` vectors and the result is padded.
            // Map those to `u64::MAX` so callers can detect them.
            if slot < 0 {
                ids.push(u64::MAX);
            } else {
                ids.push(self.slot_to_id[slot as usize]);
            }
        }
        (res.scores, ids)
    }

    pub fn contains(&self, id: u64) -> bool {
        self.id_to_slot.contains_key(&id)
    }

    pub fn len(&self) -> usize {
        self.slot_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slot_to_id.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    pub fn bits(&self) -> usize {
        self.inner.bits()
    }

    pub fn prepare(&self) {
        self.inner.prepare();
    }

    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        io::write_id_map(
            path,
            self.inner.bits(),
            self.inner.dim(),
            self.inner.len(),
            self.inner.seed(),
            self.inner.packed_codes(),
            self.inner.norms(),
            &self.slot_to_id,
        )
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let (bits, dim, n_vectors, seed, packed_codes, norms, slot_to_id) =
            io::load_id_map(path)?;
        let inner =
            RotorQuantIndex::from_parts(dim, bits, n_vectors, seed, packed_codes, norms);
        let id_to_slot: HashMap<u64, usize> =
            slot_to_id.iter().enumerate().map(|(s, &id)| (id, s)).collect();
        if id_to_slot.len() != slot_to_id.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "duplicate ids in .rvim file",
            ));
        }
        Ok(Self { inner, slot_to_id, id_to_slot })
    }
}
