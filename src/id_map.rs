//! Stable external IDs on top of [`RotorQuantIndex`].

use std::collections::HashMap;
use std::path::Path;

use crate::io;
use crate::rotor::Rotation;
use crate::{RotorQuantIndex, DEFAULT_ROTATION, DEFAULT_ROTOR_SEED};

pub struct IdMapIndex {
    inner: RotorQuantIndex,
    slot_to_id: Vec<u64>,
    id_to_slot: HashMap<u64, usize>,
}

impl IdMapIndex {
    pub fn new(dim: usize, bits: usize) -> Self {
        Self::with_options(dim, bits, DEFAULT_ROTATION, DEFAULT_ROTOR_SEED)
    }

    pub fn with_rotation(dim: usize, bits: usize, rotation: Rotation) -> Self {
        Self::with_options(dim, bits, rotation, DEFAULT_ROTOR_SEED)
    }

    pub fn with_options(dim: usize, bits: usize, rotation: Rotation, seed: u64) -> Self {
        Self {
            inner: RotorQuantIndex::with_options(dim, bits, rotation, seed),
            slot_to_id: Vec::new(),
            id_to_slot: HashMap::new(),
        }
    }

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

    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        let res = self.inner.search(queries, k);
        let mut ids = Vec::with_capacity(res.indices.len());
        for &slot in &res.indices {
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

    pub fn rotation(&self) -> Rotation {
        self.inner.rotation()
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
            self.inner.rotation_kind(),
            self.inner.seed(),
            self.inner.packed_codes(),
            self.inner.norms(),
            &self.slot_to_id,
        )
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let (bits, dim, n_vectors, rotation, seed, packed_codes, norms, slot_to_id) =
            io::load_id_map(path)?;
        let inner = RotorQuantIndex::from_parts(
            dim, bits, rotation, n_vectors, seed, packed_codes, norms,
        );
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
