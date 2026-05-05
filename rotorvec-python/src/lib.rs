#![allow(clippy::type_complexity, clippy::manual_checked_ops)]

//! Python bindings for rotorvec via PyO3.
//!
//! Exposes:
//! * `Rotation` — string-valued enum: `"planar2"`, `"rotor3"`, `"iso4"`.
//! * `RotorQuantIndex` — positional index.
//! * `IdMapIndex`      — id-addressed index.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use rotorvec_core::Rotation;

fn parse_rotation(s: &str) -> PyResult<Rotation> {
    match s.to_ascii_lowercase().as_str() {
        "planar2" | "planar" | "givens" => Ok(Rotation::Planar2),
        "rotor3" | "rotor" | "clifford" => Ok(Rotation::Rotor3),
        "iso4" | "iso" | "quaternion" => Ok(Rotation::Iso4),
        other => Err(PyValueError::new_err(format!(
            "unknown rotation '{other}': expected 'planar2', 'rotor3', or 'iso4'"
        ))),
    }
}

fn rotation_to_str(r: Rotation) -> &'static str {
    match r {
        Rotation::Planar2 => "planar2",
        Rotation::Rotor3 => "rotor3",
        Rotation::Iso4 => "iso4",
    }
}

#[pyclass]
struct RotorQuantIndex {
    inner: rotorvec_core::RotorQuantIndex,
}

#[pymethods]
impl RotorQuantIndex {
    /// Create a new index.
    ///
    /// Args:
    ///     dim: vector dimension. Must be a multiple of 8.
    ///     bits: bits per coordinate. Must be 2, 3, or 4.
    ///     rotation: 'planar2' (2D Givens), 'rotor3' (Cl(3,0), default),
    ///               or 'iso4' (quaternion).
    #[new]
    #[pyo3(signature = (dim, bits, rotation = "rotor3"))]
    fn new(dim: usize, bits: usize, rotation: &str) -> PyResult<Self> {
        let r = parse_rotation(rotation)?;
        Ok(Self {
            inner: rotorvec_core::RotorQuantIndex::with_rotation(dim, bits, r),
        })
    }

    /// Add `n = vectors.shape[0]` vectors. `vectors` must be a 2-D
    /// `float32` array of shape `(n, dim)`.
    fn add(&mut self, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = vectors.as_array();
        let slice = arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("vectors must be C-contiguous"))?;
        self.inner.add(slice);
        Ok(())
    }

    /// Top-`k` search. Returns `(scores, indices)` as `(nq, k)` arrays.
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>)> {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let slice = arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("queries must be C-contiguous"))?;
        let results = self.inner.search(slice, k);

        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .map_err(|e| PyValueError::new_err(format!("scores reshape: {e}")))?
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .map_err(|e| PyValueError::new_err(format!("indices reshape: {e}")))?
            .into_pyarray(py);

        Ok((scores, indices))
    }

    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| PyIOError::new_err(format!("{e}")))
    }

    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = rotorvec_core::RotorQuantIndex::load(path)
            .map_err(|e| PyIOError::new_err(format!("{e}")))?;
        Ok(Self { inner })
    }

    /// Eagerly populate caches (block-rotation matrices, Lloyd-Max
    /// centroids) so the first `search` call doesn't pay the init cost.
    fn prepare(&self) {
        self.inner.prepare();
    }

    /// Remove the vector at `idx` in O(1) by swap-with-last. Returns the
    /// previous index of the moved vector. Order is **not** preserved.
    fn swap_remove(&mut self, idx: usize) -> usize {
        self.inner.swap_remove(idx)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "RotorQuantIndex(dim={}, bits={}, rotation='{}', n={})",
            self.inner.dim(),
            self.inner.bits(),
            rotation_to_str(self.inner.rotation()),
            self.inner.len(),
        )
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn bits(&self) -> usize {
        self.inner.bits()
    }

    #[getter]
    fn rotation(&self) -> &'static str {
        rotation_to_str(self.inner.rotation())
    }
}

#[pyclass]
struct IdMapIndex {
    inner: rotorvec_core::IdMapIndex,
}

#[pymethods]
impl IdMapIndex {
    #[new]
    #[pyo3(signature = (dim, bits, rotation = "rotor3"))]
    fn new(dim: usize, bits: usize, rotation: &str) -> PyResult<Self> {
        let r = parse_rotation(rotation)?;
        Ok(Self {
            inner: rotorvec_core::IdMapIndex::with_rotation(dim, bits, r),
        })
    }

    /// Add vectors with stable external `uint64` ids.
    fn add_with_ids(
        &mut self,
        vectors: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<()> {
        let v = vectors.as_array();
        let v_slice = v
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("vectors must be C-contiguous"))?;
        let i = ids.as_array();
        let i_slice = i
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("ids must be C-contiguous"))?;
        self.inner.add_with_ids(v_slice, i_slice);
        Ok(())
    }

    fn remove(&mut self, id: u64) -> bool {
        self.inner.remove(id)
    }

    /// Top-`k` search returning external ids. Returns `(scores, ids)` as
    /// `(nq, k)` arrays.
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u64>>)> {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let slice = arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("queries must be C-contiguous"))?;
        let (scores, ids) = self.inner.search(slice, k);
        let effective_k = if nq == 0 { k } else { scores.len() / nq };

        let scores_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), scores)
            .map_err(|e| PyValueError::new_err(format!("scores reshape: {e}")))?
            .into_pyarray(py);
        let ids_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), ids)
            .map_err(|e| PyValueError::new_err(format!("ids reshape: {e}")))?
            .into_pyarray(py);
        Ok((scores_arr, ids_arr))
    }

    fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    fn prepare(&self) {
        self.inner.prepare();
    }

    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| PyIOError::new_err(format!("{e}")))
    }

    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = rotorvec_core::IdMapIndex::load(path)
            .map_err(|e| PyIOError::new_err(format!("{e}")))?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    fn __repr__(&self) -> String {
        format!(
            "IdMapIndex(dim={}, bits={}, rotation='{}', n={})",
            self.inner.dim(),
            self.inner.bits(),
            rotation_to_str(self.inner.rotation()),
            self.inner.len(),
        )
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn bits(&self) -> usize {
        self.inner.bits()
    }

    #[getter]
    fn rotation(&self) -> &'static str {
        rotation_to_str(self.inner.rotation())
    }
}

#[pymodule]
fn _rotorvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RotorQuantIndex>()?;
    m.add_class::<IdMapIndex>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
