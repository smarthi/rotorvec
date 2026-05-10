//! Run a single (method, dataset, bits) configuration end to end:
//! build the index, search the queries, measure recall against ground
//! truth, measure throughput.

use anyhow::Result;
use rotorvec::{Rotation, RotorQuantIndex};
use std::time::Instant;
use turbovec::TurboQuantIndex;

#[derive(Debug, Clone, Copy)]
pub enum Method {
    TurboVec,
    RvPlanar2,
    RvRotor3,
    RvIso4,
}

impl Method {
    pub fn label(&self) -> &'static str {
        match self {
            Method::TurboVec => "turbovec",
            Method::RvPlanar2 => "rotorvec planar2",
            Method::RvRotor3 => "rotorvec rotor3 ",
            Method::RvIso4 => "rotorvec iso4   ",
        }
    }
    pub fn all() -> &'static [Method] {
        &[
            Method::TurboVec,
            Method::RvPlanar2,
            Method::RvRotor3,
            Method::RvIso4,
        ]
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Result1 {
    pub method: String,
    pub bits: usize,
    pub build_secs: f64,
    pub search_secs: f64,
    pub queries: usize,
    pub qps: f64,
    pub recall_at_k: f64,
    pub k: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    method: Method,
    bits: usize,
    train: &[f32],
    _n_train: usize,
    queries: &[f32],
    n_queries: usize,
    dim: usize,
    k: usize,
    truth: &[i32],
) -> Result<Result1> {
    // ---- build ----
    let t0 = Instant::now();
    let approx = match method {
        Method::TurboVec => {
            let mut idx = TurboQuantIndex::new(dim, bits);
            idx.add(train);
            idx.prepare();
            let build = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let res = idx.search(queries, k);
            let search = t1.elapsed().as_secs_f64();
            return Ok(finish(
                method,
                bits,
                build,
                search,
                n_queries,
                k,
                &res.indices,
                truth,
            ));
        }
        Method::RvPlanar2 => build_rv(Rotation::Planar2, train, dim, bits),
        Method::RvRotor3 => build_rv(Rotation::Rotor3, train, dim, bits),
        Method::RvIso4 => build_rv(Rotation::Iso4, train, dim, bits),
    };

    let (idx, build) = approx;
    let t1 = Instant::now();
    let res = idx.search(queries, k);
    let search = t1.elapsed().as_secs_f64();
    Ok(finish(
        method,
        bits,
        build,
        search,
        n_queries,
        k,
        &res.indices,
        truth,
    ))
}

fn build_rv(rotation: Rotation, train: &[f32], dim: usize, bits: usize) -> (RotorQuantIndex, f64) {
    let t0 = Instant::now();
    let mut idx = RotorQuantIndex::with_rotation(dim, bits, rotation);
    idx.add(train);
    idx.prepare();
    (idx, t0.elapsed().as_secs_f64())
}

fn finish(
    method: Method,
    bits: usize,
    build: f64,
    search: f64,
    n_queries: usize,
    k: usize,
    approx: &[i64],
    truth: &[i32],
) -> Result1 {
    let recall = crate::ground_truth::recall_at_k(approx, truth, k);
    Result1 {
        method: method.label().trim().to_string(),
        bits,
        build_secs: build,
        search_secs: search,
        queries: n_queries,
        qps: n_queries as f64 / search,
        recall_at_k: recall,
        k,
    }
}
