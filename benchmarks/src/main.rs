//! CLI entry point: `rotorvec-bench --dataset sift-1m --bits 4 --k 10`.

#![allow(clippy::too_many_arguments)]

// Force blas-src to be linked so turbovec's `cblas_sgemm` calls resolve.
// Without this `use as _`, the rustc dependency tracker prunes blas-src
// from the final binary because we don't reference any of its types
// directly — only turbovec does, transitively.
#[cfg(any(target_os = "macos", target_os = "linux"))]
use blas_src as _;

mod datasets;
mod ground_truth;
mod output;
mod runner;

use anyhow::Result;
use clap::Parser;
use datasets::Dataset;
use runner::Method;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about = "Recall + throughput benchmarks for rotorvec")]
struct Cli {
    /// Dataset to benchmark.
    #[arg(long, default_value = "sift-1m")]
    dataset: String,

    /// Quantization bits per coordinate.
    #[arg(long, default_value_t = 4)]
    bits: usize,

    /// Top-k.
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Comma-separated method list. `all` = every supported method.
    #[arg(long, default_value = "all")]
    methods: String,

    /// Optional cap on training set size (handy for smoke tests).
    #[arg(long)]
    max_train: Option<usize>,

    /// Optional cap on number of queries.
    #[arg(long)]
    max_queries: Option<usize>,

    /// Where to write the markdown report. Defaults to
    /// `benchmarks/results/<dataset>_<bits>bit.md` relative to the workspace.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let dataset = parse_dataset(&cli.dataset)?;
    let methods = parse_methods(&cli.methods)?;

    let dataset_name = dataset.name();
    eprintln!("Loading {} ...", dataset_name);
    let mut loaded = datasets::load(dataset)?;

    // Cap sizes if requested. Both arrays are flat row-major.
    if let Some(cap) = cli.max_train {
        let cap = cap.min(loaded.n_train);
        loaded.train.truncate(cap * loaded.dim);
        loaded.n_train = cap;
        // The cached ground truth is no longer valid for a truncated set.
        loaded.ground_truth = None;
    }
    if let Some(cap) = cli.max_queries {
        let cap = cap.min(loaded.n_queries);
        loaded.queries.truncate(cap * loaded.dim);
        loaded.n_queries = cap;
        if let Some((gt, k_gt)) = loaded.ground_truth.as_mut() {
            gt.truncate(cap * *k_gt);
        }
    }

    // We index by inner product, so normalize. SIFT's official ground truth
    // is for unnormalized L2 — recompute on normalized vectors. Cache key
    // includes "norm" + sizes so we don't reuse stale results.
    eprintln!(
        "Normalizing {} train + {} query vectors (d={})...",
        loaded.n_train, loaded.n_queries, loaded.dim
    );
    datasets::normalize_inplace(&mut loaded.train, loaded.dim);
    datasets::normalize_inplace(&mut loaded.queries, loaded.dim);

    let cache_key = format!(
        "{}_norm_n{}_q{}",
        dataset_name, loaded.n_train, loaded.n_queries
    );
    let truth = ground_truth::compute_or_load(
        &cache_key,
        &loaded.train,
        loaded.n_train,
        &loaded.queries,
        loaded.n_queries,
        loaded.dim,
        cli.k,
    )?;

    eprintln!("Running {} methods at {}-bit ...", methods.len(), cli.bits);
    let mut results = Vec::new();
    for &m in &methods {
        eprintln!("  ... {}", m.label().trim());
        let r = runner::run(
            m,
            cli.bits,
            &loaded.train,
            loaded.n_train,
            &loaded.queries,
            loaded.n_queries,
            loaded.dim,
            cli.k,
            &truth.indices,
        )?;
        eprintln!(
            "      recall@{}={:.3}  build={:.2}s  qps={:.0}",
            r.k, r.recall_at_k, r.build_secs, r.qps
        );
        results.push(r);
    }

    let md = output::render_markdown(
        &dataset_name,
        loaded.n_train,
        loaded.n_queries,
        loaded.dim,
        &results,
    );
    println!("\n{md}");

    let out_path = cli.output.unwrap_or_else(|| {
        let mut p = PathBuf::from("benchmarks/results");
        let _ = fs::create_dir_all(&p);
        p.push(format!("{}_{}bit.md", dataset_name, cli.bits));
        p
    });
    fs::create_dir_all(
        out_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new(".")),
    )?;
    fs::write(&out_path, &md)?;
    eprintln!("Wrote {}", out_path.display());

    let json_path = out_path.with_extension("json");
    fs::write(&json_path, serde_json::to_string_pretty(&results)?)?;
    eprintln!("Wrote {}", json_path.display());

    Ok(())
}

fn parse_dataset(spec: &str) -> Result<Dataset> {
    match spec {
        "sift-1m" | "sift" | "sift-128" => Ok(Dataset::Sift1M),
        "glove-100" | "glove" => Ok(Dataset::GloVe100),
        other if other.starts_with("random") => {
            // Forms accepted:
            //   random          → n=100_000, dim=128
            //   random:N        → n=N, dim=128
            //   random:N:D      → n=N, dim=D
            let mut parts = other.split(':').skip(1);
            let n: usize = parts
                .next()
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(100_000);
            let dim: usize = parts.next().map(|s| s.parse()).transpose()?.unwrap_or(128);
            if dim % 8 != 0 {
                anyhow::bail!("random dataset dim must be a multiple of 8 (got {dim})");
            }
            Ok(Dataset::Random { n, dim })
        }
        other => anyhow::bail!("unknown dataset: {other}"),
    }
}

fn parse_methods(spec: &str) -> Result<Vec<Method>> {
    if spec == "all" {
        return Ok(Method::all().to_vec());
    }
    spec.split(',')
        .map(|s| match s.trim() {
            "turbovec" | "tv" => Ok(Method::TurboVec),
            "planar" | "planar2" | "rv-planar" => Ok(Method::RvPlanar2),
            "rotor" | "rotor3" | "rv-rotor" => Ok(Method::RvRotor3),
            "iso" | "iso4" | "rv-iso" => Ok(Method::RvIso4),
            other => anyhow::bail!("unknown method: {other}"),
        })
        .collect()
}
