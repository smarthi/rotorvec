//! Pretty-print benchmark results as a markdown table.

use crate::runner::Result1;

pub fn render_markdown(
    dataset: &str,
    n_train: usize,
    n_queries: usize,
    dim: usize,
    results: &[Result1],
) -> String {
    let k = results.first().map(|r| r.k).unwrap_or(10);
    let mut out = String::new();
    out.push_str(&format!(
        "## {dataset} (n={n_train}, queries={n_queries}, d={dim})\n\n"
    ));
    out.push_str(&format!(
        "| Method | Bits | Recall@{k} | Build (s) | Search (s) | QPS |\n"
    ));
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for r in results {
        out.push_str(&format!(
            "| {} | {} | {:.3} | {:.2} | {:.2} | {:.0} |\n",
            r.method, r.bits, r.recall_at_k, r.build_secs, r.search_secs, r.qps,
        ));
    }
    out
}
