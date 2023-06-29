mod egraph;
mod extract;

pub use egraph::*;
pub use extract::*;

use indexmap::IndexMap;
use ordered_float::NotNan;

use anyhow::Context;

use std::io::Write;
use std::{path::PathBuf, str::FromStr};

pub type Cost = NotNan<f64>;
pub const INFINITY: Cost = unsafe { NotNan::new_unchecked(std::f64::INFINITY) };

pub type Id = usize;

fn main() {
    env_logger::init();

    let extractors: IndexMap<&str, Box<dyn Extractor>> = [
        ("bottom-up", extract::bottom_up::BottomUpExtractor.boxed()),
        #[cfg(feature = "ilp-cbc")]
        ("ilp-cbc", extract::ilp_cbc::CbcExtractor.boxed()),
    ]
    .into_iter()
    .collect();

    let mut args = pico_args::Arguments::from_env();

    let extractor_name: String = args
        .opt_value_from_str("--extractor")
        .unwrap()
        .unwrap_or_else(|| "bottom-up".into());
    if extractor_name == "print" {
        for name in extractors.keys() {
            println!("{}", name);
        }
        return;
    }

    let out_filename: PathBuf = args
        .opt_value_from_str("--out")
        .unwrap()
        .unwrap_or_else(|| "out.json".into());

    let filename: String = args.free_from_str().unwrap();

    let rest = args.finish();
    if !rest.is_empty() {
        panic!("Unknown arguments: {:?}", rest);
    }

    let mut out_file = std::fs::File::create(out_filename).unwrap();

    let egraph = std::fs::read_to_string(&filename)
        .unwrap_or_else(|e| panic!("Failed to read {filename}: {e}"))
        .parse::<SimpleEGraph>()
        .with_context(|| format!("Failed to parse {filename}"))
        .unwrap();

    let extractor = extractors
        .get(extractor_name.as_str())
        .with_context(|| format!("Unknown extractor: {extractor_name}"))
        .unwrap();

    let start_time = std::time::Instant::now();
    let result = extractor.extract(&egraph, &egraph.roots);

    let us = start_time.elapsed().as_micros();
    let tree = result.tree_cost(&egraph, &egraph.roots);
    let dag = result.dag_cost(&egraph, &egraph.roots);

    log::info!("{filename:40}\t{extractor_name:10}\t{tree:5}\t{dag:5}\t{us:5}");
    writeln!(
        out_file,
        r#"{{ 
    "name": "{filename}",
    "extractor": "{extractor_name}", 
    "tree": {tree}, 
    "dag": {dag}, 
    "micros": {us}
}}"#
    )
    .unwrap();
}
