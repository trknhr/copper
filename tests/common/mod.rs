#![allow(dead_code)]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct HfFixture {
    pub prompt: String,
    pub input_ids: Vec<u32>,
    pub last_token_logits: Option<Vec<f32>>,

    #[serde(default = "default_atol")]
    pub atol: f32,
    #[serde(default = "default_rtol")]
    pub rtol: f32,
}

fn default_atol() -> f32 {
    1e-3
}
fn default_rtol() -> f32 {
    1e-3
}

pub fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hf_gpt2.json")
}

pub fn load_fixture(path: &Path) -> Result<HfFixture> {
    let bytes = std::fs::read(path).with_context(|| format!("read fixture {path:?}"))?;
    serde_json::from_slice(&bytes).context("parse fixture json")
}

pub fn maybe_model_dir() -> Option<PathBuf> {
    std::env::var_os("COPPER_MODEL_DIR").map(PathBuf::from)
}
