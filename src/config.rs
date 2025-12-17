use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSpec {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,

    #[serde(default)]
    pub n_positions: Option<usize>,
    #[serde(default)]
    pub n_inner: Option<usize>,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,
}

fn default_layer_norm_epsilon() -> f64 {
    1e-5
}

pub fn load_model_spec(model_dir: &Path) -> Result<ModelSpec> {
    let path = model_dir.join("config.json");
    let bytes = std::fs::read(&path).with_context(|| format!("read {path:?}"))?;
    let spec: ModelSpec =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {path:?}"))?;
    Ok(spec)
}

impl ModelSpec {
    pub fn n_positions(&self) -> usize {
        self.n_positions.unwrap_or(self.n_ctx)
    }

    pub fn n_inner(&self) -> usize {
        self.n_inner.unwrap_or(4 * self.n_embd)
    }
}
