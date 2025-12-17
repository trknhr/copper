use std::path::Path;

use anyhow::{bail, Context, Result};
use tokenizers::Tokenizer;

pub struct Gpt2Tokenizer {
    inner: Tokenizer,
}

pub fn load_tokenizer(model_dir: &Path) -> Result<Gpt2Tokenizer> {
    let tok_json = model_dir.join("tokenizer.json");
    if !tok_json.exists() {
        bail!(
            "missing tokenizer.json at {tok_json:?} (download GPT-2 tokenizer files into --model-dir)"
        );
    }
    let inner = Tokenizer::from_file(&tok_json)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("load {tok_json:?}"))?;
    Ok(Gpt2Tokenizer { inner })
}

impl Gpt2Tokenizer {
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}
