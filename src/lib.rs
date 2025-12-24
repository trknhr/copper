pub mod config;
pub mod generate;
pub mod loader;
pub mod model;
pub mod tokenizer;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use std::io::Write;

pub struct Runtime {
    pub device: Device,
    pub dtype: DType,
}

impl Runtime {
    pub fn cpu_f32() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
        }
    }
}

pub fn run(model_dir: &std::path::Path, prompt: &str, max_new_tokens: usize) -> Result<String> {
    let rt = Runtime::cpu_f32();

    let spec = config::load_model_spec(model_dir).context("load config.json")?;
    eprintln!("ModelSpec: {spec:?}");

    let tokenizer = tokenizer::load_tokenizer(model_dir).context("load tokenizer")?;
    let mut ids = tokenizer.encode(prompt).context("tokenize prompt")?;

    let weights = loader::load_gpt2_weights(model_dir, &rt).context("load weights")?;
    let model = model::Gpt2::new(spec, weights, &rt).context("build model")?;

    generate::greedy_generate_cached(&model, &tokenizer, &mut ids, max_new_tokens)
        .context("generate")?;

    tokenizer.decode(&ids).context("decode")
}

pub fn run_stream(
    model_dir: &std::path::Path,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<()> {
    let rt = Runtime::cpu_f32();

    let spec = config::load_model_spec(model_dir).context("load config.json")?;
    eprintln!("ModelSpec: {spec:?}");

    let tokenizer = tokenizer::load_tokenizer(model_dir).context("load tokenizer")?;
    let mut ids = tokenizer.encode(prompt).context("tokenize prompt")?;

    let weights = loader::load_gpt2_weights(model_dir, &rt).context("load weights")?;
    let model = model::Gpt2::new(spec, weights, &rt).context("build model")?;

    let mut out = std::io::stdout();
    write!(out, "{prompt}").context("write prompt")?;
    out.flush().context("flush prompt")?;

    generate::greedy_generate_cached_stream(&model, &tokenizer, &mut ids, max_new_tokens, &mut out)
        .context("generate stream")?;
    writeln!(out).context("write newline")?;
    Ok(())
}
