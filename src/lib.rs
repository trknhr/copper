pub mod config;
pub mod generate;
pub mod loader;
pub mod model;
pub mod tokenizer;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use std::io::{BufRead, Write};
use std::path::Path;
use tracing::debug;

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

fn init(model_dir: &Path) -> Result<(Runtime, tokenizer::Gpt2Tokenizer, model::Gpt2)> {
    let rt = Runtime::cpu_f32();
    let spec = config::load_model_spec(model_dir).context("load config.json")?;
    debug!(?spec, "loaded model spec");
    let tokenizer = tokenizer::load_tokenizer(model_dir).context("load tokenizer")?;
    let weights = loader::load_gpt2_weights(model_dir, &rt).context("load weights")?;
    let model = model::Gpt2::new(spec, weights, &rt).context("build model")?;
    Ok((rt, tokenizer, model))
}

pub fn run(model_dir: &Path, prompt: &str, max_new_tokens: usize) -> Result<String> {
    let (_rt, tokenizer, model) = init(model_dir)?;
    let mut ids = tokenizer.encode(prompt).context("tokenize prompt")?;

    generate::greedy_generate_cached(&model, &tokenizer, &mut ids, max_new_tokens)
        .context("generate")?;

    tokenizer.decode(&ids).context("decode")
}

pub fn run_stream(model_dir: &Path, prompt: &str, max_new_tokens: usize) -> Result<()> {
    let (_rt, tokenizer, model) = init(model_dir)?;
    let mut ids = tokenizer.encode(prompt).context("tokenize prompt")?;

    let mut out = std::io::stdout();
    write!(out, "{prompt}").context("write prompt")?;
    out.flush().context("flush prompt")?;

    generate::greedy_generate_cached_stream(&model, &tokenizer, &mut ids, max_new_tokens, &mut out)
        .context("generate stream")?;
    writeln!(out).context("write newline")?;
    Ok(())
}

pub fn run_chat(model_dir: &Path, max_new_tokens: usize, stream: bool) -> Result<()> {
    let (_rt, tokenizer, model) = init(model_dir)?;
    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();
    let mut out = std::io::stdout();
    let mut line = String::new();
    let mut history_ids: Vec<u32> = Vec::new();
    let mut cache = model.new_kv_cache();

    writeln!(out, "Chat mode: type `exit` or `quit` to stop").context("write chat header")?;
    writeln!(
        out,
        "Context is kept across turns. If context overflows, old turns are truncated."
    )
    .context("write chat context note")?;

    loop {
        write!(out, "> ").context("write prompt marker")?;
        out.flush().context("flush prompt marker")?;

        line.clear();
        let n = stdin.read_line(&mut line).context("read stdin line")?;
        if n == 0 {
            break;
        }
        let prompt = line.trim_end();
        if prompt.is_empty() {
            continue;
        }
        if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
            break;
        }

        let turn = format!("User: {prompt}\nAssistant:");
        let turn_ids = tokenizer.encode(&turn).context("tokenize chat turn")?;
        if turn_ids.is_empty() {
            continue;
        }

        let need = turn_ids.len() + max_new_tokens;
        let n_ctx = model.n_ctx();
        if need > n_ctx {
            writeln!(
                out,
                "[warn] prompt too long for context window ({} > {}), skipping turn",
                need, n_ctx
            )
            .context("write too-long warning")?;
            continue;
        }

        if history_ids.len() + need > n_ctx {
            let keep = n_ctx - need;
            if history_ids.len() > keep {
                let start = history_ids.len() - keep;
                history_ids = history_ids[start..].to_vec();
                cache.clear();
                if !history_ids.is_empty() {
                    model
                        .forward_cached(&history_ids, &mut cache)
                        .context("rebuild cache from trimmed history")?;
                }
                writeln!(out, "[info] trimmed older context to fit window")
                    .context("write trim info")?;
            }
        }

        history_ids.extend_from_slice(&turn_ids);

        if stream {
            let generated = generate::greedy_generate_cached_continue_stream(
                &model,
                &tokenizer,
                &turn_ids,
                max_new_tokens,
                &mut cache,
                &mut out,
            )
            .context("generate stream")?;
            history_ids.extend_from_slice(&generated);
            writeln!(out).context("write newline")?;
        } else {
            let generated = generate::greedy_generate_cached_continue(
                &model,
                &turn_ids,
                max_new_tokens,
                &mut cache,
            )
            .context("generate buffered")?;
            history_ids.extend_from_slice(&generated);
            let reply = tokenizer.decode(&generated).context("decode chat reply")?;
            writeln!(out, "{reply}").context("write chat reply")?;
        }

        let nl_ids = tokenizer
            .encode("\n")
            .context("tokenize newline separator")?;
        if !nl_ids.is_empty() {
            history_ids.extend_from_slice(&nl_ids);
            model
                .forward_cached(&nl_ids, &mut cache)
                .context("advance cache with newline separator")?;
        }
    }

    Ok(())
}
