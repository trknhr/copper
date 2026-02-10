use anyhow::{Context, Result};

use std::io::Write;

use crate::model::cache::KvCache;
use crate::model::Gpt2;
use crate::tokenizer::Gpt2Tokenizer;

pub fn greedy_generate(
    model: &Gpt2,
    _tokenizer: &Gpt2Tokenizer,
    input_ids: &mut Vec<u32>,
    max_new_tokens: usize,
) -> Result<()> {
    for _ in 0..max_new_tokens {
        let logits = model.forward(input_ids).context("forward")?;
        let seq_len = input_ids.len();
        let last = logits
            .narrow(1, seq_len - 1, 1)
            .context("slice last token")?
            .squeeze(1)
            .context("squeeze seq dim")?
            .squeeze(0)
            .context("squeeze batch dim")?;

        let v: Vec<f32> = last.to_vec1().context("logits to_vec1")?;
        input_ids.push(argmax(&v));
    }
    Ok(())
}

pub fn greedy_generate_cached(
    model: &Gpt2,
    _tokenizer: &Gpt2Tokenizer,
    input_ids: &mut Vec<u32>,
    max_new_tokens: usize,
) -> Result<()> {
    if max_new_tokens == 0 {
        return Ok(());
    }

    let mut cache = model.new_kv_cache();

    let logits = model
        .forward_cached(input_ids, &mut cache)
        .context("prefill forward_cached")?;
    let seq_len = input_ids.len();
    let last = logits
        .narrow(1, seq_len - 1, 1)
        .context("slice last token")?
        .squeeze(1)
        .context("squeeze seq dim")?
        .squeeze(0)
        .context("squeeze batch dim")?;
    let mut v: Vec<f32> = last.to_vec1().context("logits to_vec1")?;
    let mut next_id = argmax(&v);
    input_ids.push(next_id);

    for _ in 1..max_new_tokens {
        let logits = model
            .forward_cached(&[next_id], &mut cache)
            .context("decode forward_cached")?;
        let last = logits
            .squeeze(1)
            .context("squeeze seq dim")?
            .squeeze(0)
            .context("squeeze batch dim")?;
        v = last.to_vec1().context("logits to_vec1")?;
        next_id = argmax(&v);
        input_ids.push(next_id);
    }

    Ok(())
}

pub fn greedy_generate_cached_stream(
    model: &Gpt2,
    tokenizer: &Gpt2Tokenizer,
    input_ids: &mut Vec<u32>,
    max_new_tokens: usize,
    out: &mut dyn Write,
) -> Result<()> {
    if max_new_tokens == 0 {
        return Ok(());
    }

    let mut cache = model.new_kv_cache();

    let logits = model
        .forward_cached(input_ids, &mut cache)
        .context("prefill forward_cached")?;
    let seq_len = input_ids.len();
    let last = logits
        .narrow(1, seq_len - 1, 1)
        .context("slice last token")?
        .squeeze(1)
        .context("squeeze seq dim")?
        .squeeze(0)
        .context("squeeze batch dim")?;
    let mut v: Vec<f32> = last.to_vec1().context("logits to_vec1")?;
    let mut next_id = argmax(&v);
    input_ids.push(next_id);
    let token = tokenizer.decode(&[next_id]).context("decode token")?;
    write!(out, "{token}").context("write token")?;
    out.flush().context("flush token")?;

    for _ in 1..max_new_tokens {
        let logits = model
            .forward_cached(&[next_id], &mut cache)
            .context("decode forward_cached")?;
        let last = logits
            .squeeze(1)
            .context("squeeze seq dim")?
            .squeeze(0)
            .context("squeeze batch dim")?;
        v = last.to_vec1().context("logits to_vec1")?;
        next_id = argmax(&v);
        input_ids.push(next_id);
        let token = tokenizer.decode(&[next_id]).context("decode token")?;
        write!(out, "{token}").context("write token")?;
        out.flush().context("flush token")?;
    }

    Ok(())
}

pub fn greedy_generate_cached_continue(
    model: &Gpt2,
    new_input_ids: &[u32],
    max_new_tokens: usize,
    cache: &mut KvCache,
) -> Result<Vec<u32>> {
    if max_new_tokens == 0 {
        return Ok(vec![]);
    }
    if new_input_ids.is_empty() {
        return Ok(vec![]);
    }

    let logits = model
        .forward_cached(new_input_ids, cache)
        .context("prefill forward_cached")?;
    let seq_len = new_input_ids.len();
    let last = logits
        .narrow(1, seq_len - 1, 1)
        .context("slice last token")?
        .squeeze(1)
        .context("squeeze seq dim")?
        .squeeze(0)
        .context("squeeze batch dim")?;
    let mut v: Vec<f32> = last.to_vec1().context("logits to_vec1")?;
    let mut next_id = argmax(&v);
    let mut out = vec![next_id];

    for _ in 1..max_new_tokens {
        let logits = model
            .forward_cached(&[next_id], cache)
            .context("decode forward_cached")?;
        let last = logits
            .squeeze(1)
            .context("squeeze seq dim")?
            .squeeze(0)
            .context("squeeze batch dim")?;
        v = last.to_vec1().context("logits to_vec1")?;
        next_id = argmax(&v);
        out.push(next_id);
    }

    Ok(out)
}

pub fn greedy_generate_cached_continue_stream(
    model: &Gpt2,
    tokenizer: &Gpt2Tokenizer,
    new_input_ids: &[u32],
    max_new_tokens: usize,
    cache: &mut KvCache,
    out: &mut dyn Write,
) -> Result<Vec<u32>> {
    let generated = greedy_generate_cached_continue(model, new_input_ids, max_new_tokens, cache)?;
    for token_id in &generated {
        let token = tokenizer.decode(&[*token_id]).context("decode token")?;
        write!(out, "{token}").context("write token")?;
    }
    out.flush().context("flush token stream")?;
    Ok(generated)
}

fn argmax(v: &[f32]) -> u32 {
    let (mut best_id, mut best_val) = (0u32, f32::NEG_INFINITY);
    for (i, &val) in v.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_id = i as u32;
        }
    }
    best_id
}
