use anyhow::{Context, Result};

use crate::model::Gpt2;
use crate::tokenizer::Gpt2Tokenizer;

pub fn greedy_generate(
    model: &Gpt2,
    tokenizer: &Gpt2Tokenizer,
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
        let (mut best_id, mut best_val) = (0u32, f32::NEG_INFINITY);
        for (i, &val) in v.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_id = i as u32;
            }
        }
        input_ids.push(best_id);
    }
    let _ = tokenizer; // keep signature stable for future streaming hooks
    Ok(())
}
