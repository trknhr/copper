mod common;

use anyhow::{bail, Result};

#[test]
#[ignore = "requires local GPT-2 model and HF fixture"]
fn prefill_cached_matches_full_forward() -> Result<()> {
    let Some(model_dir) = common::maybe_model_dir() else {
        eprintln!("SKIP: set COPPER_MODEL_DIR to a local GPT-2 directory to run this test");
        return Ok(());
    };

    let fx_path = common::fixture_path();
    if !fx_path.exists() {
        eprintln!(
            "SKIP: missing fixture at {fx_path:?}. Generate via scripts/export_hf_fixtures.py"
        );
        return Ok(());
    }
    let fx = common::load_fixture(&fx_path)?;

    let rt = copper::Runtime::cpu_f32();
    let spec = copper::config::load_model_spec(&model_dir)?;
    let weights = copper::loader::load_gpt2_weights(&model_dir, &rt)?;
    let model = copper::model::Gpt2::new(spec, weights, &rt)?;

    let logits_full = model.forward(&fx.input_ids)?;
    let seq_len = fx.input_ids.len();
    let last_full = logits_full
        .narrow(1, seq_len - 1, 1)?
        .squeeze(1)?
        .squeeze(0)?;
    let full: Vec<f32> = last_full.to_vec1()?;

    let mut cache = model.new_kv_cache();
    let logits_cached = model.forward_cached(&fx.input_ids, &mut cache)?;
    let last_cached = logits_cached
        .narrow(1, seq_len - 1, 1)?
        .squeeze(1)?
        .squeeze(0)?;
    let cached: Vec<f32> = last_cached.to_vec1()?;

    if full.len() != cached.len() {
        bail!(
            "logits length mismatch: full {} cached {}",
            full.len(),
            cached.len()
        );
    }
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (f, c) in full.iter().zip(cached.iter()) {
        let abs = (f - c).abs();
        let rel = abs / c.abs().max(1e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        if abs > fx.atol && rel > fx.rtol {
            bail!(
                "logits mismatch: abs {abs} rel {rel} (atol {} rtol {})",
                fx.atol,
                fx.rtol
            );
        }
    }
    eprintln!("max_abs={max_abs} max_rel={max_rel}");
    Ok(())
}

#[test]
#[ignore = "requires local GPT-2 model and HF fixture"]
fn multi_step_decode_cached_matches_full_forward() -> Result<()> {
    let Some(model_dir) = common::maybe_model_dir() else {
        eprintln!("SKIP: set COPPER_MODEL_DIR to a local GPT-2 directory to run this test");
        return Ok(());
    };
    let fx_path = common::fixture_path();
    if !fx_path.exists() {
        eprintln!(
            "SKIP: missing fixture at {fx_path:?}. Generate via scripts/export_hf_fixtures.py"
        );
        return Ok(());
    }
    let fx = common::load_fixture(&fx_path)?;

    let rt = copper::Runtime::cpu_f32();
    let spec = copper::config::load_model_spec(&model_dir)?;
    let weights = copper::loader::load_gpt2_weights(&model_dir, &rt)?;
    let model = copper::model::Gpt2::new(spec, weights, &rt)?;

    let mut cache = model.new_kv_cache();
    let mut ids = fx.input_ids.clone();
    let prefill = model.forward_cached(&ids, &mut cache)?;
    let prefill_last = prefill
        .narrow(1, ids.len() - 1, 1)?
        .squeeze(1)?
        .squeeze(0)?
        .to_vec1::<f32>()?;
    let mut next_id = argmax(&prefill_last);
    ids.push(next_id);

    for _ in 0..2 {
        let decode = model.forward_cached(&[next_id], &mut cache)?;
        let decode_last = decode.squeeze(1)?.squeeze(0)?.to_vec1::<f32>()?;
        next_id = argmax(&decode_last);
        ids.push(next_id);
    }

    let full = model.forward(&ids)?;
    let full_last = full.narrow(1, ids.len() - 1, 1)?.squeeze(1)?.squeeze(0)?;
    let cached = model.forward_cached(&[next_id], &mut cache)?;
    let cached_last = cached.squeeze(1)?.squeeze(0)?;

    let v_full: Vec<f32> = full_last.to_vec1()?;
    let v_cached: Vec<f32> = cached_last.to_vec1()?;
    if v_full.len() != v_cached.len() {
        bail!("length mismatch: {} vs {}", v_full.len(), v_cached.len());
    }
    for (f, c) in v_full.iter().zip(v_cached.iter()) {
        let abs = (f - c).abs();
        let rel = abs / c.abs().max(1e-6);
        if abs > fx.atol && rel > fx.rtol {
            bail!(
                "decode mismatch: abs {abs} rel {rel} (atol {} rtol {})",
                fx.atol,
                fx.rtol
            );
        }
    }
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    let (mut best_idx, mut best_val) = (0usize, f32::NEG_INFINITY);
    for (idx, val) in v.iter().enumerate() {
        if *val > best_val {
            best_val = *val;
            best_idx = idx;
        }
    }
    best_idx as u32
}
