mod common;

use anyhow::{bail, Result};

#[test]
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
    let last_full = logits_full.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)?;
    let full: Vec<f32> = last_full.to_vec1()?;

    let mut cache = copper::model::cache::KvCache::new(model.n_layer());
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
