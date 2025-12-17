mod common;

use anyhow::{bail, Result};

#[test]
fn prefill_last_token_logits_close_to_hf() -> Result<()> {
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
    let Some(expected) = fx.last_token_logits else {
        eprintln!("SKIP: fixture has no last_token_logits");
        return Ok(());
    };

    let rt = copper::Runtime::cpu_f32();
    let spec = copper::config::load_model_spec(&model_dir)?;
    let weights = copper::loader::load_gpt2_weights(&model_dir, &rt)?;
    let model = copper::model::Gpt2::new(spec, weights, &rt)?;

    let logits = model.forward(&fx.input_ids)?;
    let seq_len = fx.input_ids.len();
    let last = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)?;
    let got: Vec<f32> = last.to_vec1()?;

    if got.len() != expected.len() {
        bail!(
            "logits length mismatch: got {} expected {}",
            got.len(),
            expected.len()
        );
    }
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (g, e) in got.iter().zip(expected.iter()) {
        let abs = (g - e).abs();
        let rel = abs / e.abs().max(1e-6);
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
