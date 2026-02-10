mod common;

use anyhow::Result;

#[test]
#[ignore = "requires local GPT-2 model and HF fixture"]
fn tokenizer_matches_hf_input_ids() -> Result<()> {
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

    let tok = copper::tokenizer::load_tokenizer(&model_dir)?;
    let ids = tok.encode(&fx.prompt)?;
    assert_eq!(ids, fx.input_ids);
    Ok(())
}
