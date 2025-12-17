`hf_gpt2.json` is intentionally not committed.

Generate it locally (requires `transformers` + `torch` and a local GPT-2 directory that contains `config.json`, `tokenizer.json`, and `model.safetensors`):

1. `python3 scripts/export_hf_fixtures.py --model-dir /path/to/gpt2`
2. `COPPER_MODEL_DIR=/path/to/gpt2 cargo test`

