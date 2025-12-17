# copper (MVP)

Minimal GPT-2 inference (CPU / f32 / greedy / batch=1) using Candle as the compute backend.

## Run

Requires a local GPT-2 directory containing at least:

- `config.json`
- `tokenizer.json`
- a single `.safetensors` file (e.g. `model.safetensors`)

Example:

```bash
cargo run -- --model-dir /path/to/gpt2 --prompt "Hello" --max-new-tokens 16
```

## HF match tests (fixtures)

1. Generate fixture JSON (requires `transformers` + `torch`):

```bash
python3 scripts/export_hf_fixtures.py --model-dir /path/to/gpt2
```

2. Run tests:

```bash
COPPER_MODEL_DIR=/path/to/gpt2 cargo test
```

