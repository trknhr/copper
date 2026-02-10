# copper (MVP)

Minimal GPT-2 inference (CPU / f32 / greedy / batch=1) using Candle as the compute backend.

## Run

Requires a local GPT-2 directory containing at least:

- `config.json`
- `tokenizer.json`
- a single `.safetensors` file (e.g. `model.safetensors`)

Example:

```bash
cargo run -- --model-dir /path/to/gpt2 --prompt "Hello" --max-new-tokens 16 --output stream
```

Use buffered output instead:

```bash
cargo run -- --model-dir /path/to/gpt2 --prompt "Hello" --output buffered
```

Set logging level:

```bash
cargo run -- --model-dir /path/to/gpt2 --prompt "Hello" --log-level debug
```

Start interactive chat mode:

```bash
cargo run -- --model-dir /path/to/gpt2 --chat --max-new-tokens 64 --output stream
```

## Quality checks

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib --bins
```

## HF parity tests (fixtures)

1. Generate fixture JSON (requires `transformers` + `torch`):

```bash
python3 scripts/export_hf_fixtures.py --model-dir /path/to/gpt2
```

2. Run ignored HF parity tests:

```bash
COPPER_MODEL_DIR=/path/to/gpt2 cargo test -- --ignored
```
