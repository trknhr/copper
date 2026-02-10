#!/usr/bin/env python3
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_once(model, tok, prompt: str, max_new_tokens: int) -> tuple[str, int, float]:
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: matches Rust behavior
            pad_token_id=tok.eos_token_id,
        )
    dt = time.perf_counter() - t0
    total_text = tok.decode(out[0], skip_special_tokens=True)
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    return total_text, int(new_tokens), dt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt", default="Hello, how are you?")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--chat", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32
    )
    model.eval()

    if not args.chat:
        text, new_tokens, dt = generate_once(model, tok, args.prompt, args.max_new_tokens)
        print(text)
        print(f"\nnew_tokens={new_tokens}, time={dt:.3f}s, tok/s={new_tokens / dt:.2f}")
        return

    print("Chat mode: type `exit` or `quit` to stop")
    history = ""
    while True:
        try:
            user = input("> ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        turn_prefix = f"User: {user}\nAssistant:"
        prompt = history + turn_prefix
        full, new_tokens, dt = generate_once(model, tok, prompt, args.max_new_tokens)
        reply = full[len(prompt) :].strip()
        print(reply)
        print(f"[perf] new_tokens={new_tokens}, time={dt:.3f}s, tok/s={new_tokens / dt:.2f}")
        history = full + "\n"


if __name__ == "__main__":
    main()
