#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("tests/fixtures/hf_gpt2.json"))
    p.add_argument("--prompt", type=str, default="Hello")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    enc = tok(args.prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits[0, -1].detach().cpu().to(torch.float32).tolist()

    fixture = {
        "prompt": args.prompt,
        "input_ids": input_ids[0].detach().cpu().to(torch.int64).tolist(),
        "last_token_logits": logits,
        "atol": 1e-3,
        "rtol": 1e-3,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

