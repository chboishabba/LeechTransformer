#!/usr/bin/env python
"""Capture last-token module outputs during Leech-LILA generation.

The saved float arrays retain their original dtype and therefore their exact
IEEE byte representation.  Hooks observe outputs after computation; they do
not alter model inputs or arithmetic.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.run_inference import _extract_model_state_dict, _load_checkpoint_compat
from models.model import LeechConfig, LeechGPT
from tokenizer.tokenizer_utils import load_tokenizer
from inference.generate import generate


def main():
    p = argparse.ArgumentParser(description="Capture real Leech-LILA module outputs")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", default="who is Lila?")
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--kv-cache", action="store_true")
    a = p.parse_args()
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sp = load_tokenizer()
    model = LeechGPT(LeechConfig(vocab_size=sp.get_piece_size())).to(device)
    model.load_state_dict(_extract_model_state_dict(_load_checkpoint_compat(a.checkpoint, device)))
    model.eval()
    captured = {}
    handles = []

    def hook(name):
        def save(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(value):
                return
            # The last position has fixed width across growing contexts.
            captured.setdefault(name, []).append(value[:, -1, :].detach().cpu().contiguous().numpy().copy())
        return save

    for name, module in model.named_modules():
        if name == "final_norm" or name.endswith(".attn") or name.endswith(".ffn") or name.startswith("blocks.") and name.count(".") == 1:
            handles.append(module.register_forward_hook(hook(name)))
    try:
        text = generate(model, sp, start_str=a.prompt, max_tokens=a.max_tokens,
                        temperature=0, top_k=None, top_p=None, device=device,
                        use_kv_cache=a.kv_cache, print_tokens=False)
    finally:
        for handle in handles:
            handle.remove()
    arrays = {name.replace(".", "__"): np.concatenate(values, axis=0)
              for name, values in captured.items()}
    arrays["metadata_json"] = np.array(json.dumps({"model": "Leech-LILA", "checkpoint": a.checkpoint,
        "prompt": a.prompt, "generated_text": text, "device": str(device), "hooks": sorted(captured)}))
    np.savez(a.out, **arrays)
    print(f"captured {sum(len(v) for v in captured.values())} module frames across {len(captured)} streams -> {a.out}")


if __name__ == "__main__":
    main()
