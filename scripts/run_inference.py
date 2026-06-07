"""
🎯 Leech-Lila DOI: 10.5281/zenodo.18784424
This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
Commercial Licensing: For proprietary R&D, integration into private AI stacks, or hardware implementation,
please contact the Architect directly.
Copyright (C) 2026 Anatolii Kornienko This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License
as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/agpl-3.0.txt/>.
"""

import argparse
import __main__
import os
import sys
import time
from typing import Any
import torch
from torch.serialization import add_safe_globals

# Добавляем корень проекта в path для импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _select_device() -> tuple[torch.device, dict[str, object]]:
    """Return best available torch device with robust ROCm availability checks."""
    debug = {
        "torch_version": torch.__version__,
        "torch_cuda_is_available": None,
        "torch_cuda_device_count": None,
        "torch_version_hip": getattr(torch.version, "hip", None),
        "torch_version_cuda": getattr(torch.version, "cuda", None),
    }

    try:
        cuda_count = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - defensive for partially initialized runtimes
        debug["torch_cuda_device_count_error"] = f"{type(exc).__name__}: {exc}"
        cuda_count = -1

    debug["torch_cuda_device_count"] = cuda_count

    try:
        is_available = torch.cuda.is_available()
    except Exception as exc:  # pragma: no cover
        debug["torch_cuda_is_available_error"] = f"{type(exc).__name__}: {exc}"
        is_available = False
    debug["torch_cuda_is_available"] = is_available

    if cuda_count > 0:
        return torch.device("cuda"), debug
    if is_available:
        return torch.device("cuda"), debug

    return torch.device("cpu"), debug


def _resolve_leech_config_for_torch_load() -> tuple[object, object | None]:
    """Ensure checkpoint torch.load can deserialize legacy __main__.LeechConfig objects."""
    safe_candidate = getattr(__main__, "LeechConfig", None)
    fallback_backup = safe_candidate

    if getattr(safe_candidate, "__module__", None) == "__main__" and getattr(safe_candidate, "__name__", None) == "LeechConfig":
        add_safe_globals([safe_candidate])
        return safe_candidate, fallback_backup

    compat_cfg = type("LeechConfig", (object,), {})
    compat_cfg.__module__ = "__main__"
    __main__.LeechConfig = compat_cfg
    add_safe_globals([compat_cfg])
    return compat_cfg, fallback_backup


def _load_checkpoint_compat(path: str, map_device: torch.device) -> dict[str, Any]:
    """Load checkpoint while tolerating torch.serialization hardening around unpickling."""
    try:
        checkpoint = torch.load(path, map_location=map_device, weights_only=True)
        return checkpoint
    except Exception as first_err:
        compat_cfg, backup_cfg = _resolve_leech_config_for_torch_load()
        try:
            # Keep full-load fallback because older checkpoints contain a pickled config object.
            return torch.load(path, map_location=map_device, weights_only=False)
        except Exception as second_err:
            raise RuntimeError(
                f"Failed to load checkpoint {path}: "
                f"weights_only=True failed ({type(first_err).__name__}: {first_err}), "
                f"full load failed ({type(second_err).__name__}: {second_err})"
            ) from second_err
        finally:
            # Restore previous module binding if we injected a compatibility symbol.
            if backup_cfg is not None:
                __main__.LeechConfig = backup_cfg
            else:
                try:
                    delattr(__main__, "LeechConfig")
                except AttributeError:
                    pass


def _extract_model_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Get model weights from known checkpoint shapes.

    Supports:
    - {'model_state_dict': {...}}
    - {'state_dict': {...}}
    - {'model': {...}}
    - {'module': {...}}
    - raw tensor dict dumps
    """
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(checkpoint).__name__}")

    for key in ("model_state_dict", "state_dict", "model", "module"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    # Fallback: collect tensors from top-level keys
    tensors = {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}
    if tensors:
        return tensors

    raise ValueError(
        f"Could not find model weights in checkpoint keys: {list(checkpoint.keys())}"
    )


def _rocm_stability_tweaks(device: torch.device, max_tokens: int, top_p: float | None) -> float | None:
    """Apply ROCm-specific sampling compatibility tweaks for long generation runs."""
    if (
        device.type == "cuda"
        and getattr(torch.version, "hip", None) is not None
        and max_tokens > 36
        and top_p is not None
        and 0.0 < top_p < 1.0
    ):
        print(
            "[compat] ROCm+long-token path detected; disabling top-p sampling for stability "
            "(use --top_p 1.0 to force explicit behavior)."
        )
        return 1.0

    return top_p

def main():
    print(f"[bootstrap] preparse is_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    parser = argparse.ArgumentParser(description="LILA-E8 inference: generate text from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default="who is Lily?", help="Start prompt")
    parser.add_argument("--max_tokens", type=int, default=112, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature (0.1-2.0)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling (0=disabled)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P (nucleus) sampling (0=disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.4, help="Penalty for repeated tokens")
    parser.add_argument("--repetition_window", type=int, default=50, help="Window for repetition penalty")
    parser.add_argument("--use_resonator", action="store_true", help="Enable LeechResonanceBiasing logits bias")
    parser.add_argument("--resonator_alpha", type=float, default=0.1, help="Initial alpha for resonance biasing")
    parser.add_argument("--bench", action="store_true", help="Print tokens/sec for this generation run")
    parser.add_argument("--bench_warmup", type=int, default=1, help="Warmup runs before timing (default: 1)")
    parser.add_argument("--quiet", action="store_true", help="Disable streaming token printing (useful for benchmarking)")
    parser.add_argument("--kv_cache", action="store_true", help="Use KV-cache for faster decoding (works reliably while total length <= block_size)")
    args = parser.parse_args()

    print(f"[bootstrap] argv={sys.argv}")
    print(f"[bootstrap] cwd={os.getcwd()}")
    print(f"[bootstrap] PYTHONHOME={os.environ.get('PYTHONHOME', '(unset)')} PYTHONPATH={os.environ.get('PYTHONPATH', '(unset)')}")

    device, debug_info = _select_device()
    if device.type == "cuda":
        print(f"[device] selected=cuda debug={debug_info}")
        if args.kv_cache and os.environ.get("LEECH_ALLOW_KVCACHE_GPU", "0") != "1":
            print(
                "[warn] --kv_cache is known to trigger GPU page faults on this ROCm path."
                " It has been disabled for this run."
            )
            args.kv_cache = False
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[device] selected=mps debug={debug_info}")
    else:
        device = torch.device("cpu")
        print(f"[device] selected=cpu debug={debug_info}")

    from models.model import LeechGPT
    from tokenizer.tokenizer_utils import load_tokenizer
    from inference.generate import generate
    from config.config import LeechConfig

    sp = load_tokenizer()
    cfg = LeechConfig(vocab_size=sp.get_piece_size())
    model = LeechGPT(cfg).to(device)
    checkpoint = _load_checkpoint_compat(args.checkpoint, device)
    model_state_dict = _extract_model_state_dict(checkpoint)
    model.load_state_dict(model_state_dict)
    model.eval()

    if device.type == "cuda" and args.max_tokens > 36:
        print(
            "[warn] On this ROCm 6.4 Polaris path, generation above 36 tokens can trigger"
            " GPU page faults. Re-run with --max_tokens <= 36 for a stable pass."
        )

    if args.use_resonator:
        model.attach_resonator(alpha_init=args.resonator_alpha)

    if args.bench and args.bench_warmup > 0:
        for _ in range(args.bench_warmup):
            _ = generate(
                model, sp,
                start_str=args.prompt,
                max_tokens=min(16, args.max_tokens),
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                top_p=_rocm_stability_tweaks(device, min(16, args.max_tokens), args.top_p if 0 < args.top_p < 1 else None),
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                device=device,
                use_resonator=args.use_resonator,
                use_kv_cache=args.kv_cache,
                print_tokens=not args.quiet,
            )

    result = generate(
        model, sp,
        start_str=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=_rocm_stability_tweaks(device, args.max_tokens, args.top_p if 0 < args.top_p < 1 else None),
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        device=device,
        use_resonator=args.use_resonator,
        use_kv_cache=args.kv_cache,
        return_stats=args.bench,
        print_tokens=not args.quiet,
    )
    # Текст уже выводится потоково внутри generate()
    if args.bench:
        _, stats = result
        toks = stats["generated_tokens"]
        sec = stats["seconds"]
        tps = (toks / sec) if sec > 0 else float("inf")
        print(f"[bench] generated_tokens={toks} seconds={sec:.4f} tok/s={tps:.2f}")

if __name__ == "__main__":
    main()
