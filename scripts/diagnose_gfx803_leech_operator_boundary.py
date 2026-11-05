"""First divergent *actual* Leech forward module on gfx803 vs CPU.

This extends the accepted dense-gate failure at batch=2, context=64,
NOT the independent convolution solver test. Reuses the frozen teacher
and tokenizer IDs from LeechTransformer. Never trains or approves GPU.

Hooks can themselves synchronize and change nondeterministic kernel
behavior: compare instrumented vs *unmodified* forward logits, and
report any observer effect. A passing hooked run cannot supersede a
failing uninstrumented run. The in-repo compatibility graph retains
its already-existing kernel profiler and crash witnesses.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.diagnose_moe_quality_collapse import load_pair
from scripts.gfx803_moe_gpu_gate import comparison, file_sha256
from scripts.gfx803_runtime_attestation import attest_runtime


def named_stages(model, max_blocks: int):
    """Actual forward submodule order, no hand-reimplemented attention."""
    if max_blocks < 1 or max_blocks > len(model.blocks):
        raise ValueError("max-blocks outside network")
    yield "tok_emb", model.tok_emb
    for index, block in enumerate(model.blocks):
        if index >= max_blocks:
            continue
        label=f"block{index}"
        yield f"{label}.ln1", block.ln1
        yield f"{label}.qkv", block.attn.qkv
        yield f"{label}.attn.out", block.attn.out
        yield f"{label}.attn", block.attn
        yield f"{label}.ln2", block.ln2
        if not isinstance(block.ffn, nn.Sequential) or len(block.ffn)<3:
            raise ValueError("dense teacher FFN layout is not expected Sequential")
        yield f"{label}.ffn.linear_in", block.ffn[0]
        yield f"{label}.ffn.linear_out", block.ffn[2]
        yield f"{label}.ffn", block.ffn
        yield f"{label}.output", block
    yield "final_norm", model.final_norm
    yield "head", model.head


def recorded_forward(model, ids, max_blocks: int):
    """Capture stage outputs as independent host snapshots.

    Every CPU copy of GPU outputs can force synchronization.
    Compare instrumented and uninstrumented forward *separately*
    before drawing conclusions from first divergent stage.
    """
    stages={}
    hooks=[]
    def make_hook(name):
        def capture(_module, _inputs, value):
            if isinstance(value,tuple):
                value=value[0]
            if not isinstance(value,torch.Tensor):
                raise RuntimeError("unexpected non-tensor stage "+name)
            if name in stages:
                raise AssertionError("stage executed more than once: "+name)
            stages[name]=value.detach().float().cpu().clone()
        return capture
    try:
        for name,module in named_stages(model,max_blocks):
            hooks.append(module.register_forward_hook(make_hook(name)))
        with torch.no_grad():
            logits=model(ids)[0].detach().float().cpu().clone()
    finally:
        for hook in hooks:
            hook.remove()
    return stages,logits


def first_failed_stage(results):
    for name,score in results.items():
        if not score.get("passed",False):
            return name
    return None


def run_differential(teacher,rows,*,max_blocks=2,repeats=3,
                     atol=1e-3,rtol=1e-3):
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("ROCm HIP GPU unavailable; no CPU fallback")
    if (rows.ndim!=2 or rows.shape[0]<1 or rows.shape[1]<2 or
            rows.shape[1]>teacher.cfg.block_size or
            not np.issubdtype(rows.dtype,np.integer) or
            np.any(rows<0) or np.any(rows>=teacher.cfg.vocab_size)):
        raise ValueError("original checkpoint-compatible real tokenizer IDs required")
    if repeats<2:
        raise ValueError("repeat count must be at least 2")
    cpu=teacher.eval().cpu()
    gpu=copy.deepcopy(cpu).eval().to("cuda")
    ids=torch.as_tensor(rows,dtype=torch.long)
    ids_gpu=ids.to("cuda")
    if hasattr(torch.backends,"cuda"):
        torch.backends.cuda.matmul.allow_tf32=False
    if hasattr(torch.backends,"cudnn"):
        torch.backends.cudnn.allow_tf32=False
    result={"admitted":False,"shape":list(ids.shape),
            "dense_only":True,"no_c2_grads_or_training":True,
            "max_blocks_profiled":max_blocks,"repeats":repeats,
            "checks":{},"stage_comparisons":{},
            "first_failed_stage":None,
            "possible_observer_effect":False}
    with torch.no_grad():
        print("[leech-stage] uninstrumented CPU and HIP logits",flush=True)
        bare_cpu=cpu(ids)[0].detach().float().cpu().clone()
        bare_gpu=gpu(ids_gpu)[0].detach().float().cpu().clone()
        result["checks"]["bare_cpu_vs_gpu"]=comparison(
            bare_cpu,bare_gpu,atol=atol,rtol=rtol)
        for i in range(1,repeats):
            next_gpu=gpu(ids_gpu)[0]
            result["checks"][f"bare_gpu_repeat_{i}"]=comparison(
                bare_gpu,next_gpu,atol=1e-5,rtol=1e-5)
        print("[leech-stage] instrumented module-hook CPU and HIP",flush=True)
        cpu_stages,cap_cpu=recorded_forward(cpu,ids,max_blocks)
        gpu_stages,cap_gpu=recorded_forward(gpu,ids_gpu,max_blocks)
        result["checks"]["instrumented_cpu_vs_bare"]=comparison(
            bare_cpu,cap_cpu,atol=1e-5,rtol=1e-5)
        result["checks"]["instrumented_gpu_vs_bare"]=comparison(
            bare_gpu,cap_gpu,atol=1e-5,rtol=1e-5)
        if list(cpu_stages)!=list(gpu_stages):
            raise RuntimeError("CPU/GPU stage hooks not executed in same order")
        for name in cpu_stages:
            result["stage_comparisons"][name]=comparison(
                cpu_stages[name],gpu_stages[name],atol=atol,rtol=rtol)
    result["first_failed_stage"]=first_failed_stage(result["stage_comparisons"])
    result["possible_observer_effect"]=not result["checks"][
        "instrumented_gpu_vs_bare"]["passed"]
    result["all_checks_pass"]=all(v["passed"] for v in result["checks"].values())
    result["all_stages_pass"]=all(v["passed"] for v in result[
        "stage_comparisons"].values())
    result["admitted"]=False
    result["scope"]="Diagnostic ONLY: no route around gfx803_moe_gpu_gate"
    result["interpretation"]=(
        "Instrumented GPU forward changed results: hooks synchronize "
        "and may mask the original race. Do NOT attribute bare-runtime "
        "failure to the first hooked stage."
        if result["possible_observer_effect"] else
        "Same-process hooked stage comparisons identify a candidate "
        "earliest operator boundary; not necessarily a root cause.")
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--teacher",type=Path,required=True)
    parser.add_argument("--student",type=Path,required=True,
                        help="Source hash only; the dense teacher is diagnosed")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--train-ids",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--batch",type=int,default=2)
    parser.add_argument("--max-blocks",type=int,default=2)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--deterministic-algorithms",action="store_true",
                        help="Opt-in separate-process torch determinism diagnostic; never admits GPU")
    args=parser.parse_args()
    out={
        "admitted":False,
        "diagnostic_only":True,
        "python":sys.version.split()[0],
        "platform":platform.platform(),
        "torch_version":torch.__version__,
        "torch_hip":torch.version.hip,
        "source_sha256":{},
        "runtime_note":"Run via gfx803_compat_graph existing Nix shell and patched-Python wrapper",
        "deterministic_algorithms_requested":args.deterministic_algorithms,
    }
    try:
        out["source_sha256"]={
            "teacher":file_sha256(args.teacher),
            "student":file_sha256(args.student),
            "train":file_sha256(args.train_ids)}
        if torch.version.hip is None or not torch.cuda.is_available():
            raise RuntimeError("No ROCm HIP GPU; no CPU fallback")
        props=torch.cuda.get_device_properties(0)
        out["gpu_name"]=torch.cuda.get_device_name(0)
        out["gcn_arch_name"]=getattr(props,"gcnArchName",None)
        ids=np.load(args.train_ids,allow_pickle=False)
        if args.batch<1 or ids.ndim!=2 or len(ids)<args.batch:
            raise ValueError("missing original token rows")
        out["probe_indices"]=list(range(args.batch))
        teacher,_=load_pair(args.teacher,args.student,args.config)
        if args.deterministic_algorithms:
            torch.use_deterministic_algorithms(True)
        out["torch_deterministic_algorithms_enabled"]=bool(
            torch.are_deterministic_algorithms_enabled())
        out["result"]=run_differential(
            teacher,ids[:args.batch],max_blocks=args.max_blocks,
            repeats=args.repeats)
    except Exception as exc:
        out["error"]=f"{type(exc).__name__}: {exc}"
        out["traceback_tail"]=traceback.format_exc()[-1600:]
    finally:
        out["effective_runtime_attestation"]=attest_runtime()
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({
        "diagnostic_only":True,
        "receipt":str(args.receipt),
        "first_failed_stage":out.get("result",{}).get("first_failed_stage"),
        "observer_effect":out.get("result",{}).get("possible_observer_effect"),
        "error":out.get("error")},indent=2),flush=True)
    # Even a fully passing diagnostic cannot authorize GPU training.
    return 0 if "error" not in out else 2


if __name__=="__main__":
    raise SystemExit(main())
