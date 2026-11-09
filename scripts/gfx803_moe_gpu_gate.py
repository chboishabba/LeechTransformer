"""Fail-closed numerical ROCm/gfx803 admission for LeechTransformer training.

This is a checkpoint/shape-specific HARD GATE, not a ROCm certification.
The existing gfx803_compat_graph Nix shell supplies the Python/runtime.
Use only authentic saved checkpoint and token windows (no random smoke
substitute). Check CPU/GPU dense and focused C2 forward, intra-GPU
repeatability, local teacher-input odd+even output, and focused
CE+KL+MSE gradients plus one AdamW update.

On the user's Polaris stack, torch.cuda.is_available() alone does
NOT establish correctness: prior Leech runs drifted before block 1.
Failed or missing checks NEVER launch training or fall back to CPU.
Original Leech-LILA author: Anatolii Kornienko.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.dense_to_signed_experts import canonical_even_input
from scripts.diagnose_moe_quality_collapse import load_pair
from scripts.gfx803_runtime_attestation import attest_runtime


def write_progress(path: Path | None, stage: str, **extra) -> None:
    """Atomically retain the last entered hardware stage if HIP kills Python."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"stage": stage, **extra}, indent=2) + "\n")
    tmp.replace(path)


def file_sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()


# Include actual solver flags in the receipt; their absence in older
# records must never be interpreted as proof of a direct-only profile.
GPU_NUMERIC_ENV_KEYS=(
    "MIOPEN_DEBUG_CONV_DIRECT",
    "MIOPEN_DEBUG_CONV_GEMM",
    "MIOPEN_DEBUG_CONV_WINOGRAD",
    "MIOPEN_DEBUG_CONV_FFT",
    "MIOPEN_DEBUG_CONV_DET",
    "MIOPEN_DEBUG_DISABLE_FIND_DB",
    "MIOPEN_FIND_ENFORCE",
    "TORCH_BLAS_PREFER_HIPBLASLT",
    "ROCBLAS_TENSILE_LIBPATH",
    "HIP_LAUNCH_BLOCKING",
    "HSA_OVERRIDE_GFX_VERSION",
    "ROC_ENABLE_PRE_VEGA",
    "PYTORCH_ROCM_ARCH",
    "ROCM_ARCH",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)

def runtime_numeric_configuration():
    return {name:os.environ.get(name) for name in GPU_NUMERIC_ENV_KEYS}


def comparison(reference,actual,*,atol=1e-3,rtol=1e-3):
    """Tensor-level test on CPU-copied values; no silent NaN approval."""
    a=reference.detach().float().cpu()
    b=actual.detach().float().cpu()
    if a.shape!=b.shape:
        return {"passed":False,"reason":"shape mismatch",
                "reference_shape":list(a.shape),"actual_shape":list(b.shape)}
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"passed":False,"finite":False,"shape":list(a.shape)}
    d=(a-b).abs()
    bound=atol+rtol*a.abs()
    passed=bool((d<=bound).all())
    return {
        "passed":passed,"finite":True,"shape":list(a.shape),
        "max_abs_diff":float(d.max()) if d.numel() else 0.,
        "max_allowed_error":float(bound.max()) if bound.numel() else 0.,
        "fail_count":int((d>bound).sum()),
        "max_reference_abs":float(a.abs().max()) if a.numel() else 0.,
        "atol":atol,"rtol":rtol}


def collect_teacher_ln2(teacher,ids,layer):
    """Uses the SAME existing normalized FFN input from the dense teacher."""
    capture=[]
    handle=teacher.blocks[layer].ln2.register_forward_hook(
        lambda _module,_args,value:capture.append(value.detach().clone()))
    try:
        with torch.no_grad():
            logits=teacher(ids)[0]
    finally:
        handle.remove()
    if len(capture)!=1:
        raise AssertionError("teacher normalized-state capture failed")
    return logits,capture[0]


def focused_loss(hybrid,teacher_logits,teacher_ffn,normalized,ids,layer):
    """Actual layer-1 matched MSE + focused token CE + teacher KL."""
    hybrid.train()
    for module in hybrid.modules():
        if isinstance(module,nn.Dropout):
            module.eval()
    for b in hybrid.blocks:
        with torch.no_grad():
            b.quality_gate.fill_(0.)
    block=hybrid.blocks[layer]
    odd=block.moe(normalized,train_sign=True)[0]
    even=block.even_student(canonical_even_input(normalized))
    local=F.mse_loss(odd+even,teacher_ffn)
    logits=hybrid(ids,student_only=True,student_only_layers=(layer,))[0]
    targets=ids[:,1:].reshape(-1)
    student_ce=F.cross_entropy(
        logits[:,:-1].reshape(-1,logits.shape[-1]),targets)
    teacher_probs=F.softmax(teacher_logits.detach(),dim=-1)
    kl=F.kl_div(F.log_softmax(logits,dim=-1),teacher_probs,
                reduction="none").sum(-1).mean()
    loss=local+student_ce+kl
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite focused training loss")
    return loss,logits


def run_gate(teacher,student,rows,*,layer=1,repeat=3,
             stability_steps=1,atol=1e-3,rtol=1e-3,dense_only=False,
             stop_after_c2_forward=False,progress_path: Path | None=None,
             router_selector="topk",stop_after_c2_backward=False):
    if not (torch.version.hip is not None and torch.cuda.is_available()):
        raise RuntimeError("ROCm/HIP PyTorch GPU is unavailable; refusing CPU fallback")
    if repeat<2 or not 0<=layer<len(teacher.blocks) or stability_steps<1:
        raise ValueError("need >=2 GPU repeats, >=1 stability step and a valid layer")
    if router_selector not in ("topk", "sort"):
        raise ValueError("router_selector must be topk or sort")
    if (rows.ndim!=2 or rows.shape[0]<1 or rows.shape[1]<2 or
        not np.issubdtype(rows.dtype,np.integer) or
        np.any(rows<0) or np.any(rows>=teacher.cfg.vocab_size) or
        rows.shape[1]>teacher.cfg.block_size):
        raise ValueError("real checkpoint-compatible token rows required")
    device=torch.device("cuda:0")
    ids_cpu=torch.as_tensor(rows,dtype=torch.long,device="cpu")
    ids_gpu=ids_cpu.to(device)
    teacher.eval()
    student.eval()
    # Disable TF32 shortcuts when present, preserving FP32 parity scope.
    if hasattr(torch.backends,"cuda"):
        torch.backends.cuda.matmul.allow_tf32=False
    if hasattr(torch.backends,"cudnn"):
        torch.backends.cudnn.allow_tf32=False
    report={"checks":{},"shape":list(ids_cpu.shape),"layer":layer}
    checks=report["checks"]
    write_progress(progress_path,"dense-forward-start")
    print("[gfx803-gate] dense CPU/GPU and layer-input comparison",flush=True)
    with torch.no_grad():
        teacher_logits_cpu,x_cpu=collect_teacher_ln2(teacher,ids_cpu,layer)
        teacher_ffn_cpu=teacher.blocks[layer].ffn(x_cpu)
        teacher_gpu=copy.deepcopy(teacher).eval().to(device)
        teacher_logits_gpu,x_gpu=collect_teacher_ln2(
            teacher_gpu,ids_gpu,layer)
        teacher_ffn_gpu=teacher_gpu.blocks[layer].ffn(x_gpu)
        checks["dense_cpu_gpu_logits"]=comparison(
            teacher_logits_cpu,teacher_logits_gpu,atol=atol,rtol=rtol)
        checks["teacher_layer_input"]=comparison(
            x_cpu,x_gpu,atol=atol,rtol=rtol)
        checks["teacher_layer_ffn"]=comparison(
            teacher_ffn_cpu,teacher_ffn_gpu,atol=atol,rtol=rtol)
        print("[gfx803-gate] repeated GPU dense forward",flush=True)
        for i in range(1,repeat):
            t=teacher_gpu(ids_gpu)[0]
            checks[f"dense_gpu_repeat_{i}"]=comparison(
                teacher_logits_gpu,t,atol=1e-5,rtol=1e-5)
        if not all(v["passed"] for v in checks.values()):
            report["admitted"]=False
            report["stage"]="teacher numerical parity"
            return report
        del teacher_gpu
        torch.cuda.empty_cache()
    write_progress(progress_path,"dense-forward-complete",checks=sorted(checks))
    if dense_only:
        report["admitted"]=False
        report["stage"]="dense-only complete"
        report["limits"]=("Only original dense CPU/GPU forward and repeatability "
                          "were measured; C2, backward and optimizer paths were skipped.")
        return report

    # Compare actual STRICT layer-1 substitution with all other layers
    # backed by dense FFNs, rather than scoring the admitted hybrid gate.
    from scripts.train_guarded_moe_continuation import build_guarded_hybrid
    hybrid_cpu=build_guarded_hybrid(teacher,student).eval()
    hybrid_gpu=copy.deepcopy(hybrid_cpu).eval().to(device)
    for hybrid in (hybrid_cpu,hybrid_gpu):
        hybrid.blocks[layer].moe.router_selection=router_selector
    report["router_selector"]=router_selector
    write_progress(progress_path,"c2-forward-start")
    print("[gfx803-gate] strict focused C2 CPU/GPU and repeatability",flush=True)
    with torch.no_grad():
        focused_cpu=hybrid_cpu(ids_cpu,student_only=True,
                               student_only_layers=(layer,))[0]
        focused_gpu=hybrid_gpu(ids_gpu,student_only=True,
                               student_only_layers=(layer,))[0]
        checks["focused_cpu_gpu_logits"]=comparison(
            focused_cpu,focused_gpu,atol=atol,rtol=rtol)
        for i in range(1,repeat):
            actual=hybrid_gpu(ids_gpu,student_only=True,
                              student_only_layers=(layer,))[0]
            checks[f"focused_gpu_repeat_{i}"]=comparison(
                focused_gpu,actual,atol=1e-5,rtol=1e-5)
        if not all(v["passed"] for v in checks.values()):
            report["admitted"]=False
            report["stage"]="focused student numerical parity"
            return report
    write_progress(progress_path,"c2-forward-complete",checks=sorted(checks))
    if stop_after_c2_forward:
        report["admitted"]=False
        report["stage"]="c2-forward-only complete"
        report["limits"]="C2 forward/repeatability only; backward and optimizer paths were skipped."
        return report

    # One REAL training gradient and optimizer update on the original
    # sampled token rows. Neither model's source checkpoint is mutated.
    cpu_logits=teacher_logits_cpu.detach()
    gpu_logits=cpu_logits.to(device)
    norm_cpu=x_cpu.detach()
    norm_gpu=x_gpu.detach()
    ffn_cpu=teacher_ffn_cpu.detach()
    ffn_gpu=ffn_cpu.to(device)
    named_cpu=(list(hybrid_cpu.blocks[layer].moe.named_parameters())+
               [("even_student."+name, value) for name,value in
                hybrid_cpu.blocks[layer].even_student.named_parameters()])
    named_gpu=(list(hybrid_gpu.blocks[layer].moe.named_parameters())+
               [("even_student."+name, value) for name,value in
                hybrid_gpu.blocks[layer].even_student.named_parameters()])
    named_cpu=[(name,value) for name,value in named_cpu if value.requires_grad]
    named_gpu=[(name,value) for name,value in named_gpu if value.requires_grad]
    if [name for name,_ in named_cpu] != [name for name,_ in named_gpu]:
        raise AssertionError("CPU/GPU C2 parameter names differ")
    params_cpu=[value for _,value in named_cpu]
    params_gpu=[value for _,value in named_gpu]
    opt_cpu=torch.optim.AdamW(params_cpu,lr=3e-4)
    opt_gpu=torch.optim.AdamW(params_gpu,lr=3e-4)
    opt_cpu.zero_grad(set_to_none=True)
    opt_gpu.zero_grad(set_to_none=True)
    write_progress(progress_path,"c2-loss-backward-start")
    print("[gfx803-gate] matched-objective local MSE + CE/KL backward",flush=True)
    loss_cpu,_=focused_loss(hybrid_cpu,cpu_logits,ffn_cpu,norm_cpu,
                            ids_cpu,layer)
    loss_gpu,_=focused_loss(hybrid_gpu,gpu_logits,ffn_gpu,norm_gpu,
                            ids_gpu,layer)
    checks["matched_objective_loss"]=comparison(
        loss_cpu,loss_gpu,atol=atol,rtol=rtol)
    loss_cpu.backward()
    loss_gpu.backward()
    grads=[]
    for (name,cpu_p),(_,gpu_p) in zip(named_cpu,named_gpu):
        if cpu_p.grad is None or gpu_p.grad is None:
            grads.append({"name":name,"passed":False,"reason":"missing gradient"})
            continue
        grads.append({"name":name,**comparison(cpu_p.grad,gpu_p.grad,
                                                 atol=2e-3,rtol=2e-3)})
    checks["focused_training_gradients"]={
        "passed":all(g["passed"] for g in grads),
        "parameter_tensors_checked":len(grads),
        "largest_max_abs_diff":max((g.get("max_abs_diff",0.) for g in grads),default=0.),
        "total_fail_count":sum(g.get("fail_count",1) for g in grads),
        "any_nonfinite":any(not g.get("finite",False) for g in grads),
        "failed_parameters":[g for g in grads if not g["passed"]]}
    if not all(v["passed"] for v in checks.values()):
        report["admitted"]=False
        report["stage"]="training backward parity"
        return report
    if stop_after_c2_backward:
        report["admitted"]=False
        report["stage"]="c2-backward-only complete"
        report["limits"]="C2 forward and loss/gradient parity only; optimizer and stability paths were skipped."
        return report
    torch.nn.utils.clip_grad_norm_(params_cpu,1.)
    torch.nn.utils.clip_grad_norm_(params_gpu,1.)
    write_progress(progress_path,"c2-optimizer-start")
    print("[gfx803-gate] one real AdamW update and post-step parity",flush=True)
    opt_cpu.step()
    opt_gpu.step()
    with torch.no_grad():
        new_cpu=hybrid_cpu.eval()(ids_cpu,student_only=True,
                                   student_only_layers=(layer,))[0]
        new_gpu=hybrid_gpu.eval()(ids_gpu,student_only=True,
                                   student_only_layers=(layer,))[0]
        checks["post_optimizer_step_logits"]=comparison(
            new_cpu,new_gpu,atol=atol,rtol=rtol)
    report["stability_steps_requested"]=stability_steps
    report["stability_steps_completed"]=1
    report["stability_trajectory"]=[]
    if not checks["post_optimizer_step_logits"]["passed"]:
        report["admitted"]=False
        report["stage"]="optimizer update parity"
        return report
    # A separate sustained focused-C2 update trajectory, using only the
    # authentic token window and identical initial optimizer state. This does
    # NOT save a trained checkpoint or bypass the full admission checks.
    for step in range(2,stability_steps+1):
        write_progress(progress_path,"c2-stability-step-start",step=step)
        opt_cpu.zero_grad(set_to_none=True)
        opt_gpu.zero_grad(set_to_none=True)
        loss_cpu,_=focused_loss(hybrid_cpu,cpu_logits,ffn_cpu,norm_cpu,
                                ids_cpu,layer)
        loss_gpu,_=focused_loss(hybrid_gpu,gpu_logits,ffn_gpu,norm_gpu,
                                ids_gpu,layer)
        loss_check=comparison(loss_cpu,loss_gpu,atol=atol,rtol=rtol)
        checks[f"stability_loss_step_{step}"]=loss_check
        if not loss_check["passed"]:
            report["stage"]=f"stability loss step {step}"
            report["admitted"]=False
            return report
        loss_cpu.backward()
        loss_gpu.backward()
        step_grad=[]
        for pc,pg in zip(params_cpu,params_gpu):
            step_grad.append(
                comparison(pc.grad,pg.grad,atol=2e-3,rtol=2e-3)
                if pc.grad is not None and pg.grad is not None
                else {"passed":False,"reason":"missing gradient"})
        gradient_ok=all(item["passed"] for item in step_grad)
        checks[f"stability_gradients_step_{step}"]={
            "passed":gradient_ok,
            "parameter_tensors_checked":len(step_grad),
            "largest_max_abs_diff":max(
                (item.get("max_abs_diff",float("inf")) for item in step_grad),
                default=0.0),
            "total_fail_count":sum(item.get("fail_count",1) for item in step_grad),
        }
        if not gradient_ok:
            report["stage"]=f"stability gradient step {step}"
            report["admitted"]=False
            return report
        torch.nn.utils.clip_grad_norm_(params_cpu,1.)
        torch.nn.utils.clip_grad_norm_(params_gpu,1.)
        opt_cpu.step()
        opt_gpu.step()
        with torch.no_grad():
            post_cpu=hybrid_cpu.eval()(ids_cpu,student_only=True,
                                       student_only_layers=(layer,))[0]
            post_gpu=hybrid_gpu.eval()(ids_gpu,student_only=True,
                                       student_only_layers=(layer,))[0]
            post_repeat=hybrid_gpu.eval()(ids_gpu,student_only=True,
                                          student_only_layers=(layer,))[0]
            delta=comparison(post_cpu,post_gpu,atol=atol,rtol=rtol)
            gpu_repeat=comparison(post_gpu,post_repeat,
                                  atol=1e-5,rtol=1e-5)
        checks[f"stability_post_step_logits_{step}"]=delta
        checks[f"stability_gpu_repeat_{step}"]=gpu_repeat
        report["stability_trajectory"].append({
            "step":step,
            "loss":loss_check,
            "gradients":checks[f"stability_gradients_step_{step}"],
            "post_logits":delta,
            "gpu_repeat":gpu_repeat,
        })
        if not (delta["passed"] and gpu_repeat["passed"]):
            report["stage"]=f"stability post-step {step}"
            report["admitted"]=False
            return report
        report["stability_steps_completed"]=step
    report["admitted"]=all(v["passed"] for v in checks.values())
    report["stage"]="complete" if report["admitted"] else "optimizer update parity"
    report["limits"]="Only tested checkpoint, device, dtype FP32, batch and token shape; not evidence of longer-run KFD/reset safety or GPU speedup."
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--student",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--train-ids",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--layer",type=int,default=1)
    p.add_argument("--probe-batch",type=int,default=2)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--stability-steps",type=int,default=1)
    p.add_argument("--dense-only",action="store_true",
                   help="Run dense forward/repeat checks only; never admit training")
    p.add_argument("--stop-after-c2-forward",action="store_true",
                   help="Run C2 forward/repeatability only; never enter loss, backward or optimizer")
    p.add_argument("--stop-after-c2-backward",action="store_true",
                   help="Run C2 forward and loss/gradient parity only; never enter optimizer or stability")
    p.add_argument("--progress-file",type=Path,
                   help="Atomically update before each hardware stage for hard-fault attribution")
    p.add_argument("--router-selector",choices=("topk","sort"),default="topk",
                   help="Explicit C2 routing implementation; sort is diagnostic-only on topk-faulting hardware")
    args=p.parse_args()
    metadata={
        "stability_steps_requested":args.stability_steps,
        "dense_only":args.dense_only,
        "stop_after_c2_forward":args.stop_after_c2_forward,
        "stop_after_c2_backward":args.stop_after_c2_backward,
        "router_selector":args.router_selector,
        "admitted":False,"workload":"LeechTransformer focused C2 training",
        "scope":"same-source CPU vs gfx803 HIP forward, repeat, backward and update",
        "torch_version":torch.__version__,"torch_hip":torch.version.hip,
        "platform":platform.platform(),
        "python":sys.version.split()[0],
        "hip_launch_blocking":os.environ.get("HIP_LAUNCH_BLOCKING"),
        "hsa_override_gfx_version":os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
        "stack_id":os.environ.get("STACK_ID"),
        "runtime_family":os.environ.get("RUNTIME_FAMILY"),
        "torch_python":os.environ.get("TORCH_PYTHON"),
        "numerical_environment":runtime_numeric_configuration(),
        "recorded_solver_flags_not_equivalent_to_full_workload_parity":True,
        "checked_train_rows":args.probe_batch}
    try:
        metadata["source_sha256"]={
            "teacher":file_sha256(args.teacher),
            "student":file_sha256(args.student),
            "train":file_sha256(args.train_ids)}
        if args.probe_batch<1 or args.repeats<2 or args.stability_steps<1:
            raise ValueError("probe-batch >=1, repeats >=2 and stability-steps >=1 required")
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("HIP GPU invisible; no CPU fallback")
        props=torch.cuda.get_device_properties(0)
        arch=getattr(props,"gcnArchName",None)
        metadata["gpu"]={
            "name":torch.cuda.get_device_name(0),
            "gcn_arch_name":arch,
            "total_memory":int(props.total_memory)}
        if arch is not None and not str(arch).startswith("gfx803"):
            raise RuntimeError("selected HIP device is not gfx803: "+str(arch))
        metadata["gpu"]["gfx803_attested_by_torch"]=(
            arch is not None and str(arch).startswith("gfx803"))
        raw=np.load(args.train_ids,allow_pickle=False)
        if raw.ndim!=2 or len(raw)<args.probe_batch:
            raise ValueError("probe batch exceeds real source rows")
        metadata["probe_indices"]=list(range(args.probe_batch))
        metadata["full_token_window_length"]=int(raw.shape[1])
        # Select repair before model construction, and never silently swap the
        # hardware backend or preserve an inherited profiler observer.
        repair=os.environ.get("GFX803_LINEAR_REPAIR")
        expected_shim=os.environ.get("GFX803_EXPECT_SHIM")
        if repair and expected_shim:
            raise RuntimeError("ROCm 6.4 CPU-linear and 10.1 shim paths are mutually exclusive")
        if repair:
            if repair != "cpu-fp32":
                raise RuntimeError("Unknown explicit linear repair mode: "+repair)
            if os.environ.get("GFX803_WGM1_QKV_JOIN"):
                provided=os.environ.get("GFX803_EXPECT_TEACHER_SHA256")
                authentic=metadata["source_sha256"]["teacher"]
                if provided and provided != authentic:
                    raise RuntimeError("WGM1 selection belongs to another teacher checkpoint")
                os.environ["GFX803_EXPECT_TEACHER_SHA256"]=authentic
            from scripts.gfx803_linear_repair import install_cpu_fp32_linear_repair
            install_cpu_fp32_linear_repair()
        if expected_shim:
            hip_prefix=os.environ.get("GFX803_EXPECT_HIP_PREFIX")
            if not hip_prefix or not str(torch.version.hip).startswith(hip_prefix):
                raise RuntimeError("Selected torch HIP ABI differs from pinned Schaka lane: " +
                                   repr(torch.version.hip) + " versus " + repr(hip_prefix))
            from scripts.gfx803_repair_runtime import verify_loaded_sgemm_shim
            metadata["shim_runtime_before"]=verify_loaded_sgemm_shim(expected_shim, require_rocblas=False)
        teacher,student=load_pair(args.teacher,args.student,args.config)
        write_progress(args.progress_file,"model-load-start")
        metadata["checks_result"]=run_gate(teacher,student,
            raw[:args.probe_batch],layer=args.layer,repeat=args.repeats,
            stability_steps=args.stability_steps,dense_only=args.dense_only,
            stop_after_c2_forward=args.stop_after_c2_forward,
            progress_path=args.progress_file,router_selector=args.router_selector,
            stop_after_c2_backward=args.stop_after_c2_backward)
        metadata["admitted"]=bool(metadata["checks_result"]["admitted"])
        if repair:
            from scripts.gfx803_linear_repair import repair_receipt
            metadata["linear_repair"]=repair_receipt()
            if metadata["linear_repair"]["gpu_linear_calls_repaired"] == 0:
                raise RuntimeError("CPU linear repair was requested but no GPU linear executed")
            if os.environ.get("GFX803_WGM1_QKV_JOIN"):
                qkv=(metadata["linear_repair"].get("wgm1_qkv") or {})
                if not isinstance(qkv.get("forced_qkv_gpu_forward_calls"),int) or \
                        qkv["forced_qkv_gpu_forward_calls"]<1:
                    raise RuntimeError("WGM1 selected but no original QKV GPU dispatch observed")
            # Hybrid numerical triage, not a fully GPU-resident LILA solution.
            metadata["admitted"]=False
            metadata["gate_only_hybrid"]=True
            metadata["checks_result"]["scope_note"]="Hybrid CPU GEMM diagnostic only"
        if expected_shim:
            from scripts.gfx803_repair_runtime import verify_loaded_sgemm_shim
            metadata["shim_runtime_after"]=verify_loaded_sgemm_shim(expected_shim)
    except Exception as exc:
        metadata["error"]=f"{type(exc).__name__}: {exc}"
        metadata["traceback_tail"]=traceback.format_exc()[-1800:]
    finally:
        if os.environ.get("GFX803_LINEAR_REPAIR") == "cpu-fp32":
            from scripts.gfx803_linear_repair import repair_receipt
            metadata["linear_repair"]=repair_receipt()
            metadata["admitted"]=False  # no hybrid GPU training promotion
        metadata["effective_runtime_attestation"]=attest_runtime()
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(metadata,indent=2)+"\n")
        write_progress(args.progress_file,"python-finally",receipt=str(args.receipt),
                       error=metadata.get("error"),
                       result_stage=metadata.get("checks_result",{}).get("stage"))
    print("[gfx803-gate] receipt saved; training remains blocked unless admitted",flush=True)
    print(json.dumps({"admitted":metadata["admitted"],
                      "receipt":str(args.receipt),
                      "stage":metadata.get("checks_result",{}).get("stage"),
                      "error":metadata.get("error")},indent=2),flush=True)
    return 0 if metadata["admitted"] else 2


if __name__=="__main__":
    raise SystemExit(main())
