"""Nonpromotional independent source/receipt verifier for 100+100 LILA smoke.

Validates exact original source hashes, two physical-process receipts,
all eight 25-step checkpoint hashes, immutable save/reload evidence,
exact resumed parent checkpoint and output identity. Never launches GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.gfx803_moe_gpu_gate import file_sha256


def verify(first_path:Path, second_path:Path):
    one=json.loads(first_path.read_text())
    two=json.loads(second_path.read_text())
    for label,r in (("first",one),("resumed",two)):
        if r.get("status") not in ("partial-restart-required","complete"):
            raise ValueError(label+": incomplete/failed training receipt")
        if not r.get("native_gpu_only") or not r.get("no_c2"):
            raise ValueError(label+": wrong trainer/run class")
        if r.get("settings",{}).get("target_steps")!=200 or r.get(
                "settings",{}).get("save_every")!=25:
            raise ValueError(label+": not requested 200-step/25-save contract")
        gpu=r.get("gpu",{})
        if gpu.get("architecture") not in (None,"gfx803") or not gpu.get("hip"):
            raise ValueError(label+": missing attested gfx803 HIP GPU")
    if one["settings"]!=two["settings"]:
        raise ValueError("original checkpoint/tokens or training hyperparameters changed")
    if (one["status"]!="partial-restart-required"
            or one["start_step"]!=0 or one["end_step"]!=100
            or one["steps_this_process"]!=100
            or one.get("checkpoint_resume_verified")):
        raise ValueError("first process must stop exactly at step 100")
    if (two["status"]!="complete"
            or two["start_step"]!=100 or two["end_step"]!=200
            or two["steps_this_process"]!=100
            or not two.get("checkpoint_resume_verified")
            or not two.get("resume_verified")
            or two.get("resumed_step")!=100):
        raise ValueError("second process did not resume once at 100 and finish 200")
    if len(one.get("trace",[]))!=100 or len(two.get("trace",[]))!=200:
        raise ValueError("incomplete real training metrics")
    for index,(x,y) in enumerate(zip(one["trace"],two["trace"][:100]),1):
        for key in ("step","train_loss_nats","grad_norm_preclip",
                    "target_tokens","row_indices"):
            if x.get(key)!=y.get(key):
                raise ValueError(f"resumed trace differs from original at {index}: {key}")
    for i,row in enumerate(two["trace"],1):
        if row.get("step")!=i or row.get("target_tokens",0)<=0 or not (
                0<=row.get("train_loss_nats",float("nan"))<float("inf")):
            raise ValueError("invalid step/finite-loss/real-token metric: "+str(i))
    cp1=one.get("checkpoints_written",[])
    cp2=two.get("checkpoints_written",[])
    if ([r.get("step") for r in cp1]!=[25,50,75,100] or
        [r.get("step") for r in cp2]!=[125,150,175,200]):
        raise ValueError("missing or out-of-order 25-step checkpoint")
    for entry in cp1+cp2:
        path=Path(entry["path"])
        if (not entry.get("reload_verified") or not path.is_file()
                or file_sha256(path)!=entry.get("sha256")):
            raise ValueError("checkpoint missing, corrupted or not reloaded: "+str(path))
    if one["settings"]["teacher_sha256"]!=two["settings"]["teacher_sha256"]:
        raise ValueError("source teacher changed")
    if two.get("resumed_checkpoint_sha256")!=cp1[-1]["sha256"]:
        raise ValueError("second run did not resume from exact first step-100 file")
    if not (one.get("mean_training_tokens_per_second",0)>0 and
            two.get("mean_training_tokens_per_second",0)>0):
        raise ValueError("missing real tokens/s")
    if not (one.get("peak_allocated_bytes",0)>0 and
            two.get("peak_allocated_bytes",0)>0):
        raise ValueError("missing GPU VRAM metric")
    return {
        "verified":True,"scope":"200-step original dense LILA GPU-native smoke only",
        "teacher_sha256":one["settings"]["teacher_sha256"],
        "train_tokens_sha256":one["settings"]["train_tokens_sha256"],
        "steps":200,"save_interval":25,"checkpoints_verified":8,
        "process_boundary_resume_step":100,
        "final_checkpoint_sha256":cp2[-1]["sha256"],
        "first_process_mean_tok_s":one["mean_training_tokens_per_second"],
        "resumed_process_mean_tok_s":two["mean_training_tokens_per_second"],
        "peak_allocated_bytes":max(one["peak_allocated_bytes"],
                                   two["peak_allocated_bytes"]),
        "first_loss_nats":two["trace"][0]["train_loss_nats"],
        "last_loss_nats":two["trace"][-1]["train_loss_nats"],
        "not_claimed":["independent heldout improvement",
                       "full C2 student quality","multi-hour reset safety",
                       "ROCm/HIP universal correctness"],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--first",type=Path,required=True)
    p.add_argument("--resumed",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to replace original verifier evidence")
    result=verify(args.first,args.resumed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
