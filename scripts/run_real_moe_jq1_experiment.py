"""One-command REAL text -> trained signed MoE -> JQ1 experiment.

The source dataset is TinyStories or FineWeb-Edu using the ORIGINAL 2048
SentencePiece tokenizer from the repo. The real best_model.pt is frozen
and its weights are reused as teacher. Runs same data/epochs/seed on
c2_invariant and punctured9_wave student configurations, exports ALL
layers, scores heldout next-token CE, captures recurrent expert state
and measures full bidirectional JQ1 costs at 1/10 Gb/s.

This is a foreground orchestration of the existing CLI tools, not a
background process, inference of pretrained MoE, or claimed speedup.
The full student MUST still achieve task-quality criteria to be promoted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def _run(args):
    print("+ "+" ".join(map(str,args)),flush=True)
    subprocess.run([str(x) for x in args],check=True)


def _digest(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk=f.read(1<<20)
            if not chunk:break
            h.update(chunk)
    return h.hexdigest()


def experiment(args):
    if args.epochs<1 or args.train_rows<2 or args.heldout_rows<2:
        raise ValueError("real experiment requires nonempty independent splits")
    if args.layer<0 or args.batch<1 or args.token_length<2:
        raise ValueError("invalid layer/batch/time")
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    data=(args.reuse_splits_dir if args.reuse_splits_dir is not None
          else out/"dataset")
    prep=[sys.executable,"-m","scripts.prepare_real_leech_tokens",
        "--source",args.source,"--output-dir",data,
        "--train-rows",args.train_rows,"--heldout-rows",args.heldout_rows,
        "--length",args.token_length,"--seed",args.seed,
        "--expected-vocab",args.expected_vocab]
    if args.tokenizer:prep+=["--tokenizer",args.tokenizer]
    if args.fineweb_config:prep+=["--fineweb-config",args.fineweb_config]
    if args.dataset_revision:prep+=["--dataset-revision",args.dataset_revision]
    if args.source=="local-text":
        if not args.train_text or not args.heldout_text:
            raise ValueError("two separate local real text sources are required")
        prep+=["--train-text",args.train_text,
               "--heldout-text",args.heldout_text]
    if args.reuse_splits_dir is None:
        _run(prep)
    provenance=json.loads((data/"provenance.json").read_text())
    if args.reuse_splits_dir is not None:
        for split in ("train","heldout"):
            path=data/f"{split}_ids.npy"
            values=np.load(path,allow_pickle=False)
            expected=provenance["split_audit"][split]["window_array_sha256"]
            if hashlib.sha256(values.tobytes()).hexdigest()!=expected:
                raise ValueError("reused split no longer matches original provenance: "+split)
    if provenance.get("not_random_token_smoke") is not True:
        raise ValueError("refuse to label synthetic token IDs as real")
    summary={
        "experiment":"real-data trained signed-wave MoE",
        "dense_checkpoint":str(args.checkpoint.resolve()),
        "checkpoint_sha256":_digest(args.checkpoint),
        "training_corpus":provenance,
        "geometry_comparisons":{},
        "data_split_is_disjoint":True}
    modes=[(geom,False) for geom in args.route_geometries]
    if args.include_shared_e8:
        modes.append(("e8_root_invariant",True))
    for geom,shared in modes:
        label="e8_shared_canonical" if shared else geom
        prefix=out/label
        prefix.mkdir(exist_ok=True)
        adapter=prefix/"trained_adapters.pt"
        student=prefix/"student_model.pt"
        capture=prefix/f"capture_layer{args.layer}.npz"
        command=[sys.executable,"-m","scripts.train_signed_wave_experts",
            "--checkpoint",args.checkpoint,"--config",args.config,
            "--token-ids",data/"train_ids.npy","--layer","-1",
            "--epochs",args.epochs,"--batch",args.batch,"--seed",args.seed,
            "--experts",args.experts,"--top-k",args.top_k,
            "--expert-width",args.expert_width,
            "--transition-rank",args.transition_rank,
            "--route-geometry",geom,"--device",args.device,
            "--output",adapter]
        if shared:
            command.append("--shared-e8-history")
        _run(command)
        _run([sys.executable,"-m","scripts.export_signed_wave_model",
            "--checkpoint",args.checkpoint,"--config",args.config,
            "--adapters",adapter,
            "--heldout-token-ids",data/"heldout_ids.npy",
            "--device",args.device,"--batch",args.batch,
            "--output",student])
        quality=json.loads(student.with_suffix(".json").read_text())
        guarded=None
        teacher_free={"attempted":False,"reason":"guarded continuation not selected"}
        if args.guarded_continuation:
            guard_file=prefix/"guarded_model.pt"
            _run([sys.executable,"-m","scripts.train_guarded_moe_continuation",
                "--teacher",args.checkpoint,"--student",student,
                "--config",args.config,
                "--train-ids",data/"train_ids.npy",
                "--heldout-ids",data/"heldout_ids.npy",
                "--epochs",args.guarded_epochs,
                "--calibration-budget-nats",args.calibration_budget_nats,
                "--batch",args.batch,"--device",args.device,
                "--output",guard_file])
            guarded=json.loads(guard_file.with_suffix(".json").read_text())
            if guarded.get("full_student_replacement_pass"):
                # Do not count a fully opened *hybrid* as a teacher-free
                # model until strict stripping and fresh parity checks pass.
                standalone=prefix/"teacher_free_model.pt"
                _run([sys.executable,"-m","scripts.promote_guarded_moe",
                    "--teacher",args.checkpoint,
                    "--guarded",guard_file,
                    "--config",args.config,
                    "--heldout-ids",data/"heldout_ids.npy",
                    "--ce-budget-nats",args.calibration_budget_nats,
                    "--batch",args.batch,"--device",args.device,
                    "--output",standalone])
                teacher_free=json.loads(standalone.with_suffix(".json").read_text())
                teacher_free["attempted"]=True
                teacher_free["model_path"]=str(standalone)
            else:
                teacher_free={"attempted":False,
                    "reason":"not all quality gates reach one with independent heldout quality; dense teacher retained"}
        _run([sys.executable,"-m","scripts.capture_joint_moe_state",
            "--checkpoint",student,"--token-ids",data/"heldout_ids.npy",
            "--layer",args.layer,"--batch",args.batch,
            "--device",args.device,"--output",capture])
        comparisons={}
        for placement in ("source","peer"):
            report=prefix/f"jq1_{placement}.json"
            bench=[sys.executable,"-m","scripts.benchmark_joint_state_quotient",
                "--capture",capture,"--history-placement",placement,
                "--rtt-ms",args.rtt_ms,"--output",report]
            if shared:bench.append("--e8-orbit-actions")
            _run(bench)
            comparisons[placement]=json.loads(report.read_text())
        summary["geometry_comparisons"][label]={
            "adapter_path":str(adapter),"student_path":str(student),
            "shared_canonical_e8_history":shared,
            "heldout_quality":quality,
            "guarded_quality":guarded,
            "teacher_free_promotion":teacher_free,
            "joint_state_capture":str(capture),
            "transport":comparisons}
        (out/"comparison.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({
        "comparison":str(out/"comparison.json"),
        "geometry_modes":list(summary["geometry_comparisons"]),
        "verification":"real sample and checkpoint; performance is empirical, no unproved semantic/j/Leech identity"
        },indent=2))
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True,type=Path)
    ap.add_argument("--config",required=True,type=Path)
    ap.add_argument("--output-dir",default=Path("artifacts/real_moe_jq1"),type=Path)
    ap.add_argument("--reuse-splits-dir",type=Path,
                    help="reuse provenance-verified saved tokens; avoid rerunning dataset subprocess")
    ap.add_argument("--source",choices=("tinystories","fineweb-edu","local-text"),
                    default="tinystories")
    ap.add_argument("--tokenizer",type=Path)
    ap.add_argument("--fineweb-config",default="sample-10BT")
    ap.add_argument("--dataset-revision")
    ap.add_argument("--train-text",type=Path)
    ap.add_argument("--heldout-text",type=Path)
    ap.add_argument("--train-rows",type=int,default=1024)
    ap.add_argument("--heldout-rows",type=int,default=256)
    ap.add_argument("--token-length",type=int,default=128)
    ap.add_argument("--expected-vocab",type=int,default=2048)
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--batch",type=int,default=2)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--seed",type=int,default=41)
    ap.add_argument("--experts",type=int,default=4)
    ap.add_argument("--top-k",type=int,default=2)
    ap.add_argument("--expert-width",type=int,default=192)
    ap.add_argument("--transition-rank",type=int,default=24)
    ap.add_argument("--route-geometries",nargs="+",
        choices=("c2_invariant","c2c3_wave","punctured9_wave","e8_root_invariant"),
        default=("c2_invariant","e8_root_invariant","punctured9_wave"))
    ap.add_argument("--include-shared-e8",action="store_true",
                    help="add E8 canonical shared-history ablation and E8 JQ1 predictors")
    ap.add_argument("--guarded-continuation",action="store_true",
                    help="also train teacher-retaining quality gates against separate train calibration")
    ap.add_argument("--guarded-epochs",type=int,default=3)
    ap.add_argument("--calibration-budget-nats",type=float,default=.02)
    ap.add_argument("--layer",type=int,default=0)
    ap.add_argument("--rtt-ms",type=float,default=5.)
    args=ap.parse_args()
    experiment(args)


if __name__=="__main__":
    main()
