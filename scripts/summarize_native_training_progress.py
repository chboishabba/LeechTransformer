"""Summarize append-only native training progress without inventing GPU metrics.

Usage:
 python3 scripts/summarize_native_training_progress.py \
   --progress artifacts/dense.progress.jsonl artifacts/c2.progress.jsonl \
   --output artifacts/native_training_summary.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def summarize_one(path: Path):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty progress receipt: " + str(path))
    last=rows[-1]
    expected=last["steps_expected"]
    if (any(r.get("schema")!=1 or r.get("kind")!="native_training_progress"
            for r in rows) or
        any(b["steps_completed"]<=a["steps_completed"]
            for a,b in zip(rows,rows[1:])) or
        last["steps_completed"]>expected):
        raise ValueError("invalid or nonmonotone progress receipt: "+str(path))
    return {
        "path":str(path.resolve()),
        "label":last["label"], "device":last["device"],
        "complete":last["steps_completed"]==expected,
        "steps_completed":last["steps_completed"],
        "steps_expected":expected,
        "target_tokens_completed":last["target_tokens_completed"],
        "measured_loop_elapsed_seconds":last["elapsed_train_seconds"],
        "average_loop_steps_per_second":last["steps_per_second"],
        "average_loop_target_tokens_per_second":last["target_tokens_per_second"],
        "last_interval_steps_per_second":last["interval_steps_per_second"],
        "last_interval_target_tokens_per_second":last["interval_target_tokens_per_second"],
        "peak_pytorch_gpu_allocated_bytes":last.get("gpu_peak_allocated_bytes"),
        "peak_pytorch_gpu_reserved_bytes":last.get("gpu_peak_reserved_bytes"),
        "timing_scope":last.get("timing_scope"),
        "training_only_not_quality_or_federation":True
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--progress",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    report={"schema":1,"runs":[summarize_one(path) for path in args.progress],
            "quality_comparison_computed":False,
            "physical_network_throughput_measured":False,
            "claim":"Step throughput and allocator memory only; do not infer student-only CE, model superiority or whole-program wall time."}
    data=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists():
            raise FileExistsError("refuse to overwrite summary "+str(args.output))
        args.output.write_text(data)
    print(data,end="")


if __name__=="__main__":
    main()
