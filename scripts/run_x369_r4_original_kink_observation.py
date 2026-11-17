#!/usr/bin/env python3
"""One-entry original fitted R4 source/ResponseKink observation pipeline.

Runs original source-owner modules in separate Python invocations:
1. repaired C2 + native SLR + fitted R4 delta strict restoration,
   with one original development document/token x + full all-E H.
2. exact original HistoryOrbitMoE four-load source vs forced topk.
3. exact original same-clock two-prefix history source vs fixed pair.

The result is an OBSERVATION receipt, positive OR negative.
No synthetic fallback, no F5, no fitted behavior claimed without bytes,
no Agda/float error/semantic/hysteresis proof by orchestration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    root=Path(__file__).resolve().parents[1]
    # Fail before touching outputs if any original owned input or
    # independently supplied hash is absent/inconsistent.
    owned=(args.repaired_c2_checkpoint,args.source_repair_receipt,
           args.native_slr_jsonl,args.fitted_delta,args.config,
           args.r1_summary,args.tokenizer)
    if any(not p.is_file() for p in owned):
        raise FileNotFoundError("original owned source assets absent")
    for path,expected in (
        (args.repaired_c2_checkpoint,args.repaired_c2_sha256),
        (args.native_slr_jsonl,args.native_slr_jsonl_sha256),
        (args.fitted_delta,args.fitted_delta_sha256)):
        if (len(expected)!=64 or any(c not in "0123456789abcdef" for c in expected)
            or sha(path)!=expected):
            raise ValueError("original owned checkpoint/SLR/delta SHA mismatch")
    if type(args.nat_scale) is not int or not 1<=args.nat_scale<=10**9:
        raise ValueError("invalid source Nat scale")
    out=args.output_dir
    if out.exists():
        raise FileExistsError("never overwrite original R4 observation directory")
    if args.layer_index<0:
        raise ValueError("original source layer index must be nonnegative")
    out.mkdir(parents=True,exist_ok=False)
    module=out/"restored_original_r4_layer.pt"
    source_receipt=out/"restored_original_r4_layer.json"
    probe=out/"original_native_slr_x_all_E_H.json"
    reset=out/"original_four_load_probe.json"
    histories=out/"original_same_clock_history_probe.json"
    final=out/"original_r4_kink_observation.json"

    def call(script,parameters):
        subprocess.run(
            [sys.executable,str(root/"scripts"/script),*parameters],
            cwd=root,check=True)
    call("export_x369_r4_restored_moe_layer.py",[
        "--repaired-c2-checkpoint",str(args.repaired_c2_checkpoint.resolve()),
        "--repaired-c2-sha256",args.repaired_c2_sha256,
        "--source-repair-receipt",str(args.source_repair_receipt.resolve()),
        "--native-slr-jsonl",str(args.native_slr_jsonl.resolve()),
        "--native-slr-jsonl-sha256",args.native_slr_jsonl_sha256,
        "--fitted-delta",str(args.fitted_delta.resolve()),
        "--fitted-delta-sha256",args.fitted_delta_sha256,
        "--config",str(args.config.resolve()),
        "--layer-index",str(args.layer_index),
        "--output-checkpoint",str(module.resolve()),
        "--output-receipt",str(source_receipt.resolve()),
        "--output-probe-input",str(probe.resolve()),
        "--r1-summary",str(args.r1_summary.resolve()),
        "--tokenizer",str(args.tokenizer.resolve()),
        "--document-id",str(args.document_id),
        "--token-position",str(args.token_position),
        "--load-levels",args.load_levels,
        "--response-kind",args.response_kind,
        "--response-index",str(args.response_index),
        "--nat-scale",str(args.nat_scale)])
    for script,target in [
       ("probe_x369_r4_kink_checkpoint.py",reset),
       ("probe_x369_r4_history_path_dependence.py",histories)]:
        call(script,[
            "--checkpoint",str(module.resolve()),
            "--input",str(probe.resolve()),
            "--layer-receipt",str(source_receipt.resolve()),
            "--output",str(target.resolve())])
    rec=json.loads(source_receipt.read_text(encoding="utf8"))
    src=json.loads(probe.read_text(encoding="utf8"))
    four=json.loads(reset.read_text(encoding="utf8"))
    path=json.loads(histories.read_text(encoding="utf8"))
    if (not four.get("source_input_bound_to_original_R4_export")
        or not path.get("source_input_bound_to_original_R4_export")
        or rec.get("module_checkpoint_sha256")!=sha(module)
        or rec.get("source_probe_input_sha256")!=sha(probe)
        or four.get("original_checkpoint_sha256")!=sha(module)
        or path.get("original_checkpoint_sha256")!=sha(module)
        or four.get("source_input_sha256")!=sha(probe)
        or path.get("source_input_sha256")!=sha(probe)
        or src.get("source_record_split")!="development"
        or not src.get("source_step_and_standalone_MoE_bitwise_replay_verified")):
        raise RuntimeError("original source observation chain not same-object")
    result={
       "schema":"x369_r4_native_observation_v1",
       "kind":"original_R4_development_reconstructed_source_observation",
       "source_fitted_R4_delta_replayed":True,
       "source_native_SLR_development_state_replayed":True,
       "source_module_checkpoint_sha256":sha(module),
       "source_probe_input_sha256":sha(probe),
       "source_checkpoint_export_receipt_sha256":sha(source_receipt),
       "original_four_load_result_sha256":sha(reset),
       "original_same_clock_history_result_sha256":sha(histories),
       "source_document_id":src["source_document_id"],
       "source_document_sha256":src["source_document_sha256"],
       "source_token_position":src["source_token_position"],
       "source_moe_layer_index":args.layer_index,
       "nat_response":[st["source_nat_response"] for st in four["stages"]],
       "four_load_kink_observed":four["nat_kink_observed"],
       "four_load_fixed_pair_kink_observed":
           four["fixed_route_nat_kink_observed"],
       "same_clock_history_response_difference_observed":
           path["same_load_and_clock_history_dependent_response_observed"],
       "same_clock_fixed_pair_ablation_difference_observed":
           path["source_selection_ablated_history_path_difference"],
       "original_checkpoint_and_trace_bytes_checked":True,
       "fitted_numerical_error_bounds_certified":False,
       "original_corpus_heldout_independence_certified":False,
       "general_hysteresis_certified":False,
       "genuine_task_semantic_action_certified":False,
       "agda_kernel_accepted_by_this_pipeline":False,
       "experimental_positive_result_required":False,
       "F5_or_reserved_data_accessed":False}
    final.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                     encoding="utf8")
    print("X369_R4_NATIVE_ORIGINAL_OBSERVATION_COMPLETE",
          final,
          "four_load_kink_observed",result["four_load_kink_observed"],
          "history_dependence_observed",
          result["same_clock_history_response_difference_observed"])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repaired-c2-checkpoint",type=Path,required=True)
    p.add_argument("--repaired-c2-sha256",required=True)
    p.add_argument("--source-repair-receipt",type=Path,required=True)
    p.add_argument("--native-slr-jsonl",type=Path,required=True)
    p.add_argument("--native-slr-jsonl-sha256",required=True)
    p.add_argument("--fitted-delta",type=Path,required=True)
    p.add_argument("--fitted-delta-sha256",required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--r1-summary",type=Path,required=True)
    p.add_argument("--tokenizer",type=Path,required=True)
    p.add_argument("--document-id",required=True)
    p.add_argument("--token-position",type=int,required=True)
    p.add_argument("--layer-index",type=int,required=True)
    p.add_argument("--load-levels",required=True)
    p.add_argument("--response-kind",choices=("coordinate","l2"),required=True)
    p.add_argument("--response-index",type=int,default=0)
    p.add_argument("--nat-scale",type=int,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
