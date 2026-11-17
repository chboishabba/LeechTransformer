#!/usr/bin/env python3
"""Export ONE genuine R4 HistoryOrbitMoE layer from original bound assets.

The original R4b source training output is not a monolithic R4 checkpoint:
it is a fitted residual delta that MUST be welded back to the authenticated
frozen repaired C2 source with the exact native SLR provenance. This exporter
reuses the ORIGINAL checkpoint_parts/infer_config/restore_residual_delta owners.

No R4 network/weight/hidden-history is fabricated. Source-independent
hidden histories and activations must be exported separately from the
actual original Leech trace and supplied to the load-response probe.
No F5/reserved data is opened.

Usage:
 python scripts/export_x369_r4_restored_moe_layer.py \
    --repaired-c2-checkpoint PATH --source-repair-receipt PATH \
    --native-slr-jsonl PATH --fitted-delta PATH \
    --fitted-delta-sha256 HEX --config PATH --layer-index N \
    --output-checkpoint PATH --output-receipt PATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.model import LeechGPT
from models.history_orbit_moe import HistoryOrbitMoE
from models.x369_fair_c3_residual import (
    restore_residual_delta, r4_geometry_source_fingerprint)
from scripts.verify_trained_federated_moe import checkpoint_parts, infer_config


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    trace_enabled=args.output_probe_input is not None
    mandatory=(args.r1_summary,args.tokenizer,args.document_id,
               args.token_position,args.load_levels,args.response_kind,
               args.nat_scale)
    if trace_enabled and any(v is None for v in mandatory):
        raise ValueError("source probe capture requires complete original R1/SLR inputs")
    if not trace_enabled and any(v is not None for v in mandatory):
        raise ValueError("source capture fields require --output-probe-input")
    outputs=(args.output_checkpoint,args.output_receipt)+(
        (args.output_probe_input,) if trace_enabled else ())
    if args.output_checkpoint.resolve()==args.output_receipt.resolve():
        raise ValueError("checkpoint and receipt must have different paths")
    if any(p.exists() for p in outputs):
        raise FileExistsError("never overwrite an existing original source receipt")
    source_sha=digest(args.repaired_c2_checkpoint)
    delta_sha=digest(args.fitted_delta)
    native_sha=digest(args.native_slr_jsonl)
    if (source_sha!=args.repaired_c2_sha256
        or delta_sha!=args.fitted_delta_sha256
        or native_sha!=args.native_slr_jsonl_sha256):
        raise ValueError("original C2/delta/native SLR byte identity mismatch")
    repair=json.loads(args.source_repair_receipt.read_text(encoding="utf8"))
    if (repair.get("kind")!="native_c2_bounded_early_even_warmstart"
        or repair.get("output_sha256")!=source_sha):
        raise ValueError("original repaired C2 source receipt mismatch")
    state, metadata=checkpoint_parts(args.repaired_c2_checkpoint)
    cfg=infer_config(state,metadata,args.config)
    if (cfg.moe_route_geometry!="c2_invariant" or not cfg.moe_guarded_ffn
        or cfg.moe_shared_e8_history or cfg.moe_late_history_cap!=64.):
        raise ValueError("R4 producer needs genuine repaired bounded C2 source")
    if not 0<=args.layer_index<cfg.n_layers:
        raise ValueError("selected original MoE layer index out of bounds")
    source=LeechGPT(cfg).eval()
    source.load_state_dict(state,strict=True)
    payload=torch.load(args.fitted_delta,map_location="cpu",weights_only=True)
    restored=restore_residual_delta(
        source,payload,source_sha256=source_sha,
        source_jsonl_sha256=native_sha)
    if restored.cfg.moe_route_geometry not in (
        "c2_c3_residual","c2_c3_mixed_null_residual","c2_c2_residual"):
        raise ValueError("restored model is not an original R4 residual arm")
    moe=restored.blocks[args.layer_index].moe
    if not isinstance(moe,HistoryOrbitMoE):
        raise TypeError("source model layer does not own HistoryOrbitMoE")
    if moe.router_selection!="topk":
        raise ValueError("only original topk model supported")
    if (moe.d_model!=cfg.d_model or moe.num_experts!=cfg.moe_experts
        or moe.top_k!=cfg.moe_top_k
        or moe.route_geometry!=restored.cfg.moe_route_geometry):
        raise ValueError("restored original R4 module contract mismatch")
    dtype=next(moe.parameters()).dtype
    if dtype not in (torch.float32,torch.float64):
        raise ValueError("source layer dtype must be float32 or float64")
    module_state={k:v.detach().cpu().clone()
                  for k,v in moe.state_dict().items()}
    if not module_state or not all(bool(torch.isfinite(v).all())
                                    for v in module_state.values()):
        raise ValueError("nonfinite or missing source module parameters")
    # The original HistoryOrbitMoE.history_cap is a NONPERSISTENT
    # source config value and MUST be included with the export.
    module_cfg={
        "d_model":moe.d_model,"num_experts":moe.num_experts,
        "top_k":moe.top_k,
        "expert_width":moe.experts[0][0].out_features,
        "temperature":float(moe.temperature),
        "route_geometry":moe.route_geometry,
        "transition_rank":moe.transition_rank,
        "router_selection":moe.router_selection}
    expected=HistoryOrbitMoE(**module_cfg).to(dtype=dtype).eval()
    expected.history_cap=moe.history_cap
    expected.load_state_dict(module_state,strict=True)
    for key,value in expected.state_dict().items():
        if not torch.equal(value,module_state[key]):
            raise AssertionError("independent source module restore differs: "+key)
    args.output_checkpoint.parent.mkdir(parents=True,exist_ok=True)
    args.output_receipt.parent.mkdir(parents=True,exist_ok=True)
    # Optional native DEVELOPMENT state: reuse the strict original
    # C2+delta restored model, source loader and scoped source step
    # observation. The source and exact MoE replay must be BITWISE
    # equal before any export bytes are written.
    if trace_enabled:
        from scripts.capture_x369_r4_original_slr_state import (
            capture_original_slr_development_probe)
        trace=capture_original_slr_development_probe(
            restored,moe,module_cfg,dtype,
            layer_index=args.layer_index,
            native_slr_jsonl=args.native_slr_jsonl,
            native_sha=native_sha,r1_summary=args.r1_summary,
            tokenizer=args.tokenizer,document_id=args.document_id,
            token_position=args.token_position,
            load_levels=args.load_levels,
            response_kind=args.response_kind,
            response_index=args.response_index,
            nat_scale=args.nat_scale)
    else:
        trace=None
    # Write the source-owned state ONLY after all restoration gates.
    torch.save(module_state,args.output_checkpoint)
    checkpoint_hash=digest(args.output_checkpoint)
    receipt={
      "schema":"x369_r4_restored_original_moe_layer_v1",
      "kind":"strict_original_C2_plus_fitted_R4_delta_layer_export",
      "source_repaired_c2_checkpoint_sha256":source_sha,
      "source_repair_receipt_sha256":digest(args.source_repair_receipt),
      "source_native_slr_jsonl_sha256":native_sha,
      "original_fitted_R4_delta_sha256":delta_sha,
      "original_R4_geometry_source_sha256":r4_geometry_source_fingerprint(),
      "restored_full_source_delta_verified":True,
      "source_model_layer_index":args.layer_index,
      "module_checkpoint_sha256":checkpoint_hash,
      "model":module_cfg,
      "dtype":str(dtype).split(".")[-1],
      "history_cap":moe.history_cap,
      "original_source_all_E_histories_included":trace_enabled,
      "original_source_activation_trace_included":trace_enabled,
      "original_source_development_step_replay_verified":trace_enabled,
      "fitted_model_output_or_kink_certified":False,
      "floating_error_intervals_certified":False,
      "source_test_split_independence_independently_certified":False,
      "F5_or_reserved_data_accessed":False
    }
    if trace is not None:
        args.output_probe_input.parent.mkdir(parents=True,exist_ok=True)
        args.output_probe_input.write_text(
            json.dumps(trace,indent=2,sort_keys=True,allow_nan=False)+"\\n",
            encoding="utf8")
        receipt["source_probe_input_sha256"]=digest(args.output_probe_input)
        receipt["source_document_id"]=trace["source_document_id"]
        receipt["source_document_sha256"]=trace["source_document_sha256"]
        receipt["source_token_position"]=args.token_position
        receipt["source_step_vs_exported_module_bitwise_equal"]=True
    args.output_receipt.write_text(
        json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n",
        encoding="utf8")
    print("X369_R4_ORIGINAL_BOUND_LAYER_EXPORTED",
          args.output_checkpoint,checkpoint_hash,
          "layer",args.layer_index)
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repaired-c2-checkpoint",required=True,type=Path)
    p.add_argument("--repaired-c2-sha256",required=True)
    p.add_argument("--source-repair-receipt",required=True,type=Path)
    p.add_argument("--native-slr-jsonl",required=True,type=Path)
    p.add_argument("--native-slr-jsonl-sha256",required=True)
    p.add_argument("--fitted-delta",required=True,type=Path)
    p.add_argument("--fitted-delta-sha256",required=True)
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--layer-index",required=True,type=int)
    p.add_argument("--output-checkpoint",required=True,type=Path)
    p.add_argument("--output-receipt",required=True,type=Path)
    p.add_argument("--output-probe-input",type=Path)
    p.add_argument("--r1-summary",type=Path)
    p.add_argument("--tokenizer",type=Path)
    p.add_argument("--document-id")
    p.add_argument("--token-position",type=int)
    p.add_argument("--load-levels",
                   help="four increasing original x multipliers, including 1")
    p.add_argument("--response-kind",choices=("coordinate","l2"))
    p.add_argument("--response-index",type=int,default=0)
    p.add_argument("--nat-scale",type=int)
    run(p.parse_args())


if __name__=="__main__":
    main()
