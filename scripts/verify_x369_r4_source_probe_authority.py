"""Bind a native original R4 x/H probe to its exact restored module bytes.

This checks identity of exported source artifacts. It does NOT sign receipts,
prove source-model quality, certify floating roundoff or infer R1 split
independence. The exporter itself must use the original verified R1,
repaired C2 and fitted delta restoration producers.
"""
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def verify_original_probe_authority(checkpoint,input_path,request,
                                    history,layer_receipt=None):
    native=request.get("source_record_split")=="development"
    if layer_receipt is None:
        if native or request.get(
                "source_step_and_standalone_MoE_bitwise_replay_verified"):
            raise ValueError("original native SLR x/H requires bound layer receipt")
        return {
            "source_input_bound_to_original_R4_export":False,
            "source_input_type":"unbound_or_synthetic",
            "source_provenance_authenticated_by_this_check":False}
    receipt=json.loads(layer_receipt.read_text(encoding="utf8"))
    if (receipt.get("schema")!="x369_r4_restored_original_moe_layer_v1"
        or receipt.get("kind")!="strict_original_C2_plus_fitted_R4_delta_layer_export"
        or receipt.get("restored_full_source_delta_verified") is not True
        or receipt.get("original_source_all_E_histories_included") is not True
        or receipt.get("original_source_development_step_replay_verified") is not True
        or receipt.get("source_step_vs_exported_module_bitwise_equal") is not True):
        raise ValueError("original C2+R4 restoration/trace receipt missing")
    if (not native or
        request.get("source_step_and_standalone_MoE_bitwise_replay_verified")
          is not True or
        request.get("F5_or_reserved_data_read") is not False):
        raise ValueError("non-native or reserved source cannot use bound receipt")
    if (receipt.get("module_checkpoint_sha256")!=sha256(checkpoint)
        or receipt.get("source_probe_input_sha256")!=sha256(input_path)):
        raise ValueError("restored checkpoint or source trace byte hash mismatch")
    if (request.get("model")!=receipt.get("model")
        or request.get("dtype")!=receipt.get("dtype")
        or request.get("history_cap")!=receipt.get("history_cap")
        or str(request.get("source_document_id"))!=str(
            receipt.get("source_document_id"))
        or request.get("source_document_sha256")!=receipt.get("source_document_sha256")
        or request.get("source_token_position")!=receipt.get("source_token_position")
        or request.get("source_native_slr_jsonl_sha256")!=receipt.get(
            "source_native_slr_jsonl_sha256")):
        raise ValueError("R4 source model/document/revision mismatch")
    source_history_hash=hashlib.sha256(
        history.contiguous().numpy().tobytes()).hexdigest()
    if (source_history_hash!=request.get("source_history_tensor_sha256")
        or not isinstance(request.get("history_revision"),str)
        or source_history_hash not in request["history_revision"]):
        raise ValueError("original all-E history is not the source trace")
    return {
       "source_input_bound_to_original_R4_export":True,
       "source_input_type":"native_slr_development",
       # A source receipt provides a checked LOCAL chain, not an external
       # cryptographic signature or independent heldout provenance.
       "source_provenance_authenticated_by_this_check":False,
       "source_bound_checkpoint_sha256":receipt["module_checkpoint_sha256"],
       "source_bound_probe_input_sha256":receipt["source_probe_input_sha256"],
       "source_bound_original_C2_sha256":
           receipt["source_repaired_c2_checkpoint_sha256"],
       "source_bound_R4_fitted_delta_sha256":
           receipt["original_fitted_R4_delta_sha256"],
       "source_bound_native_slr_sha256":
           receipt["source_native_slr_jsonl_sha256"],
       "source_bound_original_module_layer":
           receipt["source_model_layer_index"]}
