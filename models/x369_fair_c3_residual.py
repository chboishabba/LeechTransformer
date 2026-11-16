"""R4 strict same-function transplants from a *real* source C2 checkpoint.

This construction is intentionally not a claim that the physical network
learned a C3 task action. The finite trialectic feature prior does not define
a semantic label action. The new score is identically zero at initialization,
and all original C2 parameters, buffers and nonpersistent history caps retain
their meanings and values. Neither F5 nor a data split is opened here.
"""
from __future__ import annotations

from dataclasses import replace
import torch

from models.model import LeechGPT


R4_GEOMETRIES = ("c2_c2_residual", "c2_c3_residual", "c2_c3_mixed_null_residual")


def clone_c2_with_zero_c3_residual(source: LeechGPT, geometry: str) -> LeechGPT:
    """Create a strict inherited C2 function with ONE new zero score per block.

    We deliberately *do not* call load_state_dict(strict=False) and ignore its
    errors: only exactly n_layers additional residual weights are permitted.
    The model should be prepared on CPU and then moved to the single GPU
    one arm at a time; cloning on an 8GB RX 580 could otherwise cause OOM.
    """
    if geometry not in R4_GEOMETRIES:
        raise ValueError("R4 geometry must be C2 residual, coherent C3 or fixed mixed C3 residual")
    if (source.cfg.moe_route_geometry != "c2_invariant" or
            source.cfg.moe_experts < 2 or source.cfg.moe_shared_e8_history):
        raise ValueError("requires original independent-history C2 source model")
    if source.resonator is not None:
        raise ValueError("R4 source resonator must be detached or separately modelled")
    if next(source.parameters()).device.type != "cpu":
        raise ValueError("construct R4 clones on CPU; move one arm to GPU at a time")
    cfg = replace(source.cfg, moe_route_geometry=geometry)
    with torch.random.fork_rng(devices=[]):
        result = LeechGPT(cfg).to(dtype=next(source.parameters()).dtype)
    incompatible = result.load_state_dict(source.state_dict(), strict=False)
    expected_missing = {
        f"blocks.{i}.moe.c3_residual_weight"
        for i in range(source.cfg.n_layers)
    }
    if (set(incompatible.missing_keys) != expected_missing or
            incompatible.unexpected_keys):
        raise AssertionError("unexpected C2 checkpoint transplant mismatch: "
                             f"{incompatible}")
    inherited = source.state_dict()
    transplanted = result.state_dict()
    for name, value in inherited.items():
        if name not in transplanted or not torch.equal(value, transplanted[name]):
            raise AssertionError(f"source tensor changed during R4 transplant: {name}")
    for i, (old, new) in enumerate(zip(source.blocks, result.blocks)):
        if (old.moe is None or new.moe is None or
                old.moe.history_cap != new.moe.history_cap or
                new.moe.c3_residual_weight.count_nonzero().item() != 0):
            raise AssertionError(f"R4 missing C2 cap or nonzero residual at {i}")
    result.train(source.training)
    return result


@torch.no_grad()
def assert_identical_initial_student_rollout(
        source: LeechGPT, coherent: LeechGPT, null: LeechGPT,
        token_ids: torch.Tensor, c2_control: LeechGPT | None = None) -> dict:
    """Fail on any non-bitwise identical logits, hidden or expert histories.

    Inference state must be dropout-free. Token examples here are only an
    implementation witness; this cannot stand in for source-bound R4 training.
    """
    if any(m.training for m in (source, coherent, null) + (() if c2_control is None else (c2_control,))):
        raise ValueError("initial exact-comparison models must all be eval()")
    if token_ids.ndim != 2 or token_ids.numel() == 0:
        raise ValueError("nonempty [batch,time] tokens required")
    if token_ids.dtype != torch.long:
        raise ValueError("token IDs must be torch.long")
    if int(token_ids.min()) < 0 or int(token_ids.max()) >= source.cfg.vocab_size:
        raise ValueError("out-of-vocabulary parity probe")
    outputs = [
        model(token_ids, student_only=True, return_expert_histories=True)
        for model in (source, coherent, null) + (() if c2_control is None else (c2_control,))
    ]
    base = outputs[0]
    for arm in outputs[1:]:
        for i in (0, 1):  # both final logits and normalized hidden states
            if not torch.equal(base[i], arm[i]):
                raise AssertionError(f"R4 same-function failure in output {i}")
        if len(base[4]) != len(arm[4]):
            raise AssertionError("R4 expert-history layer count differs")
        for old, new in zip(base[4], arm[4]):
            if not torch.equal(old, new):
                raise AssertionError("R4 recurrent expert histories differ")
    sizes = [
        sum(block.moe.c3_residual_weight.numel() for block in model.blocks)
        for model in (coherent, null) + (() if c2_control is None else (c2_control,))
    ]
    if len(set(sizes)) != 1:
        raise AssertionError("R4 mixed-null trainable capacity mismatch")
    return {
        "kind": "x369_r4_cpu_same_function_initialisation",
        "source_c2_equals_coherent_logits_bitwise": True,
        "source_c2_equals_mixed_null_logits_bitwise": True,
        "source_c2_equals_trainable_c2_control_bitwise": c2_control is not None,
        "source_c2_equals_both_hidden_and_histories_bitwise": True,
        "residual_trainable_parameters_per_arm": sizes[0],
        "residual_weight_init": 0,
        "input_shape": list(token_ids.shape),
        "device": str(token_ids.device),
        "physical_training_or_learned_action_claim": False,
        "f5_opened": False,
    }


def make_residual_delta_payload(
        model: LeechGPT, *, source_sha256: str,
        source_jsonl_sha256: str, seed: int, best_step: int) -> dict:
    """Archive ONLY fitted residual tensors; never copy full source weights.

    Source checkpoint and original native-PG source remain independently
    immutable, SHA-addressed inputs. The payload cannot be reinterpreted as
    a full model, as a fresh sample or as evidence of a typed C3 label action.
    """
    def sha_ok(sha):
        return (isinstance(sha, str) and len(sha) == 64 and
                all(c in "0123456789abcdef" for c in sha))
    if (model.cfg.moe_route_geometry not in R4_GEOMETRIES or
            not sha_ok(source_sha256) or not sha_ok(source_jsonl_sha256) or
            type(seed) is not int or seed < 0 or
            type(best_step) is not int or best_step < 0):
        raise ValueError("R4 residual payload lacks frozen original identities")
    weights = {}
    for i, block in enumerate(model.blocks):
        if block.moe is None or block.moe.c3_residual_weight is None:
            raise ValueError("R4 residual missing from model layer")
        tensor = block.moe.c3_residual_weight.detach().cpu().contiguous().clone()
        if (tensor.shape != (1, 2 * model.cfg.d_model) or
                not bool(torch.isfinite(tensor).all())):
            raise ValueError("R4 residual tensor shape/finiteness violated")
        weights[f"blocks.{i}.moe.c3_residual_weight"] = tensor
    if len(weights) != model.cfg.n_layers:
        raise AssertionError("R4 residual layer count mismatch")
    return {
        "schema": 1,
        "kind": "x369_r4b_fitted_source_bound_residual_delta",
        "source_repaired_c2_checkpoint_sha256": source_sha256,
        "source_native_slr_jsonl_sha256": source_jsonl_sha256,
        "geometry": model.cfg.moe_route_geometry,
        "seed": seed,
        "selected_development_step": best_step,
        "feature_semantics": "x369_r4_q_gated_mixed_v2",
        "source_base_parameters_included": False,
        "F5_raw_or_reserved_opened": False,
        "typed_C3_task_action_verified": False,
        "residual_weights": weights,
    }


def restore_residual_delta(
        source: LeechGPT, payload: dict, *, source_sha256: str,
        source_jsonl_sha256: str) -> LeechGPT:
    """Fail-closed replay: exact frozen C2 source + *only* trained R4 delta.

    Caller must independently hash and authenticate the source checkpoint
    and native SLR JSONL before invoking this function. weights_only=True
    should be used for torch.load, never arbitrary pickle execution.
    """
    if (not isinstance(payload, dict) or
            payload.get("schema") != 1 or
            payload.get("kind") != "x369_r4b_fitted_source_bound_residual_delta" or
            payload.get("source_repaired_c2_checkpoint_sha256") != source_sha256 or
            payload.get("source_native_slr_jsonl_sha256") != source_jsonl_sha256 or
            payload.get("feature_semantics") != "x369_r4_q_gated_mixed_v2" or
            payload.get("source_base_parameters_included") is not False or
            payload.get("F5_raw_or_reserved_opened") is not False or
            payload.get("typed_C3_task_action_verified") is not False or
            type(payload.get("seed")) is not int or payload["seed"] < 0 or
            type(payload.get("selected_development_step")) is not int or
            payload["selected_development_step"] < 0):
        raise ValueError("wrong source or invalid R4 fitted delta identity")
    geometry = payload.get("geometry")
    if geometry not in R4_GEOMETRIES:
        raise ValueError("unknown saved R4 geometry")
    weights = payload.get("residual_weights")
    expected = {
        f"blocks.{i}.moe.c3_residual_weight"
        for i in range(source.cfg.n_layers)}
    if not isinstance(weights, dict) or set(weights) != expected:
        raise ValueError("missing, additional or mismatched R4 residual tensor")
    model = clone_c2_with_zero_c3_residual(source, geometry)
    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            key = f"blocks.{i}.moe.c3_residual_weight"
            src = weights[key]
            dst = block.moe.c3_residual_weight
            if (not isinstance(src, torch.Tensor) or
                    src.shape != dst.shape or src.dtype != dst.dtype or
                    not bool(torch.isfinite(src).all())):
                raise ValueError("unsafe residual dtype/shape/nonfinite value: " + key)
            dst.copy_(src)
    return model.eval()



def fixed_mixed_c3_weight_gauge(
        coherent_weight: torch.Tensor, base_mask: torch.Tensor) -> torch.Tensor:
    """Exact linear reparameterization of OLD fixed-mixed C3 head.

    Coherent T(R+x,h)=(r,s,f); fixed R- is (f,r,s), a
    +1 cyclic roll of the coherent directed-feature triple.
    For unrestricted separate trained weights on all three channels,
    applying the SAME roll to weights preserves their dot product.
    This is a degeneracy witness, not a reason to prefer either arm.
    New q-gated mixed control is sample-dependent and is NOT covered.
    """
    if (coherent_weight.ndim!=2 or coherent_weight.shape[0]!=1 or
            coherent_weight.shape[1]<12 or coherent_weight.shape[1]%6 or
            not torch.is_floating_point(coherent_weight) or
            not bool(torch.isfinite(coherent_weight).all())):
        raise ValueError("coherent residual must be finite [1,2d] with 3|d")
    d=coherent_weight.shape[1]//2
    if (base_mask.shape!=(d//3,) or base_mask.dtype!=torch.bool or
            base_mask.device!=coherent_weight.device):
        raise ValueError("original fixed mixed mask must have d/3 triads")
    weights=coherent_weight.clone()
    directed=weights[:,d:].reshape(d//3,3)
    converted=torch.where(
        base_mask[:,None],directed,directed.roll(1,dims=-1))
    weights[:,d:]=converted.reshape(1,d)
    return weights
