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


R4_GEOMETRIES = ("c2_c3_residual", "c2_c3_mixed_null_residual")


def clone_c2_with_zero_c3_residual(source: LeechGPT, geometry: str) -> LeechGPT:
    """Create a strict inherited C2 function with ONE new zero score per block.

    We deliberately *do not* call load_state_dict(strict=False) and ignore its
    errors: only exactly n_layers additional residual weights are permitted.
    The model should be prepared on CPU and then moved to the single GPU
    one arm at a time; cloning on an 8GB RX 580 could otherwise cause OOM.
    """
    if geometry not in R4_GEOMETRIES:
        raise ValueError("R4 geometry must be coherent or fixed mixed C3 residual")
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
        token_ids: torch.Tensor) -> dict:
    """Fail on any non-bitwise identical logits, hidden or expert histories.

    Inference state must be dropout-free. Token examples here are only an
    implementation witness; this cannot stand in for source-bound R4 training.
    """
    if any(m.training for m in (source, coherent, null)):
        raise ValueError("initial exact-comparison models must all be eval()")
    if token_ids.ndim != 2 or token_ids.numel() == 0:
        raise ValueError("nonempty [batch,time] tokens required")
    if token_ids.dtype != torch.long:
        raise ValueError("token IDs must be torch.long")
    if int(token_ids.min()) < 0 or int(token_ids.max()) >= source.cfg.vocab_size:
        raise ValueError("out-of-vocabulary parity probe")
    outputs = [
        model(token_ids, student_only=True, return_expert_histories=True)
        for model in (source, coherent, null)
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
        for model in (coherent, null)
    ]
    if sizes[0] != sizes[1]:
        raise AssertionError("R4 mixed-null trainable capacity mismatch")
    return {
        "kind": "x369_r4_cpu_same_function_initialisation",
        "source_c2_equals_coherent_logits_bitwise": True,
        "source_c2_equals_mixed_null_logits_bitwise": True,
        "source_c2_equals_both_hidden_and_histories_bitwise": True,
        "residual_trainable_parameters_per_arm": sizes[0],
        "residual_weight_init": 0,
        "input_shape": list(token_ids.shape),
        "device": str(token_ids.device),
        "physical_training_or_learned_action_claim": False,
        "f5_opened": False,
    }
