"""Finite, independently-labelled C3 task -> prediction consumer audit.

Reuses the exact nine-point Grokking task from DASHI's
ModularFourierGrokkingC3Exact and GrokkingEquivarianceDefectGeometryExact:
(a,b) -> a+b (mod 3), shift first operand -> shift output by +1.
Training observations: p00..p21; p22 independently held out.

Only evaluates already-captured [9,3] model logits: NO synthetic
fallback, neural forward, model identification or semantic-label
inference from geometric triads. A task action and model logit
consumer must be bound by the owner separately before neural release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


TASK_PAIRS=tuple((a,b) for a in range(3) for b in range(3))
TASK_LABELS=tuple((a+b)%3 for a,b in TASK_PAIRS)
TRAIN_INDICES=tuple(range(8))
HELDOUT_INDEX=8


def typed_mod3_prediction_consumer(logits: torch.Tensor) -> dict:
    """Audit a task-typed learned three-class consumer; no model inference.

    Numerical equivariance is measured using the entire predicted
    probability distribution. Hard-label equivariance is a separate
    measure and must not replace heldout target correctness.
    """
    if (logits.shape!=(9,3) or not torch.is_floating_point(logits) or
            not bool(torch.isfinite(logits).all())):
        raise ValueError("canonical nine modular-3 pairs require finite logits [9,3]")
    logprob=F.log_softmax(logits.double(),dim=-1)
    p=logprob.exp()
    predicted=logits.argmax(-1)
    labels=torch.tensor(TASK_LABELS,device=logits.device,dtype=torch.long)
    destinations=torch.tensor(
        [((a+1)%3)*3+b for a,b in TASK_PAIRS],
        device=logits.device,dtype=torch.long)
    shifted_predictions=predicted.index_select(0,destinations)
    correct_hard_action=(shifted_predictions==((predicted+1)%3))
    expected_shifted_probs=p.roll(1,dims=-1)
    actual_shifted_probs=p.index_select(0,destinations)
    actual_shifted_logprob=logprob.index_select(0,destinations)
    # Each source pair is compared with its typed action-transformed
    # source pair. KL is diagnostic, not a causal intervention estimate.
    equivariance_kl=(expected_shifted_probs*
                     (expected_shifted_probs.clamp_min(1e-300).log()-
                      actual_shifted_logprob)).sum(-1)
    correct=predicted==labels
    return {
        "schema":1,
        "kind":"x369_typed_mod3_prediction_consumer_action_audit",
        "source_carrier":"explicit nine original modular pairs a,b in Z/3Z",
        "source_action":"R(a,b)=((a+1) mod 3,b)",
        "label_action":"rho(y)=(y+1) mod 3",
        "full_task_correct":bool(correct.all()),
        "training_eight_correct":bool(correct[:8].all()),
        "heldout_p22_correct":bool(correct[8]),
        "training_accuracy":float(correct[:8].double().mean()),
        "entire_nine_accuracy":float(correct.double().mean()),
        "hard_label_action_defect_count":int((~correct_hard_action).sum()),
        "output_distribution_action_KL_mean_nats":
            float(equivariance_kl.clamp_min(0).mean()),
        "heldout_p22_target_nll_nats":float(-logprob[8,labels[8]]),
        "source_logits_checked_no_model_forward":True,
        "independent_source_pair_labels_required":True,
        "representation_was_learned":False,
        "original_LILA_task_C3_steering_measured":False,
        "sophontic_semantic_transfer_inferred":False,
        "F5_raw_or_reserved_opened":False,
    }


def _sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def run(args):
    if args.output.exists():
        raise FileExistsError("do not overwrite pre-existing action audit")
    manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
    if (manifest.get("kind")!="x369_mod3_typed_task_consumer_input_v1" or
            manifest.get("pairs")!=[list(pair) for pair in TASK_PAIRS] or
            manifest.get("target_labels")!=list(TASK_LABELS) or
            manifest.get("train_indices")!=list(TRAIN_INDICES) or
            manifest.get("heldout_indices")!=[HELDOUT_INDEX] or
            manifest.get("action")!="shift_first_operand_and_output_mod3" or
            manifest.get("logits_sha256")!=_sha(args.logits)):
        raise ValueError("task/source/label/action/logits provenance mismatch")
    # A model or tokenizer hash recorded only inside this manifest
    # does not independently authenticate the original producer.
    # Intentional refusal to silently infer a token inventory or
    # source model from a bare array. This scorer is pure audit only.
    arr=np.load(args.logits,allow_pickle=False)
    if not np.issubdtype(arr.dtype,np.floating):
        raise ValueError("source logits must be floating")
    report=typed_mod3_prediction_consumer(torch.as_tensor(arr))
    report.update({
        "manifest_sha256":_sha(args.manifest),
        "input_logits_sha256":_sha(args.logits),
        "checkpoint_source_identity_independently_verified":False,
        "no_original_neural_evaluation_claim":True,
    })
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logits",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
