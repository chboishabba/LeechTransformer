"""Typed grokking action consumer witnesses; no neural success claimed."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.audit_x369_typed_mod3_prediction_consumer import (
    TASK_PAIRS, TASK_LABELS, TRAIN_INDICES, HELDOUT_INDEX,
    typed_mod3_prediction_consumer, run)


def logits_for(labels):
    out=torch.zeros(9,3,dtype=torch.float64)
    for i,y in enumerate(labels):
        out[i,y]=12.
    return out


class X369TypedMod3ConsumerTests(unittest.TestCase):
    def test_algorithmic_task_character_rule_transports_and_composes(self):
        a=typed_mod3_prediction_consumer(logits_for(TASK_LABELS))
        self.assertEqual(a["hard_label_action_defect_count"],0)
        self.assertTrue(a["heldout_p22_correct"])
        self.assertTrue(a["training_eight_correct"])
        self.assertTrue(a["full_task_correct"])
        self.assertLess(a["output_distribution_action_KL_mean_nats"],1e-12)
        self.assertFalse(a["representation_was_learned"])

    def test_grokking_memorizer_training_fits_but_heldout_fails(self):
        # Reuses the EXACT original Agda witness p22 incorrectly maps to
        # phase0 instead of the modular-addition phase1.
        memorizer=list(TASK_LABELS)
        memorizer[8]=0
        a=typed_mod3_prediction_consumer(logits_for(memorizer))
        self.assertTrue(a["training_eight_correct"])
        self.assertFalse(a["heldout_p22_correct"])
        self.assertEqual(a["hard_label_action_defect_count"],2)
        self.assertGreater(
            a["output_distribution_action_KL_mean_nats"],.1)

    def test_perfect_argmax_does_not_entail_distribution_equivariance(self):
        a=logits_for(TASK_LABELS)
        a[1,TASK_LABELS[1]]=8.
        report=typed_mod3_prediction_consumer(a)
        self.assertTrue(report["full_task_correct"])
        self.assertEqual(report["hard_label_action_defect_count"],0)
        self.assertGreater(
            report["output_distribution_action_KL_mean_nats"],1e-5)

    def test_source_task_manifest_is_not_inferred_from_logits(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            path=root/"logits.npy"
            np.save(path,logits_for(TASK_LABELS).numpy())
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest={
                "kind":"x369_mod3_typed_task_consumer_input_v1",
                "pairs":[list(z) for z in TASK_PAIRS],
                "target_labels":list(TASK_LABELS),
                "train_indices":list(TRAIN_INDICES),
                "heldout_indices":[HELDOUT_INDEX],
                "action":"shift_first_operand_and_output_mod3",
                "logits_sha256":sha,
            }
            m=root/"task.json"
            m.write_text(json.dumps(manifest))
            args=SimpleNamespace(logits=path,manifest=m,output=root/"out.json")
            a=run(args)
            self.assertTrue(a["no_original_neural_evaluation_claim"])
            with self.assertRaises(FileExistsError):
                run(args)
            args.output=root/"wrong.json"
            manifest["target_labels"][-1]=0
            m.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                run(args)
            manifest["target_labels"][-1]=1
            manifest["logits_sha256"]="0"*64
            m.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                run(args)

    def test_malformed_nine_carrier_rejected(self):
        with self.assertRaises(ValueError):
            typed_mod3_prediction_consumer(torch.zeros(8,3))
        with self.assertRaises(ValueError):
            typed_mod3_prediction_consumer(
                torch.full((9,3),float("nan")))
        with self.assertRaises(ValueError):
            typed_mod3_prediction_consumer(torch.zeros(9,3,dtype=torch.long))


if __name__=="__main__":
    unittest.main()
