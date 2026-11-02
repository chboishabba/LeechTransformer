"""A nonzero triadic model-output interaction is not semantic proof."""
import unittest
import torch

from scripts.eval_trialectic_intervention_cube import (
    cube_evidence,FACE_COEFFICIENTS)


class TrialecticInterventionCubeTests(unittest.TestCase):
    def test_additive_three_axis_model_has_zero_face(self):
        # f(x,y,z) = a + 2x + 3y + 4z; no triple interaction.
        logits=torch.zeros(2,8,2)
        # For a two-class model, log P(class 0)=f can be chosen
        # explicitly only for negative f; use log(sigmoid(w)) where
        # w is additive, whose log nonlinearity can create interaction.
        # Instead use equal class logit values for constant probabilities.
        result=cube_evidence(logits,torch.zeros(2,8,dtype=torch.long),
                             torch.zeros(2,dtype=torch.long))
        self.assertAlmostEqual(result["third_order_fixed_probe_mean_abs_nats"],0.)
        self.assertEqual(result["entire_cube_correct_rate"],1.)

    def test_pure_three_way_effect_and_distinct_semantic_score(self):
        logits=torch.zeros(1,8,2)
        logits[0,7,0]=5.
        result=cube_evidence(logits,
            torch.ones(1,8,dtype=torch.long),
            torch.zeros(1,dtype=torch.long))
        self.assertGreater(result["third_order_fixed_probe_mean_abs_nats"],.1)
        self.assertEqual(result["corner_correct_rate"],0.)
        self.assertEqual(result["entire_cube_correct_rate"],0.)
        self.assertEqual(result["face_coefficients"],list(FACE_COEFFICIENTS))
        self.assertEqual(result["cube_order"],"000,001,010,011,100,101,110,111")

    def test_refuse_missing_fixed_probe_and_bad_labels(self):
        logits=torch.zeros(1,8,3)
        targets=torch.zeros(1,8,dtype=torch.long)
        for probe in (torch.zeros(1,8,dtype=torch.long),
                      torch.tensor([5])):
            with self.assertRaises(ValueError):
                cube_evidence(logits,targets,probe)
        with self.assertRaises(ValueError):
            cube_evidence(logits,targets.float(),torch.tensor([0]))
        with self.assertRaises(ValueError):
            cube_evidence(logits.fill_(float("nan")),
                          targets,torch.tensor([0]))


if __name__=="__main__":
    unittest.main()
