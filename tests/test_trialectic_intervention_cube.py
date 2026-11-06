"""A nonzero triadic model-output interaction is not semantic proof."""
import unittest
import torch

from scripts.eval_trialectic_intervention_cube import (
    cube_evidence,FACE_COEFFICIENTS)


class TrialecticInterventionCubeTests(unittest.TestCase):
    def test_additive_three_axis_model_has_zero_face(self):
        # Set log P(probe=0) = -3 + 0.2x + 0.3y + 0.1z.
        # This is additive in the independent axis interventions.
        logits=torch.zeros(2,8,2,dtype=torch.float64)
        for index in range(8):
            x=(index>>2)&1
            y=(index>>1)&1
            z=index&1
            value=-3.+.2*x+.3*y+.1*z
            prob=torch.exp(torch.tensor(value,dtype=torch.float64))
            logits[:,index,0]=torch.log(prob/(1.-prob))
        result=cube_evidence(logits,torch.ones(2,8,dtype=torch.long),
                             torch.zeros(2,dtype=torch.long))
        self.assertLess(result["third_order_fixed_probe_mean_abs_nats"],1e-5)
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
