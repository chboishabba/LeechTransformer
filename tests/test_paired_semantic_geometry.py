import unittest
import torch
from scripts.eval_paired_semantic_geometry import paired_evidence


class TypedPairedGeometryTests(unittest.TestCase):
    def test_joint_correctness_and_geometric_fit_are_independent(self):
        # First pair is a challenge with a genuine changed target and an
        # antipodal hidden state; second is a context addition with an
        # unchanged target but ALSO an antipodal hidden state.
        a=torch.tensor([[8.,0.,0.],[0.,8.,0.]])
        b=torch.tensor([[0.,8.,0.],[0.,8.,0.]])
        h=torch.tensor([[1.,2.,3.,4.,5.,6.],
                        [6.,5.,4.,3.,2.,1.]])
        evidence=paired_evidence(a,b,h,-h,torch.tensor([0,1]),
                                  torch.tensor([1,1]),
                                  ["challengeEvidence","addContext"])
        self.assertEqual(evidence["pairs"],2)
        self.assertEqual(evidence["joint_correct_rate"],1)
        self.assertEqual(evidence["predicted_flip_rate"],.5)
        self.assertEqual(evidence["best_action_counts"]["C2_inversion"],2)
        # Matching geometric antipodes cannot determine semantic response.
        self.assertEqual(evidence["by_response_kind"]["addContext"]["actual_changed"],0)

    def test_spurious_flips_fail_even_when_flip_rate_is_high(self):
        a=torch.tensor([[9.,0.],[0.,9.]])
        b=torch.tensor([[0.,9.],[9.,0.]])
        h=torch.randn(2,6)
        evidence=paired_evidence(a,b,h,h,torch.tensor([0,1]),
                                  torch.tensor([0,1]),
                                  ["acceptWithQualification","addContext"])
        self.assertEqual(evidence["predicted_flip_rate"],1)
        self.assertEqual(evidence["joint_correct_rate"],0)
        self.assertEqual(evidence["flip_decision_accuracy"],0)


if __name__=="__main__":unittest.main()
