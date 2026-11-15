import unittest
import torch
from models.paired_intervention_geometry_loss import paired_training_objective


class PairedTrainingObjectiveTests(unittest.TestCase):
    def test_without_action_receipt_never_imposes_geometric_antipode(self):
        logits1=torch.randn(3,5,requires_grad=True)
        logits2=torch.randn(3,5,requires_grad=True)
        h1=torch.randn(3,6,requires_grad=True)
        h2=torch.randn(3,6,requires_grad=True)
        r=paired_training_objective(logits1,logits2,
            torch.tensor([1,2,3]),torch.tensor([2,2,3]),
            h1,h2,known_actions=["none"]*3,geometry_weight=200.)
        self.assertEqual(r["licensed_action_count"],0)
        self.assertEqual(float(r["licensed_action_loss"]),0.)
        r["loss"].backward()
        self.assertIsNotNone(logits1.grad)
        self.assertIsNone(h1.grad)

    def test_explicit_signed_action_gradients_respect_given_receipt(self):
        h1=torch.tensor([[1.,2.,3.,4.,5.,6.]],requires_grad=True)
        h2=torch.tensor([[1.,2.,3.,4.,5.,6.]],requires_grad=True)
        logit1=torch.randn(1,4,requires_grad=True)
        logit2=torch.randn(1,4,requires_grad=True)
        r=paired_training_objective(logit1,logit2,
            torch.tensor([0]),torch.tensor([1]),h1,h2,
            known_actions=["C2_inversion"],geometry_weight=1.)
        self.assertGreater(float(r["licensed_action_loss"]),0.)
        r["loss"].backward()
        self.assertIsNotNone(h1.grad)
        self.assertIsNotNone(h2.grad)


if __name__=="__main__":unittest.main()
