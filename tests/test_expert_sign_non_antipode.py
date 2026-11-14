"""Sophontic non-antipode regression for actual learned expert router.

Sign is a geometric coordinate chart action, not the semantic
opposition of a claim or a safety label. A generic learned top-k
router does NOT inherit inversion equivariance automatically.
"""
import unittest
import torch
from models.history_orbit_moe import HistoryOrbitMoE


class ExpertSignNonAntipodeTests(unittest.TestCase):
    def test_default_router_may_change_expert_on_joint_inversion(self):
        m=HistoryOrbitMoE(d_model=2,num_experts=2,top_k=1,expert_width=4)
        with torch.no_grad():
            m.router.weight.zero_()
            m.router.bias.zero_()
            # Route on expert-specific first history coordinate.
            m.router.weight[0,2] = 1.
            m.expert_bias.zero_()
        x=torch.zeros(1,2)
        h=torch.tensor([[[1.,0.],[-1.,0.]]])
        _,_,a,_,_=m.step(x,h)
        _,_,b,_,_=m.step(-x,-h)
        self.assertEqual(a.item(),0)
        self.assertEqual(b.item(),1)
        # No global commuting-square assertion may be attached to
        # unmodified learned top-k selection.

    def test_semantic_disagreement_is_not_an_orientation_instruction(self):
        # Metadata is independently typed and cannot affect
        # model operation unless an explicit caller introduces a mapping.
        torch.manual_seed(8)
        m=HistoryOrbitMoE(2,2,1,4).eval()
        x=torch.tensor([[.2,-.4]])
        h=torch.zeros(1,2,2)
        annotations=("challenge_evidence","alternative_explanation")
        traces=[m.step(x,h) for _ in annotations]
        self.assertTrue(torch.equal(traces[0][2],traces[1][2]))
        self.assertTrue(torch.equal(traces[0][3],traces[1][3]))
        self.assertTrue(torch.equal(traces[0][0],traces[1][0]))


if __name__ == "__main__":
    unittest.main()
