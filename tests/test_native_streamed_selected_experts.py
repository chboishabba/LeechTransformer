"""Source parity and negative controls for real HistoryOrbitMoE selected execution.

Uses small constructed tensors as *unit tests*, not neural-quality or GPU claims.
Native device and asynchronous overlap require the separate benchmark.
"""
import unittest
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from scripts.native_streamed_selected_experts import compare


class SelectedStreamingParityTests(unittest.TestCase):
    def test_selected_execution_uses_source_routing(self):
        torch.manual_seed(42)
        moe=HistoryOrbitMoE(d_model=24,num_experts=4,top_k=2,
                           expert_width=12,route_geometry="raw").eval()
        x=torch.randn(3,24)
        h=torch.randn(3,4,24)
        with torch.no_grad():
            original, updated, ids, _, weights=moe.step(x,h,train_sign=False)
            selected=h.gather(1,ids.unsqueeze(-1).expand(-1,-1,24))
            got, got_h,_=moe.execute_selected(
                x,ids,selected,selected_weights=weights,train_sign=False)
            expected_h=updated.gather(
                1,ids.unsqueeze(-1).expand(-1,-1,24))
            self.assertTrue(torch.allclose(got,original,atol=1e-6,rtol=1e-6))
            self.assertTrue(torch.allclose(got_h,expected_h,atol=1e-6,rtol=1e-6))

    def test_parity_receipt_failure_is_visible(self):
        reference=[torch.zeros(2,24)]
        histories=[torch.zeros(2,2,24)]
        good=[(torch.zeros(2,24),torch.zeros(2,2,24),torch.zeros(2,2))]
        receipt=compare(good,reference,histories,atol=0,rtol=0)
        self.assertTrue(receipt["allclose"])
        bad=[(torch.ones(2,24),torch.zeros(2,2,24),torch.zeros(2,2))]
        receipt=compare(bad,reference,histories,atol=0,rtol=0)
        self.assertFalse(receipt["allclose"])
        self.assertGreater(receipt["max_output_abs"],0)

    def test_selected_ids_not_authenticated_by_tensor_math(self):
        moe=HistoryOrbitMoE(d_model=24,num_experts=4,top_k=2,
                           expert_width=12,route_geometry="raw").eval()
        with self.assertRaises(ValueError):
            moe.execute_selected(torch.zeros(1,24),
                                 torch.tensor([[0,0]],dtype=torch.long),
                                 torch.zeros(1,2,24),
                                 selected_weights=torch.ones(1,2)/2)


if __name__=="__main__":
    unittest.main()
