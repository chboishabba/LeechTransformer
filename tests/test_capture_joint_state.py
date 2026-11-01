import unittest
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from scripts.capture_joint_moe_state import capture_from_expert
from scripts.benchmark_joint_state_quotient import benchmark_capture

class TestCaptureJoint(unittest.TestCase):
    def test_trace(self):
        torch.manual_seed(43)
        model=HistoryOrbitMoE(48,4,2,24,route_geometry="punctured9_wave",transition_rank=24).eval()
        hidden=torch.randn(2,4,48)
        with torch.no_grad():
            expected=model(hidden)
            captured=capture_from_expert(model,hidden)
        self.assertEqual(captured["activation"].shape,(8,48))
        self.assertTrue(torch.allclose(torch.from_numpy(captured["replayed_final_history"]),expected[1],atol=1e-6))
        self.assertTrue(torch.allclose(torch.from_numpy(captured["replayed_output"]).view(2,4,48),expected[0],atol=1e-6))
        report=benchmark_capture({key:captured[key] for key in ("activation","histories","selected")})
        self.assertEqual(report["samples"],8)
        self.assertTrue(report["byte_exact_reconstruction"])
        self.assertEqual(report["reply_bytes_assumption"],2*2*48*4)

if __name__=="__main__":
    unittest.main()
