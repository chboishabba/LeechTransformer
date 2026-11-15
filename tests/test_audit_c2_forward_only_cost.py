"""Formulaic wall-cost auditor contracts. Toy receipts, NOT GPU timing."""
import unittest
from scripts.audit_c2_forward_only_cost import audit

def specimen():
    return {
      "kind":"native_gpu_cumulative_student_quality_curriculum",
      "teacher_heldout_ce":2.5834,
      "full_student_only_heldout_ce":4.763,
      "elapsed_seconds":4700,
      "stages":[
        {"stage":1,"active_layers":[1],"best_development_ce":3.0,
         "epoch_scores":[{"focused_dev_ce":3.5},{"focused_dev_ce":3.0}],
         "performance":{"steps":12,"loop_elapsed_seconds":220}},
        {"stage":2,"active_layers":[0,1],"best_development_ce":4.0,
         "epoch_scores":[{"focused_dev_ce":4.0}],
         "performance":{"steps":6,"loop_elapsed_seconds":280}}
      ]
    }
class CostAuditTests(unittest.TestCase):
    def test_observed_wall_is_not_backward_microbenchmark(self):
        r=audit(specimen(),80.1)
        self.assertEqual(r["observed_wall_seconds"],4806)
        self.assertEqual(r["total_recorded_optimizer_steps"],18)
        self.assertEqual(r["reported_per_stage_loop_seconds_sum"],500)
        self.assertFalse(r["backward_pass_fraction_measured"])
        self.assertFalse(r["gradient_free_speedup_claimed"])
        self.assertAlmostEqual(r["heldout_gap_nats"],2.1796)
    def test_only_measured_probe_break_even(self):
        r=audit(specimen(),80.1,fwd_ms=10,bwd_ms=15,probes=2)
        self.assertEqual(r["probes_forward_equivalent_break_even"],1.5)
        self.assertFalse(r["probe_compute_less_than_backward"])
        self.assertFalse(r["probe_cost_is_end_to_end_prediction"])
    def test_no_hardware_prediction_without_timing(self):
        for kwargs in ({"probes":2},{"fwd_ms":1,"probes":2},{"bwd_ms":1,"probes":2}):
            with self.assertRaises(ValueError):audit(specimen(),80.1,**kwargs)
    def test_invalid_observed_loss_fails(self):
        s=specimen();s["full_student_only_heldout_ce"]=float("nan")
        with self.assertRaises(ValueError):audit(s,80.1)
    def test_missing_timing_fails(self):
        s=specimen();s["stages"][0]["performance"]={}
        with self.assertRaises(ValueError):audit(s,80.1)

if __name__=="__main__":unittest.main()
