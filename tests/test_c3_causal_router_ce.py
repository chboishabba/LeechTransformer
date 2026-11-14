"""F3 quantitative row-paired null controls and source action probe."""
import unittest
import numpy as np
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from scripts.audit_c3_causal_router_ce import paired_cluster_interval
class C3CausalRouterProbeTests(unittest.TestCase):
    def test_paired_row_bootstrap_and_bad_inputs(self):
        x=np.asarray([1.,2.,3.,4.])
        r=paired_cluster_interval(x,x+.25,seed=8,bootstraps=39)
        self.assertAlmostEqual(r["mean_delta_ce"],.25)
        np.testing.assert_allclose(r["cluster_bootstrap_interval"],[.25,.25])
        with self.assertRaises(ValueError):
            paired_cluster_interval(x,np.asarray([1.,2.]),bootstraps=39)
    def test_original_moe_diagonal_probe_is_route_invariant(self):
        torch.manual_seed(7)
        moe=HistoryOrbitMoE(d_model=24,num_experts=4,top_k=2,
                            route_geometry="c2c3_wave").eval()
        x=torch.randn(2,24)
        h=torch.randn(2,4,24)
        with torch.no_grad():
            ids,weights=moe.select_source_experts(x,h)
            for action in ("c3_diagonal_shift","c2_diagonal_negation"):
                moe.route_probe_action=action
                i,w=moe.select_source_experts(x,h)
                self.assertTrue(torch.equal(ids,i))
                self.assertTrue(torch.allclose(weights,w,atol=1e-5,rtol=1e-5))
            moe.route_probe_action="signed_axis_permutation_x"
            i,w=moe.select_source_experts(x,h)
            self.assertEqual(i.shape,ids.shape)
            self.assertEqual(w.shape,weights.shape)
            moe.route_probe_action="nonsense"
            with self.assertRaises(ValueError):
                moe.select_source_experts(x,h)
            moe.route_probe_action=None
            i,w=moe.select_source_experts(x,h)
            self.assertTrue(torch.equal(i,ids))
            self.assertTrue(torch.equal(w,weights))
if __name__=="__main__":unittest.main()
