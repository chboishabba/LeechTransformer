"""F3 quantitative row-paired null controls and source action probe."""
import unittest
import numpy as np
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from scripts.audit_c3_causal_router_ce import paired_cluster_interval,validate_f3_probe_manifest
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
            moe.route_probe_action="c3_matched_displacement_random_x"
            moe.route_probe_seed=19
            matched_ids,matched_weights=moe.select_source_experts(x,h)
            self.assertEqual(matched_ids.shape,ids.shape)
            self.assertEqual(matched_weights.shape,weights.shape)
            self.assertTrue(torch.isfinite(matched_weights).all())
            repeated_ids,repeated_weights=moe.select_source_experts(x,h)
            self.assertTrue(torch.equal(matched_ids,repeated_ids))
            self.assertTrue(torch.equal(matched_weights,repeated_weights))
            moe.route_probe_action="c3_feature_exact_mixed_orientation_x"
            moe.route_probe_seed=19
            im,wm=moe.select_source_experts(x,h)
            self.assertEqual(im.shape,ids.shape)
            self.assertTrue(torch.isfinite(wm).all())
            repeat_i,repeat_w=moe.select_source_experts(x,h)
            self.assertTrue(torch.equal(im,repeat_i))
            self.assertTrue(torch.equal(wm,repeat_w))
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

class F3SourceDigestAdmissionTests(unittest.TestCase):
    def test_f3_f5_sha_only_provenance_without_opening_reserved_token_array(self):
        sha=lambda n:format(n,"064x")
        screen={"validation_overlap_window":16}
        manifest={"kind":"x369_f4_f6_fresh_real_document_token_split",
                  "prior_token_array_sha256":[sha(4),sha(3)],
                  "output_token_array_sha256":{"development":sha(7)},
                  "original_token_overlap_window":16,
                  "selected_token_windows_disjoint_from_all_supplied_prior_arrays":True}
        receipt=validate_f3_probe_manifest(manifest,screen,sha(7),sha(3))
        self.assertTrue(receipt["reserved_raw_tokens_not_opened_by_this_validation"])
        self.assertFalse(receipt["historical_document_independence_verified"])
        import copy
        cases=[]
        m=copy.deepcopy(manifest);m["prior_token_array_sha256"]=[sha(4)]
        cases.append(m)
        m=copy.deepcopy(manifest);m["output_token_array_sha256"]["development"]=sha(8)
        cases.append(m)
        m=copy.deepcopy(manifest);m["original_token_overlap_window"]=17
        cases.append(m)
        m=copy.deepcopy(manifest);m["selected_token_windows_disjoint_from_all_supplied_prior_arrays"]=False
        cases.append(m)
        for bad in cases:
            with self.assertRaisesRegex(ValueError,"F3 probe provenance"):
                validate_f3_probe_manifest(bad,screen,sha(7),sha(3))
        with self.assertRaisesRegex(ValueError,"digests"):
            validate_f3_probe_manifest(manifest,screen,sha(7),"7d13b31c")

if __name__=="__main__":unittest.main()
