"""X369 F3 frozen original mixed-C3 null-ensemble source regressions.

These are source-contract tests (pure numerical + stubbed original row CE).
No GPU, tokenizer, F5 array or original model accuracy is fabricated.
"""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scripts.audit_x369_f3_mixed_orientation_ensemble_ce import (
    summarize_f3_mixed_ensemble,score_frozen_f3_ensemble)

class X369F3OrientationEnsembleTests(unittest.TestCase):
    def test_clustered_ce_decomposition_and_direction(self):
        baseline=np.array([6.,5.,4.,3.])
        coherent=np.array([5.8,4.8,3.8,2.8])
        # Mixed is poorer, every original row, by 0.1 on average.
        mixed=np.stack([coherent+.15,coherent+.05,
                        coherent+.08,coherent+.12])
        result=summarize_f3_mixed_ensemble(
            baseline,coherent,mixed,seed=41,bootstraps=199)
        self.assertAlmostEqual(result["coherent_minus_baseline_mean_delta_ce"],-.2)
        self.assertAlmostEqual(result["coherent_minus_mixed_mean_delta_ce"],-.1)
        self.assertAlmostEqual(result["fraction_mixed_orientation_draws_coherent_has_lower_mean_ce"],1.)
        np.testing.assert_allclose(
            result["row_cluster_bootstrap_95pct_descriptive_interval"],
            [-.1,-.1],atol=1e-12)
        self.assertTrue(result["selected_null_after_inspecting_old_F3_is_exploratory"])
        self.assertTrue(result["no_F5_token_rows_loaded_scored_or_fit"])
        self.assertFalse(result["release_admitted"])

    def test_no_independent_mask_p_value_claim(self):
        b=np.array([2.,3.,4.])
        c=b+.1
        m=np.stack([b+.1,b+.2,b,b+.3])
        got=summarize_f3_mixed_ensemble(
            b,c,m,seed=4,bootstraps=29)
        self.assertTrue(got["this_is_not_an_exact_randomisation_test_p_value"])
        self.assertTrue(got["orientation_masks_are_not_independent_document_clusters"])
        self.assertEqual(len(got["coherent_minus_each_mixed_draw_mean_ce"]),4)
        self.assertEqual(len(got["per_original_row_coherent_minus_mixed_mean_ce"]),3)

    def test_input_fail_closed_and_bounded_draw_count(self):
        good=np.array([2.,2.5,3.])
        c=good+.1
        mixes=np.stack([good+.2]*4)
        bads=[
            (good,c,mixes[:3]),
            (good,c,np.full((4,3),float("nan"))),
            (good[:2],c[:2],mixes[:,:2]),
            (good,c,np.full((4,3),-1.)),
            (good,c[:2],mixes),
        ]
        for b,y,m in bads:
            with self.assertRaises(ValueError):
                summarize_f3_mixed_ensemble(b,y,m)
        with self.assertRaises(ValueError):
            summarize_f3_mixed_ensemble(good,c,mixes,bootstraps=1)

    def test_original_probe_schedule_multiple_controls_and_restoration(self):
        class FakeMOE:
            route_geometry="c2c3_wave"
            route_probe_action=None
            route_probe_seed=0
        fake=FakeMOE()
        model=SimpleNamespace(blocks=[
            SimpleNamespace(moe=fake)])
        seen=[]
        def fake_ce(model,rows,device):
            seen.append((fake.route_probe_action,fake.route_probe_seed))
            delta={None:0.,"c3_relative_shift":-.1,
                   "c3_feature_exact_mixed_orientation_x":.2}[
                        fake.route_probe_action]
            return [5.+delta]*len(rows)
        with patch("scripts.audit_x369_f3_mixed_orientation_ensemble_ce.row_ce",
                   fake_ce):
            data=score_frozen_f3_ensemble(
                model,np.zeros((3,64),dtype=np.int64),"cpu",
                source_seed=42,null_seed_base=700,
                null_draws=4,trace_rows=True)
        self.assertEqual(len(seen),6)
        self.assertEqual(seen[0],(None,42))
        self.assertEqual(seen[1],("c3_relative_shift",42))
        self.assertEqual([p[1] for p in seen[2:]],[700,701,702,703])
        self.assertIsNone(fake.route_probe_action)
        self.assertEqual(data["seed_schedule"],[700,701,702,703])
        self.assertAlmostEqual(np.mean(data["coherent"]),4.9)

    def test_error_midrun_restores_original_probe(self):
        class FakeMOE:
            route_geometry="c2c3_wave"
            route_probe_action=None
            route_probe_seed=0
        fake=FakeMOE()
        model=SimpleNamespace(blocks=[
            SimpleNamespace(moe=fake)])
        def raise_on_coherent(model,rows,device):
            if fake.route_probe_action=="c3_relative_shift":
                raise RuntimeError("deliberately fail once")
            return [4.]*len(rows)
        with patch("scripts.audit_x369_f3_mixed_orientation_ensemble_ce.row_ce",
                   raise_on_coherent):
            with self.assertRaisesRegex(RuntimeError,"deliberately"):
                score_frozen_f3_ensemble(
                    model,np.zeros((3,64),dtype=np.int64),"cpu",
                    source_seed=41,null_draws=4)
        self.assertIsNone(fake.route_probe_action)
        with self.assertRaisesRegex(ValueError,"null draws"):
            score_frozen_f3_ensemble(
                model,np.zeros((3,64),dtype=np.int64),"cpu",
                source_seed=41,null_draws=1)
        self.assertIsNone(fake.route_probe_action)

if __name__=="__main__":
    unittest.main()
