"""F3-B actual two-checkpoint source-level analysis acceptance.

Synthetic CE arrays test paired original document clustering and F5 firewall.
Physical immutable original RX580 seed41/42 checkpoints are not executed here.
"""
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scripts.run_x369_f3_two_seed_maxcut import pair_seed_checkpoints,run


def receipt(seed,delta=0.):
    n=6
    baseline=np.arange(2.,2.+n)
    coherent=baseline+delta
    mixed=np.stack([coherent+0.02*(j+1) for j in range(12)])
    return dict(source_seed=seed,
       frozen_F4_checkpoint_best_development_step=0 if seed==41 else 24,
       source_document_rows=n,
       original_F4_receipt_sha256="a"*64,
       original_probe_sha256="b"*64,
       original_probe_provenance_sha256="c"*64,
       reserved_F5_prior_array_sha256_only="d"*64,
       frozen_F4_checkpoint_sha256=("e" if seed==41 else "f")*64,
       null_seeds_preselected=list(range(40100,40112)),
       original_row_CE_coherent=coherent.tolist(),
       original_row_CE_baseline=baseline.tolist(),
       original_row_CE_mixed_by_seed=mixed.tolist())


class X369F3TwoSeedMaxCut(unittest.TestCase):
    def test_same_masks_paired_original_rows_and_interaction(self):
        a=receipt(41,0.)
        b=receipt(42,-.15)
        # Shared mask penalties give the SAME coherence benefit for
        # both seeds despite different overall prediction quality.
        got=pair_seed_checkpoints(a,b,bootstraps=199)
        self.assertAlmostEqual(got["checkpoint_interaction_seed42_minus_seed41_ce"],0.)
        self.assertEqual(got["orientation_draws_per_seed"],12)
        self.assertEqual(got["document_count"],6)
        self.assertAlmostEqual(got["seed41_fraction_masks_coherent_better"],1.)
        self.assertAlmostEqual(got["seed42_fraction_masks_coherent_better"],1.)
        self.assertEqual(len(got["paired_mask_interaction_CE"]),12)
        self.assertTrue(got["F5_raw_token_bytes_never_opened"])
        self.assertFalse(got["independent_source_seed_causal_training_effect_estimated"])
        self.assertFalse(got["physical_action_prediction_advantage_admitted"])

    def test_nontrivial_checkpoint_interaction_and_row_clusters(self):
        a=receipt(41)
        b=receipt(42)
        # Coherent only improves relative to mixed in second checkpoint.
        b["original_row_CE_coherent"]=[
            x-.05 for x in b["original_row_CE_coherent"]]
        got=pair_seed_checkpoints(a,b,bootstraps=199)
        self.assertAlmostEqual(
            got["checkpoint_interaction_seed42_minus_seed41_ce"],-.05)
        lo,hi=got["bootstrap_conditional_document_95pct_intervals"][
            "seed42_minus_seed41"]
        self.assertAlmostEqual(lo,-.05)
        self.assertAlmostEqual(hi,-.05)

    def test_source_identity_step_mask_and_shape_mismatches_fail_closed(self):
        a=receipt(41)
        b=receipt(42)
        cases=[]
        x=deepcopy(b);x["original_probe_sha256"]="0"*64;cases.append(x)
        x=deepcopy(b);x["null_seeds_preselected"][3]=900;cases.append(x)
        x=deepcopy(b);x["frozen_F4_checkpoint_best_development_step"]=0;cases.append(x)
        x=deepcopy(b);x["source_document_rows"]=7;cases.append(x)
        x=deepcopy(b);x["original_row_CE_mixed_by_seed"]=[[0.,1.]]*12;cases.append(x)
        x=deepcopy(b);x["original_row_CE_coherent"][2]=float("nan");cases.append(x)
        x=deepcopy(b);x["source_seed"]=41;cases.append(x)
        for item in cases:
            with self.subTest(case=cases.index(item)):
                with self.assertRaises(ValueError):
                    pair_seed_checkpoints(a,item,bootstraps=199)
        with self.assertRaisesRegex(ValueError,">=99"):
            pair_seed_checkpoints(a,b,bootstraps=2)

    def test_paired_wrapper_one_command_runs_both_and_never_reads_F5(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            for name in ("screen_receipt","source_candidate","original_train_ids",
                         "original_heldout_ids","selection_validation_ids",
                         "probe_ids","probe_provenance"):
                (root/name).write_text("pinned controlled source")
            args=SimpleNamespace(
                output_dir=root/"new",require_gfx803=False,device="cpu",
                screen_receipt=root/"screen_receipt",
                source_candidate=root/"source_candidate",
                original_train_ids=root/"original_train_ids",
                original_heldout_ids=root/"original_heldout_ids",
                selection_validation_ids=root/"selection_validation_ids",
                probe_ids=root/"probe_ids",
                probe_provenance=root/"probe_provenance",
                reserved_lockbox_sha256="d"*64,
                original_f3_receipt_41=None,original_f3_receipt_42=None,
                bootstraps=199)
            called=[]
            def fake_one(namespace):
                called.append((namespace.seed,namespace.null_seed_base,
                               namespace.null_draws,
                               namespace.reserved_lockbox_sha256))
                result=receipt(namespace.seed)
                namespace.output.write_text("{}")
                return result
            with patch("scripts.run_x369_f3_two_seed_maxcut.run_one",
                       side_effect=fake_one):
                got=run(args)
            self.assertEqual([v[0] for v in called],[41,42])
            self.assertTrue(all(v[1:]==(40100,12,"d"*64) for v in called))
            self.assertTrue((args.output_dir/"f3_two_seed_summary.json").exists())
            self.assertEqual(got["document_count"],6)
            with self.assertRaises(FileExistsError):
                run(args)

if __name__=="__main__":
    unittest.main()
