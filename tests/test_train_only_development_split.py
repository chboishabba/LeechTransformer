"""Development partition cannot consume independent final heldout."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.prepare_train_only_development import create_train_only_development


class TrainOnlyDevelopmentSplitTests(unittest.TestCase):
    def test_audited_deterministic_disjoint_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dataset=root/"data"
            output=root/"study"
            dataset.mkdir()
            rows=np.arange(20*6,dtype=np.int64).reshape(20,6)
            np.save(dataset/"train_ids.npy",rows)
            audit={"real_dataset_samples":True,
                "not_random_token_smoke":True,
                "record_separation":"train and heldout use distinct upstream splits or reserved document ranges, and duplicate text SHA256 across both is rejected",
                "split_audit":{"train":{"rows":20,
                    "window_array_sha256":hashlib.sha256(rows.tobytes()).hexdigest()}}}
            (dataset/"provenance.json").write_text(json.dumps(audit))
            report=create_train_only_development(dataset,output,
                                                 seed=7,development_fraction=.2)
            fit=np.load(output/"train_only_fit.npy",allow_pickle=False)
            dev=np.load(output/"train_only_development.npy",allow_pickle=False)
            self.assertEqual(fit.shape,(16,6))
            self.assertEqual(dev.shape,(4,6))
            self.assertEqual(len(set(report["fit_indices"]) |
                                 set(report["development_indices"])),20)
            self.assertFalse(set(report["fit_indices"]) &
                             set(report["development_indices"]))
            self.assertTrue(np.array_equal(np.load(dataset/"train_ids.npy"),rows))
            self.assertTrue(report["not_final_model_quality_evidence"])
            self.assertEqual(report["seed"],7)
            copied=create_train_only_development(
                dataset,root/"study_two",seed=7,development_fraction=.2)
            self.assertEqual(report["development_array_sha256"],
                             copied["development_array_sha256"])
            self.assertFalse((dataset/"heldout_ids.npy").exists())

    def test_bad_provenance_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            (base/"provenance.json").write_text(json.dumps({
                "real_dataset_samples":True,"not_random_token_smoke":False,
                "split_audit":{"train":{"rows":10,"window_array_sha256":"bad"}},
                "record_separation":"not audited"}))
            np.save(base/"train_ids.npy",np.arange(50).reshape(10,5))
            with self.assertRaisesRegex(ValueError,"source-audited"):
                create_train_only_development(base,base/"derived")
            with self.assertRaises(ValueError):
                create_train_only_development(base,base/"derived",
                                              development_fraction=.6)


if __name__=="__main__":
    unittest.main()
