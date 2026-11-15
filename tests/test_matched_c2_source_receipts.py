"""Ground D3 negative quality comparison in the two verbatim source JSONs.

The reference artifacts were supplied by the experiment executor,
not synthesized from the PR narrative. These CPU-development
receipts do not certify any GPU kernel or heldout promotion.
"""
from pathlib import Path
import hashlib
import json
import unittest


ROOT=Path(__file__).resolve().parents[1]
C2=ROOT/"docs/receipts/c2_focused_matched_loss_20261103.json"
DENSE=ROOT/"docs/receipts/c2_dense_matched_loss_control_20261103.json"


class MatchedC2SourceReceiptsTests(unittest.TestCase):
    def test_verbatim_source_hashes(self):
        self.assertEqual(hashlib.sha256(C2.read_bytes()).hexdigest(),
                         "610d9ae8ca94347338fb807357e1593488c2bfd02f7a5cd04cd68f39cc6ae124")
        self.assertEqual(hashlib.sha256(DENSE.read_bytes()).hexdigest(),
                         "3e998aaf5f8bc37821de07ec0853250a9ba01d1406e5df6d47503269a7192b20")

    def test_same_model_and_optimiser_training_data(self):
        c=json.loads(C2.read_text())
        d=json.loads(DENSE.read_text())
        self.assertEqual(c["source_checkpoint_sha256"],d["teacher_checkpoint_sha256"])
        self.assertEqual(c["student_candidate_sha256"],d["student_checkpoint_sha256"])
        self.assertEqual(c["exported_optimizer_fit_file_sha256"],d["fit_ids_sha256"])
        self.assertEqual(c["teacher_heldout_ce"],d["dense_teacher_development_ce_nats"])
        self.assertEqual(c["optimizer_fit_rows"],d["fit_rows"])
        self.assertEqual(d["fit_rows"],45)
        self.assertEqual(d["development_rows"],13)
        self.assertEqual(d["epochs"][-1]["target_tokens"],819)
        self.assertTrue(c["matched_control_objective"])

    def test_quality_negative_and_no_false_promotion(self):
        c=json.loads(C2.read_text())
        d=json.loads(DENSE.read_text())
        gap=c["focused_student_heldout_ce"]-d["final_control_development_ce_nats"]
        self.assertAlmostEqual(gap,0.675808358105111,places=9)
        self.assertLess(d["final_control_development_ce_nats"],c["teacher_heldout_ce"])
        self.assertGreater(c["final_heldout_ce"],c["teacher_heldout_ce"])
        self.assertGreater(c["student_only_heldout_ce"],c["focused_student_heldout_ce"])
        self.assertEqual(c["all_quality_gates"][1],0.5)
        self.assertFalse(c["full_student_replacement_pass"])
        self.assertTrue(c["retained_dense_teacher_ffns"])
        self.assertFalse(d["full_teacher_free_promotion"])
        self.assertTrue(c["focus_is_diagnostic_only"])


if __name__=="__main__":
    unittest.main()
