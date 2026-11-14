"""F4 matched-arm development-only selection and frozen F5 guards."""
import copy
import unittest
from scripts.audit_x369_f4_development_fairness import adjudicate_f4_development

H=lambda v:format(v,"064x")

def fixtures():
    values=[
        (41,"c2_invariant",5.0,4.8),
        (41,"c2c3_wave",4.7,4.6),
        (42,"c2_invariant",5.2,4.9),
        (42,"c2c3_wave",5.3,4.7)]
    arms=[{"seed":s,"geometry":g,
           "steps":24,"best_step":24,
           "trained_parameter_count":48,
           "initial_fresh_ce":init,"best_fresh_ce":best,
           "candidate_checkpoint_sha256":H(s+int(g=="c2c3_wave"))}
          for s,g,init,best in values]
    paired=[
        {"seed":41,"c2_best_ce":4.8,"c2c3_best_ce":4.6,
         "c2c3_minus_c2":-0.2},
        {"seed":42,"c2_best_ce":4.9,"c2c3_best_ce":4.7,
         "c2c3_minus_c2":-0.2}]
    screen={"kind":"original_bounded_c2_vs_c2c3_router_matched_screen",
            "release_admitted":False,"fit_steps_per_arm":24,
            "validation_rows":3,
            "fresh_validation_sha256":H(123),
            "arms":arms,"paired":paired}
    split={"kind":"x369_f4_f6_fresh_real_document_token_split",
           "output_token_array_sha256":{
               "development":H(123),"reserved":H(456)},
           "source_document_ids":{
               "development":["d0","d1","d2"],
               "reserved":["r0","r1","r2"]},
           "source_text_sha256":{
               "development":[H(1),H(2),H(3)],
               "reserved":[H(4),H(5),H(6)]},
           "source_document_identity_inventory_claimed_complete":False}
    return screen,split


class X369F4FairnessTests(unittest.TestCase):
    def test_original_two_seed_decomposition_and_development_selection(self):
        screen,split=fixtures()
        got=adjudicate_f4_development(screen,split)
        self.assertEqual(got["seed_count"],2)
        self.assertEqual(got["selected_on_development_only"],"c2c3_wave")
        self.assertAlmostEqual(got["mean_initial_c3_minus_c2"],-.1)
        self.assertAlmostEqual(got["mean_c3_minus_c2_training_gain"],.1)
        self.assertAlmostEqual(got["mean_best_c3_minus_c2"],-.2)
        self.assertTrue(got["no_F5_rows_loaded_or_scored"])
        self.assertFalse(got["f5_document_independent_release_admitted"])
        self.assertFalse(got["causal_c3_learning_proved"])

    def test_development_split_digest_or_docs_mismatch_rejected(self):
        screen,split=fixtures()
        bad=copy.deepcopy(split)
        bad["output_token_array_sha256"]["development"]=H(99)
        with self.assertRaisesRegex(ValueError,"hash"):
            adjudicate_f4_development(screen,bad)
        bad=copy.deepcopy(split)
        bad["source_document_ids"]["reserved"][0]="d0"
        with self.assertRaisesRegex(ValueError,"collision"):
            adjudicate_f4_development(screen,bad)
        bad=copy.deepcopy(split)
        bad["source_text_sha256"]["development"][1]=H(1)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            adjudicate_f4_development(screen,bad)

    def test_invalid_paired_and_missing_nonfinite_ce_fails_closed(self):
        screen,split=fixtures()
        for change in ("missing","nan","corrupt"):
            bad=copy.deepcopy(screen)
            if change=="missing":bad["paired"][0].pop("c2c3_minus_c2")
            if change=="nan":bad["paired"][0]["c2c3_minus_c2"]=float("nan")
            if change=="corrupt":bad["paired"][0]["c2c3_minus_c2"]=-.3
            with self.assertRaisesRegex(ValueError,"paired"):
                adjudicate_f4_development(bad,split)
        bad=copy.deepcopy(screen)
        bad["arms"][0]["trained_parameter_count"]=123
        with self.assertRaisesRegex(ValueError,"parameter"):
            adjudicate_f4_development(bad,split)

    def test_best_ce_not_better_than_initial_rejected(self):
        screen,split=fixtures()
        screen["arms"][0]["best_fresh_ce"]=5.1
        with self.assertRaisesRegex(ValueError,"inconsistent"):
            adjudicate_f4_development(screen,split)

    def test_duplicate_seed_arm_and_missing_checkpoint_rejected(self):
        screen,split=fixtures()
        screen["arms"][1]["geometry"]="c2_invariant"
        with self.assertRaisesRegex(ValueError,"duplicate"):
            adjudicate_f4_development(screen,split)
        screen,split=fixtures()
        screen["arms"][0]["candidate_checkpoint_sha256"]=None
        with self.assertRaisesRegex(ValueError,"checkpoint"):
            adjudicate_f4_development(screen,split)

if __name__=="__main__":
    unittest.main()
