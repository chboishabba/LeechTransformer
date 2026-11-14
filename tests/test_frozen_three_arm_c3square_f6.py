"""F6 independent three-arm winner precommit / paired provenance contracts."""
import unittest
from scripts.score_frozen_three_arm_c3square_f6 import (
    freeze_three_arm_winner,frozen_three_arm_checkpoints)
class ThreeArmF6FrozenContracts(unittest.TestCase):
    def test_development_only_winner_and_conservative_tie(self):
        s={"kind":"original_bounded_c2_c3_c3square_three_arm_development_screen",
           "paired":[
             {"seed":41,"c2_best_ce":3.1,"c2c3_best_ce":3.0,"c3square_best_ce":2.9},
             {"seed":42,"c2_best_ce":3.0,"c2c3_best_ce":3.1,"c3square_best_ce":2.8}]}
        chosen,means,seeds=freeze_three_arm_winner(s)
        self.assertEqual(chosen,"punctured9_wave_matched")
        self.assertEqual(seeds,{41,42})
        self.assertAlmostEqual(means[chosen],2.85)
        s["paired"][0]["c3square_best_ce"]=3.1
        s["paired"][1]["c3square_best_ce"]=3.0
        self.assertEqual(freeze_three_arm_winner(s)[0],"c2_invariant")
    def test_missing_or_duplicate_frozen_candidates_fail_closed(self):
        s={"kind":"original_bounded_c2_c3_c3square_three_arm_development_screen",
           "paired":[{"seed":41,"c2_best_ce":1.,"c2c3_best_ce":2.,
                      "c3square_best_ce":3.}]}
        with self.assertRaises(ValueError):
            frozen_three_arm_checkpoints({"arms":[]},{41})
        x={"seed":41,"geometry":"c2_invariant","candidate_checkpoint":"a",
           "candidate_checkpoint_sha256":"sha"}
        with self.assertRaises(ValueError):
            frozen_three_arm_checkpoints({"arms":[x,x]},{41})
        with self.assertRaises(ValueError):
            freeze_three_arm_winner({"kind":"wrong","paired":s["paired"]})
if __name__=="__main__":unittest.main()
