import os
import unittest
import struct
import json
from scripts.ieee_sign_orbit_codec import flip_ieee_signs
from scripts.punctured_nine_exact_site_actions import (
    ACTIONS,SITES,site_permutation,permute_punctured_nine_bytes)
from scripts.federated_joint_state_quotient import (
    JointState,pack_joint_selected,unpack_joint_selected)


class ExactPuncturedSiteOrbitTests(unittest.TestCase):
    def test_all_actions_are_pointwise_bijections_and_rot3_cubed(self):
        for name in ACTIONS:
            self.assertEqual(set(site_permutation(name)),set(range(8)))
        raw=os.urandom(24*4*4)
        c=permute_punctured_nine_bytes(raw,4,"rot3")
        c=permute_punctured_nine_bytes(c,4,"rot3")
        c=permute_punctured_nine_bytes(c,4,"rot3")
        self.assertEqual(c,raw)
        self.assertEqual(permute_punctured_nine_bytes(
            permute_punctured_nine_bytes(raw,4,"point_neg"),4,"point_neg"),raw)
        self.assertEqual(permute_punctured_nine_bytes(
            permute_punctured_nine_bytes(raw,4,"swap"),4,"swap"),raw)

    def test_joint_geometric_orbit_predictor_preserves_ieee_bits(self):
        x=os.urandom(384*4)
        selected=(0,2)
        hist0=permute_punctured_nine_bytes(x,4,"rot3")
        hist2=flip_ieee_signs(permute_punctured_nine_bytes(x,4,"swap"),4)
        state=JointState(x,{0:hist0,2:hist2})
        packet=pack_joint_selected(state,selected=selected,expert_count=4,
            itemsize=4,block_width=24,model_revision="model7",
            history_revision="history7")
        meta_length=struct.unpack("<I",packet[4:8])[0]
        header=json.loads(packet[8:8+meta_length])
        self.assertEqual(header["codec"],"orbit")
        self.assertTrue(header["site_actions"])
        self.assertTrue(any(recipe["reference"].startswith("sheet_x:") or
             recipe["reference"].startswith("neg_sheet_x:")
             for recipe in header["recipes"]))
        recovered=unpack_joint_selected(packet,expected_model="model7",
            expected_history="history7",expected_selected=selected)
        self.assertEqual(recovered,state)
        # Every bit preserved, including any IEEE NaN/sign payload.
        self.assertLess(len(packet),len(x)*2)

    def test_no_site_actions_ablation_and_layout_rejection(self):
        x=os.urandom(384*4)
        h=permute_punctured_nine_bytes(x,4,"shear")
        state=JointState(x,{0:h})
        old=pack_joint_selected(state,selected=(0,),expert_count=2,
            itemsize=4,block_width=24,model_revision="m",history_revision="h",
            use_site_actions=False)
        self.assertEqual(unpack_joint_selected(old,expected_model="m",
            expected_history="h").histories[0],h)
        with self.assertRaises(ValueError):
            permute_punctured_nine_bytes(b"short",4,"rot3")
        with self.assertRaises(ValueError):
            site_permutation("not-a-real-action")


if __name__=="__main__":
    unittest.main()
