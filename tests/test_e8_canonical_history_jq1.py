"""E8 Weyl shared history is a constructed trainable state invariant."""
import os
import struct
import json
import unittest
import torch
from models.e8_orbit_actions import (
    WEYL_ACTIONS,e8_orbit_bytes,e8_orbit_transform)
from models.e8_root_prior import e8_roots
from models.e8_shared_orbit_history_moe import E8SharedOrbitHistoryMoE
from scripts.federated_joint_state_quotient import (
    JointState,pack_joint_selected,unpack_joint_selected)


class E8CanonicalReuseTests(unittest.TestCase):
    def test_weyl_byte_actions_preserve_bits_and_root_set(self):
        x=os.urandom(8*4*9)
        for action in WEYL_ACTIONS:
            self.assertEqual(e8_orbit_bytes(
                e8_orbit_bytes(x,4,action),4,action),x)
        roots=e8_roots(dtype=torch.float64)
        allowed=set(map(tuple,roots.tolist()))
        for name in WEYL_ACTIONS:
            out=e8_orbit_transform(roots,name)
            self.assertEqual(set(map(tuple,out.tolist())),allowed)
        self.assertNotEqual(e8_orbit_bytes(x,4,"flip01"),x)

    def test_shared_expert_memories_form_exact_action_orbit_over_time(self):
        torch.manual_seed(18)
        model=E8SharedOrbitHistoryMoE(48,4,2,24,
            route_geometry="e8_root_invariant",transition_rank=24).eval()
        tokens=torch.randn(2,5,48)
        with torch.no_grad():
            baseline=model(tokens)
            self.assertTrue(torch.isfinite(baseline[0]).all())
            memory=baseline[1]
            base=model.extract_base(memory)
            self.assertTrue(torch.equal(memory,model.expand_history(base)))
            # A cached 2-token continuation uses only the canonical
            # history; regenerated E relative histories stay identical.
            p=model(tokens[:,:3])
            continued=model(tokens[:,3:],p[1])
            self.assertTrue(torch.allclose(
                torch.cat((p[0],continued[0]),dim=1),baseline[0],atol=1e-6))
            self.assertTrue(torch.allclose(continued[1],baseline[1],atol=1e-6))

    def test_independent_history_candidate_is_rejected_not_coerced(self):
        model=E8SharedOrbitHistoryMoE(48,4,2,24)
        bad=torch.randn(1,4,48)
        with self.assertRaises(ValueError):
            model.extract_base(bad)

    def test_jq1_e8_weyl_cross_expert_reopens_exact_bytes(self):
        x=os.urandom(48*4)
        raw_base=os.urandom(48*4)
        h0=raw_base
        h1=e8_orbit_bytes(raw_base,4,"swap01")
        h2=e8_orbit_bytes(raw_base,4,"flip01")
        h3=e8_orbit_bytes(raw_base,4,"negative8")
        snapshot=JointState(x,{0:h0,1:h1,2:h2,3:h3})
        packet=pack_joint_selected(snapshot,selected=(0,1,2,3),
            expert_count=4,itemsize=4,block_width=24,
            model_revision="E8-v1",history_revision="H-v1",
            use_e8_actions=True,use_site_actions=False)
        restored=unpack_joint_selected(packet,
            expected_model="E8-v1",expected_history="H-v1")
        self.assertEqual(restored,snapshot)
        headerlen=struct.unpack("<I",packet[4:8])[0]
        meta=json.loads(packet[8:8+headerlen])
        self.assertEqual(meta["codec"],"orbit")
        self.assertTrue(any(str(recipe["reference"]).startswith("e8_prev:")
                            for recipe in meta["recipes"]))
        raw=pack_joint_selected(snapshot,selected=(0,1,2,3),
            expert_count=4,itemsize=4,block_width=24,
            model_revision="E8-v1",history_revision="H-v1",
            permit_compression=False,use_e8_actions=True,
            use_site_actions=False)
        self.assertLess(len(packet),len(raw))


if __name__=="__main__":
    unittest.main()
