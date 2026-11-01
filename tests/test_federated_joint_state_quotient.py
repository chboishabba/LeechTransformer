"""JQ1 whole-state consumer quotient regression.

These are *constructed deterministic protocol examples*, not trained
checkpoint compression ratios or live network measurements.
"""
import json
import os
import struct
import unittest
from scripts.ieee_sign_orbit_codec import flip_ieee_signs
from scripts.federated_joint_state_quotient import (
    JointState,pack_joint_selected,unpack_joint_selected)


SIZE=384*4
SELECTED=(0,2)


def encode(state,**extra):
    return pack_joint_selected(state,selected=SELECTED,expert_count=4,
        itemsize=4,block_width=24,model_revision="weights-v1",
        history_revision="history-v7",**extra)


def decode(data,**extra):
    return unpack_joint_selected(data,expected_model="weights-v1",
        expected_history="history-v7",expected_selected=SELECTED,**extra)


class JointStateQuotientTests(unittest.TestCase):
    def setUp(self):
        # Explicit IEEE bit patterns (zero, signed zero, qNaN payload),
        # so sign-bit conjugation must preserve exact bytes.
        words=(b"\\x00\\x00\\x00\\x00"+b"\\x00\\x00\\x00\\x80"+
               b"\\x11\\x22\\xc0\\x7f"+b"\\x00\\x00\\x80\\x3f")
        x=bytearray(os.urandom(SIZE))
        x[:len(words)]=words
        x=bytes(x)
        self.state=JointState(x,{0:x,1:b"unused",
                                2:flip_ieee_signs(x,4),3:b"unused"})

    def test_joint_orbit_reuses_one_activation_for_two_histories(self):
        p=encode(self.state)
        reconstructed=decode(p)
        self.assertEqual(reconstructed.activation,self.state.activation)
        self.assertEqual(reconstructed.histories[0],self.state.histories[0])
        self.assertEqual(reconstructed.histories[2],self.state.histories[2])
        self.assertEqual(set(reconstructed.histories),set(SELECTED))
        self.assertLess(len(p),SIZE//2)
        hlen=struct.unpack("<I",p[4:8])[0]
        self.assertEqual(json.loads(p[8:8+hlen])["codec"],"orbit")

    def test_exact_reopen_with_unrelated_input_and_no_inflation(self):
        x=os.urandom(SIZE)
        state=JointState(x,{0:os.urandom(SIZE),2:os.urandom(SIZE)})
        p=encode(state)
        self.assertEqual(decode(p),state)
        # Worst case falls back to a complete *shared* raw packet.
        raw=encode(state,permit_compression=False)
        self.assertLessEqual(len(p),len(raw))

    def test_acknowledged_temporal_reuse_with_residual(self):
        baseline=self.state
        x=bytearray(baseline.activation)
        x[17]^=0x01
        h0=bytearray(baseline.histories[0]);h0[47]^=0x20
        h2=baseline.histories[2]
        now=JointState(bytes(x),{0:bytes(h0),2:h2})
        p=encode(now,acknowledged_baseline=baseline,
                 acknowledged_revision="ack-6")
        r=decode(p,acknowledged_baseline=baseline,
                 expected_ack_revision="ack-6")
        self.assertEqual(r,now)
        with self.assertRaises(ValueError):decode(p)
        with self.assertRaises(ValueError):decode(p,
            acknowledged_baseline=baseline,expected_ack_revision="stale")
        corrupted=JointState(baseline.activation,{0:baseline.histories[0],
                       2:b"0"*SIZE})
        with self.assertRaises(ValueError):decode(p,
            acknowledged_baseline=corrupted,expected_ack_revision="ack-6")

    def test_wrong_consumer_model_history_or_source_digest(self):
        p=encode(self.state)
        for values in ({"expected_model":"other"},
                       {"expected_history":"stale"},
                       {"expected_selected":(1,2)}):
            with self.assertRaises(ValueError):
                kwargs=dict(expected_model="weights-v1",
                            expected_history="history-v7",
                            expected_selected=SELECTED)
                kwargs.update(values)
                unpack_joint_selected(p,**kwargs)
        with self.assertRaises(ValueError):
            decode(p[:-1]+bytes([p[-1]^7]))

    def test_selected_only_and_duplicate_or_invalid_ids(self):
        with self.assertRaises(ValueError):
            pack_joint_selected(self.state,selected=(2,2),expert_count=4,
                itemsize=4,block_width=24,model_revision="w",
                history_revision="h")
        with self.assertRaises(ValueError):
            pack_joint_selected(self.state,selected=(1,2),expert_count=4,
                itemsize=4,block_width=24,model_revision="w",
                history_revision="h")
        with self.assertRaises(ValueError):decode(b"BAD")
        with self.assertRaises(ValueError):
            unpack_joint_selected(encode(self.state),
                expected_model="weights-v1",expected_history="history-v7",
                max_uncompressed=SIZE)


if __name__=="__main__":
    unittest.main()
