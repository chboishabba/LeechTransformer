"""End-to-end exact selection, revision policy, MAC and replay regression.

Only in-process evidence; does not claim real WAN or durable peer commit.
"""
import struct
import unittest

from scripts.federated_authenticated_session import SessionReceiver
from scripts.federated_joint_admitted_session import (
    pack_authenticated_selected, unpack_authenticated_selected)
from scripts.federated_joint_state_quotient import JointState


class FederatedAdmittedSelectedSessionTests(unittest.TestCase):
    def setUp(self):
        self.key=b"k"*32
        self.session=b"independent-session-1"
        self.state=JointState(
            activation=struct.pack("<24f",*[1.]*24),
            histories={1:struct.pack("<24f",*[2.]*24)})
        self.common=dict(state=self.state, model_revision="model-sha256:v7",
                         history_revision="history-sha256:v11",
                         rate_gbps=1,key=self.key,session=self.session,
                         sequence=0,selected=(1,),expert_count=3)
        self.receiver=SessionReceiver(
            self.key,self.session,b"history-sha256:v11")
        self.read=dict(receiver=self.receiver,expected_model="model-sha256:v7",
                       expected_history="history-sha256:v11",
                       expected_selected=(1,))

    @staticmethod
    def policy(choice="raw"):
        return {"model_revision":"model-sha256:v7",
                "history_revision":"history-sha256:v11",
                "heldout_latency_policy":{
                    "status":"measured-local-codec/ideal-link, not WAN",
                    "links_gbps":{"1":{"chosen_before_encoding":choice},
                                  "10":{"chosen_before_encoding":"raw"}}}}

    def test_raw_fallback_reopens_selected_state(self):
        packet,receipt=pack_authenticated_selected(
            policy=self.policy(),**self.common)
        self.assertEqual(receipt["policy_choice"],"raw")
        self.assertGreater(receipt["authenticated_envelope_bytes"],
                           receipt["inner_packet_bytes"])
        self.assertEqual(
            unpack_authenticated_selected(packet,**self.read),self.state)
        # Verification alone is NOT remote execution / ACK.
        self.assertEqual(self.receiver.next_sequence,0)
        self.assertEqual(self.receiver.expected_history_revision,
                         b"history-sha256:v11")

    def test_quotient_policy_uses_prior_choice_and_reopens(self):
        packet,receipt=pack_authenticated_selected(
            policy=self.policy("quotient"),**self.common)
        self.assertEqual(receipt["policy_choice"],"quotient")
        self.assertEqual(
            unpack_authenticated_selected(packet,**self.read),self.state)

    def test_revision_mismatch_and_tamper_fail_closed(self):
        bad=dict(self.policy())
        bad["model_revision"]="unapproved-model"
        with self.assertRaisesRegex(ValueError,"different revision"):
            pack_authenticated_selected(policy=bad,**self.common)
        pkt,_=pack_authenticated_selected(policy=self.policy(),**self.common)
        changed=bytearray(pkt)
        changed[-1]^=1
        with self.assertRaisesRegex(ValueError,"authentication failure"):
            unpack_authenticated_selected(bytes(changed),**self.read)
        with self.assertRaisesRegex(ValueError,"history revisions differ"):
            unpack_authenticated_selected(
                pkt,receiver=self.receiver,
                expected_model="model-sha256:v7",
                expected_history="wrong-history",expected_selected=(1,))
        self.assertEqual(self.receiver.next_sequence,0)

    def test_exact_replay_and_sequence_advance_only_after_commit(self):
        pkt,_=pack_authenticated_selected(policy=self.policy(),**self.common)
        self.assertEqual(unpack_authenticated_selected(pkt,**self.read),
                         self.state)
        self.receiver.commit(b"history-sha256:v12")
        self.assertEqual(self.receiver.next_sequence,1)
        with self.assertRaises(ValueError):
            unpack_authenticated_selected(
                pkt,receiver=self.receiver,expected_model="model-sha256:v7",
                expected_history="history-sha256:v12",expected_selected=(1,))


if __name__=="__main__":
    unittest.main()
