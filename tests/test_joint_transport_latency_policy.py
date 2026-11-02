"""Deterministic policy tests: packet bytes alone must not win."""
import unittest

from scripts.benchmark_joint_state_quotient import heldout_latency_policy


class JointLatencyPolicyTests(unittest.TestCase):
    @staticmethod
    def sample(*, encode_ms, decode_ms=0., raw_encode_ms=.001,
               raw_decode_ms=.001, wire_bytes=100, raw_wire_bytes=200):
        return {
            "wire_bytes":wire_bytes,"raw_wire_bytes":raw_wire_bytes,
            "encode_ms":encode_ms,"decode_ms":decode_ms,
            "raw_encode_ms":raw_encode_ms,"raw_decode_ms":raw_decode_ms}

    def test_small_packet_loses_when_encoding_dominates(self):
        samples=[self.sample(encode_ms=1.) for _ in range(8)]
        policy=heldout_latency_policy(samples,rtt_ms=5.,reply_bytes=1000)
        self.assertEqual(policy["status"],
                         "measured-local-codec/ideal-link, not WAN")
        for rate in ("1","10"):
            report=policy["links_gbps"][rate]
            self.assertTrue(report["raw_fallback"])
            self.assertEqual(report["chosen_before_encoding"],"raw")
            self.assertAlmostEqual(report["validation_policy_gain_ms"],0.)

    def test_only_calibration_frames_choose_mode(self):
        # The first half clearly benefits from JQ1 at a 1Gb link.
        # Later frames are deliberately worse, but cannot retroactively
        # change the already-selected pre-encoding policy.
        good=self.sample(encode_ms=.001,wire_bytes=100,
                         raw_wire_bytes=20000)
        bad=self.sample(encode_ms=2.,wire_bytes=100,
                        raw_wire_bytes=20000)
        policy=heldout_latency_policy([good]*4+[bad]*4,
            rtt_ms=5.,reply_bytes=1000,minimum_gain_ms=.01)
        rate=policy["links_gbps"]["1"]
        self.assertEqual(rate["chosen_before_encoding"],"quotient")
        self.assertLess(rate["validation_policy_gain_ms"],0.)
        self.assertEqual(rate["validation_policy_wire_bytes"],400)
        self.assertEqual(policy["untouched_validation_frames"],4)

    def test_insufficient_evidence_defaults_to_raw(self):
        report=heldout_latency_policy(
            [self.sample(encode_ms=0.) for _ in range(3)],
            rtt_ms=5.,reply_bytes=1000)
        self.assertTrue(report["raw_fallback"])
        self.assertEqual(report["minimum_samples"],4)


if __name__=="__main__":
    unittest.main()
