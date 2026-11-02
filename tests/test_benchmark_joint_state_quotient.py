import unittest
import numpy as np
from scripts.benchmark_joint_state_quotient import benchmark_capture


class JointCaptureBenchmarkTests(unittest.TestCase):
    def capture(self):
        # Constructed case with shared activation, sign-related histories.
        rng=np.random.default_rng(17)
        x=rng.normal(size=(3,384)).astype("<f4")
        histories=np.empty((3,4,384),dtype="<f4")
        histories[:,0]=x
        histories[:,1]=rng.normal(size=(3,384)).astype("<f4")
        histories[:,2]=-x
        histories[:,3]=0
        selected=np.tile(np.array([[0,2]],dtype=np.int64),(3,1))
        return dict(activation=x,histories=histories,selected=selected)

    def test_exact_capture_and_complete_reply_cost(self):
        data=self.capture()
        r=benchmark_capture(data,history_placement="source")
        self.assertTrue(r["byte_exact_reconstruction"])
        self.assertGreater(r["quotient_vs_selected_complete_ratio"],1.)
        self.assertEqual(r["reply_bytes_assumption"],2*2*384*4)
        self.assertTrue(r["reply_includes_updated_histories"])
        self.assertIn(r["modeled_link"]["1"]["faster_in_measured_codec_plus_ideal_link"],
                      ("quotient","raw"))

    def test_peer_retained_history_requires_ack_replay_provision(self):
        data=self.capture()
        r=benchmark_capture(data,history_placement="peer",ack_bytes=64)
        self.assertEqual(r["reply_bytes_assumption"],2*384*4+64)
        self.assertTrue(r["peer_retained_history_requires_authenticated_ack"])
        data["baseline_activation"]=data["activation"].copy()
        data["baseline_histories"]=data["histories"].copy()
        with self.assertRaises(ValueError):
            benchmark_capture(data)
        again=benchmark_capture(data,ack_revision="remote-h17")
        self.assertTrue(again["acknowledged_baseline_supplied"])

    def test_reject_noninteger_routes_and_invalid_shapes(self):
        data=self.capture()
        data["selected"]=np.ones((3,2),dtype=np.float32)
        with self.assertRaises(ValueError):benchmark_capture(data)
        data=self.capture()
        data["selected"][:,0]=3
        with self.assertRaises(ValueError):benchmark_capture(data)


if __name__=="__main__":
    unittest.main()
