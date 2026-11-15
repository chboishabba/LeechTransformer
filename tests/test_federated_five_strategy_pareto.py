import unittest
from scripts.federated_five_strategy_pareto import Strategy, compare


class FederatedFiveStrategyTests(unittest.TestCase):
    def test_decimal_line_speed(self):
        s=Strategy("one_way_megabyte",1_000_000,0,round_trips=0)
        self.assertAlmostEqual(s.latency_s(1e9,0),0.008)
        self.assertAlmostEqual(s.latency_s(1e10,0),0.0008)

    def test_rtt_dominates_token_requests(self):
        result=compare(activation_bytes=1536,bit_rate=1e9,rtt_s=.005)
        rows={x["name"]:x for x in result["strategies"]}
        self.assertAlmostEqual(rows["remote_raw"]["latency_ms"],5.024576)
        self.assertAlmostEqual(rows["remote_codec"]["latency_ms"],5.012288)
        self.assertEqual(result["best_modeled"],"local")

    def test_codec_cost_can_reverse_bytes_winner(self):
        result=compare(activation_bytes=1536,bit_rate=10e9,rtt_s=.005,
                       codec_overhead_s=.001,local_compute_s=.05)
        rows={x["name"]:x for x in result["strategies"]}
        self.assertLess(rows["remote_raw"]["latency_ms"],
                        rows["remote_codec"]["latency_ms"])

    def test_all_five_strategies_available(self):
        result=compare(activation_bytes=1_000_000,bit_rate=1e9,rtt_s=.005,
                       acknowledged_delta_bytes=1000,action_bytes=30)
        self.assertEqual(len(result["strategies"]),5)

if __name__=="__main__":
    unittest.main()
