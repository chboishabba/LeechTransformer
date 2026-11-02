import unittest
from scripts.punctured_nine_federated_cost import compare


class PuncturedNineTransferCostTests(unittest.TestCase):
    def test_per_expert_projection_vs_shared_raw(self):
        r=compare(384,4,2)
        self.assertEqual(r["one_expert_raw_pair_bytes"],3072)
        self.assertEqual(r["one_expert_router_only_quotient_bytes"],2304)
        self.assertEqual(r["projected_reduction_fraction_vs_unshared_pair"],.25)
        self.assertEqual(r["all_experts_shared_raw_bytes"],7680)
        self.assertEqual(r["all_experts_router_only_features_bytes"],9216)
        self.assertEqual(r["selected_activation_and_histories_bytes"],4608)
        self.assertEqual(r["selected_execution_source_owned_return_bytes"],6144)
        self.assertEqual(r["selected_complete_source_owned_bidirectional_bytes"],10752)
        self.assertEqual(r["selected_complete_peer_retained_bidirectional_bytes"],7744)
        self.assertFalse(r["complete_model_compute_available_from_router_quotient"])
        self.assertLess(r["link_serialization_ms"]["10"]["shared_full_inputs"],
                        r["link_serialization_ms"]["1"]["shared_full_inputs"])


if __name__=="__main__":
    unittest.main()
