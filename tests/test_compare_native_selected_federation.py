"""Strict offline provenance comparison tests; no hardware implied."""
import unittest
from scripts.compare_native_selected_federation import compare

def receipts():
    local={
        "kind":"native_gpu_local_resident_selected_expert_baseline",
        "teacher_sha256":"a","student_sha256":"b","token_source_sha256":"c",
        "layer":1,"rows_completed":256,"source_verify_every":16,
        "rows_with_original_source_expert_parity_checked":16,
        "max_output_abs_error_on_verified_rows":0.,
        "elapsed_seconds":10.,"completed_rows_per_second":25.6,
        "physical_multiworker_execution":False,
        "p95_row_latency_seconds":.04}
    remote={
        "kind":"native_gpu_multiworker_source_selected",
        "source_teacher_sha256":"a","student_sha256":"b","token_source_sha256":"c",
        "layer":1,"rows_completed":256,"source_verify_every":16,
        "rows_with_original_source_expert_parity_checked":16,
        "max_output_abs_error_on_verified_rows":0.,
        "elapsed_seconds":8.,"completed_rows_per_second":32.,
        "workers":2,"all_configured_workers_responded":True,
        "hmac_authenticated":True,"physical_peer_device_independence_unverified":True,
        "request_bytes":4096,"response_bytes":4096,
        "p95_row_latency_seconds":.03}
    return local,remote

class ComparisonTests(unittest.TestCase):
    def test_same_work_ratio(self):
        local,remote=receipts()
        result=compare(local,remote)
        self.assertAlmostEqual(result["observed_remote_over_local_throughput_ratio"],1.25)
        self.assertFalse(result["hardware_independent_workers_attested"])
    def test_wrong_teacher_rejected(self):
        local,remote=receipts();remote["source_teacher_sha256"]="wrong"
        with self.assertRaises(ValueError):compare(local,remote)
    def test_wrong_layer_rejected(self):
        local,remote=receipts();remote["layer"]=2
        with self.assertRaises(ValueError):compare(local,remote)
    def test_different_verification_budget_rejected(self):
        local,remote=receipts();remote["rows_with_original_source_expert_parity_checked"]=1
        with self.assertRaises(ValueError):compare(local,remote)
    def test_unverified_rejected(self):
        local,remote=receipts();remote["max_output_abs_error_on_verified_rows"]=None
        with self.assertRaises(ValueError):compare(local,remote)
    def test_nonresponse_rejected(self):
        local,remote=receipts();remote["all_configured_workers_responded"]=False
        with self.assertRaises(ValueError):compare(local,remote)
    def test_nonmatching_scope_rejected(self):
        local,remote=receipts();local["kind"]="unrelated"
        with self.assertRaises(ValueError):compare(local,remote)

if __name__=="__main__":unittest.main()
