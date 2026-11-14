"""CPU-only anti-promotion tests for Schaka's *actual* SGEMM kernel shim.

These tests exercise trace provenance, raw tensor hashes, the 100+ repeat
baseline and the same-image controls. They do not establish hardware parity.
"""
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.parse_gfx803_qkv_shim_ab import join, summarize, trace_kernels


class ShimABTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def trace(self, arm, name):
        root = self.root / arm
        root.mkdir()
        with (root / "1_marker_api_trace.csv").open("w", newline="") as f:
            c = csv.DictWriter(f, fieldnames=("Function", "Start_Timestamp", "End_Timestamp"))
            c.writeheader()
            c.writerow({"Function": "leech:qkv:sgemm-shim:" + arm,
                        "Start_Timestamp": 100, "End_Timestamp": 200})
        with (root / "1_kernel_trace.csv").open("w", newline="") as f:
            c = csv.DictWriter(f, fieldnames=("Kernel_Name", "Start_Timestamp",
                                               "End_Timestamp", "Correlation_Id"))
            c.writeheader()
            c.writerow({"Kernel_Name": name, "Start_Timestamp": 150,
                        "End_Timestamp": 160, "Correlation_Id": "99"})
        return root

    def values(self, arm, status="correct-at-tested-shape"):
        return {"mode": "shim", "arm": arm,
                "fixture_manifest_sha256": "b"*64, "teacher_sha256": "t"*64,
                "input_sha256": "i"*64, "weight_sha256": "w"*64,
                "rocblas_sha256": "a"*64, "shim_sha256": "c"*64,
                "shim_realpath": "/nix/store/source/libgfx803_sgemm_shim.so",
                "rocblas_realpath": "/nix/store/source/librocblas.so",
                "required_image_root": "/nix/store/source",
                "coherent_mapped_rocm_libraries": {
                    "librocblas.so": ["/nix/store/source/lib/librocblas.so"],
                    "libamdhip64.so": ["/nix/store/source/lib/libamdhip64.so"]},
                "torch_hip": "10.1.0", "gpu_arch": "gfx803",
                "gpu_device": "Radeon gfx803",
                "threshold": {"atol": 0.001, "rtol": 0.0001},
                "shim_actually_mapped": True, "admission": "denied",
                "repeats_requested": 100, "repeats_completed": 100,
                "status": status, "raw_first_output_sha256": "e"*64,
                "output_sha256_by_repeat": ["e"*64]*100}

    def test_per_arm_kernel_and_original_bytes(self):
        trace = trace_kernels(self.trace("on", "gfx803_sgemm_kernel"), "on")
        self.assertEqual(trace["status"], "identified")
        self.assertEqual(trace["kernels"][0]["name"], "gfx803_sgemm_kernel")
        report = join("on", self.values("on"), self.values("on"), trace)
        self.assertTrue(report["attribution_complete"])
        self.assertTrue(report["interposer_kernel_identified"])

    def test_profiler_observer_change_refuses_attribution(self):
        trace = trace_kernels(self.trace("off", "Cijk_Tensile_WGM8"), "off")
        unprofiled = self.values("off", status="incorrect-at-tested-shape")
        profiled = dict(unprofiled, output_sha256_by_repeat=["f"*64]*100,
                        raw_first_output_sha256="f"*64)
        self.assertFalse(join("off", unprofiled, profiled, trace)["attribution_complete"])

    def test_missing_selected_marker_refuses_attribution(self):
        trace = trace_kernels(self.trace("on", "gfx803_sgemm_kernel"), "off")
        self.assertNotEqual(trace["status"], "identified")

    def test_profiler_renamed_marker_is_not_physical_kernel(self):
        trace = trace_kernels(self.trace("on", "leech:qkv:sgemm-shim:on"), "on")
        self.assertNotEqual(trace["status"], "identified")

    def test_different_library_or_fixture_refuses_same_image_comparison(self):
        trace = {"status": "identified", "unique_kernel_names": ["gfx803_sgemm_kernel"]}
        first = self.values("on")
        second = dict(first, rocblas_sha256="f"*64)
        with self.assertRaisesRegex(ValueError, "rocblas_sha256"):
            join("on", first, second, trace)
        with self.assertRaisesRegex(ValueError, "threshold"):
            join("on", first, dict(first, threshold={"atol": 5}), trace)

    def test_truncated_or_replaced_raw_tensor_rejected(self):
        trace = {"status": "identified", "unique_kernel_names": ["gfx803_sgemm_kernel"]}
        first = self.values("on")
        with self.assertRaisesRegex(ValueError, "complete 100"):
            join("on", dict(first, repeats_completed=99), first, trace)
        with self.assertRaisesRegex(ValueError, "first output SHA"):
            join("on", dict(first, raw_first_output_sha256="z"*64), first, trace)
        with self.assertRaisesRegex(ValueError, "mapping"):
            join("on", dict(first, shim_actually_mapped=False), first, trace)

    def test_missing_or_foreign_mapped_runtime_is_rejected(self):
        trace={"status":"identified", "unique_kernel_names":["gfx803_sgemm_kernel"]}
        original=self.values("on")
        with self.assertRaisesRegex(ValueError,"missing actual mapped"):
            join("on", dict(original, coherent_mapped_rocm_libraries={}), original, trace)
        with self.assertRaisesRegex(ValueError,"foreign"):
            join("on", dict(original, coherent_mapped_rocm_libraries={
                "librocblas.so":["/foreign/librocblas.so"],
                "libamdhip64.so":["/nix/store/source/lib/libamdhip64.so"],
            }), original, trace)

    def test_shim_effect_requires_two_measured_distinct_kernels(self):
        ontrace = trace_kernels(self.trace("on", "gfx803_sgemm_kernel"), "on")
        offtrace = trace_kernels(self.trace("off", "Cijk_Tensile_WGM8"), "off")
        on = join("on", self.values("on"), self.values("on"), ontrace)
        offvals = self.values("off", status="incorrect-at-tested-shape")
        off = join("off", offvals, offvals, offtrace)
        summary = summarize(on, off)
        self.assertTrue(summary["shim_effect_at_original_QKV_shape"])
        self.assertFalse(summary["release_accepted"])
        self.assertFalse(summary["gpu_training_admitted"])
        self.assertFalse(summarize(on, dict(off, shim_sha256="another-build"))
                         ["shim_effect_at_original_QKV_shape"])
        self.assertFalse(summarize(on, dict(off, input_sha256="different-input"))
                         ["shim_effect_at_original_QKV_shape"])
        self.assertFalse(summarize(on, dict(off, threshold={"atol": 900}))
                         ["shim_effect_at_original_QKV_shape"])
        self.assertFalse(summarize(on, dict(off, unprofiled_repeats_completed=2))
                         ["shim_effect_at_original_QKV_shape"])
        self.assertFalse(summarize(on, dict(off, interposer_kernel_identified=True))
                         ["shim_effect_at_original_QKV_shape"])

    def test_both_correct_requires_profiler_and_does_not_claim_shim_causality(self):
        ontrace = trace_kernels(self.trace("on", "gfx803_sgemm_kernel"), "on")
        offtrace = trace_kernels(self.trace("off", "Cijk_Tensile_WGM1"), "off")
        on = join("on", self.values("on"), self.values("on"), ontrace)
        off = join("off", self.values("off"), self.values("off"), offtrace)
        s = summarize(on, off)
        self.assertEqual(s["status"], "both-arms-correct-no-interposer-necessity")
        self.assertTrue(s["both_shim_arms_correct_at_tested_QKV_shape"])
        self.assertFalse(s["shim_effect_at_original_QKV_shape"])
        self.assertFalse(s["gpu_training_admitted"])
        self.assertEqual(summarize(on, dict(off, attribution_complete=False))["status"],
                         "inconclusive-or-incomplete")

    def test_no_kernel_evidence_is_not_a_fix(self):
        on = join("on", self.values("on"), self.values("on"),
                  {"status": "missing-profile-kernel", "unique_kernel_names": []})
        offvals = self.values("off", status="incorrect-at-tested-shape")
        off = join("off", offvals, offvals,
                   {"status": "missing-profile-kernel", "unique_kernel_names": []})
        self.assertFalse(summarize(on, off)["shim_effect_at_original_QKV_shape"])


if __name__ == "__main__":
    unittest.main()
