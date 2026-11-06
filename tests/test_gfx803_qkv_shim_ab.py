"""CPU-only regression tests for original Schaka SGEMM shim A/B attribution.

A shim path in LD_PRELOAD is not an admitted numerical result: require
actual mapped library, traced kernel and profile-vs-unprofiled equality.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.parse_gfx803_qkv_shim_ab import join, summarize, trace_kernels


class ShimABTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def trace(self, arm, name):
        root=self.root / arm
        root.mkdir()
        with (root/"1_marker_api_trace.csv").open("w",newline="") as f:
            c=csv.DictWriter(f,fieldnames=("Function","Start_Timestamp","End_Timestamp"))
            c.writeheader()
            c.writerow({"Function":"leech:qkv:sgemm-shim:"+arm,
                        "Start_Timestamp":100,"End_Timestamp":200})
        with (root/"1_kernel_trace.csv").open("w",newline="") as f:
            c=csv.DictWriter(f,fieldnames=("Kernel_Name","Start_Timestamp",
                                            "End_Timestamp","Correlation_Id"))
            c.writeheader()
            c.writerow({"Kernel_Name":name,"Start_Timestamp":150,
                        "End_Timestamp":160,"Correlation_Id":"99"})
        return root

    def values(self, arm, status="correct-at-tested-shape"):
        return {"mode":"shim","arm":arm,"fixture_manifest_sha256":"b"*64,
                "rocblas_sha256":"a"*64,"shim_sha256":"c"*64,
                "shim_realpath":"/nix/store/source/libgfx803_sgemm_shim.so",
                "rocblas_realpath":"/nix/store/source/librocblas.so",
                "required_image_root":"/nix/store/source",
                "status":status,"output_sha256_by_repeat":["e"*64,"e"*64]}

    def test_per_arm_kernel_and_original_bytes(self):
        trace=trace_kernels(self.trace("on","gfx803_sgemm_gfx803"),"on")
        self.assertEqual(trace["status"],"identified")
        self.assertEqual(trace["kernels"][0]["name"],"gfx803_sgemm_gfx803")
        report=join("on",self.values("on"),self.values("on"),trace)
        self.assertTrue(report["attribution_complete"])
        self.assertTrue(report["profiler_observer_effect_absent"])

    def test_no_profiler_change_must_be_observed(self):
        trace=trace_kernels(self.trace("off","Cijk_Tensile_WGM8"),"off")
        unprofiled=self.values("off",status="incorrect-at-tested-shape")
        profiled=dict(unprofiled,output_sha256_by_repeat=["f"*64])
        self.assertFalse(join("off",unprofiled,profiled,trace)["attribution_complete"])

    def test_missing_selected_marker_refuses_attribution(self):
        trace=trace_kernels(self.trace("on","gfx803_sgemm"),"off")
        self.assertNotEqual(trace["status"],"identified")

    def test_different_library_or_fixture_refuses_same_image_comparison(self):
        trace={"status":"identified","kernels":[{"name":"gfx803_sgemm"}]}
        first=self.values("on")
        second=dict(first,rocblas_sha256="f"*64)
        with self.assertRaisesRegex(ValueError,"rocblas_sha256"):
            join("on",first,second,trace)

    def test_shim_effect_requires_same_image_and_two_observer_free_arms(self):
        trace={"status":"identified","kernels":[{"name":"gfx803_sgemm"}]}
        on=join("on",self.values("on"),self.values("on"),trace)
        offvals=self.values("off",status="incorrect-at-tested-shape")
        off=join("off",offvals,offvals,trace)
        summary=summarize(on,off)
        self.assertTrue(summary["shim_effect_at_original_QKV_shape"])
        self.assertFalse(summary["release_accepted"])
        self.assertFalse(summary["gpu_training_admitted"])
        off=dict(off,shim_sha256="another-build")
        self.assertFalse(summarize(on,off)["shim_effect_at_original_QKV_shape"])

    def test_no_kernel_evidence_is_not_a_fix(self):
        on=join("on",self.values("on"),self.values("on"),
                {"status":"missing-profile-kernel","kernels":[]})
        offvals=self.values("off",status="incorrect-at-tested-shape")
        off=join("off",offvals,offvals,
                 {"status":"missing-profile-kernel","kernels":[]})
        self.assertFalse(summarize(on,off)["shim_effect_at_original_QKV_shape"])


if __name__ == "__main__":
    unittest.main()
