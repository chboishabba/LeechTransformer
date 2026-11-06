"""CPU-only contracts for exact-original-QKV WGM solution harness.

No GPU skip masquerades as a numerical pass. These tests validate byte
layout, source/fixture identity, rocBLAS beta ABI signature, profiler trace
join and release non-promotion, NOT real 6.4 or 10.1 kernel correctness.
"""
import csv
import ctypes
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.export_gfx803_qkv_fixture import verify_fixture, write_f32
from scripts.gfx803_qkv_solution_ab import (
    ABRefused, OP_T, OP_N, F32, SOLUTION_INDEX, make_rocblas_ex_arguments,
    make_signatures, bound_rocblas_path, allclose_diagnostics, verify_sample_fixture,
)
from scripts.parse_gfx803_qkv_solution_ab import (
    match_tensile, join_profile, summarize,
)


class FrozenQKVContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.folder = self.base / "fixture"
        self.folder.mkdir()
        m, n, k = 2, 4, 3
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype="<f4")
        w = np.arange(n*k, dtype="<f4").reshape(n, k) * .01
        ref = inp @ w.T
        tensors = {}
        for key, data in {"input": inp, "weight": w, "reference_f32": ref,
                          "reference_f64_cast_f32": ref.astype("<f4"),
                          "reference_full_torch": ref}.items():
            tensors[key] = write_f32(self.folder / (key + ".f32"), data)
        self.meta = {
            "schema_version": 1, "kind": "leech-qkv-original-fp32",
            "tensors": tensors, "gemm_logical_m_n_k": [m,n,k],
            "rocblas": {"transA_transB": ["T","N"], "M":n,"N":m,"K":k,
                        "lda":k,"ldb":k,"ldc":n},
            "teacher_sha256":"a"*64, "batch":1, "tokens":2,
        }
        (self.folder/"manifest.json").write_text(json.dumps(self.meta))

    def test_exact_leech_matmul_layout_and_input_sha(self):
        entry = verify_sample_fixture(self.folder,
                                      self.meta["tensors"]["input"]["sha256"])
        self.assertEqual(entry["gemm_logical_m_n_k"], [2,4,3])
        self.assertEqual(entry["rocblas"]["transA_transB"], ["T","N"])

    def test_corrupt_operand_rejected(self):
        path = self.folder/"weight.f32"
        b = path.read_bytes()
        path.write_bytes(b[:-1] + bytes([b[-1]^1]))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            verify_fixture(self.folder)

    def test_different_original_input_cannot_use_old_solution(self):
        with self.assertRaisesRegex(ValueError, "wrong original input"):
            verify_fixture(self.folder, expected_input_sha="b"*64)

    def test_dimension_or_transposition_mismatch_rejected(self):
        self.meta["rocblas"]["transA_transB"] = ["N","N"]
        (self.folder/"manifest.json").write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, "T,N"):
            verify_fixture(self.folder)


class RocblasBetaABIContractTests(unittest.TestCase):
    def test_original_qkv_transpose_and_pointer_order(self):
        a = ctypes.c_float(1)
        b = ctypes.c_float(0)
        args = make_rocblas_ex_arguments(ctypes.c_void_p(9),a,0xAAA,0xBBB,b,
                                          0xCCC,128,1152,384)
        self.assertEqual(len(args), 22)
        self.assertEqual((args[1],args[2]), (OP_T, OP_N))
        self.assertEqual((args[3],args[4],args[5]), (1152,128,384))
        self.assertEqual((args[8],args[9],args[11],args[12]), (F32,384,F32,384))
        self.assertEqual((args[16],args[19],args[21]), (1152,1152,SOLUTION_INDEX))
        self.assertEqual(args[7].value, 0xAAA)
        self.assertEqual(args[10].value, 0xBBB)
        self.assertEqual(args[17].value, 0xCCC)

    def test_exact_beta_signature_argument_counts(self):
        class Symbol:
            pass
        class ABI:
            rocblas_create_handle=Symbol()
            rocblas_destroy_handle=Symbol()
            rocblas_set_stream=Symbol()
            rocblas_gemm_ex=Symbol()
            rocblas_gemm_ex_get_solutions=Symbol()
        make_signatures(ABI)
        self.assertEqual(len(ABI.rocblas_gemm_ex.argtypes), 24)
        self.assertEqual(len(ABI.rocblas_gemm_ex_get_solutions.argtypes), 25)

    def test_rocblas_realpath_must_be_single_source(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            f=base/"librocblas.so.4.4"
            f.write_bytes(b"mock")
            maps=f"ab-ab r-xp 0 00:00 1 {f}\n"
            self.assertEqual(bound_rocblas_path(maps,base), str(f.resolve()))
            other=base.parent/"untrusted/librocblas.so.10"
            with self.assertRaises(ABRefused):
                bound_rocblas_path(maps + f"bc-bc r-xp 0 00:00 1 {other}\n",base)

    def test_bad_float_result_does_not_pass(self):
        x=np.array([1.,2.,3.], dtype=np.float32)
        self.assertTrue(allclose_diagnostics(x,x,1e-3,1e-4)["pass"])
        self.assertFalse(allclose_diagnostics(x+2,x,1e-3,1e-4)["pass"])
        y=x.copy(); y[1]=np.nan
        self.assertFalse(allclose_diagnostics(y,x,1e-3,1e-4)["pass"])


class RealTraceFormatTests(unittest.TestCase):
    def make_trace(self, base, symbol="Cijk_Alik_Bljk_SB_ISA803_K1_KLA_WGM8"):
        trace=base/"trace"
        trace.mkdir()
        with (trace/"123_marker_api_trace.csv").open("w",newline="") as f:
            wr=csv.DictWriter(f,fieldnames=["Domain","Function","Start_Timestamp","End_Timestamp"])
            wr.writeheader()
            wr.writerow(dict(Domain="MARKER_CORE_RANGE_API",
                             Function="leech:qkv:solution:7",
                             Start_Timestamp=99,End_Timestamp=99))
            wr.writerow(dict(Domain="MARKER_CORE_RANGE_API",
                             Function="leech:qkv:solution:7",
                             Start_Timestamp=100,End_Timestamp=200))
        with (trace/"123_kernel_trace.csv").open("w",newline="") as f:
            wr=csv.DictWriter(f,fieldnames=["Kernel_Name","Start_Timestamp",
                                            "End_Timestamp","Correlation_Id"])
            wr.writeheader()
            wr.writerow({"Kernel_Name": symbol,"Start_Timestamp":120,
                         "End_Timestamp":130,"Correlation_Id":48})
        return trace

    def test_real_rocprofv3_interval_join(self):
        with tempfile.TemporaryDirectory() as td:
            trace=self.make_trace(Path(td))
            r=match_tensile(trace,7)
            self.assertEqual(r["status"],"identified")
            self.assertEqual(r["wgm"],8)
            self.assertEqual(r["kernels"][0]["correlation_id"],"48")

    def test_renamed_kernel_is_not_claimed_as_wgm(self):
        with tempfile.TemporaryDirectory() as td:
            trace=self.make_trace(Path(td),symbol="leech:qkv:solution:7")
            self.assertNotEqual(match_tensile(trace,7)["status"],"identified")

    def test_wrong_solution_id_does_not_match_marker(self):
        with tempfile.TemporaryDirectory() as td:
            trace=self.make_trace(Path(td))
            self.assertEqual(match_tensile(trace,8)["status"],
                             "missing-or-ambiguous-marker")

    def test_observer_effect_guard(self):
        one={"solution_id":7,"rocblas_sha256":"a"*64,
             "rocblas_realpath":"/nix/store/mock/librocblas.so",
             "fixture_manifest_sha256":"b"*64, "input_sha256":"c"*64,
             "weight_sha256":"d"*64,"status":"incorrect-at-tested-shape",
             "output_sha256_by_repeat":["e"*64]}
        two=dict(one)
        kernel={"status":"identified","wgm":8,"unique_kernel_names":["Cijk...WGM8"]}
        self.assertTrue(join_profile(one,two,kernel)["attribution_complete"])
        two=dict(one,output_sha256_by_repeat=["f"*64])
        self.assertFalse(join_profile(one,two,kernel)["attribution_complete"])
        with self.assertRaisesRegex(ValueError,"library/fixture"):
            join_profile(one,dict(one,rocblas_sha256="f"*64),kernel)

    def test_summary_never_promotes_gpu_training(self):
        common={"rocblas_sha256":"a"*64, "fixture_manifest_sha256":"b"*64,
                "attribution_complete":True,
                "profile_observer_effect_absent_for_first_output":True}
        a=dict(common,solution_id=7,unprofiled_numerics="incorrect-at-tested-shape",
               trace={"status":"identified","wgm":8,
                      "unique_kernel_names":["Cijk_KLA_WGM8"]})
        b=dict(common,solution_id=8,unprofiled_numerics="correct-at-tested-shape",
               trace={"status":"identified","wgm":1,
                      "unique_kernel_names":["Cijk_KLA_WGM1"]})
        s=summarize([a,b])
        self.assertTrue(s["wgm8_wrong_and_wgm1_correct"])
        self.assertFalse(s["repair_accepted"])
        self.assertTrue(s["no_training_admission"])
        b=dict(b,rocblas_sha256="not-the-same-library")
        self.assertFalse(summarize([a,b])["wgm8_wrong_and_wgm1_correct"])


if __name__ == "__main__":
    unittest.main()
