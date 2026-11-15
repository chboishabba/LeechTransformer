"""CPU-only checks for *effective* gfx803 profile and ROCm library attestation.

These tests do not represent hardware measurements or change admission.
"""
import unittest
from unittest import mock

from scripts.gfx803_runtime_attestation import (
    SAFE_DEFAULTS, attest_runtime, mapped_gpu_libraries,
)


class EffectiveRuntimeAttestationTests(unittest.TestCase):
    def test_matching_flags_are_reported_without_approving_computation(self):
        result = attest_runtime(environ=SAFE_DEFAULTS, proc_maps="")
        self.assertTrue(result["matches_legacy_wrapper_defaults"])
        self.assertEqual(result["deviations_from_wrapper_defaults"], {})
        self.assertFalse(result["loaded_library_paths_observed"])
        self.assertIn("not-a-correctness-claim", result["scope"])

    def test_explicit_override_is_visible_not_silently_reset(self):
        env = dict(SAFE_DEFAULTS, MIOPEN_DEBUG_CONV_GEMM="1")
        result = attest_runtime(environ=env, proc_maps="")
        self.assertFalse(result["matches_legacy_wrapper_defaults"])
        self.assertEqual(result["deviations_from_wrapper_defaults"][
            "MIOPEN_DEBUG_CONV_GEMM"]["actual"], "1")

    def test_missing_flag_does_not_become_a_claimed_default(self):
        env = dict(SAFE_DEFAULTS)
        del env["MIOPEN_DEBUG_CONV_DIRECT"]
        result = attest_runtime(environ=env, proc_maps="")
        self.assertIsNone(result["selected_flags"]["MIOPEN_DEBUG_CONV_DIRECT"])
        self.assertFalse(result["matches_legacy_wrapper_defaults"])

    def test_loaded_rocm_libs_can_reveal_nix_store_mix(self):
        maps = (
            "7f4a000-7f4b000 r-xp 00000000 00:14 512 "
            "/mnt/truenas/comet-net/gfx803_compat_graph/lib-compat/librocblas.so.5\n"
            "7f4c000-7f4d000 r-xp 00000000 00:14 513 "
            "/nix/store/zzzzzzz-rocm/lib/libamdhip64.so.6\n"
            "7f4e000-7f4f000 r-xp 00000000 00:14 514 "
            "/usr/lib/libdrm.so.2\n"
        )
        result = attest_runtime(environ=SAFE_DEFAULTS, proc_maps=maps)
        self.assertEqual(result["nix_store_gpu_library_paths"], [
            "/nix/store/zzzzzzz-rocm/lib/libamdhip64.so.6"])
        self.assertIn("librocblas.so.5", result["loaded_library_paths"])
        self.assertIn("libdrm.so.2", result["loaded_library_paths"])

    def test_does_not_pretend_unmapped_library_was_loaded(self):
        self.assertEqual(mapped_gpu_libraries(
            "7f000-7f999 r-xp 0 00:00 7 /opt/other/libunrelated.so\n"), {})

    def test_env_profile_function_is_pure(self):
        old = dict(SAFE_DEFAULTS)
        env = dict(old)
        attest_runtime(environ=env, proc_maps="")
        self.assertEqual(env, old)


if __name__ == "__main__":
    unittest.main()
