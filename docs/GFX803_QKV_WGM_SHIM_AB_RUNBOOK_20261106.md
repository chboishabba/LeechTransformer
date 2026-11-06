# Original Leech QKV — gfx803 WGM and Schaka shim A/B operator runbook
**6 November 2026 — source written, GPU execution open. GPU training blocked.**

## Attribution and claim boundary

**Schaka** authored the [gfx803 WGM investigation](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/patches/rocblas/wgm-miscompute.sh), measured **0/14 WGM8 vs 39/39 WGM1** assembly solutions, and wrote the [WorkGroupMapping fix](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/patches/rocblas/wgm-miscompute-source.patch). He also independently documented the [separate FP32 SGEMM miscompute](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/rocm6.4.4/KERNEL_BUGS.md) and wrote the [SGEMM preload shim](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/patches/rocblas/sgemm-shim/sgemm_shim.cpp). The beta solution scan reuses [his original same-build methodology](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/rocm6.4.4/tools/rocblas_solution_scan.cpp), not an independently invented discovery.

**First-party Leech/gfx803** experiments identify the original `ISA803...WGM8` QKV dispatch and the completed CPU-copied-input result: input SHA256 identical, yet CPU↔GPU QKV error **4.127624988555908**. That is **not** a WGM1 repair demonstration. Hardware A/B experiments below have *not yet run*.

## A. Same-build original ROCm 6.4 WGM scan

From the original detached Leech worktree:

```bash
cd /home/c/Documents/code/.leech-gpu-work

bash scripts/run_gfx803_qkv_solution_ab_nix.sh \
  --out-dir artifacts/gfx803_qkv_wgm_ab \
  --teacher /home/c/Documents/code/DASHIg/LeechTransformer/data/best_model.pt \
  --config artifacts/gfx803_c2/leech_dense_config.json \
  --token-ids artifacts/gfx803_c2/dataset/train_ids.npy \
  --batch 2 --tokens 64 --repeats 100
```

This script uses the original `gfx803-pytorch-stack` and `host-docker-python.sh` 6.4 control, not 10.1. It freezes *real* CPU LayerNorm1 QKV input, checkpoint weights and bias, plus CPU FP32/FP64 references; checks checkpoint and input SHA256; enumerates `rocblas_gemm_ex_get_solutions` within the *loaded same library*; and runs 100+ repetitions per actual solution. It records non-passing candidates and any hang/timeouts as negative outcomes, then uses ROCTX + rocprofv3 to identify the *actual* selected Tensile and WGM kernel. A profiled result must match the unprofiled output hash before the kernel is attributed.

For a bounded preliminary scan, append `--limit 8`. A run covering only part of the solution list cannot claim the WGM8-bad/WGM1-good contrast. To choose a known enumerated same-build index, append `--solution-id INDEX`. **Never reuse a solution index across rocBLAS releases.** Note that forced `rocblas_gemm_ex` is *not by itself* the original traced `rocblas_sgemm` path; see [AMD's documented beta API](https://rocm.docs.amd.com/projects/rocBLAS/en/docs-6.4.0/reference/beta-features.html).

Keep `fixture/manifest.json`, `enumeration.json`, `solutions/*/{unprofiled,profiled,joined}.json`, `wgm_ab_summary.json` and `artifact-index.json`. A marked WGM1 result passing the exact original operands would support a *scoped hypothesis*, not model-wide training clearance.

## B. Schaka patched 10.1 shim ON/OFF test

**Prerequisite:** a coherent, *actually imported* and checked 10.1 Schaka ROCm root plus matching Python/Torch environment. The image digest and physical runtime acceptance in gfx803 Nix PR #5 are currently open. **Do not** preload 10.1 into the existing 6.4 Python.

After that runtime exists, supply its *actual* Nix store paths in place of the placeholders:

```bash
cd /home/c/Documents/code/.leech-gpu-work

bash scripts/run_gfx803_qkv_shim_ab.sh \
  --out-dir artifacts/gfx803_qkv_shim_ab \
  --fixture artifacts/gfx803_qkv_wgm_ab/fixture \
  --python /nix/store/ACTUAL-10_1-PYTHON/bin/python \
  --rocm-root /nix/store/ACTUAL-10_1-ROCM \
  --shim /nix/store/ACTUAL-10_1-ROCM/lib/libgfx803_sgemm_shim.so \
  --hip-prefix 10.1 --repeats 100
```

The on/off arms use separate processes and *the same* runtime, source fixture and mapped SGEMM shim. The **ON** arm sets `GFX803_SGEMM_SHIM_DISABLE=0`; the **OFF** control sets `=1`, which is Schaka's own documented escape hatch. The diagnostic verifies the mapped rocBLAS and shim paths plus HIP version, records actual numerical outputs, uses ROCTX/rocprofv3, and compares profiled/unprofiled hashes. It does not prove every other process dependency is coherent; full Nix closure and device acceptance are separate gates.

Keep `on/unprofiled.json`, `off/unprofiled.json`, `on/trace_join.json`, `off/trace_join.json`, `shim_ab_summary.json` and `artifact-index.json`. Even a shim-on passing / shim-off failing observation is valid **only** for the tested QKV input and dimensions.

## Available CPU-only checks (not physical acceptance)

```bash
python -m pip install 'numpy>=1.24,<3'
python -m unittest discover -s tests -p 'test_gfx803_qkv_*ab.py' -v
bash -n scripts/run_gfx803_qkv_solution_ab_nix.sh
bash -n scripts/run_gfx803_qkv_shim_ab.sh
```

The [QKV contract CI workflow](../.github/workflows/gfx803-qkv-ab-contracts.yml) is source-written; no successful branch run has been verified. Source tests establish byte/layout/API/parser and non-promotion contracts—not physical GPU correctness. Original ComfyUI, kernel reset and C2 admission remain independent. **Do not launch GPU training** until original bare dense parity and later C2-specific gates pass.

## Source-implementation max-cut hardened on 6 November (hardware still open)

The source-only implementation now rejects **replaying old output directories**: give each invocation a fresh `--out-dir`; retain old directories untouched. Both operators reject fewer than **100** repeated unprofiled evaluations. WGM traces are bound to the same teacher, checkpoint input/weight SHA, original fixture, actual loaded rocBLAS real path and SHA, HIP architecture, GEMM geometry and `atol/rtol`. A joined result also requires a completed >=100-repeat unprofiled run, one completed profiled dispatch, saved first-result tensor SHA, equality of numerical classifications, and absence of profiler-induced first-output changes. The WGM parser's `supports-scoped-mapping-hypothesis` is never a complete-model fix and **never** a training gate.

The 10.1 SGEMM shim comparator separately requires a complete >=100-repeat baseline for **both** ON and OFF, identical actual ROCm/torch/device and tolerance identities, an actual mapped interposer, and raw-output fingerprints. ON must dispatch Schaka's real `gfx803_sgemm_kernel` from [his pinned header](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/patches/rocblas/sgemm-shim/gfx803_sgemm.h); OFF must not. Marker labels, memcpy/fill kernels and profiler-renamed symbols cannot impersonate the interposer. A scoped ON-good/OFF-bad contrast is not full dense/C2 proof.

The independent `gfx803_compat_graph` [candidate verifier](https://github.com/chboishabba/gfx803_compat_graph/blob/feat/gfx803-nix-binary-factory-20261104/scripts/gfx803_leech_qkv_admission.py) reopens the profiler CSV and original GPU FP32 tensors, checks the v2 joined-arm contract, reloads runtime library hashes and recomputes raw byte-level error. Do **not** copy or synthesize an acceptance receipt from the source-only CPU mocks.

**Remaining executable boundary:** run A on real 6.4 gfx803 (a WGM1 may or may not beat WGM8); resolve and lock the real Schaka 10.1 OCI digest and import a coherent Nix runtime; run B with shim ON/OFF; independently rehash and verify the bundle; then run bare dense, C2 gradient/optimizer, long-run stability and full-core Nix/Cachix gates. At present none of those new physical operations has been executed or claimed successful. Both PRs remain draft. No GPU training.

### Runtime-provenance hardening for the patched 10.1 comparison

Invoke the shim runner with **no inherited `LD_PRELOAD` in its supervisor**,
for example `env -u LD_PRELOAD bash scripts/run_gfx803_qkv_shim_ab.sh ...`.
Only the ON/OFF Python child receives Schaka's interposer; the timeout process
does not. The profiler must resolve from the **same selected 10.1 Nix runtime**
(`core-10.1/bin/rocprofv3` or `bin/rocprofv3` beneath the runtime root).
Using a host ROCm 6.4 profiler is deliberately refused. If the imported image
does not provide this profiler, `--skip-profiling` can collect numeric-only
evidence but **cannot** produce a kernel-attributed shim comparison.

Both Python processes now inspect their actual mapped HIP/rocBLAS and other
ROCm libraries in `/proc/self/maps`. A foreign mapped ROCm component,
even with a correct version label and correct selected rocBLAS path, is a
refusal. The independent gfx803 release-side verifier checks the reported
paths under the selected Nix root and matches them to the source trace record.
These source controls are not a claim that the original hardware has passed.
