# LILA gfx803: two repair candidates (6 November 2026)

## Evidence separation

The reported **100-repeat full original-QKV solution scan** found an
incorrect WGM8 solution and a numerically correct WGM1 solution; subsequent
**3-repeat scan** completed 32/32 solutions. These are operator-scoped
measurements, not proof that default torch dispatch or C2 training is fixed.
The older real-model receipts failed with max ~4.127625 original QKV output
difference and ~28.56 logit difference in the dense gate. The patches in
this branch are **source-written, not a newly hardware-accepted environment**.

Schaka owns both his WGM mapping fix and separate FP32 SGEMM replacement,
including the 10.1 `libgfx803_sgemm_shim.so`; our code provides gated
consumer integration and truthful instrumentation.

## Path A: ROCm 6.4 numerical repair isolation

```bash
cd /path/to/LeechTransformer
python3 -m unittest tests.test_gfx803_lila_repair_contract -v
env -u LD_PRELOAD bash scripts/run_gfx803_lila_repair_gate.sh \
  --lane 6.4-cpu-linear \
  --receipt artifacts/gfx803_lila_64_hybrid_new.json \
  --repeats 3 -- \
  --teacher /path/to/data/best_model.pt \
  --student /path/to/C2/student_model.pt \
  --config /path/to/config.json \
  --train-ids /path/to/data/train_ids.npy
```

This **replaces all gfx803 FP32 `torch.nn.functional.linear` calls with
differentiable CPU GEMM**. It leaves non-linear, attention, GPU tensor,
optimizer and other kernels on the original GPU; backward propagates through
device copies to GPU parameters. It must not be described as forcing WGM1.
This is a slow hybrid *diagnostic* to identify what still breaks after the
known risky FP32 linear calculation is replaced. It **always denies training
admission**, even if every numeric comparison passes. Use it to isolate
residual GPU problems and decide whether a genuine WGM1 dispatch interposer
is worthwhile; do not benchmark it as GPU GEMM throughput.

For a genuine GPU-native WGM1 fix the original QKV `rocblas_gemm_ex`
solution IDs are library-local and original torch uses `rocblas_sgemm`.
A shape-aware dispatcher cannot be certified from IDs alone. A new
interposer would need to handle ordinary `rocblas_sgemm` and batched GEMMs,
forward and backward, mapped-library provenance, kernels and non-QKV shapes.
**That extension remains open.**

## Path B: ROCm 10.1 original Schaka SGEMM replacement

First establish an immutable 10.1 OCI image digest and import through
`gfx803_compat_graph` PR #5. The current checked-in source lock has no
accepted OCI image digest, and these commands cannot fill one in by guessing.
The importer supports `--with-venv`; confirm its selected Python is
executable with the imported ROCm and that the original shim is present.

Before doing model gates, run the separate original-QKV same-image shim
ON/OFF in `scripts/run_gfx803_qkv_shim_ab.sh`, using the matched selected
10.1 ROCm root and Python and one real kernel profiler. See
`docs/GFX803_QKV_WGM_SHIM_AB_RUNBOOK_20261106.md`. The direct LILA gate:

```bash
env -u LD_PRELOAD bash scripts/run_gfx803_lila_repair_gate.sh \
  --lane 10.1-schaka \
  --python /nix/store/...-selected-venv/bin/python \
  --rocm-root /nix/store/...-selected-rocm \
  --shim /nix/store/...-selected-rocm/lib/libgfx803_sgemm_shim.so \
  --receipt artifacts/gfx803_lila_101_shim_new.json \
  --repeats 3 -- \
  --teacher /path/to/data/best_model.pt \
  --student /path/to/C2/student_model.pt \
  --config /path/to/config.json \
  --train-ids /path/to/data/train_ids.npy
```

This runner strictly enables Schaka's genuine interposer before the selected
Python process starts, refuses inherited foreign preloads and old 6.4
`lib-compat`, and selects one pinned ROCm root. The original complete
`gfx803_moe_gpu_gate` then performs CPU↔GPU **dense forward, focused C2
forward, repeated GPU forwards, input probes, focused gradients and one AdamW
update**, using the authentic teacher/student/token files and without
launching training.

The gate's process checks that the shim is *actually mapped*, matches its
selected root and is not disabled, and—after model execution—that mapped
HIP/rocBLAS and the other relevant ROCm libraries come from the same root.
The presence of a library is not proof its SGEMM implementation executed.
Use the independent ON/OFF profiler evidence for that fact.

## Exact admission cut

- **A: hybrid repair logic source-written.** Real on-card checkpoint run
  and any resulting operator parity remain unexecuted here. Neither a passing
  hybrid test nor WGM1 solution index itself proves full GPU-native repair.
- **B: shim/runtime guards and original model gate source-written.** The
  actual digest lock, import/relocation, profiler-verified shim ON/OFF,
  hardware CPU/GPU comparisons and full stability qualification remain open.
- No training is launched by either runner; the existing checkpoint-bound
  trainer launcher retains its original full preflight gate. No CI/GPU
  acceptance is claimed merely because scripts were committed.
- Preserve fresh receipt paths and independent original logs. A 3-repeat
  diagnostic may build on the prior 100-repeat baseline, but cannot silently
  replace or impersonate that baseline.

To check the source-level contract:
`python3 -m unittest tests.test_gfx803_lila_repair_contract -v`
and `bash -n scripts/run_gfx803_lila_repair_gate.sh`.
