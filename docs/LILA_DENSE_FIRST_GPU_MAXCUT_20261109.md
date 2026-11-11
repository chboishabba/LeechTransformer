# LILA dense-first GPU research and C2-independent max-cut — 9 Nov 2026

**Scope:** Original Leech-LILA neural research only. ROCm/HIP compiler,
TopK C++ dispatch, Nix binary-factory packaging, Cachix and upstream
Schaka patches are owned by the separate gfx803 worker. Do not edit
their code from this lane. Golden Agdas `{-# OPTIONS --safe #-}`
remain authoritative for formal claims; **no new Agda theorem or
kernel result is asserted here**. Original Leech-LILA author:
Anatolii Kornienko (DOI 10.5281/zenodo.18784424).

## Evidence already completed

- The restored relocated Schaka ROCm 10.1 candidate correctly
  computes **the original dense LILA forward** at tested shape
  batch 2 × sequence 64, using the source-locked teacher:
  CPU/HIP logit max abs error `2.288818359375e-05`;
  normalized layer input `1.9073486328125e-06`;
  teacher FFN `5.364418029785156e-07`;
  3 forward executions showed zero observed GPU drift.
  See `docs/receipts/gfx803_relocated_schaka_qkv_dense_reported_20261109.json`.
- The shim-ON/OFF isolated original QKV trial passed both arms
  with max error `3.0994415283203125e-06`.
  This demonstrates numerical correctness of this fixture,
  **not** that the shim is causally responsible.
- Separately, original Leech dense checkpoint quality and
  CPU-only capacity-matched control studies are settled enough
  to prevent false promotion of the signed/history C2 student.
  For the exact train-derived matched-development experiment:
  dense teacher CE **2.51320**, focused C2 hybrid **2.51585**,
  strict focused C2 **2.89186**, matched ordinary dense FFN
  control **2.21605**. Strict C2 loses to matched dense by
  **0.675808** CE nats. The C2 gate at 0.5 retains the teacher.
  Full student-only CE remains **11.55993**.
- The physical C2 GPU path has a separate `warptopk`
  crash attributed by the GPU worker. Their PR #5 at
  `5080c1c` proposes a gfx803-only dispatch replacement
  to use upstream `sbtopk`. Patch applicability and their
  reported 88 source tests **are not a physical C2 success**.
  Do not wait for that separate wheel rebuild to study the
  **dense** LILA model.

## New P0 owner: original dense training numerical parity

`scripts/lila_dense_training_parity.py` is a **standalone
no-C2 Python diagnostic**. It uses existing original-model
owners `checkpoint_parts`, `infer_config`, `LeechGPT`,
`gfx803_moe_gpu_gate.comparison`, and the source gate's
atomic progress receipt writer. It does NOT import a C2
checkpoint, call `torch.topk` routing or patch ROCm.

For **all original trainable dense parameters**, on both
CPU and HIP, it now checks:

1. Original teacher and real consecutive token-window hashes,
   strict dense checkpoint load, valid vocabulary/shape,
   actual mapped model/runtime identity.
2. At every step, genuine autoregressive cross-entropy,
   full CPU↔GPU logits and strict same-GPU repeated forward.
3. Every named trainable parameter gradient after backward;
   missing/nonfinite/out-of-tolerance gradients fail closed.
4. Identical AdamW learning rate, parameter ordering,
   gradient clipping; compare every parameter tensor *after*
   the update and the resulting original dense logits.
5. Continue for 20 matched real-token updates only if every
   preceding measurement passed. Atomically checkpoint
   stages in a separate progress JSON to retain evidence
   across process crashes, and never mutate the checkpoint
   on disk or save promoted model weights.

It records `admitted=true` **only for the specified bounded
dense numerical trajectory**, not for production GPU
training or general-purpose runtime correctness. Its
numerical thresholds follow existing per-element gate
defaults: forward/loss 1e-3 abs+rel, gradients and parameter
updates 2e-3 abs+rel, same-device repeats 1e-5 abs+rel.
These tolerances are not relaxed to excuse a GPU error.

### Existing relocated 10.1 runtime (not a new ROCm task)

The **other worker** supplies a validated Python, relocated
ROCm root and shim path; use the exact receipt-identified
store paths (never guess a store hash). Example once those
variables have been exported from the same proven runtime:

```bash
cd /path/to/LeechTransformer

# Export all three from the established Schaka 10.1 runtime:
# PY101=/nix/store/<actual-venv>/bin/python
# ROCM101=/nix/store/<actual-relocated-staged-rocm>
# SHIM101=$ROCM101/lib/libgfx803_sgemm_shim.so
# TEACHER=/path/to/the/original/data/best_model.pt
# IDS=/path/to/original/train_ids.npy
# CONFIG=/path/to/leech_dense_config.json

env -u GFX803_LINEAR_REPAIR \
  GFX803_EXPECT_ROCM_ROOT="$ROCM101" \
  GFX803_EXPECT_SHIM="$SHIM101" \
  GFX803_SGEMM_PRELOAD_MODE=shim-only \
  GFX803_SGEMM_SHIM_DISABLE=0 \
  HIP_LAUNCH_BLOCKING=1 \
  HSA_OVERRIDE_GFX_VERSION=8.0.3 \
  ROC_ENABLE_PRE_VEGA=1 \
  TORCH_BLAS_PREFER_HIPBLASLT=0 \
  LD_PRELOAD="$SHIM101" \
  LD_LIBRARY_PATH="$ROCM101/lib:$ROCM101/core-10.1/lib" \
  PYTHONPATH="$PWD" \
  "$PY101" -m scripts.lila_dense_training_parity \
    --teacher "$TEACHER" --config "$CONFIG" --train-ids "$IDS" \
    --batch 2 --steps 20 --repeats 3 \
    --progress-file artifacts/gfx803_lila_dense_train_20.progress.json \
    --receipt artifacts/gfx803_lila_dense_train_20.json
```

This is **not a full training run**. If the reporter exits
nonzero or a hardware reset prevents the final receipt,
do not infer admission. The input source hashes must match
the intended physical dense experiment. Pin exact Python,
ROCm and copied `librocblas` paths and retain their
attestation with the result. On an 8 GiB Polaris card,
verify VRAM headroom and keep batch 2/sequence 64 before
increasing size; record peak allocation and throughput
in a separate bounded smoke.

The CPU-only contract regression is:
```bash
python -m unittest tests.test_lila_dense_training_parity -v
```
It can validate the instrumented algorithm, **not**
replace a physical HIP run.

## Distinct P0: C2 model and quality work

The C2 `warptopk` dispatch fault belongs to the gfx803
worker; this thread should not patch Torch dispatch or
rebuild a wheel. Once their candidate wheel is admitted,
the existing C2 gate provides separate checkpoints for
strict C2 forward → local MSE+CE+KL → all named C2
gradients → one AdamW step → 20-step trajectory.

**Even then**, the dominant *scientific* C2 max-cut is
not merely a functioning GPU kernel. The strict student
must recover the **matched ordinary dense control's
development quality** and eventually satisfy the
all-layer teacher-free criterion on an independent final
heldout split. Opening a guard to 0.5 cannot count.

## Next Pareto / snowball cut

| Gate | Status as of 9 Nov | Decisive work |
| --- | --- | --- |
| L0 Original dense GPU forward parity | **Passed** at original shape | Preserve exact receipt |
| L1 Independent dense gradient+AdamW parity | **Source written**, not executed | On-card authentic 20-step receipt |
| L2 Bounded training throughput / KFD stability | **Open** | Peak VRAM, tokens/s, repeatability, reset logs |
| L3 Original dense training on additional real windows | **Open** | Multi-batch, source-provenance-preserving model checkpoint |
| C0 C2 warptopk compatibility | GPU worker's source patch, no physical verdict | Independent wheel and router acceptance |
| C1 Strict C2 GPU forward→backward→20 steps | **Open** | Model-side original shape parity after C0 |
| C2 C2 geometry/history quality superiority | **Failed tested comparison** | New hypothesis and genuinely matched negative control |
| C3 Fully teacher-free 12-layer model | **Open** | No dense FFNs, true teacher-free CE/quality, final heldout |
| F0 Golden Agda formal transports | Conditional source leaves, no new kernel claim | Concrete carrier/learned operator correspondence |
| P0 Leech codec / WAN Pareto | Earlier raw codec faster at 1/10 Gbit/s | Genuine measurement, not an inferred compression victory |

No ROCm/HIP/Nix/Cachix build or vendor patch is owned by
this LILA lane. Keep Schaka's WGM and separate SGEMM
corrections attributed to him, and keep DASHI/DASHI
geometry and other independent owners credited by name.


## 11 November 2026 — true original dense checkpoint/resume trainer

**P0 implementation complete at source level:** the separate
`scripts/train_lila_dense_gpu_smoke.py` now loads the
authentic original **dense** `best_model.pt` strictly,
without a C2 student, fake data, fresh randomly initialised
model or CPU fallback. All neural computation and AdamW
training reside on `cuda:0` in the existing relocated
gfx803 environment. It uses saved consecutive real
token-ID windows and the *original model's own*
`forward(inputs,targets=targets)` objective (next-token CE
plus optional configured resonance regularisation).
Per-step receipts report CE separately from the total
objective, gradient norm, true GPU-synchronized tokens/s
and peak allocated/reserved VRAM.

Its required **200-step smoke with exactly one process
restart** is run 0→100 then resume 100→200, with
25-step checkpoints. Every checkpoint atomically
includes model, full config, AdamW state, original data/
checkpoint/config digests, RNG state and completed
absolute step; every write is safely reloaded and
tensor-verified before proceeding. Existing checkpoint
and receipt filenames fail closed. The resumed process
checks original data/weights/hyperparameters, loads
both model and optimizer strictly and restores
Torch CPU/CUDA, NumPy and Python RNG states before
performing step 101.

Added algorithmic CPU-only tests
`tests/test_train_lila_dense_gpu_smoke.py`
(continuous 4-step vs 2+2 resume **exact model/optimizer/
loss equality**, with nonzero resonance weight) and
`tests/test_verify_lila_dense_gpu_smoke.py`
(incomplete/wrong-parent/missing-file negative cases).
`scripts/verify_lila_dense_gpu_smoke.py` is independent
of runtime and checks all eight 25-step source-hashed
checkpoints, 200 complete loss rows, and the exact
step-100 parent SHA. Runbook:
`docs/LILA_GPU_DENSE_200_STEP_SMOKE_20261111.md`.

**This tranche does not edit gfx803_compat_graph,
ROCm, HIP, TensorTopK/sbtopk, or Nix/Cachix.**
That is the separate worker's domain.
The already measured C2 admission and development
quality result remain independent:
C2 focused CE about **2.8899**, matched ordinary
dense CE **2.21605**. Training a dense checkpoint
does not establish geometric C2 quality improvement,
full teacher-free deployment or Golden Agda completion.

**Unpaid:** execute the physical 0→100 and
100→200 processes, retain eight real checkpoint
files/receipts and independently verify them; then
evaluate a newly trained model on source-disjoint
development/heldout data and benchmark broader
GPU shapes/reset endurance. This GitHub-only tranche
did not run the new tests or a GPU job.


### 11 November C2 status correction

The user reports **completed C2 GPU admission** and
a new quality rerun: focused C2 CE **2.8899**
versus parameter-matched dense CE **2.21605**.
Thus C2 GPU execution is no longer the missing
admission gate in this LILA roadmap. Do not wait
for additional ROCm/TopK work to run original
dense GPU-native training. The C2 architectural
quality claim remains unsupported. The separate
gfx803 worker retains ownership of upstream runtime
patches and broader compatibility.
