# gfx803 / Polaris: Nix-owned, numerical-quality-gated C2 training

## Verified repository archaeology

The user's existing runtime owner is
[`chboishabba/gfx803_compat_graph`](https://github.com/chboishabba/gfx803_compat_graph),
NOT a new wheel stack. Its
`gfx803_flake_v1/flake.nix` supplies:

- `.#gfx803-pytorch-stack` — frozen extracted ROCm **6.4** Python/PyTorch
  control with old HSA/HIP compatibility userspace
- `.#gfx803-pytorch-stack-upgrade` — explicitly experimental preserved
  old-ABI upgrade lane (same extracted framework)
- `.#base` / `verify-gfx803-host` — device visibility diagnostics
- `scripts/host-docker-python.sh` — authoritative extracted torch
  launcher, library-path normalization and GCN workaround flags

The normal `nixpkgs` ROCm/PyTorch package is **not** interchangeable
with this old-ABI extracted runtime. The wrapper strips incompatible
Nix ROCm/device-library variables before invoking its own Python.
Do **not** install another PyTorch into the Nix shell or splice in
newer ROCm runtime libraries to make this experiment start.

**Critical historic observation from that same repository:**
`torch.cuda.is_available()==True` and a completed Leech generation
have NOT meant numerically correct outputs on the user's Polaris
host. Extracted ROCm 6.4 diverged from CPU as early as the first
block's `q_raw`; extracted 5.7 still suffered repeated-run drift.
`HIP_LAUNCH_BLOCKING=1` helped a narrower layout repro, not
the full end-to-end numerical proof. Long async workloads have
also triggered GPU reset / VRAM-loss incidents. The current
repo states CPU is its only trustworthy Leech output path until
a real numeric-gate receipt overturns that finding for a
precisely specified workload.

## Exact input/results provenance, 3 November 2026

The two newly uploaded source receipts are committed without
alteration as:
- `docs/receipts/c2_focused_matched_loss_20261103.json`
  original file SHA-256
  `610d9ae8ca94347338fb807357e1593488c2bfd02f7a5cd04cd68f39cc6ae124`;
- `docs/receipts/c2_dense_matched_loss_control_20261103.json`
  original file SHA-256
  `3e998aaf5f8bc37821de07ec0853250a9ba01d1406e5df6d47503269a7192b20`.

The receipt pair agrees **exactly** on the teacher checkpoint digest
`581fdc3b4b6d169a03e2b06896e10d454fd4f819b40a2836c1d5b76849e4c674`,
student checkpoint digest
`fc0187ec8f96359788a555f99e5fb86e7c95f1bb51572a77ef45bec537737b6d`,
and exported C2 optimizer-fit file SHA-256
`2f9aaae431d3bbb48805362bc331b9575a97218124af2d9d78541c85258b9ba6`.
This is a **train-only** comparison using 45 optimizer rows,
13 development rows and 819 target tokens. The same
model/training corpus and 3 epochs/seed/batch were used with a
local MSE + focused-student CE + teacher KL objective.

| Model | Train-only development CE, nats |
| --- | ---: |
| Frozen dense teacher | 2.513200809 |
| C2 guarded hybrid, gate at layer 1 = 0.5 | 2.515849615 |
| C2 strict focused layer-1 student | 2.891861335 |
| C2 all-layer full student-only rollout | 11.559929965 |
| Ordinary parameter-matched dense FFN control | **2.216052976** |

The dense control wins by **0.675808358** nats against strict
focused C2. Student odd+even parameter count **1,185,606**;
dense FFN **1,185,413** (193 difference). The C2 0.5 gate
passes internal CALIBRATION budget but does not improve CE on
this development split and keeps the dense teacher active.
The original code's `teacher_heldout_ce` field here means the
user-supplied train-derived **development** split, not a
previously untouched final test set. The full-student CE
of 11.56 is much worse and must not be promoted.

The negative quality finding is independent of ROCm compatibility:
moving this failed student to GPU does not make its loss smaller.
A successful GPU run would accelerate *further controlled tests*,
not validate a new compression/semantic advantage.

## New hard admission gate

`scripts/gfx803_moe_gpu_gate.py` uses the **existing** strict
`load_pair` and `build_guarded_hybrid` owners. It consumes
actual checkpoint files and consecutive tokenizer ID windows.
It refuses fake/synthetic data and CPU fallback. Tests:

1. ROCm HIP torch GPU visibility and device metadata, including
   `gcnArchName` where the runtime exposes it
2. Exact CPU ↔ GPU dense logits, normalized layer-1 input and
   teacher FFN outputs at the real input token shape
3. Repeated **same-GPU** dense forward, with strict 1e-5
   absolute/relative parity tolerances
4. CPU ↔ GPU STRICT focused C2 layer-1 logits and repeated same-GPU
   focused forward
5. CPU ↔ GPU local teacher-input MSE + focused next-token CE +
   teacher-logit KL, checked jointly
6. All learned selected-layer MoE/even gradient tensors;
   missing/NaN/Inf or differences over tolerance reject
7. A full, actual single-step AdamW optimizer update, comparing
   resulting CPU/GPU focused logits.

Default CPU/GPU forward and post-step tolerances are
`atol=rtol=1e-3`; gradient tolerances
`atol=rtol=2e-3`; repeated same-GPU
`atol=rtol=1e-5`. This is a quality *screen*, not a proof
of global FP32 equivalence. Any failed stage or HIP exception
writes a nonadmitted JSON receipt (when the process remains
alive) and returns nonzero. A GPU/system reset can prevent
receipt finalization. The script's result is bounded by
checkpoint, device, runtime, token shape and one training
step. Passing cannot certify long-running KFD safety.

`scripts/run_gfx803_moe_nix.sh`:
- enters **existing** `gfx803_flake_v1#gfx803-pytorch-stack`;
- runs the installed `scripts/host-docker-python.sh` (no
  alternative wheel/pip/Nix package);
- defaults to `HIP_LAUNCH_BLOCKING=1`;
- checks the **same** teacher/student/checkpoint/token training
  files for preflight and training, including batch size;
- only if the preflight returns PASS, runs the allowed
  `scripts.train_guarded_moe_continuation` module with
  forced `--device cuda`.

No command silently switches to CPU. No failed or missing
gate can create a promoted GPU training result.

### Exact host commands — first check, then train

These paths are based on the user's existing named project
and saved experimental artifacts. Run from LeechTransformer
checkout. No CPU hardware specifications need to be retyped.

```bash
# One-time if not already enabled in the existing compat workspace:
# cachix use gfx803-rocm

# Current host checkout; the launcher also detects this path automatically.
export GFX803_COMPAT_ROOT="/mnt/truenas/comet-net/gfx803_compat_graph"

# The control Nix shell and a basic visibility check:
(cd "$GFX803_COMPAT_ROOT/gfx803_flake_v1" &&
  nix develop .#base -c verify-gfx803-host)

# IMPORTANT: check-only first. No training will execute.
bash scripts/run_gfx803_moe_nix.sh \
  --teacher data/best_model.pt \
  --student /tmp/real_text_moe_jq1_smoke/c2_invariant/student_model.pt \
  --config /tmp/leech_dense_config.json \
  --train-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/matched_optimizer_fit.npy \
  --probe-batch 2 --repeats 3 \
  --receipt /tmp/real_text_moe_jq1_smoke/c2_invariant/gfx803_gpu_gate.json

# Only after the above command emits admitted=true can the
# same launcher run a checkpoint-bound real training command:
bash scripts/run_gfx803_moe_nix.sh \
  --teacher data/best_model.pt \
  --student /tmp/real_text_moe_jq1_smoke/c2_invariant/student_model.pt \
  --config /tmp/leech_dense_config.json \
  --train-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/matched_optimizer_fit.npy \
  --probe-batch 2 --repeats 3 \
  --receipt /tmp/real_text_moe_jq1_smoke/c2_invariant/gfx803_gpu_gate.json \
  -- scripts.train_guarded_moe_continuation \
  --teacher data/best_model.pt \
  --student /tmp/real_text_moe_jq1_smoke/c2_invariant/student_model.pt \
  --config /tmp/leech_dense_config.json \
  --train-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/matched_optimizer_fit.npy \
  --heldout-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/train_only_development.npy \
  --focus-layers 1 --matched-control-objective \
  --epochs 3 --seed 41 --batch 2 --learning-rate .0003 \
  --output /tmp/real_text_moe_jq1_smoke/c2_focused_gfx803.pt
```

The above retrains from the specified candidate checkpoint.
It does not resume an unseen saved optimizer state.
Training with 45 rows internally allocates its own calibration
subset, so this is a GPU smoke/quality investigation, NOT an
exact 45-optimizer-step replay. To make a truly identical
CPU↔GPU scientific comparison, use the same top-level training
input arrays, calibration indices and recorded optimizer-step
schedule for both sides, and report all actual row counts.

If the frozen control lane fails (historically likely),
run the same gate on the explicit old-ABI upgrade lane
by setting `GFX803_NIX_STACK=gfx803-pytorch-stack-upgrade`.
Do not reinterpret a lane-specific pass as universal.
The 5.7 and latest hybrid lanes remain diagnostic-only
until independently admitted.

### Tests

```bash
python -m unittest tests.test_gfx803_moe_gpu_gate -v
bash -n scripts/run_gfx803_moe_nix.sh
```

Local CPU tests only check guardrail code and shell grammar.
A real machine GPU receipt is required for admission.
Retain failed receipts and prior runtime notes in
`gfx803_compat_graph`; its
`DASHI/ComputerScience/GFX803CompatibilityReceiptExact.agda`
connection already distinguishes visibility, numerical
correctness, and reset safety. Neither source import nor
a Python unit test proves GPU truth or a trusted training
result on a particular physical Polaris host.

## Exact remaining max-cut

- **G0**: run the real GPU gate on the frozen ROCm 6.4
  Nix control and retain the full receipt
- **G1**: if G0 fails, locate first diverging tensor by reusing
  `gfx803_compat_graph/scripts/debug-leech-layerwise-divergence.py`
  and `scripts/capture-leech-minimal-repros.sh`; do not paper
  over mismatches with wider numeric tolerances
- **G2**: obtain GPU/CPU forward/backward parity, post-step
  parity and repeatability at actual batch/sequence sizes
- **G3**: complete a no-reset multi-epoch GPU/CPU comparison
  with matched schedule and data; benchmark tokens/s, peak
  VRAM, power/thermal state and wall-clock separately
- **G4**: only then admit a GPU workload-specific
  `ExecutionObservation` in the compat graph and update
  the train-quality roadmap; no change to Golden Agda
  proof status from GPU observations alone

**Status on 3 November:** source code and documentation
committed, existing frozen control/upgrade Nix entrypoints
verified in repository, new physical-device numeric gate NOT
executed. Earlier physical host measurements are contrary
to broad claims of correctness. CPU continues to be the
trusted baseline until G0–G3 are demonstrated.
