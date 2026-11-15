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


## 4 November 2026: physical GPU rejection, same-shape differential and Schaka PR #5 join

The **uploaded physical `gfx803_gpu_gate.json`** is preserved
**byte-identically** in
`docs/receipts/gfx803_control_leech_c2_rejected_20261104.json`,
SHA-256
`1e3b5640995eaf7ca93517a19cad7f85639926d1e9b817c8b6c0b296e516aa79`.
The host now correctly resolves the existing extracted
runtime under `/mnt/truenas/comet-net/gfx803_compat_graph`.
Actual runtime: `gfx803_pytorch_stack_control`,
`torch 2.6.0+gitdae14f9`, HIP `6.4.43483-a187df25c`,
Linux `7.3.0-2-cachyos-bore`, HIP launch blocking 1,
AMD gfx803 with 8 GiB. **No GPU training was admitted.**

Five dense teacher comparisons FAILED:

| Comparison | Max absolute difference | Status |
| --- | ---: | --- |
| CPU/HIP teacher logits, 2×64×2048 | 28.5572929382 | FAIL |
| CPU/HIP normalized layer-1 input, 2×64×384 | 2.3883841038 | FAIL |
| CPU/HIP teacher FFN output, 2×64×384 | 1.1904456615 | FAIL |
| GPU same-input repeat 1 | 0.2023429871 | FAIL |
| GPU same-input repeat 2 | 0.2370784283 | FAIL |

The original gate exited at `teacher numerical parity`,
**before running ANY C2 forward or gradient comparison**.
Thus this is evidence against the ordinary dense Leech
compute stack on this exact device/runtime/checkpoint and
NOT evidence that C2 introduces those errors.

The teacher SHA is
`581fdc3b4b6d169a03e2b06896e10d454fd4f819b40a2836c1d5b76849e4c674`.
The GPU gate's C2 candidate hash
`f26a5d9246bd0e1f89330b4864124835e3df82bb86e4d1da23fdf5fb3c09224c`
is **different** from the earlier matched CPU development
candidate (`fc0187ec8f96359788a555f99e5fb86e7c95f1bb51572a77ef45bec537737b6d`).
The teacher gate failure stands independently, but future
C2 GPU-vs-CPU quality comparisons must use the same
candidate identity on both sides.

The previous test flags established zero drift for
their scoped **MIOpen convolution** direct-only test.
They do not certify QKV/FFN/vocabulary GEMMs or attention.
The original physical gate did not record individual
`MIOPEN_DEBUG_CONV_*` settings, even though the operator
reports the direct-only profile was used.
Future receipts now include an explicit
`numerical_environment` map covering MIOpen, rocBLAS,
HIP and device overrides. Never reconstruct missing old
variables from an assumption.

### 5 November: was the old safe/working profile omitted?

**Source audit says probably not.** The Leech GPU launcher enters
`gfx803_flake_v1#gfx803-pytorch-stack`, and invokes the original
`gfx803_compat_graph/scripts/host-docker-python.sh`. That wrapper
explicitly supplies these default child-process values:

| Variable | Expected default |
| --- | --- |
| `HSA_OVERRIDE_GFX_VERSION` / `ROC_ENABLE_PRE_VEGA` | `8.0.3` / `1` |
| `MIOPEN_DEBUG_CONV_DIRECT` / `MIOPEN_DEBUG_CONV_GEMM` | `1` / `0` |
| `MIOPEN_DEBUG_CONV_WINOGRAD` / `MIOPEN_DEBUG_CONV_FFT` | `0` / `0` |
| `MIOPEN_DEBUG_CONV_DET` / `MIOPEN_DEBUG_DISABLE_FIND_DB` | `1` / `1` |
| `MIOPEN_FIND_ENFORCE` / `TORCH_BLAS_PREFER_HIPBLASLT` | `3` / `0` |
| `CUBLAS_WORKSPACE_CONFIG` / `HIP_LAUNCH_BLOCKING` | `:4096:8` / `1` |

The original November 4 receipt confirms blocking and GPU identity,
**but not individual MIOpen values**: a caller may override the wrapper's
defaults. Thus source defaults strongly indicate the stability configuration
was intended and used; they do not retrospectively prove the exact child
process environment. Nor does a safe MIOpen *convolution* solver imply
reliable rocBLAS GEMM/QKV/attention, as March 2026 records already warn.

A further issue is **dynamic-library provenance**. The original Nix
shell can contain ROCm development packages, while the extracted Python
wrapper explicitly prepends `lib-compat` and cleans device-library hints.
It does not itself prove *every* dynamically loaded HIP/rocBLAS library
came from the extracted control, because other inherited library search
paths can remain. A mixed ABI is a *testable hypothesis*, not an
established explanation for the failure.

The new read-only
[`gfx803_runtime_attestation.py`](../scripts/gfx803_runtime_attestation.py)
records effective legacy flags and the HIP/rocBLAS/MIOpen/LLVM libraries
actually mapped in the Leech process. Both the numerical gate and the
existing operator diagnostic now include
`effective_runtime_attestation` **after execution**, if the host
survives. This is runtime evidence, not a correctness pass. A separate
opt-in `--deterministic-algorithms` option on the *diagnostic only*
sets PyTorch's deterministic-algorithms mode. The prior gate did not
assert this mode; it must not be silently added to historical claims.

**Minimal same-model physical next cut**, preserving exactly the
uploaded checkpoint SHA256 and token-array SHA256:

1. Run the existing dense-only diagnostic **unchanged**, preserving the
   bare (unhooked) parity and repeatability measurements and the new
   effective runtime attestation.
2. Run it as a **separate process** with
   `--deterministic-algorithms`, same inputs and batch. Record whether
   PyTorch rejects unsupported kernels, which is a useful failure
   rather than an excuse to switch to CPU or train.
3. Inspect the actual first divergent module *only when the hooks do
   not perturb the uninstrumented GPU output*. If `tok_emb` or
   `block0.ln1` fails, investigate embedding/position/addition/normalization
   before attributing the failure to QKV GEMM. If `block0.qkv` is first,
   collect its selected rocBLAS/Tensile kernel and WGM solution identity.
4. If the loaded-map output contains a suspect foreign ROCm library,
   isolate that runtime ABI choice in an independent Nix/driver test.
   Do not disable or overwrite the original safe legacy control.
5. None of these diagnostics changes the **rejected G0** training
   receipt; exact same-object C2 parity, long-run stability and an
   accepted patched ROCm 10.1 runtime remain open.

The loaded library set is only a snapshot of libraries present when the
receipt is written; libraries not yet loaded must not be asserted absent
from the whole execution.

### First actual operator-boundary probe

Rather than retrying C2 training, use
`scripts/diagnose_gfx803_leech_operator_boundary.py`
and its Nix-only helper
`scripts/run_gfx803_dense_diagnosis_nix.sh`.
They use the **original model's own nn.Module forward
hooks**, not a hand-reimplemented attention formula, to
compare the same saved teacher/input on CPU and HIP:
embedding, block-0 LN1 and QKV, attention, LN2 and
both FFN linear projections, and the final head.
They also compare *uninstrumented* forwards before
using hooked traces: copying GPU tensors to CPU at
each hook can serialize execution, so a seemingly
improved trace can be a debugger **observer effect**.
A passing diagnostic never overrides the failed
main numerical gate, and this code **cannot launch
training**.

From LeechTransformer's root on the real host:

```bash
# The wrapper auto-detects /mnt/truenas/comet-net/gfx803_compat_graph,
# otherwise set GFX803_COMPAT_ROOT to the original checkout explicitly.
bash scripts/run_gfx803_dense_diagnosis_nix.sh \
  --teacher "$TEACHER_CHECKPOINT" \
  --student "$C2_CHECKPOINT" \
  --config "$LEECH_CONFIG" \
  --train-ids "$TOKEN_ID_WINDOWS" \
  --batch 2 --max-blocks 2 --repeats 3 \
  --receipt /tmp/gfx803_leech_operator_boundary.json
```

**Path contract:** set `TEACHER_CHECKPOINT`,
`C2_CHECKPOINT`, `LEECH_CONFIG` and
`TOKEN_ID_WINDOWS` to the actual files used in
the failed 4 November gate; these environment variables
are not inferred from its working directory. The
authoritative identity is the three SHA256 fields in
the original receipt. Reproduce all three to compare
precisely. The `--student` argument is needed by the
existing saved-pair checkpoint loader; the diagnostic
evaluates the dense teacher only.

This is a **new source-written diagnostic**, not a
record of hardware execution. The CPU-only stage
regressions live in
`tests/test_gfx803_leech_operator_boundary.py`.

### QKV kernel-attribution probe

After a hook-free operator diagnosis identifies `block0.qkv` as the first
observed divergent operation, use the QKV-only profiler probe. It derives the
input through the model's own embedding, positional addition and `block0.ln1`,
then calls the original `block0.attn.qkv` inside an ROCTX range named
`leech:block0:qkv`. No Python forward hooks are installed.

```bash
bash scripts/run_gfx803_qkv_kernel_profile_nix.sh \
  --out-dir artifacts/gfx803_qkv_profile \
  --teacher "$TEACHER_CHECKPOINT" \
  --config "$LEECH_CONFIG" \
  --token-ids "$TOKEN_ID_WINDOWS" \
  --batch 2 --repeats 3
```

It writes an unprofiled receipt, a profiled receipt, the `rocprofv3` trace and
`kernel_candidates.json`. The current real workload has logical QKV GEMM
dimensions `M=128, N=1152, K=384` for batch 2 and 64-token windows. A named
Tensile/WGM candidate is not yet a causal finding: the unprofiled/profiler
checksums and repeatability must first show that profiling did not change the
execution. GPU training remains blocked regardless of this diagnostic.

#### Measured QKV attribution (2026-11-06)

The persistent local receipt at `artifacts/gfx803_qkv_profile_unrenamed/`
passed that narrow observer-effect check: profiled and unprofiled GPU QKV
SHA-256 values were both
`7c8cd6717509955b38492c6c4c63923b9d062e243d94b45164aa883ed6ce94e6`, while
the CPU/GPU maximum absolute error remained `4.127624988555908`.

The sole QKV GEMM dispatch in the selected ROCTX region was:

```
Cijk_Alik_Bljk_SB_MT64x128x16_SN_AMAS3_BL1_BS1_EPS1_GLVWA4_GLVWB4_GRVWn1_GSU1_GSUASB_ISA803_K1_KLA_LPA4_LPB4_LRVW4_MMFGLC_NLCA1_NLCB1_PGR1_PLR1_SIA1_SU32_SUS256_SVW4_TT4_8_USFGROn1_VAW1_VSn1_VW4_VWB4_WG16_16_1_WGM8
```

This establishes the specific rocBLAS/Tensile WGM8 implementation executed by
the failing QKV shape under the recorded ROCm 6.4 compatibility runtime. It
does **not** establish that the kernel is the root cause: a replacement-library
or solution-forcing A/B run must still make this exact QKV receipt pass before
the attribution becomes causal.

### Schaka patch and binary-factory phase

[gfx803_compat_graph PR #5](https://github.com/chboishabba/gfx803_compat_graph/pull/5)
has a pinned Schaka ROCm 10.1 source lock,
rootless OCI-to-Nix importer, release evidence
policy and optional Cachix packages. At present
those binaries have **not** been imported, admitted
on the user's device, or published as a stable
verified binary release.

**History (before 6 November QKV tracing):** the initial Leech gate had not identified the selected Tensile solution, despite March profiler WGM8 precedent. The later real ROCTX/QKV trace **did identify** `Cijk_Alik_Bljk_...ISA803_K1_KLA...WGM8` and `rocblas_sgemm(T,N,1152,128,384)` inside the QKV range. That observation is our **independent first-party diagnostic**, not the authorship or causal validation of **Schaka's** original gfx803 assembly WGM!=1 investigation and source patch. His **separate FP32 SGEMM failure and replacement shim** must also be credited and tested independently; WGM1 alone is not established as the Leech repair. See the [primary-source ownership ledger](https://github.com/chboishabba/gfx803_compat_graph/blob/feat/gfx803-nix-binary-factory-20261104/docs/GFX803_QKV_ATTRIBUTION_LEDGER_20261106.md).
The distinct Schaka fused-Winograd patch concerns
convolutions, not automatically a dense transformer.

The native test sequence remains:
1. Preserve the negative ROCm 6.4 source receipt;
2. Identify the actual first failing original module
   **without suppressing bare forward drift**;
3. Trace selected Leech GEMM and its ISA/WGM/shape,
   reuse the first-party March 2026 kernel probes;
4. Import and integrity-check Schaka's pinned OCI/Nix
   candidate and run his original correctness suite;
5. Rerun the **same original dense** CPU/GPU and
   intra-GPU tests with exact model/token hashes;
6. Only after teacher parity passes, run the strict
   C2 gradient/AdamW gate and independent extended
   no-reset stability test; then measure throughput.

Golden Agda compatibility receipts remain query-scoped:
machine-visible ≠ numerically correct ≠ no reset.
The actual student's CPU quality still loses to the
matched ordinary dense control by ~0.676 CE nats on
train-derived development data. Better GPU runtimes
alone cannot discharge that model-quality gap.


## 6 November: exact QKV WGM8 assembly dispatch, two profiler archives

The uploaded user-produced *raw* profiler tarballs were examined directly,
rather than inferring kernel identity from the renamed trace:

- `gfx803_qkv_profile_unrenamed.tar.xz`, SHA256
  `20abfb4c72b2cde0da42e5a1e1792ece80a162911bc59d9119a5fade22c3d9aa`,
  preserves the **actual kernel name** `Cijk_Alik_Bljk_..._ISA803_K1_KLA_..._WGM8`.
- `gfx803_qkv_profile.tar.xz`, SHA256
  `53a564ea41201cd13db862706d9acb5eac216e0e6a4bcd0290da13051c20a538`,
  has profiler-renamed `Kernel_Name` labels (`leech:block0:qkv` and the
  `rocblas_sgemm` call); its empty kernel-candidate list cannot
  disprove the real assembly dispatch in the unrenamed trace.
- Profiler kernel dispatch start/end
  `223164589324724..223164589502806` is contained within
  the QKV ROCTX interval
  `223164558211470..223164589563416` and
  the rocBLAS SGEMM interval
  `223164562508475..223164589518217`.
- Column-major rocBLAS shape:
  `sgemm(T,N,M=1152,N=128,K=384)`;
  original logical `nn.Linear` shape: `128×384 → 128×1152`.
  Assembly tile `MT64×128×16` and `WGM8`.
- Profiler exits 0. Both tarballs' unprofiled and profiled QKV
  receipts agree on output SHA256: GPU
  `7c8cd6717509955b38492c6c4c63923b9d062e243d94b45164aa883ed6ce94e6`,
  CPU
  `1d73efbaf9d36813acf05a30817f4ce48d4da8bf168ac4dee3f8c8fa7165f3e2`.
  CPU↔GPU maximum absolute error is **4.1276249886**;
  isolated GPU repeats measured **0.0, 0.0**.
- **Historical native-input confounder, resolved by the later CPU-copy control:** CPU LN1 and native GPU LN1 QKV
  inputs were *not byte-identical* (respective SHA256
  `6e33dd589ce2fdc69f953fe1e195016076b883a7948f72a002de05fb49056cb5`,
  `0d38c2af1e2260d455f5730a6d4227594867ae4360b1b021e11b2ed96d8f7fd0`).
  Prior LN1 max absolute difference was only
  `3.5762786865234375e-7`, but this still does not amount to a
  strict same-GEMM-input test.

All source values and caveats have been recorded at
[`docs/receipts/gfx803_qkv_wgm8_attribution_20261106.json`](receipts/gfx803_qkv_wgm8_attribution_20261106.json).
No physical A/B replacement has run; these receipts are **attribution**,
not evidence that the WGM patch fixed the Leech model.

### Next physically executable maximum cut

1. **Input identity — DONE on the physical ROCm 6.4 control (see the 6 November CPU-copy result below):** without changing the teacher, QKV weights
   or tensor shape: `scripts/profile_gfx803_leech_qkv.py` now takes
   `--input-source cpu-copy` to compute the CPU-produced LN1 input and
   copy its exact FP32 bytes to the GPU. Compare against the original
   `--input-source native` mode in separate processes. For both,
   collect unprofiled, profiler-visible, max error, output hashes,
   actual loaded rocBLAS and any selected Tensile kernel. Neither
   mode can admit training.
2. **Reuse Schaka's 6.4.4 enumerator** at
   [`rocm6.4.4/tools/rocblas_solution_scan.cpp`](https://github.com/Schaka/rocm-gfx803/blob/main/rocm6.4.4/tools/rocblas_solution_scan.cpp)
   rather than reimplementing the beta `rocblas_gemm_ex_get_solutions`
   API. Adapt to the exact **T,N 1152×128×384** and the real saved QKV
   operand bytes, and explicitly execute candidate **WGM1** and **WGM8**
   solutions from the **same rocBLAS build** for 100+ repetitions.
   The solution index is **library-local** and must not be copied
   across ROCm 6.4 and 10.1. Check actual return statuses.
3. **Do not mistake WGM1 for the whole repair.** Schaka also showed
   a distinct pervasive standard-path FP32 GEMM miscompute and
   maintains `libgfx803_sgemm_shim.so` (built in his release),
   which the [final Dockerfile](https://github.com/Schaka/rocm-gfx803/blob/main/docker/final.Dockerfile)
   activates using `LD_PRELOAD`. On one **coherent** patched 10.1
   image run exact QKV with shim ON (intended runtime), then shim OFF
   via `GFX803_SGEMM_SHIM_DISABLE=1` as a negative-control
   *separate process*. Do **not** preload a 10.1 library into the
   original 6.4 process. Also record which kernel actually dispatches.
4. Only after original QKV and bare dense teacher parity and
   repeatability pass should the independent **C2-specific gradient
   and optimizer** admission test be rerun. Independent long-run
   GPU-reset safety and current Nix binary publication remain open.

The [gfx803 Nix PR #5](https://github.com/chboishabba/gfx803_compat_graph/pull/5)
now requires the published image's SGEMM preload shim to exist,
its OCI runtime environment to match, and eventual real preload
correctness receipts. Merely copying the image's `/opt/rocm` files
would otherwise discard the original container's essential `LD_PRELOAD`.

### 6 November implementation: executable original-QKV WGM and SGEMM A/B (operator-run)

**All source is committed; the operator must execute it on the physical card.**
This is a diagnostic implementation, not a statement that WGM1 or the shim
has already repaired the actual Leech model. The prior **same-input**
QKV test is already complete as reported in commit `469bf00`. It remains
**broken** (4.1276249886 output error), with the original `WGM8` kernel.

The new implementation preserves **two independent repair hypotheses**
and the original authorship:

- **Schaka** first established gfx803 assembly `WorkGroupMapping != 1`
  miscompute (his 0/14 WGM8 vs 39/39 WGM1) and authored
  [the WGM1 fix](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/patches/rocblas/wgm-miscompute-source.patch).
- **Schaka** independently diagnosed the remaining pervasive FP32
  SGEMM corruption and wrote
  [the replacement/interposer](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/patches/rocblas/sgemm-shim/sgemm_shim.cpp).
  The latter is not merely a WGM1 parameter change.

### Run 1 — original ROCm 6.4, same-build WGM solution scan

Use the **already-restored** legacy extracted runtime and real dense
checkpoint. Run this from the Leech PR worktree:

```bash
cd /home/c/Documents/code/.leech-gpu-work

bash scripts/run_gfx803_qkv_solution_ab_nix.sh \
  --out-dir artifacts/gfx803_qkv_wgm_ab \
  --teacher /home/c/Documents/code/DASHIg/LeechTransformer/data/best_model.pt \
  --config artifacts/gfx803_c2/leech_dense_config.json \
  --token-ids artifacts/gfx803_c2/dataset/train_ids.npy \
  --batch 2 --tokens 64 --repeats 100
```

It freezes **actual CPU LN1, weight, bias (if any), and both CPU
references** as little-endian float32 with individual SHA-256, and
refuses a checkpoint/token/config mismatch on resumed runs.
The old CPU-copy input hash
`6e33dd589ce2fdc69f953fe1e195016076b883a7948f72a002de05fb49056cb5`
and original logical QKV dimensions `128×384 → 128×1152` are
mandatory. There are no synthetic random operands.

For each *actual selected-library-local* solution ID, the runner:

1. Enumerates via the same rocBLAS 6.4 beta APIs used in
   [Schaka's original 6.4.4 solution scanner](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/rocm6.4.4/tools/rocblas_solution_scan.cpp).
   It does **not** transplant solution IDs to a different rocBLAS build.
2. Runs an independent GPU process for each ID at **100 repetitions**
   (customizable), recording actual return statuses, max and
   relative error against the original CPU reference, output hashes,
   and numerically clean/incorrect/inapplicable status.
3. Runs the same ID once under `rocprofv3 --selected-regions`, marking
   `leech:qkv:solution:ID`, and joins actual trace timestamps to
   `Cijk...WGM1`/`Cijk...WGM8` kernel symbols. A symbol absent due to
   profiler renaming is **not** presumed to be WGM1. The profiler
   observer-effect gate compares GPU output hashes.
4. Writes each `solutions/ID/unprofiled.json`,
   `solutions/ID/profiled.json`, `solutions/ID/joined.json`,
   `wgm_ab_summary.json`, and a complete `artifact-index.json` with
   source log hashes. Partial GPU crashes/timeouts remain in the
   per-solution directories; no silent success.

**Important API caveat:** the direct forced-solution API is
`rocblas_gemm_ex`, while the original torch QKV call selected
`rocblas_sgemm`. Only the profiler-identified kernel name, complete
same-object inputs, consistent shared library SHA, and actual
differential performance justify an attribution. Even an observed
WGM1 pass is **scope-specific**, not model- or product-level approval.

For a short preflight use `--limit 4 --repeats 3`, but it will *not*
complete the causal WGM1/WGM8 acceptance test unless the selected
solutions happen to include both classes. The default is all
solutions with 100 repetitions each. The scan is intentionally
resumable, bounded per solution by `--timeout-seconds 240`.

### Run 2 — Schaka 10.1, SAME image, shim enabled versus disabled

**Run only after the digest-locked 10.1 Nix import has a runnable
Python/ROCm closure.** Do not substitute the older extracted Python
or preload the new shim into ROCm 6.4. The command takes the *same*
frozen fixture created by Run 1:

```bash
bash scripts/run_gfx803_qkv_shim_ab.sh \
  --out-dir artifacts/gfx803_qkv_shim_101 \
  --fixture artifacts/gfx803_qkv_wgm_ab/fixture \
  --python /nix/store/EXACT-ROCM101-VENV/bin/python \
  --rocm-root /nix/store/EXACT-ROCM101-ROCM \
  --shim /nix/store/EXACT-ROCM101-ROCM/lib/libgfx803_sgemm_shim.so \
  --hip-prefix 10.1 --repeats 100
```

Replace the *literal placeholders* only with **real same-image store
paths from the imported candidate**; none are asserted to exist now.
Each arm runs as a separate process with the stock original
`LD_PRELOAD` mapping and `GFX803_SGEMM_SHIM_DISABLE=0` (ON)
versus `1` (OFF). It checks that the original shim is actually
mapped, the loaded rocBLAS belongs to the declared runtime, and
the HIP build corresponds to ROCm 10.1. The same original Torch
QKV projection runs 100 times per arm. Profiler markers
`leech:qkv:sgemm-shim:on/off` identify dispatched kernels, while
profiled and unprofiled outputs must agree to claim a scoped
interposer effect.

### Interpreting receipts without promoting an unproved fix

| Gate | Source / physical state |
| --- | --- |
| G0 exact CPU/GPU LN1 QKV input | **First-party physical control complete**, 4.1276249886 error still present |
| G1 actual WGM8 dispatch | **First-party physical trace complete**, original input/output errors independently scoped |
| G2 same-source rocBLAS forced WGM1 versus WGM8 | **Implemented**, no physical solution-scan receipts yet |
| G3 coherent 10.1 SGEMM shim ON versus OFF | **Implemented**, 10.1 imported runnable Python/OCI closure and real A/B still open |
| G4 full dense parity, C2 and extended no-reset stability | **Blocked until G2/G3 discriminators and original dense admission pass** |
| G5 stable Nix/Cachix release | **Open**, do not treat source implementation as hardware acceptance |

Tests: `python -m unittest discover -s tests -p 'test_gfx803_qkv_*ab.py' -v`;
`bash -n scripts/run_gfx803_qkv_solution_ab_nix.sh` and
`bash -n scripts/run_gfx803_qkv_shim_ab.sh`. GitHub CI tests source
contracts only; it cannot produce a hardware pass.

GPU training remains **blocked**. The original 2025 upstream-fixed
KFD callback regression and later independent WhisperX reset
observations remain distinct; none of the Schaka WGM/SGEMM results
is credited to the Nix integration project.

## 6 November: exact CPU-input QKV control executed

The first-party physical `--input-source cpu-copy` experiment completed under the same
ROCm 6.4 compatibility runtime (reported in [commit `469bf00`](https://github.com/chboishabba/LeechTransformer/commit/469bf0026a05bb4ed57271a9aa1b9d038128dd79); [bounded reported receipt](receipts/gfx803_qkv_cpu_copy_reported_20261106.json)). It copied the CPU-generated FP32 LN1 output
to the GPU before the original QKV linear layer. The copied CPU and GPU input
SHA-256 values both equal
`6e33dd589ce2fdc69f953fe1e195016076b883a7948f72a002de05fb49056cb5`, and
the receipt records `qkv_input_bytes_match: true` and maximum input difference
`0.0`.

Despite that exact input identity, the QKV CPU/GPU maximum absolute error
remained `4.127624988555908`; GPU repeats were `0.0, 0.0`. Both profiled and
unprofiled runs produced GPU output SHA-256
`5cf48cfd8c88fc29fbe4f10131232f7a6c245a5291039debada0e5f5349a1607` and
the selected region again dispatched the same `ISA803 ... WGM8` Tensile
kernel. The receipt and trace remain at
`artifacts/gfx803_qkv_cpu_copy/` locally.

Therefore, the previous sub-micro-unit native CPU/GPU LN1 difference is not
the cause of this isolated QKV miscompute. This is still not proof that WGM8
is the sole root cause: the required next causal A/B is a same-build WGM1
solution versus the observed WGM8 solution, followed by the coherent patched
runtime's SGEMM-shim on/off control.


### Attribution boundary and remaining production evidence

**Schaka** originated and validated the gfx803 assembly WGM!=1 investigation (his original RX 470 scan: 0/14 WGM8 vs 39/39 WGM1 assembly solutions, with 2/2 source-language WGM8 correct) and the **distinct** repeated FP32 SGEMM correctness failure/SGEMM preload shim. Primary immutable sources are his [WGM study](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/patches/rocblas/wgm-miscompute.sh), [WGM patch](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/patches/rocblas/wgm-miscompute-source.patch), [independent FP32 research](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/rocm6.4.4/KERNEL_BUGS.md) and [SGEMM shim](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/patches/rocblas/sgemm-shim/sgemm_shim.cpp). These patches and his measurements are **not** authored by LeechTransformer or the Nix factory.

**First-party result** from [Leech commit `469bf00`](https://github.com/chboishabba/LeechTransformer/commit/469bf0026a05bb4ed57271a9aa1b9d038128dd79): exact CPU-generated QKV inputs copied into the user's physical gfx803 control produce unchanged 4.1276249886 error on the selected WGM8 assembly dispatch. This eliminates upstream LN1 input-byte mismatch as the *isolated* QKV explanation but **does not identify a repaired WGM1 solution, prove the WGM bug is solely responsible, establish that Schaka's SGEMM replacement passes this model, or admit C2 training**. The report names local `artifacts/gfx803_qkv_cpu_copy/`; the raw local files were not uploaded/committed with that report, so the linked JSON explicitly records the evidence as a first-party account, **not independently rehashed raw receipt**. The original pre-control input-mismatch record remains unmodified for historical provenance.

**Next exact physical cut:** original same-input/shape checkpoint → WGM8 versus **same-rocBLAS-build WGM1 candidate** via Schaka's enumerator (100+ repetitions, real solution names/statuses) → coherent pinned ROCm 10.1 **shim ON versus OFF** in separate processes → original bare dense parity → C2 gradient/optimizer gate if all prior gates pass. Retain the original 6.4 rejection, 2025 upstream KFD fix, and later WhisperX hangs as separate ownership/evidence tracks. Until then, no accepted GPU training or modern Nix/Cachix release.

## 6 November — completed QKV A/B operator implementation (hardware A/B still open)

The [original-operand WGM / Schaka SGEMM-shim runbook](GFX803_QKV_WGM_SHIM_AB_RUNBOOK_20261106.md) now provides two distinct operator commands. They reuse [Schaka's original rocBLAS beta solution-scanning method](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/rocm6.4.4/tools/rocblas_solution_scan.cpp), without claiming authorship of either his WGM mapping patch or independent FP32 SGEMM replacement.

Source added: `export_gfx803_qkv_fixture.py` (strict checkpoint and input SHA, real QKV operands and CPU reference), `gfx803_qkv_solution_ab.py` (same-rocBLAS-library enumerator/100-repetition solution selection and separate 10.1 on/off shim), `gfx803_qkv_roctx.py`, both kernel-trace parsers, `run_gfx803_qkv_solution_ab_nix.sh`, and `run_gfx803_qkv_shim_ab.sh`. Numeric output bytes, selected source library SHA, profiler observer effects, kernel names and process failures are reported separately. The shim wrapper now applies `LD_PRELOAD` only to the actual child process and fails closed on incomplete profile joins. CPU-only contracts and CI source are present.

**Not claimed:** neither new hardware A/B has been run, the 10.1 image digest/closure is not accepted, no Leech dense parity or C2 gradient gate passed, and no GPU training started. The original measured 4.127625 QKV failure remains the expected negative control until the new physical runs establish otherwise.
