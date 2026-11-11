# LILA: original dense GPU-native 200-step checkpoint/resume smoke

**11 November 2026.** LILA research lane only. No ROCm/HIP,
TopK/sbtopk, Schaka, wheel, Nix or Cachix changes in this PR.
The existing **relocated Schaka candidate** is provided by the
separate gfx803 worker. No CPU/GPU comparison or C2 router is needed
to execute this bounded training workflow.

Original neural model: **Anatolii Kornienko, Leech-LILA**,
DOI 10.5281/zenodo.18784424. Source model `LeechGPT` must be loaded
**strictly** from the original `best_model.pt`, not reconstructed
with randomly initialised weights. The user-executed GPU quality rerun gives ordinary dense
control **2.21605 CE**, focused signed C2 **2.8899 CE**.
The user reports **C2 GPU admission complete**; a
working C2 GPU execution path is not evidence of a
quality advantage.
No trained C2 architectural superiority is inferred here.

## Implementation contracts

- `scripts/train_lila_dense_gpu_smoke.py`: HIP-only
  (`torch.version.hip` and `torch.cuda.is_available()` required),
  strict original dense teacher, existing safe checkpoint/config
  readers, consecutive **original** saved token-ID windows,
  native `cuda:0` model and AdamW tensors; no CPU fallback
  or `topk` routing.
- **Exact 200 steps in two explicit process runs:** run 1 ends
  at global step 100, run 2 resumes exactly at 100 and ends
  at 200. Same source checkpoint/token file SHA256,
  hyperparameters, model state, optimizer state,
  all Torch/NumPy/Python RNG state, and absolute-step
  deterministic row permutation. The run optimises the ORIGINAL
  LeechGPT target loss: next-token CE plus its configured optional
  geometric resonance regulariser. It is not an activation MSE diagnostic.
- **25-step checkpoint cadence:** 25,50,75,100,
  125,150,175,200. Each checkpoint contains model
  and AdamW state, original complete LeechConfig, RNG state, source/training settings,
  completed global step, full per-step trace. Each is
  atomically saved and immediately safely reloaded with
  byte-exact tensor/value verification before continuing.
  Existing checkpoint files or receipt paths are NEVER
  overwritten. The second process must verify a genuine
  continuation, not a fresh model training restart.
- Each step measures true GPU-synchronized training time,
  target tokens/s, next-token CE in nats **separately from
  the native CE+optional geometry training objective**,
  pre-clip gradient norm, row indices, peak allocated
  and reserved VRAM.
  NaN/Inf logits, loss, missing gradients and nonfinite
  norm abort the process. **Dropout remains enabled
  in training mode** on the real dense model, and RNG
  state is restored when resuming.
- `scripts/verify_lila_dense_gpu_smoke.py` verifies the
  exact 100+100 process boundary, shared source hashes,
  complete 200-step traces, eight checkpoint file hashes,
  reload acceptance, 100-step parent checkpoint SHA,
  real throughput and VRAM. The verifier uses no HIP
  runtime and makes no C2 or long-run acceptance claim.
- `tests/test_train_lila_dense_gpu_smoke.py` is the
  source contract: an original tiny dense `LeechGPT`
  is trained uninterrupted for four CPU test steps
  and separately restarted after two. It requires
  **identical learned model state, optimizer state and
  all per-step loss scalars** after the split run.
  CPU testing is exclusively algorithm/regression testing,
  not production model training or hardware admission.

## Real host execution

Use the **existing** verified relocated runtime, with
the exact `PY101`, `ROCM101` and `SHIM101`
paths from the GPU worker's successful dense-receipt.
Do not substitute assumed Nix hashes or mix extracted
6.4 libraries with the imported 10.1 stack.
All commands run from the LeechTransformer checkout.
Set actual source paths first:

```bash
export PY101=/nix/store/ACTUAL-IMPORTED-VENV/bin/python
export ROCM101=/nix/store/ACTUAL-RELOCATED-STAGED-ROCM
export SHIM101="$ROCM101/lib/libgfx803_sgemm_shim.so"
export TEACHER=/path/to/LeechTransformer/data/best_model.pt
export CONFIG=/path/to/LeechTransformer/artifacts/gfx803_c2/leech_dense_config.json
export ROWS=/path/to/the/original/train_only_fit.npy

# Must be a new, empty experiment directory.
export RUN="$PWD/artifacts/lila_dense_smoke_200"
mkdir -p "$RUN"
```

First bounded GPU-native process:

```bash
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
  "$PY101" -m scripts.train_lila_dense_gpu_smoke \
    --teacher "$TEACHER" --config "$CONFIG" \
    --train-ids "$ROWS" --steps 200 --stop-after 100 \
    --save-every 25 --batch 2 --seed 41 \
    --learning-rate 0.0003 --grad-clip 1.0 \
    --output-dir "$RUN/checkpoints" \
    --receipt "$RUN/first_100.json"
```

**Do not launch the resumed process unless**
`first_100.json` reports
`status=partial-restart-required`,
`end_step=100`, and
`checkpoints/dense_step_000100.pt` is present.
Restart in a **new Python process** with the SAME
runtime, source files and hyperparameters:

```bash
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
  "$PY101" -m scripts.train_lila_dense_gpu_smoke \
    --teacher "$TEACHER" --config "$CONFIG" \
    --train-ids "$ROWS" --steps 200 \
    --resume "$RUN/checkpoints/dense_step_000100.pt" \
    --save-every 25 --batch 2 --seed 41 \
    --learning-rate 0.0003 --grad-clip 1.0 \
    --output-dir "$RUN/checkpoints" \
    --receipt "$RUN/resumed_100_to_200.json"
```

Independently verify the two process receipts (this
check may run in an ordinary Python environment
with PyTorch/NumPy installed, not necessarily on GPU):

```bash
python -m scripts.verify_lila_dense_gpu_smoke \
  --first "$RUN/first_100.json" \
  --resumed "$RUN/resumed_100_to_200.json" \
  --output "$RUN/verified_200.json"
```

Expected final evidence: `verified=true`, step count 200,
eight independently rehashed files, step-100 parent SHA
matching the second process's source, and finite loss,
target tokens/s and VRAM for both runs. The trained
model state lives in
`checkpoints/dense_step_000200.pt`; it is not an
independently held-out final-model evaluation result.
It contains an optimizer/checkpoint wrapper, not a
plain standalone `best_model.pt`: use the new training
reader or extract `model_state_dict` through
`torch.load(..., weights_only=True)` when
performing independent inference.

## Status & remaining research cut

| Gate | Status | Evidence needed |
| --- | --- | --- |
| G0 Original dense GPU forward | **PASSED** previous physical test | Preserved old receipt |
| G1 Native 100-step dense training | **Source implemented, hardware open** | First GPU process and four reload checks |
| G2 Resumed global steps 101–200 | **Source implemented, hardware open** | Second GPU process, full optimizer/RNG restoration |
| G3 Independent 200-step source verifier | **Source implemented, execution open** | Eight file hashes and full trace |
| G4 Sustained/representative dense training | **Open** | Broader batches and longer hardware stability |
| C0 C2 GPU admission | **User reports COMPLETE** | Preserve original GPU receipt; runtime owned by other worker |
| C1 C2 matched quality improvement | **FAILED** latest GPU rerun | Focused CE 2.8899 versus matched dense 2.21605 |
| C2 Full teacher-free rollout | **Open** | No frozen FFNs and competitive independent heldout CE |

A successful bounded smoke establishes GPU-native
**checkpoint/resume mechanics and measured throughput**
for this original checkpoint and token shape. It does
not establish universal ROCm numerical correctness,
a lack of GPU resets over hours/days, or that signed
C2 is a quality improvement. No CPU↔HIP parity
rerun is required by THIS script.
