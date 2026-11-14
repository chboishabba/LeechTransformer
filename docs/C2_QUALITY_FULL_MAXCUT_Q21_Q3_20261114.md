# Leech C₂ quality maximum cut Q2.1–Q3 — 14 November 2026

## Source owners and scientific target

Original model: Anatolii Kornienko, LeechTransformer / LILA. This integration
extends the existing `HistoryOrbitMoE` and `LeechGPT` under PR #4. Dust
activation credit and forward-only algorithm belong to Dahal, Mandal,
Gülbahar, Vegesna; in-repo Agda/Python donors reside on DASHI PR #1265.
Do not equate Sovereign-LILA E8GPT `from_e8.weight` and Leech C2
`even_student[-1]`: they are different original operators.

**Measured existing native run (user):** 80.1 min, dense teacher held-out
CE 2.5834, strict full C2 student held-out CE 4.7630, promotion FAIL,
gap +2.1796 nats. Earlier 11.565138 student CE is from a different
reported run; do not claim a matched before/after without source SHA.

**Where:** `/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113/`
on the real TrueNAS host. The connected GitHub tool can modify this repo but
cannot access this on-prem checkpoint. All newly authored scripts are
**source-committed**, not host-executed. Follow the single integrated suite.

## Q2.1 original observations / timing reconciliation

Existing `scripts/audit_native_c2_quality_stages.py` and
`scripts/audit_c2_forward_only_cost.py` consume `candidate.json`.
New `scripts/reproduce_c2_quality_and_layers.py` validates teacher,
train and heldout SHA against the saved curriculum receipt, reconstructs
the actual trained model and teacher through strict state loading, computes
tokenwise FP64 log probabilities and digest, and checks the source reported
heldout CE within declared 0.005 tolerance. It writes exact source
candidate/token hashes and a detailed train-derived development diagnostic.
No optimizer runs or checkpoint mutations.

Wall 4,806s includes stage initialization, three forward streams (teacher,
hybrid and strict focused student), backward, development ranking/evaluation
and checkpoint save. `TrainingProgress.summary()` contains stage-loop
wall time including interepoch validation; **not a backward timing profile**.

## Q2.2 remaining quality localisation

The same reproduction script reports, on **train-derived development only**:
- Isolated individual-layer student replacements, compared to original dense.
- Cumulative teacher-to-student replacements using the actual trained stage
  schedule, rather than a made-up ordering.
- Teacher-input local odd, even, combined and teacher FFN RMS/residual,
  relative MSE, odd/even cross-term.
- Source router top-k decision-boundary margins and near-tie frequency
  captured from original `HistoryOrbitMoE.router`; per-expert counts,
  route entropy and final history norm/amplification.
- Actual teacher/student logits and CE reproduced on heldout strictly as a
  **reporting-only** final calculation. Never rank/change hyperparameters
  against independent heldout data.

An isolated replacement is an intervention with all other source layers
retaining dense FFNs, but FFN teacher-input residual and route margin alone
do not prove a causal model-error decomposition. No Lipschitz theorem
claimed for top-k ties.

## Q2.3 and Q2.4 actual forward-only competition

New `scripts/compare_c2_forward_only_optimizer.py` uses four arms with the
**same trained full C2 checkpoint**, same fit-derived examples and order,
same trainable parameter subset, and no heldout scoring:

1. AdamW with exact `loss.backward()`, only the chosen even-channel final
   linear weight/bias trainable.
2. Restricted Dust-style forward-only output activation perturbation with
   K independent probes, per-token population-centred loss, optional future
   reward discount gamma, and input-to-output outer-product update.
3. Same with K/2 antithetic perturbation pairs.
4. Weight-space SPSA with K forward probes per step (K/2 central pairs).

This is restricted Dust activation-credit, **not** the complete source
attention/cached-head/temporal algorithm. In all arms the C2 model computes
the actual full strict teacher-free rollout, not an E8GPT analogue.
`torch.no_grad()` applies to all forward-only update functions; the AdamW
arm explicitly uses reverse-mode. Report source inputs, SHA, parameter
scope, baseline/final dev CE, time, actual forward/backward calls and
next-token evaluations, allocator peak and per-step traces. This experiment
cannot be described as equal-compute simply because steps are equal: probe
count and forward cost differ. No intrinsic gain is asserted before the
measured CE improvement **per second and forward-token cost**.

## Q2.5 whole learner and Q3 actual teacher-free export

The existing `scripts/train_native_student_quality_curriculum.py` accepts
the **trained `candidate.pt` itself as --student**, so we can continue
training without resetting the C2/odd/even model. For a full-stage
continuation choose an explicit comma-separated **all-layer** list in
`--stages` (e.g., for 12 layers `"0,1,2,3,4,5,6,7,8,9,10,11"`).
Never rerun automatic 1/2/4/... from a random initial student without a
reason. This method still uses backward and retains teacher for the
calibration objective; it is the established quality control while
forward-only is tested.

New `scripts/export_c2_teacher_free.py` uses the candidate checkpoint's
**immutable `quality_curriculum_receipt`** to require the original
independent heldout SHA and teacher CE. It physically removes all
`blocks.*.ffn.*` and `blocks.*.quality_gate` state keys, instantiates a
**new** non-guarded LeechGPT with C2 odd and even branches, strict-loads
the remainder, and proves observed logit/CE parity on the actual tokens.
It reports dense versus pure state tensor bytes, checkpoint file size,
hash and quality eligibility, but keeps `release_admitted=false`.

New `scripts/benchmark_c2_teacher_free.py` runs source gated student and
pure checkpoint **one model at a time**. Records separate teacher-forced
prefill target-tokens/s, **genuine greedy KV+expert-history generated
tokens/s**, model resident and peak PyTorch GPU allocated memory, next-token
CE and exact generated-token identity. A full source machine acceptance
still requires matched numerical logits/state histories across cached
execution, stable longer-context behavior and resource/quality gates;
two different timing scopes must never be mixed.

## One consolidated host execution (no 80-minute retrain for Q2.1–Q2.4)

Use full paths from the already successful run; these are variables, not
invented file names. If source `CONFIG` is embedded in checkpoints, omit
`--config` from commands that accept it.

```bash
cd /path/to/LeechTransformer
git fetch origin
git switch experiment/consumer-preserving-multiscale-codec-20261031
git pull --ff-only
bash scripts/check_c2_quality_maxcut.sh

export ROOT=/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113
export CANDIDATE="$ROOT/candidate.pt"
export RECEIPT="$ROOT/candidate.json"
# Substitute the existing exact real source files from the original run.
export TEACHER=/actual/original_dense_teacher.pt
export TRAIN_IDS=/actual/train_ids.npy
export HELDOUT_IDS=/actual/independent_heldout_ids.npy
export CONFIG=/actual/teacher_config.json

python3 scripts/audit_native_c2_quality_stages.py \
  --receipt "$RECEIPT" --output "$ROOT/q2_stage_audit.json"
python3 scripts/audit_c2_forward_only_cost.py \
  --receipt "$RECEIPT" --wall-minutes 80.1 \
  --output "$ROOT/q2_cost_audit.json"

bash scripts/run_native_schaka101_lila.sh \
  --module scripts.reproduce_c2_quality_and_layers -- \
  --candidate "$CANDIDATE" --teacher "$TEACHER" --config "$CONFIG" \
  --train-ids "$TRAIN_IDS" --heldout-ids "$HELDOUT_IDS" \
  --curriculum-receipt "$RECEIPT" \
  --output "$ROOT/q2_original_quality_reproduction.json"

# Choose LAYER after inspecting q2_original_quality_reproduction.json's
# train-derived single-layer and cumulative layer quality metrics.
export LAYER=1

bash scripts/run_native_schaka101_lila.sh \
  --module scripts.compare_c2_forward_only_optimizer -- \
  --candidate "$CANDIDATE" --teacher "$TEACHER" --config "$CONFIG" \
  --train-ids "$TRAIN_IDS" --heldout-ids "$HELDOUT_IDS" \
  --curriculum-receipt "$RECEIPT" \
  --layer "$LAYER" --steps 8 --batch 1 --probes 4 \
  --sigma .01 --gamma 0 --lr 1e-4 \
  --output "$ROOT/q2_forward_only_competition.json"

# An independently loadable, physically teacher-free checkpoint,
# even though it is NOT quality promoted at 4.763 CE:
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.export_c2_teacher_free -- \
  --candidate "$CANDIDATE" --token-ids "$HELDOUT_IDS" \
  --output "$ROOT/c2_pure_unpromoted.pt"

bash scripts/run_native_schaka101_lila.sh \
  --module scripts.benchmark_c2_teacher_free -- \
  --guarded "$CANDIDATE" --pure "$ROOT/c2_pure_unpromoted.pt" \
  --token-ids "$HELDOUT_IDS" --generate 8 \
  --output "$ROOT/q3_native_teacher_free_inference.json"
```

**Pause if any strict source hash or parity check fails.** Do not repeat the
original 80.1-minute curriculum by default. Re-run all forward-only arms
from the exact original candidate, *never sequentially updating the same
weights across methods*. The output for the four arms is **not** a trained
checkpoint; it is a fair local optimizer discriminator. Promotion Q2.5
requires a follow-on full-student trained checkpoint and a fresh
predeclared heldout evaluation only after model selection.

## Exact status and outstanding mathematics

| Gate | Source deliverable | Scientific/physical acceptance |
|---|---|---|
| Q2.1 `tokenwiseC2SourceLogitReconstruction` | Exact token logprob digest + CE score + source SHA | Original candidate GPU receipt to be compared; cost breakdown lacks measured backward fraction |
| Q2.2 `conditionalLayerErrorPropagationOriginalModel` | Original router/history/FFN/odd-even diagnostics; isolated/cumulative interventions | Model-specific verified route stability and downstream error bound, causal attribution |
| Q2.3 `actualLeechC2DustActivationCreditSource` | Real chosen even-linear activation credit, population-centred future reward and outer update | Original model GPU pass, estimator validity for nonlinear/recurrent objectives |
| Q2.4 `measuredForwardOnlyVsAdamWMatchedCost` | Four arms starting same checkpoint with counted forwards and timed dev CE | Native quality/time results and repeated seeds, energy accounting |
| Q2.5 `recurrentHistoryFutureRewardCredit` | Full trained C2 architecture supported by existing continuation | Genuine source temporal credit and complete C2/full-model quality promotion |
| Q3 `teacherStrippedStudentCheckpointBidi` | Strictly loaded independent non-guarded model, logits/CE and source SHA checks | Real GPU parity + full KV/history trace + quality/efficiency promotion |

Existing Agda PR #1265 owns finite affine Dust/Walsh/semantic observer proofs;
**do not claim** its source-written unverified Agda modules have new kernel
receipts. All Leech source modules in this cut are committed, pending host
tests and actual native execution. The previous 106 named max-cut programme
ledger is not inflated by adding a second catalogue.

### Post-implementation source integrity hardening

The native teacher-free inference comparator checks the saved pure export's
original guarded `candidate.pt` SHA and independent original test-token SHA.
For every greedy generation step it hashes not merely token IDs but **entire
native float32 next-token logit vectors and each expert-history layer state**,
checking digest equality against the guarded student's strict student-only
runtime. Added `test_guarded_and_pure_cached_generation_history_bidi` to
the CPU semantic suite. The physical GPU native output and its actual trace
fingerprints are **not yet observed**, and no cache parity claim is promoted
before the test passes in the user's original checkout.
