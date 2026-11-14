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

## 14 Nov user-run hardware result and next acceptance discriminator

All six integrated phases were executed on original RX580 HIP7.16 native
TrueNAS artifacts. The run reported: 825 optimiser steps, 80.1min total
(4,806 s), stage loops 4,685.3 s and 120.7 s outside stage loops.
Independent source CE reproduces 2.58340826 (teacher) and 4.76298302
(student) within about 1e-7 nats of initial receipts. Worst isolated
developer-layer substitutions are layer1 (CE 5.2445) and layer0 (4.6620);
last layer 11 alone is better than dense source on the selected development
rows (CE 2.4059). Layer1 local teacher-input relative FFN MSE 127.02 and
odd/even cross -0.4916 indicate a concrete source mismatch for investigation.

**History scope correction:** the layer10 history RMS 107,049 and layer11
RMS 1.28e9 arise from a teacher-forced local MoE call in
`reproduce_c2_quality_and_layers.py`; they are **not** full student
autoregressive history trajectories. Add genuine `return_expert_histories`
trace with actual C2 recurrent student and compare layerwise RMS, max,
top-k boundary margin and normalised state growth before claiming dynamical
explosion.

One-readout training (same source checkpoint/batches, eight optimizer steps):
AdamW 43.19s, ΔdevCE +0.016925; independent K4 Dust 75.00s, Δ+0.000031;
antithetic Dust 70.92s Δ−0.000861; central SPSA 59.36s Δ+0.003265.
Forward-only peak GPU allocated ~238.6–238.8 MB versus AdamW 277.5 MB,
but this implementation recomputes whole-model forward for each perturbation.
**Backward-free update demonstrated; competitive optimisation not established.**

The physically teacher-stripped 112.2 MB `c2_pure_unpromoted.pt` removed
56.7 MB / 33.61% source tensor storage. Original and pure student matched
heldout CE 4.762983, 0.0 logit error and entire 8-step greedy token, logit,
and expert-history digests; release/quality admission remains false.

**Performance attribution needs a separate test:** initial native runner
always executed guarded first and pure second, only eight generated
tokens; guarded 2.82 tok/s versus pure 36.43 tok/s = 12.9x observed
single-order ratio. The original guarded `student_only=True` path already
does not execute frozen dense teacher FFNs. Thus deleting the fallback
tensors cannot itself explain 12.9x *eliminated dense FFN computation*.
New `scripts/benchmark_c2_teacher_free_abba.py` warms both source
models and measures alternate-order ABBA/BAAB repeats, compares full
per-step logits/history SHA, and preserves the distinction between
the observed speed ratio and an asserted causal speedup.

Run after pulling current PR #4 source, in the already working native wrapper:

```bash
ROOT=/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.benchmark_c2_teacher_free_abba -- \
  --guarded "$ROOT/candidate.pt" \
  --pure "$ROOT/c2_pure_unpromoted.pt" \
  --token-ids "$HELDOUT_IDS" \
  --generate 16 --cycles 3 --batch 1 \
  --output "$ROOT/q3_native_teacher_free_abba.json"
```

The `HELDOUT_IDS` environment variable must resolve to the **same original
file whose SHA was admitted by the pure checkpoint**. This is a compact
measurement, not a repeat of the 80.1-minute curriculum. Do not conflate
resident allocator stats while both models are loaded into VRAM with the
previous isolated model memory experiment.

Updated near-term research cut: original layers 0/1 local FFN and
odd/even cancellation; true recurrent history accumulation; source
Dust credit/gradient direction against reverse AD on identical batches;
cached-prefix/suffix probes; larger matched training budget only after
the useful local correction is established.

---

## 2026-11-14 D4/B4–C3–E8/Leech–theta/zeta–Sophontic integration (shared roadmap)

**Canonical formal ledger:** [DASHI LILA–Sophontic–MoE maximum-cut roadmap](https://github.com/chboishabba/dashi_agda/blob/master/Docs/roadmaps/LILASophonticMoESOTAMaxCut20261111.md), section `2026-11-14 integrated D4/B4–C3–E8–Leech–theta/zeta–Sophontic–LILA maximum cut`. Exact subtarget IDs M1a–M7f and A/B/C phase order live there; this runbook does not invent a second proof inventory.

### Interpret the 4.763 checkpoint first, not a new geometry by assumption

The native original 4.762983-CE model was trained with sign-invariant C2 source architecture. A source implementation of `c2c3_wave` in `HistoryOrbitMoE` **does not prove** that the checkpoint used it: inspect its original configuration/`moe_route_geometry` before retraining or constructing output comparisons. Preserve the original trainer settings, original SHA256 on teacher/student and fit/development/heldout files, selected expert convention, odd/even output and recurrent history semantics.

- C2 source action is antipodal sign `(x,H)→(−x,−H)` and C2-invariant router score. It is not the same as 3-cycle.
- Commuting independent C3 120° rotation in an explicit real 2D phase plane plus central `−I` gives C2×C3≅C6. **A C2-constrained vector-valued network may learn such C3 structure**; parity alone does not force it.
- D4 root-system diagram triality instead requires a reflection with `srs=r⁻¹` and nonabelian `S3`. A central sign flip does not invert a nontrivial faithful C3 rotation. Use another reflection action and explicit source parameter/intertwiner, not just rename C2.
- The D4/B4/F4 triality formal route is distinct from the dihedral D4 square action and from a coincident number of experts. Check source `TrialecticD4C3ResidualBoundaryExact` before importing any related abstraction.

### The mathematical donors already present (do not rerun foundational programmes)

DASHI `RootSystemB4Carrier`, `RootSystemB4WeylAction`,
`TrialecticD4C3ResidualBoundaryExact`,
`C3FourierConjugationExact`,
`JInvariant369C6TenRankWeightTwelveCrossPollinationExact`,
`MonsterThreeLocalE8LeechBridgeExact`,
`Monster3BCyclicFourierDyadicBridgeExact`,
`ExceptionalLatticeGrokkingProtocolExact`,
`TriadicEisensteinTransformationTheorem`,
`DeltaNormalizedWeight12SameObjectExact`,
`LilaE8ThetaJBridgeSurface`,
`RiemannReflectionC3OrbitShapeBridgeExact`,
`RiemannMaassMoonshineCrossPollinationExact` and the
Sophontic `ObserverRefinementLatticeExact` /
`LILAForwardOnlyDustAdapterExact` are source owners.
Numbers `240·3(1+16+16²)=196560` do not supply actual Leech minimal-vector
coordinates or a C3-equivariant three-family action. Character phases
`ζ₃`, analytic `ζ(s)`, lattice Epstein `Z_L(s)`, and Monster character
multiplicities are different sorts; never conflate them.

The classical same-object targets are
`Theta_E8=E4`,
`Theta_Leech=E4³−720 Delta=Delta(J+24)`,
`Z_E8(s)=240·2^(−s) ζ(s) ζ(s−3)` for `Re(s)>4`
under norm-two E8 and `q=exp(2πiτ)`. These are **mathematically established formulas** but the inspected original DASHI `LilaE8ThetaJBridgeSurface`
is diagnostic only: its exact original E8/Leech source carrier,
lattice modularity and coefficient/Mellin cross-language welds are OPEN.
Pinned Lean eta24=Delta does not automatically prove lattice theta,
nor does theta equality prove RH or model CE improvement.

### New maximum-cut machine experiments under M5–M7

| Phase | Original source experiment and controlled comparison | Deliverable / acceptance |
|---|---|---|
| S0 identity | Load frozen original `candidate.pt` plus original source config; record `moe_route_geometry`, tensor SHA, active source channel and source triple interpretation | Clear whether the tested checkpoint was `c2_invariant` or `c2c3_wave`, no assertion by code presence |
| S1 actions | Construct paired vector-valued model-state probes for `−I`, 3-cycle, separate reflection. Check router scores, top-k indices/ties, odd/even output and history, without asserting end-to-end equivariance | Source residual per layer and counterexamples; local C3 action only where tensor triple meaning is explicitly specified |
| S2 observed error | Re-run original train-derived layers 0/1 odd/even teacher FFN MSE, CE isolate, cumulative replacements, true cached full student history norms | Distinguish teacher-forced 1.28e9 local history from recurrent student history; identify bad subspaces and actual decision boundaries |
| S3 C2→C3 possible, not automatic | From **same source checkpoint** compare frozen-feature C2, commuting C6 and D4/S3-compatible directed features and equivalent parameter-count nulls | Does geometric feature injection improve development loss? No posthoc heldout selection |
| S4 Sophontic semantics | Compare before-router paired intervention (routing can change) with after-router intervention (past route fixed) for 2/3/S3 cycles and directed semantic preservation | Source semantic law or explicit negative counterexample with typed observer output |
| S5 local correction | Warm-start targeted layer0/1 readout/output, matched AdamW and cached prefix/suffix forward-only Dust; record CE gain per wall second/token/VRAM | Full-student development CE improves, not only local MSE or parameter symmetry |
| S6 physical quality and Pareto | Independent all-student heldout CE, genuine pure teacher-stripped BIDI, warmed ABBA runtime and longer-context selected histories | Original heldout CE approaches 2.583408 and resource gains survive repeated source-matched measurement |

**Source-proof interface welds owned by formal roadmap:** M5a `sophonticC2SemanticInterventionDescent`; M5b `sophonticC3DirectedCycleIntervention`; M5c `sophonticS3ReflectionReversesCycle`; M5d `leechC2CanRepresentC3WithoutForcingIt`; M6a `leechC2C3RouterInvariantUnderOriginalAction`; M6b `leechTrialityExpertHistoryTransport`; M6c `leechLilaRepresentationIntertwiner`; M7a `trialityQualityLayer01Ablation`; M7b `originalC2EarlyLayerOddEvenCorrection`; M7c `originalStudentRecurrentHistoryNormBoundOrCounterexample`; M7d `originalDustCreditMatchesSourceGradient`; M7e general BIDI; M7f full quality.

**Status:** analytical roadmap/source references updated, **no new physical C3/S3/lattice training or Agda 2.9 same-carrier proof run is claimed**. The 4.763 checkpoint's verified C2 Q2.1/Q3 finite-trace receipts remain credited. Federation and actual block diffusion are parked. **Priority:** diagnose early-layer semantic/geometry discrepancy and measured recurrent history, then a matched C2/C6/S3/exceptional-lattice ablation; modular theta/zeta same-object formal work proceeds separately and should not hold up the quality experiment.
