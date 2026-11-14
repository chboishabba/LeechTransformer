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

---

## 14 Nov next implemented maximum cut — actual C2/C3 router and recurrent student dynamics

Two new source probes (theorems remain separate from physical results):

1. `scripts/audit_original_c2c3_router_actions.py`: strict-load the **trained** model, bind checkpoint SHA and `cfg.moe_route_geometry`, test actual router selected experts and weights under simultaneous sign, C3 cyclic triads and a separate S3 reflection. Include exact finite `r³=s²=1`, `srs=r⁻¹` and commuting central antipodal sign laws. The `c2c3_wave` candidate source features are C2 and C3 invariant on specified triadic hidden coordinates; directed channels are **not** generally S3 reflection-invariant. A checkpoint trained under `c2_invariant` does not automatically obtain C3-equivariant routing from the fact that optional `c2c3_wave` source code exists. This experiment uses synthetic seeded hidden states, NOT raw token-space semantic C3 interventions, a pretrained D4 triality action or an all-layer neural equivariance theorem.
2. `scripts/audit_actual_student_recurrent_history.py`: run the original trained strict student, one token at a time with genuine `past_key_values` and `past_expert_histories`, record per-layer retained-history RMS/max, actual router top-k boundary margins, next-token outputs and source SHA. This **does not** reuse the previously measured teacher-forced local FFN history to imply student recurrent explosion; it measures the previously unpaid physical dynamical quantity. Both source corpus-token continuation and greedy continuation are available. Run `--steps 32` only if original block size is at least 32.

`tests/test_original_c2c3_source_actions.py` contains C3/S3 finite-action, reflection-negative-control, original `HistoryOrbitMoE` route, C2-does-not-force-C3 and actual two-layer source cached-history regression tests. The existing consolidated `scripts/check_c2_quality_maxcut.sh` runs the new test suite.

Native RX580 execution, using only the **original** source model and previously identified full teacher/checkpoint/token paths:

```bash
cd /home/c/Documents/code/DASHIg/LeechTransformer
git fetch origin
git switch experiment/consumer-preserving-multiscale-codec-20261031
git pull --ff-only
bash scripts/check_c2_quality_maxcut.sh

ROOT=/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.audit_original_c2c3_router_actions -- \
  --candidate "$ROOT/candidate.pt" \
  --rows 8 --seed 41 \
  --output "$ROOT/c2c3_source_action_audit.json"

bash scripts/run_native_schaka101_lila.sh \
  --module scripts.audit_actual_student_recurrent_history -- \
  --candidate "$ROOT/candidate.pt" \
  --teacher "$TEACHER" --config "$CONFIG" \
  --token-ids "$HELDOUT_IDS" --steps 32 \
  --output "$ROOT/actual_student_history_32.json"
```

`TEACHER`, `CONFIG`, `HELDOUT_IDS` are the original externally defined actual source paths, not new files invented for this command; omit config flag if embedded. This is read-only/diagnostic with **no new 80-minute GPU training**. Make any later geometry or parameter selection only using the original fit-derived development split, not the independent heldout.

Parallel mathematical receipt now committed to DASHI master:
`scripts/check_e8_leech_theta_finite_same_norm.py` and
`tests/test_e8_leech_theta_finite_same_norm.py`. E8 even-integer/half-integer coordinate cosets **actually enumerate** finite shells 240,2160,6720,17520,30240 and compare with E4=1+240Σσ3(n)q^n at norm²=2n; Leech modular **candidate** E4³−720Delta yields coefficient 0 at q and 196560 at q². The latter is *not* a Leech lattice enumeration; full all-index theta/Mellin proofs remain M3a–M3d OPEN. See canonical DASHI integrated A3/B2/C1 ledger for exact source proof targets and acceptance.

**Verification status**: source-committed; independent integer calculations for E8 n<=5 and modular Leech q<=4 were performed in this work session. The just-added regression source, original checkpoint router, actual native recurrent histories and full Agda kernel bridge have not yet been host-executed; no new GPU parity/quality improvement claimed.

---

## 2026-11-14 physical original student dynamics result and immediate quality-repair max-cut

**New user-executed source receipts:**
- DASHI original E8/Leech finite tests **5/5 PASS**; full E8 integer norm-two through norm-ten shell count `[240,2160,6720,17520,30240]` matches `240·σ₃(n)`; Leech **modular candidate** `E4³−720Δ` coefficients through `q⁵`: `[1,0,196560,16773120,398034000,4629381120]`. These do not prove all-index E8/Leech theta theorems.
- Leech `check_c2_quality_maxcut.sh`: **29/29 PASS**. Note these contracts were executed before the *new history-cap repair tests* were added in this tranche; latest suite requires its own new run.
- Original RX580 `candidate.pt`: `cfg.moe_route_geometry="c2_invariant"`, depth 12, experts 4/top2. Native source router antipodal C2 selected IDs/weights **identical in all 12 layers**, `max_topk_weight_abs_delta=0.0`. C3 cyclic triad rotation changes expert IDs and top-k weights (max delta ~0.065–0.189), and S3 reflection changes weights (~0.038–0.187). This is an empirical probe on synthetic hidden-state triples, **not** proof of global C2/C3 model equivariance or Sophontic semantic invariance.
- Original RX580 `actual_student_history_32.json`: genuine strict student-only recurrent histories with real KV cache, 32 steps; at layer10 history RMS ~0.2136 initial, ~10.6802 at step16, **251.0700** at step32, max **1212.2051**, router margin **278.5124**; layer11 RMS ~0.3162 initial, ~46.2432 step16, **10990.6821** step32, max **42608.7734**, margin **27628.9336**. Layers 0–9 had step32 history RMS below 0.9 **on this particular trace**. This confirms a severe *actual model trajectory* in layers10/11; it does not prove every input is unstable, nor the eigenvalue/spectral radius of a fixed matrix. This system is nonlinear, gated and switched with inputs. Source tracer uses actual original token IDs from a real sequence by default, so **model histories are recurrent but token continuation is data/teacher-forced** unless the script's `--greedy` flag is supplied. A separate free-running greedy experiment remains a useful control.

### High-alpha repaired inference intervention (now source committed; native run pending)

New `scripts/screen_c2_history_even_repair.py` imports the original quality checkpoint and reads the exact original training receipt to recover the **train-derived development indices**. Never use independent heldout for selecting a cap/scale. All interventions use reversible forward hooks, no checkpoint mutation, no new training and no new source file rewrite:

1. **Late history state norm cap** on output of each original transition in layers10 and11: `h_new ← h_new·min(1,R/||h_new||)`, `R∈{1,4,16,64}`. The source may carry large routed history into intermediate FFN evaluation on the current step **before** the cap acts, so this only enforces a post-transition retained-state bound for subsequent steps; it does *not* bound intermediate expert activations or guarantee stability of unmodified dynamics.
2. **Late history attenuation** `h_new ← α·h_new`, `α∈{0.1,0.5}`. Attenuation alone without a cap does **not** establish boundedness of an affine/input-driven recurrent switching system.
3. **Early even-channel readout scaling** `even_student(x)←β·even_student(x)`, separately for layers 0 and1 with `β∈{0,0.5,0.75,1.25,1.5}`. This tests whether the observed layer1 odd/even cancellation is consequential; a negative odd/even cross-term alone does not prove cancellation is harmful.
4. Small combined late-cap + early-even screen on the same original development split, with unmodified identity source as a control. Each configuration measures full **student-only** source development CE, not merely local MSE. Do not infer matched cost from a single CE output; the screen is diagnostic and can be followed by restricted trained repair if favourable.

`tests/test_c2_history_even_repair.py` checks exact bounded selected histories by 25 source `HistoryOrbitMoE.step` recurrent updates, C2 oddness of the clipped transition, source antipodal output/history transport and rejection of invalid caps. The new tests are registered in `scripts/check_c2_quality_maxcut.sh` and the new module is permitted by `scripts/run_native_schaka101_lila.sh`. **No native result or successful latest test count is asserted here.**

With the latest PR #4 branch checked out, run the new source suite, then:

```bash
ROOT=/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113
bash scripts/check_c2_quality_maxcut.sh

bash scripts/run_native_schaka101_lila.sh \
  --module scripts.screen_c2_history_even_repair -- \
  --candidate "$ROOT/candidate.pt" \
  --teacher "$TEACHER" --config "$CONFIG" \
  --train-ids "$TRAIN_IDS" --curriculum-receipt "$ROOT/candidate.json" \
  --batch 1 --output "$ROOT/c2_history_even_dev_repair_screen.json"
```

As before, `TEACHER`, `CONFIG`, `TRAIN_IDS` must resolve to actual prior input files; omit `--config` if the original teacher contains it. Source hash mismatches and existing output overwrite fail closed. The **first** quality decision is whether any conservative cap improves or harms full-student development CE; then test early-even readout scales alone/combined. Re-run `--greedy` recurrent trace as a separate original-model control, and apply the winning configuration to a full history trace before considering a trained-student variant. The original heldout CE remains 4.763, **not improved by source-writing this intervention**.

**Precise mathematical consequence:** If initial expert histories have per-expert norm ≤R and every selected transition output is capped at R, while unselected states are retained unchanged, induction on token steps yields `∀e,t: ||h_e(t)||≤R`. The radial cap is odd `clip_R(−h)=−clip_R(h)`, so this modification preserves the signed source transition symmetry for antipodal inputs. It does **not** establish `ρ(A)≤1`, global Lipschitz contraction, or small model CE. There is no single fixed linear `A` for the full switched MoE to which the observed 32-step growth alone determines spectral radius.

**Next formal welds:** exact source radial-cap/retained-history induction; full source C2 route/tie consistency; source-normalised early-layer representation bound; recurrent source Lipschitz or explicit counterexample across selected switching; matched gradient-free and AdamW update fidelity; original full-student independent CE promotion.

### Direct history-control measurement after development selection

The original `scripts.audit_actual_student_recurrent_history` now admits
`--late-cap`, `--late-scale`, `--layer0-even-scale`, and
`--layer1-even-scale` using the *same reversible hook implementation* as the
development-screen tool. It records the requested intervention parameters
in the resulting JSON and, with `--late-scale 1`, asserts that every
layer-10/11 stored history max-coordinate is no greater than the selected
radius (within floating tolerance). The actual selected retained-state
**Euclidean norm** is bounded because the radial transition clipping
operates per expert; coordinate maxima are a necessary observed test,
not the full numerical norm certificate.

For example, only **after** reviewing development screen output and with a
new output filename (do not overwrite the 32-step source baseline):

```bash
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.audit_actual_student_recurrent_history -- \
  --candidate "$ROOT/candidate.pt" --teacher "$TEACHER" \
  --config "$CONFIG" --token-ids "$HELDOUT_IDS" \
  --steps 32 --late-cap 4 \
  --output "$ROOT/actual_student_history_32_cap4_observation.json"
```

This uses the independent heldout token sequence for **post-selection
observational diagnostics only**, not to choose the cap. For optimization
or model selection, use the same train-derived development data as the
screen. Run `--greedy` separately to distinguish natural source-token
continuation from free-running model-generated tokens.

---

## 14 Nov 17:31 AEDT user RX580 repair screen — acceptance and bounded warm-start implementation

**Verified user-host receipt** `c2_history_even_dev_repair_screen.json`; original `candidate.pt`; source suite at head `c3e3899` **33/33 PASS**. Development uses **only 9 distinct train-derived token windows** and 21 probed interventions, so the selected best score carries multiple-comparison/selection uncertainty. Neither its value nor the original independent heldout CE can be advertised as a heldout quality improvement. No new heldout tuning.

- Frozen strict student baseline dev CE **4.791580**.
- Best single intervention **`even_0_scale_0`** dev CE **4.771828**, improvement **0.019753 nats** on the screened 9 windows. Important: exact zero output gate destroys gradients into that even branch; this is a *diagnostic suppression*, not a trainable warm-start configuration. `even_0_scale_0.75` dev CE **4.782848** is a weaker but differentiable seed, vs layer1 1.25 gain **4.785780**. Zeroing layer1 even branch yields dev CE **4.942530**, a strong negative control. The term "destructive interference" is descriptive of a local mismatch and cannot be inferred to explain all 2.18 heldout nats.
- `late_cap_64` alone dev CE **4.791690** (0.000110 worse); `late_cap_4` dev CE **4.802763** (0.011183 worse). Inference-only cap without re-optimising is **not a proven quality-improving correction**. Most severe caps regress dev CE. Combined `even_0_scale_0 + late_cap_64` was **not among the 21 scored interventions**, hence its actual full-dev CE is UNKNOWN.
- Source 32-token recurrent histories: baseline layer10 RMS **251.07**, layer11 **10990.68**, layer11 router top-k boundary margin **27628.93**; `even0=0` layer10 RMS **114.94**, layer11 **15419.25**, margin **38719.12**; `even0=0,cap64` layer10 RMS **2.73**, layer11 **2.83**, margin **1.18**. This directly verifies finite-trace state bounding and lower route margins under the combined intervention. It does **not** prove stationary invariant distribution, `ρ(A)≤1`, successful free-running generation or lower full-student dev/heldout CE.

### Source-implemented native persistent repair (new cut; user host tests pending)

Changed **native** `HistoryOrbitMoE.cap_history` and both local `step` and externally selected `execute_selected` history updates. When config `moe_late_history_cap=None` (default), **old candidate weights/state semantics remain unchanged**. When a finite positive cap is in the saved `LeechConfig`, only the original layer10/11 MoE transitions apply C2-odd Euclidean radial projection to retained state; all other layers unchanged. New `moe_even_layer0_scale` and `moe_even_layer1_scale` default to 1 and multiply even branch outputs in the actual `LeechBlock.forward`. Both controls are serialized as **config scalars**: no hidden inference-time hook required and no new state tensors. Independent source unguarded pure checkpoint can reconstruct them and check BIDI.

New `scripts/train_c2_bounded_quality_repair.py` warm-starts **directly from original 4.763 candidate**, strict-loads source weights and immutable original teacher/train SHA; recovers **exact original fit/dev indices** (and checks duplicate windows); requires RX580 GPU; trains only layer0/1 even channel weights, optionally layer10/11 transition parameters, with clipped full-model next-token CE, source cap (R=64), best-step development rollback and no heldout read. It compares same-source unmodified dev CE with modified pretrain dev CE. Default layer0 even gain **0.75** allows gradient flow; selecting zero is **fail-closed** because a fixed zero gate would block all even-channel gradients. Save new `candidate_bounded_repair.pt` and JSON sidecar, source original candidate left untouched, `release_admitted=false`. Original independent heldout SHA and original dense teacher CE are **inherited from the original immutable receipt for later, one-shot export/validation**, not used to fit/select.

`scripts/export_c2_teacher_free.py` now accepts either exact original curriculum receipt or the new bounded repair receipt, using original independent heldout SHA and teacher CE, and preserves stored cap/even scales on an independent strict-loaded pure C2 model. This is *source-ready* teacher stripping; it does not imply the warmed repair will improve heldout CE or yield an admissible full model. `tests/test_c2_native_bounded_warmstart.py` exercises default source parity, native bounded C2 transport, persisted config/BIDI and permitted trainable scope; `scripts/check_c2_quality_maxcut.sh` now includes these five new tests. **Do not claim 38/38 PASSED until host executes the new suite**.

### Native bounded first experiment; preserve independent heldout for final only

```bash
cd /home/c/Documents/code/DASHIg/LeechTransformer
git fetch origin
git switch experiment/consumer-preserving-multiscale-codec-20261031
git pull --ff-only
bash scripts/check_c2_quality_maxcut.sh

ROOT=/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.train_c2_bounded_quality_repair -- \
  --candidate "$ROOT/candidate.pt" \
  --teacher "$TEACHER" --config "$CONFIG" \
  --train-ids "$TRAIN_IDS" --original-receipt "$ROOT/candidate.json" \
  --history-cap 64 --layer0-even-scale 0.75 \
  --layer1-even-scale 1 --train-late-transitions \
  --epochs 2 --max-steps 24 --dev-every 4 --batch 1 \
  --output "$ROOT/candidate_bounded_repair.pt"
```

`TEACHER`, `CONFIG`, `TRAIN_IDS` remain the *actual original* source paths recovered from the prior successful run. Omit `--config` when teacher embeds it. No new full 80min curriculum required. The warm-start must first beat **its own matched pretrain cap+even development CE**, and preferably the original unmodified **4.791580** dev CE, before larger trials. The nine-window dev set is tiny and was already screened over 21 candidates. Use a **fresh, duplicate-group-disjoint train-derived validation subset** or cross-validation, with no independent heldout exposure, before selecting further hyperparameters and interpreting any 0.02-nat improvement. Full-student independent heldout 4.762983 from the original candidate remains baseline until final single-shot evaluation.

**Exact remaining gates:** `originalC2EarlyLayerOddEvenCorrection` becomes original native **source implementation complete, measured warmstart quality OPEN**; `originalStudentRecurrentHistoryNormBoundOrCounterexample` has physical 32-token runaway witness plus finite C2-odd cap in source, native capped trained model proof/quality OPEN; `teacherStrippedStudentCheckpointBidi` prior unmodified checkpoint finite-source PASS, repaired config pure export/BIDI OPEN; `fullStudentHeldoutQualityPromotion` FAIL; `conditionalLayerErrorPropagationOriginalModel` and source non-linear stability remain OPEN. Do not misattribute the measured original history norm to spectral-radius theorem. Formal D4/F4, E8/Leech all-index theta/zeta and Sophontic C3/triality work remains parallel and must not block this measured C2 quality repair.

### Native memory-control hardening after source review

The warm-start runner now constructs both source models on CPU, measures the
unmodified baseline on GPU, then **unloads the control model and clears its
allocator cache before moving the bounded repair to GPU**. This avoids
holding two complete 12-layer source models on the RX580 during training
and prevents an unnecessary VRAM comparison confound. The run remains
development-only and non-promotional. Native acceptance of the newly
added 5 contracts and the 24-step repair is **still pending**.

---

## 2026-11-14 original balanced ternary / modular Fourier grokking / C2 branch symmetry maximum cut

**Canonical formal owner:** DASHI `Docs/roadmaps/LILASophonticMoESOTAMaxCut20261111.md` has the exact G1–G12 (subtargets of M5–M7) source lemmas, A/B/C stages, and finite versus learned-mechanism nonpromotion boundary.

### Paper source and reuse

Nanda et al. (*Progress measures for grokking via mechanistic interpretability*, arXiv:2301.05217) identify an actual modular-addition transformer Fourier circuit, phase composition and Fourier-space ablations; their three train-dynamics stages are memorisation, circuit formation and cleanup. Gromov (*Grokking modular arithmetic*, arXiv:2301.02679) constructs/examines sinusoidal modular feature maps and phase-alignment/interference under training. DASHI already proves the **task-level** C3 character composition and 8/9 fitting memorizer counterexample in `ModularFourierGrokkingC3Exact`; actual zero versus two action equivariance defects in `GrokkingEquivarianceDefectGeometryExact`; and cyclotomic character multiplication in `CyclotomicGrokkingActionLinearizationExact`. No new theorem is needed that C3 *can* solve the finite 9-point modular task. The open physical question is whether **original model activations and/or trained source weights actually implement and benefit from such a phase circuit**.

New `scripts/audit_balanced_ternary_c3_grokking_fourier.py` creates the finite nine-input sine/cosine receipt:
```text
u(k) = (cos(2πk/3), sin(2πk/3))
u(a) * u(b) = u(a+b mod3)
memorizer fits all 8 admitted inputs but fails (2,2)
structural C3 translation defect = 0, memorizer defect = 2
```
It also distinguishes `k↦−k` in the C3 *residue group*, which becomes **phase reflection** `K(c,s)=(c,−s)` satisfying `KRK=R⁻¹`, from the trained network's central C2 `−I`, which satisfies `(-I)R=R(-I)`. Thus balanced residue inversion, coordinatewise sign, rotating a triple's coordinates, and *semantic disagreement* are **four different operations** until a typed map identifies any two. The finite arithmetic was independently checked for exact discrete defect 2 and maximum floating Fourier product residual ~`8.01e−16`. No learned network was trained by this check.

New `scripts/audit_c2_parity_c3_fourier_teacher_inputs.py` loads original source dense teacher and strict C2 student on **original SHA-checked train-derived development** token rows, records the actual teacher's layer-0 and layer-1 pre-FFN `ln2` hidden states, and separately evaluates original student `HistoryOrbitMoE` odd branch and `canonical_even_input` even branch at **zero initial local history** on paired `x,-x`. It records actual teacher parity projections `F_even=(F(x)+F(-x))/2`, `F_odd=(F(x)-F(-x))/2`, source mean `<E(x),O(x)>`, paired cross-cancellation, per-component MSE and original C3 wave decomposition (DC vs rotating residual energy). It separately measures C3 equivariance defects of both source branches on coordinate triples. Note: **zero initial history / teacher-input local** is a diagnostic, not the previously established true recurrent 32-token full student history. Source hidden coordinates in consecutive triples are **not** assumed semantically assigned, modular residue embeddings or E8/Leech vectors. No heldout model selection or new student training.

Four source regressions in `tests/test_balanced_ternary_c3_grokking_source.py` test Fourier composition, training-fit-vs-full-task, C3 inversion reflection versus central C2 commuting, and constructed paired original MoE/even parity. The latest `scripts/check_c2_quality_maxcut.sh` now incorporates them, and the native Schaka wrapper permits both scripts. Earlier 33/33 native acceptance predates the five warm-start tests and these four new tests; **there is no new current PASS count until latest host run**.

### Native low-cost evidence commands

```bash
cd /home/c/Documents/code/DASHIg/LeechTransformer
git fetch origin && git switch experiment/consumer-preserving-multiscale-codec-20261031
git pull --ff-only
bash scripts/check_c2_quality_maxcut.sh

ROOT=/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.audit_balanced_ternary_c3_grokking_fourier -- \
  --output "$ROOT/balanced_ternary_c3_grokking_fourier.json"

bash scripts/run_native_schaka101_lila.sh \
  --module scripts.audit_c2_parity_c3_fourier_teacher_inputs -- \
  --candidate "$ROOT/candidate.pt" \
  --teacher "$TEACHER" --config "$CONFIG" \
  --train-ids "$TRAIN_IDS" \
  --original-receipt "$ROOT/candidate.json" \
  --rows 2 --device cuda \
  --output "$ROOT/c2_layer01_parity_c3_teacher_input.json"
```

Set `TEACHER`, `TRAIN_IDS`, `CONFIG` from previously verified original source files, not invented new paths; omit config flag if embedded. The second script reads only original train-derived development rows and does not require or sample independent heldout data.

### Interpretation and subsequent quality experiment

The observed original layer-1 negative odd/even cross-term means **pointwise cancellation**, not inherently bad symmetry. On paired inputs, exact C2 parity forces `<E(-x),O(-x)>=-<E(x),O(x)>`; a truly antipodal-symmetric input distribution therefore has expected cross-term zero. The new original-source paired audit will test its finite reality. Frozen layer-0 `even_scale=0` improved development CE by ~0.019753 over **only nine** windows, while `even_1_scale=0` regressed ~0.150950: cancellation should never be globally suppressed merely on sign. A strictly balanced-trit scalar zero also cannot replace a *nonzero* opposite pattern, a C3 rotating phase, an irreducible triadic face, or tetralemma both/neither evidence qualification.

After numerical parity/phase-source reports: compare matched original layer0/1 student teacher-FFN residuals and **full-student dev CE** under (A) unchanged C2 odd+even, (B) C2 plus a non-learned 2D C3 phase diagnostic/projection, (C) genuinely learned C3 phase transport, and (D) explicit typed triadic-face/evidence fibre controls, all with parameter/resource-matched negative baselines. Add a modular synthetic task with original 8/9 train/heldout C3 rule test before claiming a learned grokking circuit. Use Fourier-ablation and out-of-split validation to isolate algorithmic mechanism from increased capacity. Only the actual repeated original-model source data and a robust independent quality receipt can promote 4.763 checkpoint quality; no claim of progress from paper analogy or C3 source availability alone.

Formal source `DASHI/Foundations/BalancedTernaryC3OppositionGrokkingBridgeExact.agda` provides a new balanced-triad C3 rotation/reflection plus exact candidate finite proofs and imports of existing grokking/argument/face no-collapse theorems. **New Agda 2.9 kernel replay pending**. E8/Leech all-index theta–zeta and physical D4 triality action remain separate owner obligations.

---

## 2026-11-14 19:24 AEDT — Physical bounded C2 success and 369 atomic/NS/quantum/YM character bridge

This supersedes the earlier status "bounded warm-start pending." The **user-supplied original TrueNAS native hardware transcript** reports the following physical outcomes (not freshly executed by this source-document update):

| Gate / native original receipt | Result | Interpretation |
|---|---|---|
| Pinned DASHI Agda 2.9 bridge | `BalancedTernaryC3OppositionGrokkingBridgeExact` checked via repository-sanctioned parallel entrypoint, **310/310 interfaces, exit 0** | Kernel PASS on reported local checkout; does not prove original Leech neural C3 representation |
| Q2.5 repair `c2_bounded_repaired.json` | RX580 24 steps / **146.94s**; fit **55**, dev **9**; original dev CE 4.7916 → intervened initial 4.7831 → trained best **4.6332** | Real trained improvement; dev split tiny and 21-variant screen previously consulted |
| Q3 `c2_repaired_pure_unpromoted.json` | Pure source 112.2MB vs guarded 168.9MB; logit **max abs 0.0** tested; heldout CE original **4.7630**, repaired **4.5966**, dense teacher **2.5834** | Exact finite BIDI and new measured heldout gain 0.1664; remaining teacher gap **~2.0132**; `quality_pass=false`, `release_admitted=false` |
| C1 `actual_student_history_32_bounded_repaired.json` | layer10 RMS 251.07→2.8363; margin 569.21→1.1719; layer11 RMS 10990.68→2.8341; margin 27628.93→0.0759 | Measured 32-step controlled history and router desaturation, NOT stationary-measure/attractor or Jacobian contraction |
| Source programme | Original trained configuration `c2_invariant`, not `c2c3_wave`; independent heldout has now been evaluated for the repaired variant | No C3 learned-circuit proof; do not use viewed heldout as a hyperparameter selection set |

**Original files:** `/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113/c2_bounded_repaired.pt`, `c2_bounded_repaired.json`, `c2_repaired_pure_unpromoted.pt`, `c2_repaired_pure_unpromoted.json`, `actual_student_history_32_bounded_repaired.json`. Preserve the original `candidate.pt` as baseline, never overwrite receipts. The documented "stationary attractor" wording in the pasted host summary is stronger than the evidence: a finite forced-token recurrent trace with a projected state is not a stationary distribution or global attraction proof. Report **bounded selected retained states and observed lower 32-token router margins**.

### Four-way 369 cross-lane Fourier interpretation

At the exact character level, `χ(k+l)=χ(k)χ(l)`, `1+ω+ω²=0`, and finite `C3×C3` Fourier convolution law are reusable across the following **typed carriers**:

- **Quantum/qutrit:** `DASHI/ComputerScience/BalancedTernaryC2C3DihedralCodecBridgeExact.agda` gives C2 inversion, C3 cyclic shift and a qutrit basis transport (same *three-point trit carrier*, with S3 reflection). Source Hilbert amplitude and general QFT remain distinct.
- **369/Monster:** `DASHI/Wikimedia/IbrahimC3ZetaRegularCharacterOEISQuantumGRBidiExact.agda` gives exact `53 + 65610(1+ζ₃+ζ₃²)` representation regular-part cancellation. Cyclotomic ζ₃ is not analytic ζ(s).
- **NS:** `DASHI/Physics/Closure/NSTriadKNLuoGalerkinFourierModePairExact.agda` owns *physical periodic Fourier mode* addition, directional cross defects, explicit high-high→zero-output mode pairs. A 3×3 torus modulo-three toy Fourier convolution has no physical viscosity, no 3D Galerkin solution or nonlinear rate bound.
- **YM:** `DASHI/Physics/YangMills/BalabanP33ThreeCycleBlockTorusCrossPollinationExact.agda` owns an index-prototype `C3×C3` commuting block translation. It explicitly **does not construct** the physical side-four SU(N) Wilson/Balaban RG source or physical TV/Dk inequality.
- **Atomic:** `DASHI/Physics/Chemistry/AtomicPeriodicTable369GenerativeExact.agda` owns staged 369 substrate, representation, fermion antisymmetry, shell occupancies, valence/observable quotient; finite ternary phase does not itself determine a hydrogen Hamiltonian or orbital selection.
- **LILA grokking:** the new paired student C2 source Fourier probe remains UNEXECUTED according to supplied host transcript; the existing C3 modular 8/9 training-fit counterexample is a finite character result, not trained C3 performance.

A new exact-integer helper in DASHI `scripts/check_369_c3x3_cyclotomic_fourier_crosslane.py` computes `Z3×Z3` character orthogonality, translation eigenphase, finite Fourier inversion, Parseval, convolution and a high-high opposite-frequency zero-output witness using integer pairs in `Z[ω]` (not floating sine), with four regression tests. **Committed but host execution pending.** The same-object transfer from this finite character algebra to physical NS/YM and original learned LILA is **still OPEN**. The canonical DASHI roadmap defines X369-A..G gates/ownership; do not duplicate physics proof obligations here.

### New physical/algorithmic decision and guardrails

The repaired student genuinely gains ~0.1664 heldout nats but remains substantially weaker than the dense teacher. That justifies a short **new protocol**, not blindly repeating 80min or interpreting 369 as a neural circuit:

1. Lock the already-viewed heldout data as an observational receipt. Predefine **fresh train-derived, duplicate-disjoint validation** for future parameter/geometry decisions; retain original unchanged baseline and source SHA.
2. Run `scripts/audit_c2_parity_c3_fourier_teacher_inputs.py` against actual matched teacher/learned layer0/1 activations: paired branch parity, odd/even cancellation and residual DC/rotating-phase energy. Distinguish local teacher input from genuine recurrent student.
3. Evaluate the trained capped model's real full-source C2 and cyclic/S3 router action (optional architecture is not automatically trained). Cross-check **greedy token policy** history; prior 32-token trace uses original corpus tokens by default.
4. Construct a matched, preregistered C2 vs `c2c3_wave` vs S3/face experiment with *task-level* group actions and independent semantic face/qualification channels only where original interventions are meaningful; no C3 equivariance or RL/sophontic claim without empirical source data.
5. Continue mathematical A3 E8/Leech all-index theta and physical NS/YM input-frequency/localisation gates in separate proof tracks; they need same-object physical estimates and do not follow from a C3 Fourier identity.

**Required Agda policy correction:** earlier instructions in this runbook suggesting a bare `agda -i .` invocation are superseded. Always use `DASHI_NO_TMUX=1 scripts/run_agda29_parallel_check.sh <target>` under DASHI's official toolchain. This is not optional.

---

## 2026-11-14 X369-F matched C2 versus C2×C3 source router test — ready for native admission, not yet run

**New original experiment owner** `scripts/screen_bounded_c2_vs_c2c3_router_fresh.py` and unit tests `tests/test_bounded_c2_c2c3_matched_screen.py`. Current source checkpoint is the user's **bounded repaired 4.5966 heldout CE** `c2_bounded_repaired.pt`, `c2_invariant` router, cap R=64 on layers10/11, source early gains .75/1.00. Do **not** overwrite it or tune on original heldout.

The full nine-row paired teacher-input audit `c2_layer01_parity_c3_teacher_input_full9.json` is now reported **executed**: ~66.8–67.4% of FFN approximation residual squared norm lies in the 2D C3 rotating plane for both early layers. **This matches the isotropic geometric null of 2/3 and therefore does not establish a C3-specific concentration or a learned C3 Fourier circuit.** Original layer1 odd-even cross-mean is ~−112.86 on actual x and zero to numerical precision after antipodal pair averaging, as C2 parity predicts. The huge cross-term is not directly comparable to the earlier normalised cross-value −0.49 without matching reduction/normalisation. These are local *teacher-input*, zero-expert-history probes, not free-running recurrent source observations.

DASHI native `C3×C3` test has **4/4 PASS** and exact character, Parseval, inversion and convolution checks. Its mathematical law does not itself provide physical NS/YM/quantum carrier transport or a trained C3 mechanism.

**Screen design:** keep **identical weights** when cloning the guarded trained 12-layer model, and select geometry `c2_invariant` versus `c2c3_wave`. Both source `HistoryOrbitMoE` routers have precisely `Linear(2*d_model,1)` plus the same expert_bias count; only original router weights and biases are trainable, all other experts, recurrent transitions, source layer0 even scales, cap R=64 and token model are frozen. For each of 2 predeclared seeds, optimise each arm for **24 AdamW steps**, same original fit rows, batch, LR, zero weight decay, dev interval and gradient clipping. The head-to-head score is `c2c3_wave.best_fresh_ce - c2_invariant.best_fresh_ce`, with lower preferred.

**Data admission:** `--fresh-validation-ids` is a **new, previously unused corpus/split** with the same tokenisation and sequence width, not the 9 repeatedly screened development rows or the original already-viewed heldout. The script fail-closes if any exact contiguous 16-token span overlaps either original training file or prior heldout, if validation rows duplicate, or if SHAs for original repaired source/train/heldout mismatch the original immutable receipt. An n-gram guard does **not** substitute for document-level/near-duplicate provenance; acquire and verify corpus identity separately. This fresh file is a validation set used for best-step choice, **not a subsequent untouched promotion test**. The original heldout file is loaded only to reject overlapping ngrams and must not be scored.

**Host command (only after obtaining a verified, fresh data file)**:

```bash
cd /home/c/Documents/code/DASHIg/LeechTransformer
git fetch origin
git switch experiment/consumer-preserving-multiscale-codec-20261031
git pull --ff-only
bash scripts/check_c2_quality_maxcut.sh

ROOT=/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.screen_bounded_c2_vs_c2c3_router_fresh -- \
  --candidate "$ROOT/c2_bounded_repaired.pt" \
  --source-repair-receipt "$ROOT/c2_bounded_repaired.json" \
  --teacher "$TEACHER" --config "$CONFIG" \
  --original-train-ids "$TRAIN_IDS" \
  --original-heldout-ids "$HELDOUT_IDS" \
  --fresh-validation-ids "$FRESH_VALIDATION_IDS" \
  --steps 24 --seeds 2 --batch 1 --device cuda \
  --models-dir "$ROOT/x369_f_matched_models" \
  --output "$ROOT/x369_f_c2_vs_c2c3_fresh.json"
```

`TEACHER`, `CONFIG`, `TRAIN_IDS`, `HELDOUT_IDS` remain actual previously verified source paths; the **fresh validation path must be newly supplied and independent**. Do not invent or re-label existing sources as fresh. No native GPU experiment has been executed in this chat; new suite expected 3 additional tests but **not yet run**. Source models are CPU cloned and only one arm occupies RX580 GPU at a time. Optional arm checkpoints are saved with source SHA, geometry/seed, cap-preserving config, best-step provenance and `release_admitted=false`. The screen uses fresh validation for model selection; a further locked new heldout (not previously viewed) is required before a conclusive quality comparison.

**Decision rule:** C3 is useful only if phase routing lowers the same fresh CE against the **matched retrained C2 router** consistently over seeds, preserving C2 antipodal routing, bounded R64 histories, native guarded/pure parity, and resource cost. The ~67% rotating-plane energy by itself is uninformative against a 2/3 isotropic null. If both arms tie or C2 wins, investigate whether a task-defined C3 action or true triadic face labels are missing rather than forcing an architectural C3 promotion. `X369-F` remains OPEN pending physical results.
