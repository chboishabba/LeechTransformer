# Executable C2 early-layer quality rescue and formal max-cut

**2026-11-02 — PR #4.** This file is a reproducibility/acceptance protocol,
not a report of new training success. Original neural model: Anatolii
Kornienko, *Leech-LILA*, DOI 10.5281/zenodo.18784424.

## D1: source evidence is now actually in-repository

User-executed original C2 report is preserved **verbatim** at
`docs/receipts/c2_train_only_quality_diagnosis_20261102.json`,
13,137 bytes, with original uploaded SHA-256
`e31dec9a6135d1e9b0a819e93947d8ebfb610491473c83e118fcbe3d58aff4fc`.
The report contains 4032 train-only token targets, dense teacher
CE 2.5456663616, complete C2 student CE 8.6084148468,
uniform CE 7.6246189862, isolated layer 1 delta +3.4202431119,
prefix 0..1 delta +4.0067042245, and a 2-row teacher-forced
probe (relative layer-1 MSE 86.61932, teacher/student RMS ratio
9.273). All logits finite. The original report does NOT contain
teacher/checkpoint/token file hashes; do not imply the source SHA
is a model provenance receipt or infer which saved weights were used.
The user reported focused regressions 34/34 and full PR #4 suite
125 tests + 7 subtests on their revision. No code in THIS later
tranche has been executed by the GitHub connector.

## D2: actual saved-weight counterfactuals, no retraining

`scripts/ablate_early_layer_quality.py` uses the existing
`load_pair`, `substituted_ffns`, `token_cross_entropy`,
`HistoryOrbitMoE`, and `canonical_even_input`.
The **same** trained dense/odd/even weights are interrogated with:
- `full`: original learned MoE + even student at the chosen layer;
- `even_only`: teacher backbone with solely the trained even channel;
- `odd_only`: solely trained routed signed experts;
- `reset_history`: zero per-expert memory BEFORE EVERY token;
- `balanced_router`: fixed top-k by batch row, uniform test of
  which experts are selected (not a deployable router);
- `zero_orientation`: score zero invokes the existing
  first-nonzero-component sign tie-break;
- `flip_orientation`: opposite learned orientation weight.
Each intervention restores source module references and parameters,
including upon an exception. Every variant reports real model CE,
teacher-input FFN RMS/relative MSE/uncentered cosine, learned
route-context scope and the same-probe least-squares scale oracle.
The scalar is fit on the evaluated probe, so it is NOT heldout or
deployment evidence. The probe also receives a separately verified
double-precision pointwise
`|NLL_student−NLL_teacher| ≤ 2 ||logits_student−logits_teacher||∞`
certificate (**F08 executable inequality**, not Agda kernel proof).
The actual response-quality metric remains CE across the full
provided TRAIN rows, not the small teacher-input probe.

Ablations require the student and teacher to have **identical**
non-FFN pretrained shared weights. Any architecture/weight mismatch
fails closed rather than reporting a false layer-level cause.
These tests identify sensitivity, not a unique causal or learning
mechanism; frozen-weight interventions are not retraining experiments.

```bash
python -m unittest \
  tests.test_moe_quality_collapse_diagnostics \
  tests.test_early_layer_quality_ablations \
  tests.test_guarded_moe_continuation -v

# The user owns and checks actual local checkpoint/config file paths.
python -m scripts.ablate_early_layer_quality \
  --teacher data/best_model.pt \
  --student /tmp/real_text_moe_jq1_smoke/c2_invariant/student_model.pt \
  --config /tmp/leech_dense_config.json \
  --train-ids /tmp/real_text_moe_jq1_smoke/dataset/train_ids.npy \
  --layer 1 --probe-rows 2 --batch 2 --device cpu \
  --output /tmp/real_text_moe_jq1_smoke/c2_invariant/layer1_ablation.json

# Independently repeat layer 0 by changing --layer 0 and --output.
```

## D3: graduated full-logit learning without full replacement confusion

`models/model.py` provides
`student_only=True, student_only_layers=(1,)` for a **guarded**
hybrid. The chosen block uses the strict student forward; other
blocks run their *calibrated* dense fallback, initially all zero
gates. The full teacher is not deleted. No deployed gate is opened
by the training forward. A second rollout with all layers selected
continues to represent the eventual full student.

`train_guarded_moe_continuation.py --focus-layers 1`
freezes every other MoE and even student parameter, trains the
selected block with end-to-end next-token CE, teacher KL and local
FFN parity, and only calibrates the selected gate using the separate
train-derived calibration partition. It reports both the full-student
CE and the **focused hybrid student-only** CE. A focused run is
unconditionally marked `full_student_replacement_pass=false`,
regardless of its result: a one-layer improvement is NOT a
teacher-free twelve-layer model.

Use the new `scripts/prepare_train_only_development.py` to make the
TRAIN-only partition from the existing `prepare_real_leech_tokens.py`
provenance. It verifies the original train-array hash and inherits
that source preparer's one-distinct-document-per-row rule. It never
opens `heldout_ids.npy`, and records every selected source row and
both output-file hashes. The resulting development set is explicitly
not a final independent heldout set.

```bash
python -m scripts.prepare_train_only_development \\
  --dataset /tmp/real_text_moe_jq1_smoke/dataset \\
  --output /tmp/real_text_moe_jq1_smoke/c2_train_study \\
  --seed 41 --development-fraction .2
# train_only_fit.npy and train_only_development.npy live in this folder.
```

To avoid adaptive use of the actual final heldout data during
the architecture/layer sweep, create a reproducible TRAIN-only
development split: disjoint fit/development token rows, and use
those two arrays in the focused CLI below. The final independently
heldout data must remain untouched until the selected full model
is trained/evaluated. Since the tokenizer and document-level
provenance are important, keep the originating document IDs in
the experiment receipt; row-level token disjointness alone does
not prove document-level disjointness.

```bash
python -m scripts.train_guarded_moe_continuation \
  --teacher data/best_model.pt \
  --student /tmp/real_text_moe_jq1_smoke/c2_invariant/student_model.pt \
  --config /tmp/leech_dense_config.json \
  --train-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/train_only_fit.npy \
  --heldout-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/train_only_development.npy \
  --focus-layers 1 --epochs 3 --batch 2 \
  --student-ce-weight 1 --student-kl-weight 1 \
  --output /tmp/c2_focused_layer1_guarded.pt

# Repeat independently with --focus-layers 0, then --focus-layers 0 1.
# After inspecting TRAIN-only development results, train a full model
# WITHOUT --focus-layers on the original full train set and then
# perform the ONE independent final heldout evaluation and strict
# full-gate teacher-free promotion.
```

The **full** student promotion condition is unchanged:
`all(gate == 1)` + independent heldout CE within the declared
dense baseline +0.02 nats + standalone checkpoint parity/teacher
parameter erasure. Even a promising focused intervention or low
local FFN error cannot discharge it.

## D2-D3 negative-control training (matched parameter count)

`scripts/train_early_layer_dense_control.py` reuses the same
checkpoint loader and teacher-normalized `layer_inputs` hook.
It creates a NEW ordinary dense GELU FFN at the chosen layer,
with trainable parameters no more than the current odd-MoE plus
even student and fewer than `2*d_model+1` parameters of slack.
It freezes all other backbone and FFN parameters.
It uses the same train-derived fit/development token arrays as
focused MoE continuation, with teacher-input local MSE plus
full next-token CE and teacher-logit KL. It compares an
ordinary, non-geometric student against a geometric one at a
similar parameter budget; the number of optimizer steps, input
length, seed and data must match. This is a necessary STEELMAN
control; it neither reproduces the original dense teacher FFN
width nor certifies that history/routing is useful.

```bash
python -m scripts.train_early_layer_dense_control \\
  --teacher data/best_model.pt \\
  --student /tmp/real_text_moe_jq1_smoke/c2_invariant/student_model.pt \\
  --config /tmp/leech_dense_config.json \\
  --fit-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/train_only_fit.npy \\
  --development-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/train_only_development.npy \\
  --layer 1 --epochs 3 --batch 2 --learning-rate 0.0003 \\
  --output /tmp/c2_dense_matched_layer1_control.pt

# The focused signed-history contender:
python -m scripts.train_guarded_moe_continuation \\
  --teacher data/best_model.pt \\
  --student /tmp/real_text_moe_jq1_smoke/c2_invariant/student_model.pt \\
  --config /tmp/leech_dense_config.json \\
  --train-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/train_only_fit.npy \\
  --heldout-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/train_only_development.npy \\
  --focus-layers 1 --epochs 3 --batch 2 --learning-rate 0.0003 \\
  --output /tmp/c2_focused_layer1_guarded.pt
```

**Fairness boundary:** matching trainable parameter count does
not match FLOPs, wall-clock compute, history capacity, or inductive
bias. Record those separately before any efficiency claim.
Results are development-set diagnostic scores, not final heldout
CE. Newly written tests `tests/test_early_layer_dense_control.py`
check parameter slack, frozen backbone and one-epoch execution.

## Exact theorem cut and ownership

- **F01:** even/odd real-additive reconstruction, with named inversion
  on both activation and history. Reuse the established parity owner.
- **F02a new source:** `LeechOddOnlyApproximationObstructionExact.agda`
  proves an odd history-student cannot equal a non-odd target at both
  mirrored states; imports the existing `invertHistory`. Requires
  Agda kernel acceptance.
- **F02b still open:** quantitative real inner-product lower bound
  on two-point MSE by even-component squared norm. This is NOT
  established by the algebraic nonfit lemma.
- **F03–F06:** existing conditional signed-chart, selection and gate
  equivalences, their concrete source↔PyTorch refinement and full
  numeric admission remain separate.
- **F07 new source:** `LeechLayerErrorTransportExact.agda`
  proves a full finite trace bound given explicit triangle,
  monotonicity and **actual layer-specific Lipschitz witnesses**.
  No concrete numerical Lipschitz factors are supplied; kernel
  check remains open.
- **F08 partial executable:** stable logsoftmax diagnostic checks the
  sharp, standard 2×logit infinity bound on actual probed tokens.
  A formal real/logsumexp proof and explicit F07→actual PyTorch
  tensor/CE correspondence remain open.
- **F09–F12:** existing source-owned E8 canonical history,
  consumer-scoped quotient, authenticated packet and independently
  source-labelled Sophontic/trialectic cases are not new fitted
  model successes.

The new F02a and F07 owners are imported into
`DASHI.Unified.LeechQualityGuardTOEBridgeExact` and therefore
the same golden `Unifier` and `EverythingTOESymmetryResolved369`
import graph. This is **source-connected**, not kernel-accepted
in this session.

## Exact remaining measured frontier

**D2 ablation receipt not yet executed.**
D3 actual train-only focused learning trajectories not yet executed.
D4 full 12-layer teacher-free model not accepted. D5 Agda kernel
and concrete model refinement not executed. D6 whole socket/WAN,
durable ACK/recovery and trained E8 history Pareto not measured.
D7 independent paired/cube semantic study not measured.
The observed real C2 result still fails spectacularly and must
never be relabelled successful merely because a new theorem or
training method exists.

Original Kornienko and E8/Leech mathematics ownership,
in-repo source attribution and Ibrahim Monster 3B authorship
are unchanged. No ESD files edited.


## 3 November 2026 — executed matched layer-1 quality result and corrected steelman

**User-executed experiment, reported 3 November 2026.** Saved local
receipts are
`/tmp/real_text_moe_jq1_smoke/c2_focused_layer1_guarded.json` and
`/tmp/real_text_moe_jq1_smoke/c2_dense_matched_layer1_control.json`.
Those machine-local paths are NOT accessible via this GitHub
connector. Values below are user-reported summaries, not parsed
receipt hashes, independently reproduced scores or final heldout
measurements. Progress printing was tested and pushed at
`d13d3f3` (verified PR #4 head at time of reconciliation).

| Train-only development model | CE nats | Delta from dense teacher |
| --- | ---: | ---: |
| Frozen dense teacher | 2.513 | 0 |
| Focused signed C2, guarded hybrid | 2.486 | -0.027 |
| Focused signed C2, strict layer-1 student (all other FFNs dense) | 2.833 | +0.320 |
| One parameter-matched ordinary dense GELU FFN | **2.191** | **-0.322** |

The ordinary control has **1,185,413** trainable parameters versus
**1,185,606** in the signed C2 odd-plus-even student (difference
**193**, about 0.016%). Both reportedly used the same seed,
three epochs, CPU and deterministic train-only development split.
The control outperforms the strict layer-1 C2 student by **0.642**
nats and even beats the teacher by **0.322** nats. The guarded
hybrid advantage relative to teacher is only 0.027 nats and
**retains the original FFN**: a 0.5 gate is a train-only
calibration admission, not teacher erasure or a compact deployment.

**Negative conclusion, without overgeneralization.** The current
experimental procedure does NOT support a layer-1 quality advantage
from signed/history-bearing C2. Do not claim any such advantage in
the paper/abstract, deploy it, use the hybrid CE as standalone
student CE, or run a transport-quality promotion from this receipt.
The original 12-layer student remains unfit (prior CE 8.608).

**Important confounds found by checking source.**
The executed focused C2 procedure had (i) a separate
calibration holdback from its fit rows; the ordinary dense control
trained on all supplied fit rows, and (ii) C2's loss contained
an averaged all-layer local MSE plus hybrid CE/KL and separate
focused student-only CE/KL; the dense control used the chosen
layer's local teacher-input MSE plus one focused-model CE/KL.
Thus same parameter count/seed/epochs/CPU does not establish
identical *optimizer data or objective*. In addition, C2
reused an earlier candidate's learned weights while the
dense control was newly initialized, and matched parameter
count alone does not match sparse MoE FLOPs/time/routing.
One observed seed cannot estimate uncertainty.
These confounds do not erase the empirical ranking; they
limit attribution of the difference to the C2 inductive prior.

### New opt-in matched-objective follow-up, not a retroactive correction

The original C2 training behaviour remains the default.
`train_guarded_moe_continuation.py` now has
`--matched-control-objective` for exactly **one** focus layer.
It replaces the training objective with
**one focused-layer teacher-input MSE + strict focused-student
next-token CE + strict focused-student teacher-logit KL**
at weights 1/1/1, temperature=1. For training only, it zeroes
every quality gate before each epoch so the local teacher
activation is not contaminated by the last epoch's calibrated
hybrid. Subsequent gate calibration still uses the C2-only
holdback and preserves fail-closed admission; no full student
promotion occurs. It exports the actual C2 **optimizer-fit
row indices** and SHA-256 of their token window bytes.
`--export-optimizer-fit-ids` writes the exact optimized rows
for the ordinary dense control, so BOTH optimizers now see
identical windows, row ordering, epochs, seed, local loss
scale and CE/KL terms. The saved comparison is still
training-development evidence; the model classes and
initializations differ.

```bash
# After prepare_train_only_development has created c2_train_study:
python -m scripts.train_guarded_moe_continuation \
  --teacher data/best_model.pt \
  --student /tmp/real_text_moe_jq1_smoke/c2_invariant/student_model.pt \
  --config /tmp/leech_dense_config.json \
  --train-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/train_only_fit.npy \
  --heldout-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/train_only_development.npy \
  --focus-layers 1 --matched-control-objective \
  --export-optimizer-fit-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/matched_optimizer_fit.npy \
  --epochs 3 --batch 2 --seed 41 --learning-rate 0.0003 \
  --output /tmp/real_text_moe_jq1_smoke/c2_focused_matched_loss.pt

python -m scripts.train_early_layer_dense_control \
  --teacher data/best_model.pt \
  --student /tmp/real_text_moe_jq1_smoke/c2_invariant/student_model.pt \
  --config /tmp/leech_dense_config.json \
  --fit-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/matched_optimizer_fit.npy \
  --development-ids /tmp/real_text_moe_jq1_smoke/c2_train_study/train_only_development.npy \
  --layer 1 --epochs 3 --batch 2 --seed 41 --learning-rate 0.0003 \
  --output /tmp/real_text_moe_jq1_smoke/c2_dense_matched_loss_control.pt

python -m unittest tests.test_guarded_moe_continuation \
  tests.test_early_layer_dense_control \
  tests.test_train_only_development_split -v
```

**Acceptance decision:** Treat D3 as **executed but failed C2
quality superiority**, not completed teacher-free quality.
D4 requires all twelve gates exactly 1, teacher FFNs deleted,
strict logit parity and independent final heldout CE within
the declared budget. It remains OPEN. F02/F07 source lemmas
are consistent with this negative empirical result but
provide no explanation or numerical quality guarantee.
Do not spend D6/D7 evidence on a failed student-quality premise.
Keep the classical Leech/E8 authorship and original
Kornienko model credit intact.

**Remaining evidence:** Execute the newly added matched-mode
regressions and the matched-objective second comparison.
If the dense control continues to win across a seed panel and
appropriate FLOP/time normalization, stop promoting C2 as a
quality prior; retain C2 only where it independently earns
its symmetry, codec or error-correction value. Preserve final
heldout data for the selected complete candidate, not for
adaptive ablation selection.
