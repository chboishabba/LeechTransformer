## 20 November — current native CPU acceptance and source authority

Local operator report: corrected synthetic native fixture and both standalone probes PASS, 10/10 strict boundary cases, former tied fixture rejected, four-load Nat [0,2,9,0], same-clock Nat [0,9]; portable CPU contracts and C2 quality max-cut reported exit 0; projected score weld 6/6 after precision correction. These are reported local results, not independently rerun on this reconstructed commit. Detached a77bf4d was not available remotely; the documented precision and runbook corrections are reconstructed here. Actual fitted restoration, native SLR development replay, independent error bounds and Agda acceptance remain OPEN.

Original model blob remains a2a250462ce8e663591cd8a8f784fad9bde1b15d. Only scoped dispatcher observation is active. Cutoff t=14/5 is ambiguous; zero response is guaranteed strictly above it. Old dated execution statuses below are historical. SLR relation/mention transport is a separate upstream provenance lane; it does not replace the R4 native development JSONL, original tokenizer, checkpoint or histories.

---

# 17 November — single-command original fitted-source observation entry point

Use `scripts/run_x369_r4_original_kink_observation.py` to run
the currently implemented COMPLETE original fitted-source
observation pipeline in one ordered process. It does not ask
for a fictitious standalone R4 checkpoint, and it refuses
all missing/bad-hash source assets BEFORE creating an
output directory. It reuses the exact exporter, original
SLR development trace capture, four-load reset and
same-clock history probes, and checks their returned
module/probe SHA identities before its final receipt.

```bash
python scripts/run_x369_r4_original_kink_observation.py \
  --repaired-c2-checkpoint ORIGINAL_REPAIRED_C2.pt \
  --repaired-c2-sha256 EXPECTED_REPAIRED_C2_SHA256 \
  --source-repair-receipt ORIGINAL_REPAIR.json \
  --native-slr-jsonl ORIGINAL_NATIVE_R1.jsonl \
  --native-slr-jsonl-sha256 EXPECTED_NATIVE_JSONL_SHA256 \
  --fitted-delta ORIGINAL_FITTED_R4_DELTA.pt \
  --fitted-delta-sha256 EXPECTED_FITTED_DELTA_SHA256 \
  --config ORIGINAL_SOURCE_CONFIG.json \
  --r1-summary ORIGINAL_R1_SUMMARY.json \
  --tokenizer ORIGINAL_SENTENCEPIECE.model \
  --layer-index 0 \
  --document-id ORIGINAL_DEVELOPMENT_DOC_ID \
  --token-position 4 \
  --load-levels 0,1,2,3 \
  --response-kind coordinate \
  --response-index 0 \
  --nat-scale 10 \
  --output-dir .leech-r4-original-kink-observation
```

The output directory contains byte-bound intermediate
restored layer, native source x/all-E history, reset
load-probe receipt, same-clock history-probe receipt,
and `original_r4_kink_observation.json`.
The latter reports observed positive OR negative
four-load kink, fixed-route contrast and history-path
dependence, without requiring a positive experiment
to count as a valid source observation.

**Acceptance boundaries:**
- Actual native source files and SHA256 must exist and
  be independently supplied; no synthetic substitution
- Original repaired C2 producer receipt, fitted delta
  geometry fingerprint and R1/development split are
  consumed through their ORIGINAL owners
- The float32/float64 source activation and full all-E
  pre-step histories are captured from an original
  native development record/token, not invented
- Both probes verify the same restored checkpoint
  and native trace byte hashes
- Original topk ties/zero margins are refused
- The pipeline neither proves roundoff intervals,
  heldout generalisation, task semantics, intrinsic
  hysteresis nor Agda kernel acceptance
- It is SOURCE WRITTEN. Actual fitted original
  files were NOT available to execute this runner
  during this chat. Do not report the result as passed

---

# 17 November — authoritative fitted-source workflow (supersedes optional logits-return API)

**Important correction:** `models/history_orbit_moe.py` was RESTORED
byte-for-byte to the original fingerprint-bound blob
`a2a250462ce8e663591cd8a8f784fad9bde1b15d`.
The prior experimental `return_logits=True` model signature
mutation was REMOVED because it alters a file inside the
`r4_geometry_source_fingerprint()` source hash owned by the
existing **fitted R4 residual-delta producer**.

The currently supported read-only observer is
`scripts/observe_x369_r4_original_topk.py`.
It uses a scoped `TorchDispatchMode` to observe the REAL
`aten.topk` input tensor while the unchanged original
`HistoryOrbitMoE.select_source_experts(x,h)` runs.
It refuses missing/duplicate observations, nonfinite logits,
unexpected IDs/weights and the opt-in `sort` fallback.
The old 2-output source method API is EXACTLY unchanged.
An isolated original-shaped CPU dispatcher smoke passed **3/3**
on PyTorch 2.10.0; the entire original Leech source regression
remains NOT run in this assistant container.

## Reconstruct a real fitted R4 layer from existing original owners

The actual source R4b training pipeline stores ONLY the
`x369_r4b_fitted_source_bound_residual_delta` payload
and the frozen REPAIRED guarded C2 source checkpoint,
not a full fitted standalone `HistoryOrbitMoE` checkpoint.

New source tool:
`scripts/export_x369_r4_restored_moe_layer.py`.
It REUSES original
`scripts.verify_trained_federated_moe.checkpoint_parts`,
`infer_config`, and
`models.x369_fair_c3_residual.restore_residual_delta`.
It checks actual repaired C2 checkpoint, native SLR JSONL
and fitted R4 delta SHA256 against independently
supplied expected hashes; validates the repaired
C2 source receipt; enforces original R64 guarded C2
configuration, original source R4 delta semantics/
geometry source fingerprint, exact tensor shapes,
and independently strict module state restoration.
No accidental `strict=False` model fit, no fabricated
source checkpoint, and no F5/reserved data.

Without a development state, export the fitted MoE layer:

```bash
python scripts/export_x369_r4_restored_moe_layer.py \
  --repaired-c2-checkpoint ORIGINAL_REPAIRED_C2.pt \
  --repaired-c2-sha256 EXPECTED_REPAIRED_C2_SHA256 \
  --source-repair-receipt ORIGINAL_REPAIR.json \
  --native-slr-jsonl ORIGINAL_NATIVE_R1.jsonl \
  --native-slr-jsonl-sha256 EXPECTED_NATIVE_JSONL_SHA256 \
  --fitted-delta ORIGINAL_R4_DELTA.pt \
  --fitted-delta-sha256 EXPECTED_FITTED_DELTA_SHA256 \
  --config ORIGINAL_SOURCE_CONFIG.json \
  --layer-index 0 \
  --output-checkpoint .leech-r4-kink/restored_original_layer0.pt \
  --output-receipt .leech-r4-kink/restored_original_layer0.json
```

To also CAPTURE actual native source x/H in the same
strictly restored model, supply ALL these extra arguments
to the same command:

```bash
  --output-probe-input .leech-r4-kink/original_dev_load_probe.json \
  --r1-summary ORIGINAL_NATIVE_R1_SUMMARY.json \
  --tokenizer ORIGINAL_SENTENCEPIECE.model \
  --document-id ORIGINAL_DEVELOPMENT_DOC_ID \
  --token-position 4 \
  --load-levels 0,1,2,3 \
  --response-kind coordinate \
  --response-index 0 \
  --nat-scale 10
```

Choose an ORIGINAL development document/position and
response observer from the real source receipts.
The original `validate_source_docs` owner rejects unknown,
reserved, prefix-only or repeated source rows. The actual
original `LeechGPT` runs `student_only=True` with
`return_expert_histories=True`, and the selected layer's
original `HistoryOrbitMoE.step` is temporarily observed
**without replacing its computation**. The exact selected
activation and all-E pre-step histories are exported.
An independently reconstructed original MoE layer MUST
replay the captured original step BITWISE before those
source inputs are emitted.

The four explicit load multipliers MUST contain `1`,
identifying the actually observed activation; other
loads are **counterfactual activation interventions**,
not naturally observed physiological or time-varying load.
Their effects require same-source replay rather than
treating them as existing original SLR corpus observations.

Once the real source module and x/H have been emitted,
run both original consumer probes with the SAME receipt:

```bash
python scripts/probe_x369_r4_kink_checkpoint.py \
  --checkpoint .leech-r4-kink/restored_original_layer0.pt \
  --input .leech-r4-kink/original_dev_load_probe.json \
  --layer-receipt .leech-r4-kink/restored_original_layer0.json \
  --output .leech-r4-kink/original_reset_kink.json

python scripts/probe_x369_r4_history_path_dependence.py \
  --checkpoint .leech-r4-kink/restored_original_layer0.pt \
  --input .leech-r4-kink/original_dev_load_probe.json \
  --layer-receipt .leech-r4-kink/restored_original_layer0.json \
  --output .leech-r4-kink/original_same_clock_history.json
```

Shared `verify_x369_r4_source_probe_authority.py` refuses
a native SLR source trace without the exact restored
layer receipt. It checks module-checkpoint bytes, probe-input
bytes, original checkpoint/delta/SLR identity fields,
model configuration/dtype, nonpersistent history cap,
source-document identity and actual all-E history tensor
SHA256. Synthetic standalone fixtures still run but
report `source_input_bound_to_original_R4_export=false`.
The verifier does NOT independently authenticate a
digital signature, prove heldout independence or
produce a certified Torch floating error interval.

### Gate status

- Original fingerprint-bound model restored: **done**, actual
  original Git blob SHA checked
- Isolated CPU dispatcher capture: **3/3 local checks passed**
- Full-load ideal-real SymPy: **53/53 direct local checks passed**
- Float32/float64 source JSON tensor roundtrip: **2/2 local checks passed**
- Strict repaired C2→R4 delta→layer exporter:
  **source-written, actual files absent; execution OPEN**
- Native SLR development x/H capture and bitwise replay:
  **source-written; real file execution OPEN**
- Original native Torch checkpoint and history probes:
  **source-written; full-repo execution OPEN**
- Independent floating error, fitted response,
  Agda kernel, heldout and on-policy hysteresis: **OPEN**

Do not mistake this source-constrained producer and
mathematical architectural possibility for a
measured fitted-model kink.

---



## Exact continuous-load routing shape and native threshold regression

DASHI PR #1526 now has
`scripts/check_x369_r4_full_load_kink_envelope_sympy.py`,
an exact real algebraic source-route partition for the
REPAIRED source-shaped synthetic checkpoint.

The five exact rational topk transition loads are:
`5/4, 19/9, 5/2, 19/7, 14/5`.
The six strict original ordered topk2 regions over t>=0
are `(0,1),(1,0),(1,2),(1,3),(3,1),(3,2)`.
The selected output rises to
`sigmoid(31/9)≈0.969065` at `t=19/9`,
declines as faster-growing other experts enter
top2, and finally falls to zero after
`t=14/5`. The latter cutoff itself
is a **source-ambiguous torch.topk tie**
between positive and zero-output
experts: the left limit is
`sigmoid(-3/5)≈0.354344`,
the right limit is 0, and neither
a deterministic tie choice nor a
continuous physical model is claimed
at the exact boundary.

The original native CPU fixture
`scripts/check_x369_r4_native_topk_kink.py`
now probes BOTH sides of every
threshold with ±1e-4 load offsets
through the actual original
`select_source_experts` and `step`.
All ten cases assert unique
ordered source IDs and strictly
positive source score margins.
They are **source-written tests,
not an execution receipt** in this
session. Independent exact affine
route-side calculations passed
10/10 locally, and independent
piecewise symbolic verification
passed 27 checks.

Do not treat a four-load kink
as evidence that the original
trained model globally suppresses,
or force the original source's
unspecified topk tie choice into
an invented semantic convention.


## 17 November second matched original-source ablation

The repaired same-clock/two-prefix probe now computes BOTH
native original source `step` and an intervened fixed selected
pair through the original `execute_selected` receiver.
The reference pair is the **low-prefix path's original
second-step selection**, and is held fixed on the high-prefix
probe while weights are **recomputed from the HIGH path's
own original source logits**, using its own retained
selected histories. No source routing oracle or FFN
implementation is duplicated.

For the disclosed corrected synthetic source parameters:
- the native low-prefix and high-prefix responses are
  respectively `Nat(0)` and `Nat(9)`;
- the fixed shared second-step selected pair `(0,2)`
  yields `Nat(0)` and `Nat(0)` on the same two histories.

Thus a source-shaped fixture can display a history-dependent
**native routing** result whose discrepancy disappears
under matched selected-ID intervention. This is a
finite **mechanism-discrimination witness**.
A fitted network's selected FFNs can still
depend on signed histories and may not
agree under fixed IDs; the source result
is observational and does not establish
a trained-model-only causal theorem.

The source JSON reports
`same_load_fixed_pair_response_abs_difference`,
`source_selection_ablated_history_path_difference`,
fixed pair IDs, original reweighted probabilities
and signs, and
`source_selection_only_causal_theorem_certified=false`.
The synthetic native source test asserts the
paired difference/zero difference and the
earlier tied source negative control.
DASHI reuses
`DASHI.Core.IntersectionalNonFactorability`
for BOTH the native no-factorisation and
fixed-pair positive factorisation:
`sourceFixtureResponseCannotFactorThroughLoad`
versus
`fixedProbePairDoesFactorThroughCurrentLoad`.
This comparison is exposed in the two TOE
import graphs, pending kernel acceptance.

**Execution boundary:** independently checked original
score arithmetic and strict margins;
the exact committed native Torch source tests
remain not run in the model container.
## Strict-topk correction — 17 November 2026

The **previous** synthetic source construction used expert 3
`(s3,b3)=(10,-13)`. At the second-step probe after the low
prefix, that produced logits `(8,3,7,7)`: experts 2 and 3
TIED for the second selected slot. The original
`torch.topk` is not an authenticated tie-break and that
configuration did **not** establish a deterministic
history-path selected ID or Nat observation.

The **current corrected native fixture** changes *only* expert 3
to `(s3,b3)=(11,-16)`, retaining the top-2
four-load kink and fixed-route control. The four reset-state
score vectors are now
`(8,3,-11,-16)`, `(8,7,-2,-5)`,
`(8,11,7,6)`, `(8,15,16,17)`.
The same-clock probe vectors are now
`(8,3,7,6)` and `(8,11,-11,-16)`,
both with **unique** ordered selected pairs and strictly
positive selected/unselected separation. The new shared
`strict_source_order_margin` admission in the original
four-load and history-path producers fails closed on any
future source topk tie or ambiguous cutoff.

An independent local exact source-formula audit passed
**6/6** strict-selection/Nat checks for the corrected
fixture; the exact native Torch committed files and
Agda kernels remain not executed in this session.
Do NOT cite the old tied two-prefix witness as closed.

---

# X369 R4 early ResponseKink: native checkpoint and forced-route source probe

**Source authority:** original `HistoryOrbitMoE` on this LeechTransformer branch.
**Mathematical consumer:** existing DASHI
`DASHI/Cognition/ResponseKink.agda` and
`DASHI/Cognition/PNF/X369R4TopKSourceKinkWitnessExact.agda`.
**Research attribution:** Leech-LILA by Anatolii Kornienko.

This is a source-native PROBE, not an approximation or a new gate:
`scripts/probe_x369_r4_kink_checkpoint.py` calls
`HistoryOrbitMoE.select_source_experts`, `step` and
`execute_selected`, and never replaces their router implementation.

**SUPERSEDED historical API description — do not implement or use.** The original two-value API is restored; use the scoped observer instead.

The only backward-compatible change to original
`models/history_orbit_moe.py` is optional
`select_source_experts(..., return_logits=True)`.
The ordinary default still returns `(ids,weights)`.
When requested, it returns `(ids,weights,logits)` from the SAME
source computation. That is needed to compute a forced subset's
original-source score/softmax weights, not just freeze its old
weights. Source `torch.topk` is retained; ties are not redefined.

## First run the genuine source class against a synthetic checkpoint

From the original LeechTransformer repository root,
using the pinned Torch environment:

```bash
python scripts/check_x369_r4_native_topk_kink.py \
  --emit-probe-dir .leech-r4-kink
python scripts/probe_x369_r4_kink_checkpoint.py \
  --checkpoint .leech-r4-kink/synthetic_history_orbit_moe_state_dict.pt \
  --input .leech-r4-kink/synthetic_original_load_input.json \
  --output .leech-r4-kink/synthetic_probe_result.json
```

The first script creates **source-legal but entirely synthetic**
weights: shared C2 baseline zero, coherent C3 directed residual,
four expert biases, source-oriented sign=+, FFN constant output
for expert 1, and zero transition. The strict source logits are

    L(t)=(8,3+4t,-11+9t,-16+11t)

at t=0,1,2,3; ordered top2 are

    [0,1],[0,1],[1,0],[3,2].

Selected scalar outputs are
`1/(1+exp(5))`, `1/(1+exp(1))`,
`1/(1+exp(-3))`, 0.
Floor(10×output) gives the canonical early facilitation–
suppression Nat shape `[0,2,9,0]`.
With the **SAME source logits/FFNs/histories** but forcibly
replaying pair [0,1] at every load, the scalar selected output is
`sigmoid(4t-5)`, strictly increasing; hence the downturn
disappears. This proves an ORIGINAL R4 architectural possibility
and its matched topk-switch intervention, not an actual
learned model observation.

## Probe a real original R4 state

Use an existing original R4 checkpoint with byte-level ownership
and **actual** frozen x/H values; model must be an exact
`HistoryOrbitMoE` state_dict (or a dict keyed
`state_dict`/`model_state_dict`). If it belongs to
a full LeechGPT checkpoint, supply the exact source key prefix
with `--state-prefix` and ensure strict loading succeeds.
Do not silently fill missing tensors or discard unexpected keys.

The JSON input `--input` contains exact `model` constructor
arguments, `base_x` of dimension d, ALL E expert histories,
exactly four ascending `load_levels`, an explicit
`history_revision`, and `response` configured as a
nonnegative scalar `coordinate` or `l2` norm, with a declared
integer `scale` for the old Agda Nat projection.

The experiment uses independent reset states
`x_t=load_t*base_x` and THE SAME complete initial H
for every load. It is NOT a sequential recurrent trace.
Source `step` and `execute_selected` selected replay
must agree (including signs and updated selected histories);
if not, the probe refuses the result.

For each load the probe returns native logits,
selected IDs, softmax weights, orientation signs,
weighted response, **matched forced-route response** using
the original candidate logits for the fixed pair, and the
finite Nat response when nonnegative. The forced route is
an intervention; it is not a native selection at later loads.

The output includes SHA256 of original checkpoint BYTES and
input JSON, and of initial history tensor bytes. It records
the given authority revision. These hashes and strict load
give stronger provenance than manually entered strings,
but do not establish dataset independence, original R1
heldout compliance, trained semantic or behavioral value.
Run the pre-existing R4 R1/checkpoint admission pipeline
for those separate gates.

## Acceptance and non-promotion

- An observed source kink means all four source Nat
  responses have the strict old original shape. This alone
  does NOT prove which architecture component caused it.
- An absent kink is a negative model-specific result
  for the chosen checkpoint/load family, not a no-go for
  the architecture as a whole.
- A comparison with the fixed-route counterfactual
  isolates a possible routing-selection contribution.
  It does not rule out other changes in FFN activation,
  sign or history: examine each recorded source quantity.
- Source float measurements are not certified error intervals.
  `certified_float_error_intervals=false`; any rigorous
  Nat threshold margin must be provided independently.
- No genuine kernel-checked Agda theorem, GPU latency,
  physical energy, clinical physiology, Grokking semantics,
  or future on-policy recurrence is claimed by this source
  probe.

**Next physical source gates:** actual checkpoint and same-revision
history evidence, independent precision/rounding certificate,
four-load Nat cell error margins, forced-route ablation with
independent chosen load family, and on-policy recurrent-history
hysteresis studied separately. No F5/reserved data.


## Critical nonpersistent source-history cap

`HistoryOrbitMoE.history_cap` is owned by the ORIGINAL
source model configuration but is intentionally not
present in the `state_dict`. Therefore a fitted
checkpoint hash plus strict parameter load alone
cannot guarantee the SAME history transition
if that configuration is missing.

The input JSON must now explicitly specify

    "history_cap": null

when disabled, or a finite strictly positive
numeric cap when enabled. The original
checkpoint probe refuses to run without
the field, sets `model.history_cap` before
native replay, and records the exact
source cap selection in output metadata.
The synthetic source emitter writes
`history_cap: null` explicitly.
For genuine source revisions, recover
this value from the original owning
runner/config; never guess the default.


## Same-clock, same-load original recurrent-history probe

New source-owned execution:
`scripts/probe_x369_r4_history_path_dependence.py`.
This is the next gate after a four-load RESET comparison.

With exactly the SAME original checkpoint, dtype,
nonpersistent cap, initial all-E history and reference
input `x0`, run two original `HistoryOrbitMoE.step`
trajectories:

    path A: [load_0, load_2]
    path B: [load_3, load_2].

Both paths reach the shared probe `x=load_2*x0`
at the SAME clock index 1. Any difference in
the resulting full retained histories is caused
by the different original first-step intervention,
not by a different checkpoint or observation time.
Differences in the second response (if present)
show the source output is not a function of
CURRENT LOAD ALONE, conditional on this checkpoint.
That is **history path-dependence**; it is
NOT a universal bistability/hysteresis theorem.

The native synthetic fixture added to
`scripts/check_x369_r4_native_topk_kink.py`
now checks an analytically disclosed example
with original strict topk:

| First load | First selected | Same-load t=2 second-step selected | Second-step Nat |
|---|---|---|---:|
| 0 | (0,1) | (0,2) | 0 |
| 3 | (3,2) | (1,0) | 9 |

Under the explicitly synthetic zero selected-history
transitions, a selected expert's retained history
becomes zero after selection, whereas inactive
experts retain their original history. Thus:
- prefix0: updated e0/e1 zero, second logits
  `(8,3,7,6)`;
- prefix3: updated e3/e2 zero, second logits
  `(8,11,-11,-16)`.

The source output differs at exactly the
same probe activation and time; this demonstrates
the relevance of retaining all E histories,
not the existence of an actual fitted/modelled
trauma, biological process or recurrent
clinical mechanism. The presence of a
strict source-output difference does NOT
by itself establish the sign of a general
hysteresis loop or asymptotic multistability.

Run AFTER the synthetic checkpoint emission:

```bash
python scripts/probe_x369_r4_history_path_dependence.py \
  --checkpoint .leech-r4-kink/synthetic_history_orbit_moe_state_dict.pt \
  --input .leech-r4-kink/synthetic_original_load_input.json \
  --output .leech-r4-kink/synthetic_history_path_result.json
```

This producer calls the ACTUAL original source
`step` and `select_source_experts`;
records the two first selected sets, original
second-step logits, selected IDs, signs, weights,
full retained-history byte hashes, response
and Nat response, all original checkpoint/input
SHA256 values, dtype and declared history cap.
It explicitly marks float-error intervals,
general hysteresis, fitted mechanism and
biological interpretation NOT certified.
The same original source prompt/data restrictions
and native checkpoint SHA gates apply.

**Formal cross-weld:** DASHI PR #1526 adds
`DASHI/Cognition/PNF/X369R4HistoryLoadNonFactorabilityExact.agda`,
reusing the PREEXISTING
`DASHI.Core.IntersectionalNonFactorability`
owner to show the finite source-shaped
same-load response `0≠9` cannot factor
through LOAD ALONE. The proposition is
about declared finite fixture outputs,
not a kernel-checked PyTorch theorem.
It is exposed in both TOE import graphs.

### Actual tool execution status

- Original source native CPU scripts:
  **committed, not yet run from the
  exact original GitHub source files**.
- Independent local Python reproduction of
  the same two-step synthetic source
  score equations:
  **4 exact fixture checks passed**,
  with second source logits and selected
  IDs as disclosed above.
- The underlying `torch.tensor(["1",...])`
  input-parser bug was reproduced with
  installed CPU PyTorch, then the original
  source probe was patched to convert
  finite numeric JSON strings explicitly.
  An independent local CPU PyTorch
  recreation of the new parser passed
  **7 checks** (valid vector/matrix,
  invalid NaN/Inf/bool/shape).
- No actual fitted checkpoint or
  trusted original floating-point
  rounding certificate was available.


## GPU-first X369 shared-head acceptance (2026-11-20)

Host CPU acceptance reported by the maintainer at 8af8d278: SymPy 25 exact residuals, rational enclosure reference controls, full synthetic consumer parity (18 cases), C2 quality maxcut and portable CPU contracts. These are reported host results, not independently replayed GPU evidence. C5 remains a finite recurrent witness, not general future-history sufficiency.

GPU-only producer: `python -m scripts.check_x369_shared_head_gpu --require-gfx803 --dtype float32 --selection topk --steps 20`.
The explicit sort diagnostic is not a native topk substitute. Independent activation/history/parameter gradients, source and fused Nat cells, intervention fallback, and post-update parameters are checked. G0-G4 remain OPEN until actual hardware receipt.

Performance producer: `python -m scripts.benchmark_x369_shared_head_gpu --require-gfx803 --batch 8 --width 384 --experts 4`. Separate selection and full-step inference measurements, warmups, alternating order, synchronization, median latency, peak incremental allocator bytes. No kernel-count, backward-performance, physical energy or general training claims. G6 remains OPEN until hardware measurements.

G5 native arithmetic enclosure and B7 recurrent composition remain OPEN. Ideal-real rational intervals are not native HIP error budgets. G7 fitted assets and Agda kernel acceptance remain OPEN. Original fingerprint owners and F5 exclusion unchanged. No CI requirement added.

### Physical host receipt (maintainer reported, tested 8cafe0aa)
RX 580 gfx803 Polaris 10 8GB; Schaka 10.1 HIP 7.16.26385; Torch 2.14.0+gitd70fd61. G0-G3 synthetic discrete fixture and recurrence accepted; G4 FP64 20/20 at 1e-10, 3.855 s combined. FP32 strict original harness failed; differential report: residual head gradient max difference 1.86e-8, parameter drift step 1 3.26e-5 and step 20 1.58e-4. Other listed gradients had zero measured discrepancy in the component experiment. Reduction-order explanation is plausible, not independently isolated. Do not promote FP32 matched training to PASS.

| Experts | scope | original peak bytes | fused peak bytes | original ms | fused ms |
|---|---|---:|---:|---:|---:|
|4|selection|442368|197120|0.930|1.008|
|4|step|442368|203776|4.007|4.672|
|8|selection|884736|393728|0.629|0.730|
|8|step|884736|393728|5.350|6.752|
|16|selection|1769472|786944|0.747|0.809|
|16|step|1769472|786944|10.688|11.169|

G6a/b measured; allocation reduction ~54-55.5%, no latency speedup observed. G5, G6c/d, G7 and B7 remain open. Next: FP32 optimizer sensitivity isolation, native error producer, dispatch reduction and separate backward profiling. Do not repeat baseline GPU acceptance without a regression reason. Raw host logs not attached to this receipt.

## Original-source two-channel codec learner cut
Formal owners: DASHI #1526 at 274faf58, X369DirectedRouterConditionalCodecExact and X369CenteredDirectedConsumerMDLExact. Two-channel score = (ws-wr)s+(wf-wr)f+wr D, D=DC(x)DC(h). Aggregate weighted DC equality suffices for relative ideal scores, not necessarily training gradients. Local SymPy 1.14.0: 33 residuals zero and aggregate cancellation witness. Independent CPU reference of the same component formula: FP32/FP64 forward and x/h/w gradient numerical comparisons pass (8 checks); this is not full native GPU acceptance. Pointwise common DC gradient counterexample: D_i=(1,4), D_j=(4,2), wr=(2,3), both 14, gradient difference (-3,2). Do not drop DC in unrestricted training.

GPU command: `python -m scripts.check_x369_two_channel_gpu --require-gfx803 --dtype float64 --steps 20`; repeat with float32 for measured drift, not certified parity. A original/B shared-head/C explicit DC evolve independently; D exact-dyadic omission shadow only, returns original route on unequal DC or observed native weights/IDs mismatch. It computes original reference and host audit, hence is NOT a GPU production codec or performance claim. Full original history retained; no future-state sufficiency promotion. Original fingerprint owners untouched; no F5, no backend work. GPU execution, fitted DC distribution, native arithmetic certificate, production omission and measured cost remain OPEN.

### Physical two-channel learner receipt (maintainer reported, tested 0f3c694)
SymPy 33/33; gfx803 FP64 original/shared-head/two-with-DC matched AdamW 20/20, max gradient error 4.77e-16 and parameter error 1.86e-11. FP32 had matching IDs/signs for 20 updates, parameter drift approximately 1.16e-3, not certified training parity. Aggregate DC shadow admitted FP64 and rejected FP32 native rounding discrepancy despite exact dyadic equality; unequal DC rejected. Host C2 quality and portable CPU contracts, native kink and shared-head consumer, projected score weld (6/6) reported PASS. Raw logs not independently retrieved. Next learner target: retain differential DC backward information, quantized consumer margins, and future-history sufficiency; no backend investigation. Pointwise common DC does not imply gradient equality. For shift-invariant consumer upstream lambda with sum lambda=0, DC weight derivative sum_e lambda_e D_e = sum_e lambda_e (D_e-D_ref). This is ideal-real algebra, not native float or unconditional training admission.

### L1 explicit differential backward (first-order)
New models/x369_differential_backward.py: shared circulant forward, aggregated-history backward, reverse head derivative reconstructed from DC-self-forward. Absolute mode admits arbitrary upstream gradients; relative mode uses the exact pullback (-sum of nonreference upstream, remaining upstream), not an assumed floating softmax zero sum. Full-score adapter keeps DC and original interventions; no pointwise omission in training. Local independent reference: 14 SymPy zero residuals, 16/16 forward/VJP comparisons at width 384, four experts, FP32/FP64 and absolute/relative modes. CPU-only local checks; physical GPU and full learner acceptance OPEN; first-order only, no higher-order or production omission claim. Run python -m scripts.check_x369_differential_backward --gpu. No original fingerprint owner changed. L3-L7 remain open.

### L2-L6 consolidated learner experiment cut
- da5f8b91: differential VJP is fourth independent GPU learner arm alongside original, shared and explicit two+DC. FP64 strict numerical acceptance; FP32 measured trajectories, not certified parity. Existing three-arm physical host evidence retained.
- bc3e33ef: GPU training profiler, routing/full-step, forward/backward latency, saved tensor calls/logical bytes/unique referenced storage and incremental allocator peak. Gradient admission before timing, no AdamW timing. Referenced storage is not incremental allocated memory.
- d20eea56: exact rational nearest-grid/ties-even two-channel and DC quantizer reference; joint pair hull for original and candidate, existing strict order/softmax/response/Nat contracts. Caller must supply actual source/candidate rounding and common consumer errors; no native HIP certification or physical codec cost claim. Local independent SymPy signed residual identity passed (1.14.0); committed Python reference controls not yet executed locally.
- b0cceae2: actual finite modular prediction objective, reuses TASK_PAIRS/TASK_LABELS/TRAIN_INDICES and typed prediction audit. Two recurrent token steps, all E histories retained, four matched computational arms, fixed horizon/seeds, no heldout training loss, no checkpoint selection. Synthetic modular learner, not original fitted LILA or semantic action evidence. Audit is scoped to output logits; learning producer separately supplies training evidence. No F5.

Prepared commands (physical GPU execution still OPEN):
```bash
python -m scripts.check_x369_differential_backward --gpu
python -m scripts.x369_quantized_consumer
python -m scripts.check_x369_two_channel_gpu --require-gfx803 --dtype float64 --steps 20
python -m scripts.check_x369_two_channel_gpu --require-gfx803 --dtype float32 --steps 20
python -m scripts.benchmark_x369_training_gpu --require-gfx803 --experts 4
python -m scripts.train_x369_mod3_codec_gpu --require-gfx803 --steps 2000 --seeds 41 42 43
```
L2c training omission, native rounding producers, future sufficiency, fitted model evidence and Agda kernel gates remain OPEN. No architecture/backend duplication. No new physical baseline rerun requested in this implementation tranche.

### L3 and L5 native observation / non-descent packet (23 Nov)
- `scripts/observe_x369_quantized_gpu.py`: original scoped TopK logit observer plus synthetic GPU source state, quantized two-directed-channel scores and exact dyadic residuals, each tied to original activation/history byte SHA256. Only retrospective comparisons with stored floating operands; this is **not** an upstream native-rounding theorem, a physical storage measurement, a fitted checkpoint receipt, or a Nat certificate. The reference `scripts/x369_quantized_consumer.py` passed the maintainer's local self-test, and `scripts/check_x369_differential_backward.py` passed 14/14 symbolic plus CPU forward/VJP parity. No repeat of architecture conformance.
- `scripts/check_x369_future_reactivation_gpu.py`: original-source, synthetic two-prefix witness. Present x=0 selects [0,1] with identical selected-only packet and output even when inactive expert-2 history differs. After actual original history transition, the same subsequent x=2 gives different selected IDs [0,2] versus [0,3]. Thus the selected-only present packet is not automatically sufficient for future original routing. Existing consumer fibre-stability/commuting-transition conditions from DASHI #1526 remain necessary. This is not fitted future behaviour or general hysteresis.
- These scripts are GPU-only and fail closed without GPU; physical execution remains OPEN. Original fingerprint owners, attribution and F5 firewall preserved. Rounding certification, quantizer-to-Nat budgets, trained history, real task semantics, and Agda kernel acceptance remain OPEN.

Execution after implementation readiness:
```bash
python -m scripts.observe_x369_quantized_gpu --require-gfx803 --dtype float64 --quantum 1/256
python -m scripts.observe_x369_quantized_gpu --require-gfx803 --dtype float32 --quantum 1/256
python -m scripts.check_x369_future_reactivation_gpu --require-gfx803 --dtype float64
```

### 23 Nov physical source observations and retrospective source-bound margins
Maintainer physical gfx803 HIP host receipts at 8f51c302: original two-history selected-only packet matches at x=0, first experts [0,1], inactive e2 state differs (9 vs 0), next same x=2 selects [0,2] and [0,3], in both FP32 and FP64; confirms a **finite native non-descent witness**, NOT universal hysteresis. Original quantization observer at quantum 1/256 produced matching native and candidate ordered topk in 3 FP64 and 3 FP32 samples, with reported candidate membership margins ~0.0198/0.0550/0.172 (FP64) and ~0.133/0.0741/0.109 (FP32). These candidate-only margins DO NOT constitute a joint source/candidate certificate or a prospective native error budget. Raw receipts not imported.

The observer now records an exact-dyadic retrospective pairwise certificate: observed source ordered gaps, quantized candidate ordered gaps, worst observed pair perturbation, strict-positive membership AND internal selected-slot tests, with explicit tie rejection and no prospective native-rounding claim. It includes a hash of all observed state_dict tensor names/shapes/dtypes/bytes, in addition to original x and all-E history hashes. This binds the *synthetic observed state*, not an original fitted checkpoint, model source SHA or a transitive model provenance chain. Regression: `python -m scripts.check_x369_retrospective_pairs` (positive, tie, wrong order, invalid IDs, exact rational controls). Local run pending. Then physical observation with current head: `python -m scripts.observe_x369_quantized_gpu --require-gfx803 --dtype float64 --quantum 1/256` and float32. Earlier physical logs do not attest these NEW fields. All L5 finite non-descent result accepted; future-positive sufficiency still OPEN. L3 prospective error and L4 GPU learning backward economics remain OPEN. Original fingerprint owners unchanged; backend investigation delegated to gfx803 worker.

### 23 Nov quantized source-observation replay hardening
After six previously reported gfx803 matches, observer now records original-vs-quantized exact-dyadic *ordered* pair margins, actual pairwise perturbations, model state_dict SHA256, nonpersistent history_cap and routing configuration. Write-once output supported (`--output` fails on existing file). Twelve standalone pure rational pairwise controls executed locally PASS; the newly committed repository wrapper tests the same cases but has not separately been run from a full checkout. Offline `scripts/verify_x369_quantized_receipt.py` reconstructs baseline, candidate, exact directed score identity, selected order and all retrospective pair-certificate fields, rejects inconsistencies. It verifies internal arithmetic, not source authenticity or HIP error bounds. The new receipt fields and offline verifier have not yet been tested on the physical GPU. Commands:
```bash
python -m scripts.check_x369_retrospective_pairs
python -m scripts.observe_x369_quantized_gpu --require-gfx803 --dtype float64 --quantum 1/256 --output .leech-r4-kink/quantized_fp64_20261123.json
python -m scripts.verify_x369_quantized_receipt .leech-r4-kink/quantized_fp64_20261123.json
```
Do not overwrite earlier source receipts; preserve original model fingerprint and F5 firewall. L3 prospective quantization arithmetic/error producer and future-positive sufficiency still OPEN; no backend work transferred from gfx803 worker.

## L2/L4/L6 consolidated physical acceptance and large-bite learnability plan (23 Nov 2026)
Maintainer physical RX 580 gfx803 FP64 four-arm matched learner at 1563930b: 20/20 PASS with exact routing/signs/gradient presence, max gradient error 4.68e-16 and differential-VJP parameter delta 2.07e-11 (not bitwise floating agreement). L4 profiler on 4 experts, width 384: original routing saved calls 34, logical bytes 1454464, referenced storage 864512, incremental allocation 1161728, backward 4.90ms; differential VJP 19, 618880, 594176, 924160, 4.14ms respectively. This is 57.5% lower logical saved tensor bytes, 31.3% lower referenced storage, 20.5% lower routing allocator peak, 15.5% lower routing backward latency. Full-step backward: 13.38ms to 12.70ms (~5.1%). Inference and full-step total timings are distinct and not inferred.
L6 previous one-seed 200-step actual typed mod3 prediction: all four computation arms agreed but training accuracy remained 0.375 from step 1 through step 200, training loss 1.3008 -> 0.9973, heldout NLL 1.8784 -> 8.7853. This is NOT a training-fit phase, NOT memorization, and NOT grokking evidence. Do not say pre-grokking training fit.

New large-bite experiment: `scripts/run_x369_mod3_maxcut_gpu.py`. Original eight training pairs and heldout (2,2), independently sourced from typed consumer owner. Six compared trainable arms: original/shared-head/two+DC/differential VJP, an additive linear control (tests absent explicit pair interaction) and a learnable pair MLP control (tests task learnability under a more flexible interaction). Seventh, a **nonlearned analytic** C3 Fourier modular law control establishes task/label consistency; it does not count as a learned generalization result. Four learner arms start each seed from identical state_dict hashes; controls use independent, named initializations; fixed steps and fixed report cadence, no heldout training or early-stop. Classifier marks NO_TRAIN_FIT whenever no logged checkpoint has 8/8 training correctness and explicitly withholds grokking claims even if delayed heldout transition is observed. It does not identify semantic/Leech equivariance. CPU smoke is explicitly nonacceptance; GPU is required normally.

Consolidated execution at current branch (user owns physical gfx803, no baseline rerun necessary):
```bash
python -m scripts.run_x369_mod3_maxcut_gpu --require-gfx803 --dtype float32 --steps 2000 --report-every 100 --seeds 41 42 43 --output x369_mod3_maxcut_3seed.jsonl
```
Use `--cpu-smoke --width 24 --steps 2 --report-every 1 --seeds 41` only for smoke if debugging, never physical acceptance. No claim that 2000 steps is enough; no adaptive continuation based on heldout. Source model fingerprint and F5 preserved. New experiment remains physically UNRUN. Separate remaining obligations: prospective numerical certificate, future-positive recurrent-state sufficiency, original fitted R4 source-bound learning, and Agda kernel acceptance. No gfx803 backend work in this lane.

## 23 November — Formal-first roadmap priority (supersedes previous GPU-run-first recommendation)
Original P0 mathematical owner: DASHI #1526 `Docs/roadmaps/X369ConsumerSufficientCompressionProofMaxCut20261117.md` (latest proof-first revision at `d9f95478`), new `X369InputIndexedFutureConsumerExact.agda` (source-written, kernel OPEN). Reuse established original observability rank `det(M_w)=d q`, singular squares `d²,q,q`; do NOT confuse the 3→2 *directed feature score with known DC* and rank-two *full original recurrent history*.

Top-priority proof packet, in order: (F1) kernel-check all-input finite trace descent / maximal behavioural fibre; (F2) instantiate observable/state/transition using original R4 topk, C2 abs history, signed GELU/FFN and retained all-E histories, then prove the positive stable quotient or exhibit its exact source counterexample; (F3) connect pairwise relative error/oscillation softmax, chart parity and transition error with precise Nat-cell admission; (F4) distinguish full-channel gauge from genuine C3-constrained learning law and type independently labelled task actions. Original fitted R4 source authentication and Agda TOE checks remain separate. Physical GPU L2 20/20 and L4 savings are already accepted scoped engineering results and should not be re-run absent regression. L6 at 200 steps remained 3/8 training-fit, NOT memorisation or grokking. The unexecuted six-arm mod3 runner remains an optional downstream diagnostic, NOT the P0 formal blocker. Attribution: original Leech-LILA by Anatolii Kornienko; original fingerprint sources, fitted-delta layout, F5 exclusion intact.


## 28 November — explicit native gfx803 fitted language execution

The existing C2+fitted R4 restoration / full native SLR development capture / four-load / same-clock observation pipeline now supports `--device gfx803` on its original one-command entry point. The default `--device cpu` is preserved only for historical portable fixtures. Do not equate CPU probes with GPU execution. Actual original source C2+delta is restored and checked on CPU, then one full fitted arm is moved to the verified gfx803 HIP device. Full LeechGPT development trace and independent restored HistoryOrbitMoE layer replay execute on that same device, with true raw-byte equality (including signed zeros). Both probes require matching original capture/export runtime and exact source activation/history bytes; no silent CPU fallback. Host transfer is only for artifact serialization and hashing, not inference.

The command in the original 17 November runbook remains the real fitted-language gate: add `--device gfx803` and run with the existing native gfx803 ROCm Python environment. Supply ACTUAL repaired C2 checkpoint/receipt, native development SLR JSONL with R1 summary, fitted R4 delta, source config, tokenizer, development doc and token position; caller expected hashes remain mandatory. Do not substitute fa473dca translated ordered-source candidate or synthetic histories. No F5/reserved data, training, release or heldout quality promotion. New device tests test_x369_r4_execution_device and old source-runner regression must pass first. Native fitted assets unavailable to assistant; new code not executed on real model.
