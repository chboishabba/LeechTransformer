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

    L(t)=(8,3+4t,-11+9t,-13+10t)

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
  `(8,3,7,7)`;
- prefix3: updated e3/e2 zero, second logits
  `(8,11,-11,-13)`.

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
