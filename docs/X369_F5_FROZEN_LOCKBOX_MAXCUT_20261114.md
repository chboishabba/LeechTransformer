# X369-F5 frozen matched C2 / C2xC3 lockbox (Leech PR #4)

> **X369 14 November latest A/B/C update:** The authoritative 53-obligation C2/C3/C3² roadmap, code owners, exact remaining formal lemmas, F1 full covariance, F2 task-orbit supervised hidden readout, F3 route-input causal controls, F4/F5 independent two-arm protocol, F6 separate three-arm and independent frozen final protocol are in [X369_C2_C3_C3SQUARED_FULL_MAXCUT_20261114.md](X369_C2_C3_C3SQUARED_FULL_MAXCUT_20261114.md), latest Leech PR #4 head `1420b940`. This older section remains historical/source-specific. **New full-repo host tests, original RX580 empirical CE, Agda kernel and independent learned C3 mechanism are still OPEN**; do not promote script existence or earlier 55/55 host results into acceptance on the latest head.


## Decision boundary

F0/F1 are source residual statistical audits, not learned symmetry claims.
F2 requires a typed **task** input action and matching output/hidden action
on unseen language/modular examples; rotating raw hidden axes is not a
semantic C3 action. F3 requires a **learned** phase law plus causal
interventions, not the already-exact hand-coded Fourier identity.

F4 (`scripts/screen_bounded_c2_vs_c2c3_router_fresh.py`) trains only
matched router parameters on the original bounded R64 source with the same
fit records, matched seeds and parameter count. Its fresh validation CE
**selects the best step** of each arm. It is a development screen, not an
independent final test; reported improvements are selection-exposed.

F5 (`scripts/score_frozen_c2c3_lockbox.py`) is separate and fail-closed:
- Require the exact F4 screening receipt and saved frozen best-step
  checkpoints for both geometries at every seed.
- Select the winning geometry from F4 **development mean CE**, with C2
  winning ties; lock this decision before scoring any new test data.
- Hash-bind original fit, previously viewed heldout, F4 development set,
  original repaired source checkpoint, and each saved arm checkpoint.
- Reject width mismatches, duplicate lockbox rows and any overlapping
  contiguous token windows (original fit, prior heldout, development set).
- Score both frozen geometries, paired by seed, plus the original repaired
  C2 checkpoint on the **same new lockbox**. Report CE differences; never
  compare a new-test CE against 4.5966 from a different prior heldout.
- Never train, choose steps, or change geometry using lockbox values.
  Output is non-overwriting and marks `release_admitted=false`.

Example (supply actual independent corpus token file and frozen screen receipt):

```bash
ROOT=/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.score_frozen_c2c3_lockbox -- \
  --screen-receipt "$ROOT/x369_f_c2_vs_c2c3_fresh.json" \
  --original-train-ids "$TRAIN_IDS" \
  --original-heldout-ids "$HELDOUT_IDS" \
  --selection-validation-ids "$FRESH_VALIDATION_IDS" \
  --locked-test-ids "$LOCKED_TEST_IDS" \
  --source-candidate "$ROOT/c2_bounded_repaired.pt" \
  --teacher "$TEACHER" --config "$CONFIG" \
  --output "$ROOT/x369_f5_frozen_lockbox.json"
```

**Source-provenance firewall:** byte- and token-span separation is *not*
document-level disjointness. An external original-document identity/record
manifest, corpus revision and tokenizer digest are required for final
admission. The code deliberately reports this as unverified. The test must
be selected *once*, then remain unconsulted for architecture tuning;
non-overwrite cannot prevent a user from starting a new run with a
different output path. Preserve one official lockbox receipt.

**No premature F5 closure:** the 32+ token original-source state bound,
pure model-to-formal BIDI, measured resource accounting and task-specific
causal phase probes remain distinct open obligations. A negative paired
lockbox delta would support useful routing features under the specified
task and budget, not prove learned compositional C3 or semantic triality.

## Required F2/F3 activation follow-through

1. Specify a typed task-level C3 action `R` on actual input examples,
   its label/output action `rho(R)`, and an independently learned
   representation readout `u`. For ordinary arbitrary text, such an
   action cannot be inferred from coordinate triads.
2. Evaluate normalized `h(Rx)-rho(R)h(x)` on unseen inputs and evaluate
   `u(a)u(b)-u(a+b mod 3)` on learned phases, with a memoriser and a
   hand-coded Fourier reference as **different controls**.
3. At the actual router-only site, inject norm-preserving phase/channel
   perturbations against independent equal-norm random controls. Compare
   task CE and off-target effects on disjoint rows; report paired seed
   intervals. Source-level `rotate3` invariance of engineered features
   by itself is not evidence of trained task equivariance.

## Attribution

Original Leech-LILA neural model and claims: Anatolii Kornienko
(DOI: 10.5281/zenodo.18784424). DASHI Agda mathematical C3/C3² and
balanced ternary owners, DASHIg/dashifine modular-grokking controls and
Leech PR #4 empirical work remain separate. This script does not
attribute original LILA architecture or a trained modular circuit to DASHI.


## Canonical complete X369 gate ledger

The full programme, proposed exact lemma signatures, parallel work packages, F6 genuine C3² architecture split, and acceptance/falsification criteria are tracked in [X369 full max-cut](X369_C2_C3_C3SQUARED_FULL_MAXCUT_20261114.md). This narrower F5 protocol remains the execution-specific owner.
