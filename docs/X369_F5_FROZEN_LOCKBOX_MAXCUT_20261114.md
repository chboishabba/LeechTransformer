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
  --screen-receipt "$ROOT/x369_f4_c2_vs_c2c3_fresh.json" \
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

## 15 November 2026 actual F4/F3 receipts — keep F5 LOCKED

The physical original RX 580 F4 screen and the development-only fairness adjudicator have now completed on fresh 32 × 64 TinyStories development rows, original 55 fit rows, two seeds (41,42), 24 optimizer steps. The reported mean best C3-minus-C2 CE is **+1.7829 nats**. C2 initial/best 4.9903 (step0), C3 initial 6.8026; C3 seed42 improves by only 0.0588 to 6.7438, seed41 best remains step0. Thus **C2 is locked as the F4 development winner** under this protocol; the two seeds share one identical original checkpoint and initial function, so no claim of two independent initial-effect replications or convergence.

The F3 probe was done on the **unselected, best-step-zero** C3 seed41 comparator, never on the winning C2 model. It reports baseline 6.7296, relative-C3 CE change −0.0644 with interval crossing zero, and activation-displacement-matched random change −0.1823. Relative-minus-random paired CE is +0.1179, **unadjusted 95%** interval [+0.0504,+0.1898]. This does not demonstrate learned C3 causal specificity. Moreover, because the router consumes nonlinear C3 invariants and directed activation/history products, matching activation norm/displacement does **not** match actual router-feature displacement. The separate experimental cut now measures that in the already-viewed F3 probe with `scripts/audit_x369_f3_live_router_feature_metric.py`. Never infer a 99% familywise significance or exact p-value from the unadjusted 95% interval.

The reported reserved F5 source is `x369_fresh_splits/reserved_ids.npy`, 32 × 64, SHA256 prefix `7d13b31c73a7bc9d`; it remains **unopened and unscored** in the provided host receipts. Original historical training-document identifiers are not independently verified; only exact 16-token window nonoverlap against supplied original source arrays has been established. This is a **hard release block**. Do not run `score_frozen_c2c3_lockbox` merely because C2 wins development; keep the F5 set sealed until historical document lineage, the frozen model/seed/metric decision, and one-time final scoring protocol have been accepted. Reusing F5 to tune a new feature normalization or router geometry would invalidate its role.

No change to the prior lockbox mechanism is needed. This update corrects the old example filename typo to `x369_f4_c2_vs_c2c3_fresh.json` and records explicit user-provided F4/F3 evidence. Current PR source-head native tests added after the host-reported 118/118 must still replay. A separate new geometry-calibration experiment needs NEW development documents and a fresh, preregistered validation/final split, not this F5 reserved sample.

## Late 15 Nov — F3 raw-F5 read correction and exact-feature null firewall

The previous F3 *diagnostic code*, `audit_c3_causal_router_ce.py`, accepted a required `--reserved-lockbox-ids` and loaded that `.npy` via `rows_from_file` to exclude overlapping 16-token windows from the F3 exploratory set. Therefore the historical phrase **"F5 never opened or loaded" is factually overbroad** for the original F3 data-preparation/guard path. The available report supports **"reserved F5 rows never scored for CE, fitted against, or used for geometry selection"**, not raw-byte unreadness. The F5 trial remains **unscored**; preserve its sole official CE evaluation for a separately approved protocol. The F3 split producer also used prior token sources to check overlap; exclude these *preparation* accesses from claims of no prior inspection.

To stop future F3 scorers from opening the raw reserved array, source now rejects `--reserved-lockbox-ids` and requires `--reserved-lockbox-sha256` plus `--probe-provenance`. It verifies the *original F3 producer's* `prior_token_array_sha256` contains the exact F5 digest, `output_token_array_sha256.development` matches the F3 token file, and the same window width / overlap-exclusion flags are present. This is **source-bound reported overlap prevention**, not independent proof of all historical source documents. No F5 tokens are loaded by the updated F3 scorer.

The new original `c3_feature_exact_mixed_orientation_x` F3 negative control preserves per-triad `||x||`, `||R_+x-x||` and the **actual C3 route feature-map displacement norm for every expert** exactly, by applying random ±120° rotations *per original triad* instead of arbitrary random tangent vectors. This isolates global coherent phase direction from perturbation size **without** matching router-logit displacement. The old random null achieved lower CE (an improvement, not degradation) but changed the nonlinear feature norm 2.0–2.2× more in the reported live audit. New post-hoc F3 paired contrasts on already inspected 16 rows are descriptive/exploratory and may not determine release. They must be executed on **new output files**; never overwrite the original receipt or use F5 to select the null.

**Operational status:** F5 32×64 rows unscored in received host evidence; independent original historical document-ID crosswalk, one-time blind final protocol, and C2 winning-geometry decision still require release governance. Keep original F5 CE evaluation gated.
