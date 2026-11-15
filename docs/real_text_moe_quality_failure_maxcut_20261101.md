# Real-text JQ1 result, quality-regression diagnosis and next exact max-cut — 2026-11-01

## User-executed evidence (64 train / 32 held-out TinyStories documents, 1 epoch)

The **real text** preparation succeeded. A local runner dataset subprocess hit an
ITIR-venc Python-finalization crash AFTER the datasets were written; remaining
stages were run manually on the saved verified splits. This is not evidence
that the dataset samples are synthetic, but the old all-in-one runner is not
certified reliable under that local Python environment.

The frozen dense `best_model.pt` teacher scored held-out next-token CE=2.58.
Two independently trained 12-layer parity-MoE candidates:
- C2-invariant routing CE=8.63 (delta +6.05).
- punctured-nine wave routing CE=17.30 (delta +14.72).

Uniform 2048-class CE is log(2048)=7.6246 nats, so both candidates are
worse than this calibration baseline. This is a major functional failure
of **the one-epoch full-FFN replacement procedure**, not a proof that
DASHI's punctured-nine prior cannot help a properly trained learner.

The two actual JQ1 captures contained 2,048 frames EACH (4,096 total).
Every frame reopened exactly. Every selected codec was plain **joint zlib**;
no exact learned sign, site-action, previous-expert or history-baseline orbit
candidate won. The observed/request wire data were approximately
9.25/9.26 MB versus 10.07 MB raw framed (~8.1% saving). At modelled 1Gb/s
including the user's measured local codec overhead, JQ1 ~11.42 ms versus
raw ~5.16 ms. These are measured codec costs plus ideal-link/RTT
**model assumptions**, not a real WAN benchmark. No quality or latency
improvement is established.

## Precise explanatory gap that must not be skipped

The current signed expert module enforces a *counterfactual* C2 law,
`F_e(-x,-h)=-F_e(x,h)`, for one expert's signed chart and compatible
history action. **It does NOT imply** `H_a(x)=±H_b(x)` for two
independently trained expert slots at the same observation/state. The
current `HistoryOrbitMoE` has separate expert FFNs and history
transitions, and the source/receiver histories are independent by default.
Thus exact within-capture cross-expert orbit compression cannot be
assumed from a per-expert equivariance theorem. To obtain exact reuse
without paying arbitrary XOR residuals, one must either:
(a) **tie** canonical histories/expert operators by an explicit learned
group orbit, coordinate action, and update-confluence rule, including
when multiple slots from the same orbit route in one token; or
(b) learn a predictive residual with measured bound/side information,
or (c) certify a selected consumer-sufficient quotient that does not
need the missing bytes. An 8-point punctured-sheet count or sign chart
alone is insufficient to establish cross-expert redundancy.

## Immediate P0: isolate the quality regression, NOT add epochs blindly

Inspect actual source `scripts/train_signed_wave_experts.py`:
it distils each FFN separately against normalized activations captured
from the frozen DENSE teacher, with a local MSE target, then
`export_signed_wave_model.py` replaces **all 12** FFN blocks at once.
It does not train the joint 12-layer output/logits or directly minimize
sequence-level CE/KL. Its history-aware recurrent student is asked to
imitate a history-free teacher target. It also has a narrow even channel
versus the original 4d FFN, and a rank-24 history state bottleneck;
those are hypotheses to test, not exact capacity equivalences.

The `punctured9_wave` route feeds quadratic energy/dot/wedge features
to a linear router; feature magnitudes and recurrent history norms are
not bounded by that group-invariance construction. In the reported
one-epoch study, the larger regression of the punctured-nine route
could stem from input scales, history amplification, route collapse,
insufficient training or error propagation; **no root cause is proven yet**.

New `scripts/diagnose_moe_quality_collapse.py` loads the already-saved
teacher and candidate, keeps shared teacher weights fixed and runs:
- dense CE and full student CE on the same verified heldout tokens;
- **single layer at a time** substituting its trained MoE+even pathway
  into an otherwise dense teacher, recording CE delta;
- prefixes of 1..12 substituted layers, measuring how failure compounds;
- each trained student FFN on the exact SAME teacher normalized inputs
  as the trained FFN: RMSE, relative MSE and output scales;
- every layer's expert route counts/entropy, C2 sign balance,
  maximum recurrent history norm and per-step weight maxima.

This is a DIAGNOSTIC, not new teacher distillation, so it isolates
which failure mechanism to attack with minimal extra training.
`tests/test_moe_quality_collapse_diagnostics.py` is source written.
No checkout execution of these new files has occurred.

## Next Pareto cut based on diagnosis

1. Verify whether one or a few replacement layers cause the collapse.
   If yes, keep most dense layers, distil/train only the highest-gain
   layer or use progressive replacement, and require a tolerance of
   heldout CE before exporting/deploying each addition.
2. If errors compound without a dominant layer, train end-to-end
   with an explicit dense logit KL + next-token CE objective on a
   quality-preserving residual/gated continuation (initial gate=0,
   source teacher retained until candidate passes).
3. Investigate feature-scale normalization/clip and history stability
   for punctured9_wave, while retaining exact C2/C3 feature invariance.
   Compare `c2_invariant`, `punctured9_wave`, normalized phase/wedge
   features, and dense-only control with matched data, seeds and updates.
4. Train on meaningful real-text data; 64 sequences and one epoch are
   integration-level evidence, not convergence. Keep provenance and
   held-out records unchanged.
5. Once language quality approaches dense teacher, measure whether
   the trained expert histories really occupy shared C2/C3/site-action
   orbits, including exact/near residual distributions; do not infer
   correlation from equivariant routing alone.
6. Only after quality and orbit reuse demonstrate an advantage:
   optimize codec dispatch, early-out, packed residuals and return
   history ownership; report full bidirectional latency at 1/10Gb/s.

Source attribution: original Leech-LILA trained architecture is
Anatolii Kornienko; DASHI 369/wave/SSP/consumer-fibre owners retain
their original attribution. No analytic j, Monster/Leech
same-object proof or semantic-opposition mapping is inferred by these
test results.

### Execute the diagnostic on saved artifacts

```bash
git fetch origin experiment/consumer-preserving-multiscale-codec-20261031
git switch experiment/consumer-preserving-multiscale-codec-20261031
python -m unittest tests.test_moe_quality_collapse_diagnostics -v

python -m scripts.diagnose_moe_quality_collapse \
  --teacher data/best_model.pt \
  --student /tmp/real_text_moe_jq1_smoke/punctured9_wave/student_model.pt \
  --config /tmp/leech_dense_config.json \
  --heldout /tmp/real_text_moe_jq1_smoke/dataset/heldout_ids.npy \
  --device cpu --mode both \
  --output /tmp/punctured9_quality_diagnosis.json
```

Adjust artifact filename layouts to the actual local receipt paths;
only the user workstation can verify these `/tmp` locations.
Run the same command for `c2_invariant`, replacing `--student` and
`--output` accordingly. The diagnostic may take nontrivial CPU time
because it evaluates 12 isolated plus 12 prefix hybrid passes;
`--mode isolated` or `--mode prefix` provides narrower subsets.
