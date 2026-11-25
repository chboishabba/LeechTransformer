# X369 explanatory formalism — source-coupled neural geometry, routing and steering

**16 November 2026.** Owner: LeechTransformer PR #5, cross-source donors
DASHI Agda and original Leech-LILA by Anatolii Kornienko
(DOI: 10.5281/zenodo.18784424). This document formalises the
interpretation of ORIGINAL observed negative and positive evidence.
It is not a new Agda kernel receipt, causal steering experiment,
or physical C3 release. Reuse existing donors; do not edit ESD.

## 0. Evidence status at host-accepted 7fee764 and later proposed R2 cut

- User-host R1/R4a/b/c: **11/11 tests PASS** at exact `7fee764`,
  portable source and C2 quality regression exit 0. Synthetic/contract
  acceptance, not trained-original R4 result.
- Historical original F4 24-step source screen: C2 best fresh
  development CE=4.9903, C3 seed42 best 6.7438, seed41 6.8026,
  so mean C3 disadvantage about +1.7829 nats. Different
  initial functions/feature systems: NOT a geometry-only causal
  treatment effect.
- Original full hidden covariance commutator D0=1.33227,
  D1=1.41233 vs declared tolerance .05; fixed-basis global C3
  invariance falsified on those sampled states, not an
  impossibility theorem for an independently learned chart.
- Original F3 exploratory frozen unselected C3 seed41 step0:
  relative-input shift delta CE about -0.0644, random
  displacement-matched delta -0.1823, relative-minus-random
  +0.1179. The old random comparator was NOT matched in the
  actual nonlinear router feature-space displacement; later
  +/-120 per-triad null fixes that metric, but does not match
  projection onto learned router weights, top-k margins or
  semantic label action. All original F3 reused probe data
  remain descriptive and selected/exploratory.
- F3 historical preparation accessed F5 reserved **raw bytes**
  for overlap exclusion, although available evidence does not
  establish official F5 CE scoring/training/model selection.
  Current corrected F3 code only receives its prior SHA, and
  one-time final R5 independence remains unverified.

## 1. State and interpretation types

For a frozen checkpoint revision `v`, token prefix `p_t`,
layer `l`, and expert set `E`, define an ORIGINAL state

`Z(v,l,t)=(x_{l,t},(h_{l,t,e})_{e in E}, q_{l,t},
            theta_v, num_v, dataSHA, tokenSHA)`.

Here `x` is the actual layer/site activation, `h_e` the
per-expert recurrent history, `q` the selected routes and
tie/overflow policy; `num` records precision/backend.
All interfaces carry site `(l, t, pre/post router,
before/after expert/FFN, residual)`. Site omissions are
type errors for causal interpretations.

The source C2 router score at pre-router site is

`s_e^0(x,h_e)=w_l · |[x;h_e]| + b_l + beta_{l,e}`.

The source dispatch is `D_l(Z)=TopK_k(s_e)`, with
source-specific ties, clipping, mixture normalization,
conditional expert and state transitions. The model output
`Y_v(p)=softmax(L_v(p))` is the FULL recurrent trajectory,
not just a local router-invariant score.

The C3 chart is the product of **fixed coordinate cyclic
permutations of real triads** `R^3=id`.
The analytic real Fourier chart `wave3` conjugates R to a
2D +/-120-degree rotation plus unchanged DC. This chart
does not assert semantic rotation of an original text input,
nor a learned latent C3 representation. Distinguish
`C2=-I`, `C3=<R>`, dihedral `S3=C3 semidirect C2`
(where the reflection inverts R), and commuting C2xC3.
A coordinate number 3,6,9 alone induces no cross-domain map.

## 2. Exact finite identities: what they really say

Let `T(x,h)=(<x,h>,<x,Rh>,<x,R^-1 h>)` for each original
triad and `q(a,b,c)=a²+b²+c²-ab-bc-ca`.

Then, in exact arithmetic, the directed features satisfy

`T(Rx,Rh)=T(x,h)` (diagonal action), and

`||T(R^{+/-1}x,h)-T(x,h)||² = 2 q(x) q(h)`.

Unary `(|DC|, Re²+Im², |abc|)` channels are R-invariant.
These are exact *feature-geometry* facts in the finite
carrier. They say nothing about the learned projection
`w·delta T`, router top-k crossing, expert history,
consumer logits, text-level label action or future
representational persistence.

Use the existing finite Agda donors:
`DASHI/Physics/Common/FiniteThreeCycleTorusExact.agda`;
`DASHI/Algebra/TriadicDepthOneCharacters.agda`;
`DASHI/Cognition/PNF/ModularFourierGrokkingC3Exact.agda`;
`DASHI/Cognition/PNF/GrokkingEquivarianceDefectGeometryExact.agda`;
`DASHI/ComputerScience/BalancedTernaryC2C3DihedralCodecBridgeExact.agda`.
Source-facing identity code lives in
`models/trialectic_wave_prior.py`.
Proving a theorem about this finite source chart is NOT
proof of a learned neural mechanism.

## 3. Explanation of the observed F1/F3/F4 outcomes

**Q/K gauge negative control.** If W is orthogonal and
the exact SAME W acts on Q,K, attention logits obey
`(QW)(KW)^T=QK^T`. In ordinary exact dot-product
attention, this basis change cannot by itself improve
attention scores. Nonzero regularisation during training,
non-gauge Q/K transforms, different seeds or resonator
dynamics need separate matched controls.

**F1 fixed-basis symmetry test.** For an original hidden
empirical covariance `Sigma_l`, define
`D_l = ||R Sigma_l R^T - Sigma_l||_F /
              max(||Sigma_l||_F,epsilon)`.
The measured original full-covariance defects are
large relative to the preregistered tolerance for
those observed states. A neural hidden representation
need not choose this predetermined chart; a learned
intertwiner `U_l(g)` remains an independent open
question. No general universal negative theorem follows.

**F3 relative-phase test.** The diagonal action
`(x,h)->(Rx,Rh)` preserves directed channels.
One-sided `(x,h)->(Rx,h)` can change them, so
a one-sided relative-phase probe is the relevant
intervention. Random matched `||delta x||` need not
match `||delta F||`; the analytic mixed +/-
rotation now matches the latter *per expert/triad*,
but not `w·delta F` or downstream logits.
F3 is on an unselected, frozen comparator and
does not establish a learned semantic action.

**F4 non-identification.** The two geometry arms
had different initial prediction functions:
`L_C2(theta0,p) != L_C3(theta0,p)`.
Consequently `CE_C3-CE_C2` combines prior
feature-map differences, initial score changes,
dispatch/top-k differences and training dynamics.
The negative F4 outcome is valid descriptive
evidence against that specific protocol, but
not an isolated estimate of C3 learning advantage.

**R4b fairness repair.** Each arm now begins from the SAME
original `s0` and adds only `theta^T phi_j(x,h)`,
`theta=0`. j is C2 extra capacity, coherent C3,
or mixed-orientation C3 null, all `2d*n_layers`
trainable coefficients. This gives identical
initial complete student logits/recurrent histories
in source tests, and keeps capacity equal.
It does NOT equalize feature rank/statistics,
gradient scale, effective parameterization,
router-logit perturbations, sample efficiency or
physical compute. Matched controls and source
instrumentation must measure these remaining
possible explanations.

The estimand is *predeclared source-disjoint heldout*
`Delta_j = E_doc[CE_j(doc)-CE_C2capacity(doc)]`
along with a coherent-minus-mixed contrast.
The R4b best-development CE is model-selection evidence
only; its R4c compact learned delta permits exact
same-device replay of the selected function, but is
not independent heldout or neuron-steering evidence.

## 4. What counts as an actual neuron-steering test?

Let `a_s(p)` be the ACTUAL frozen model activation at
named site `s=(layer,token,operator,coordinate-system)`.
Let `I_{s,d,alpha}` replace that site's activation by
`a_s(p)+alpha*d`, or ablate a single coordinate / a
source-validated sparse dictionary feature. Run the
full SAME original model, source tokens, precision and
state update, with the intervention propagated to its
downstream consumer `L_v^{I}(p)`.

A router-input-only probe
`F(Rx,h)` where the FFN receives unchanged `x`
is a *routing policy intervention*, not an identified
single-neuron or SAE-feature steering result.
Input-cube 2x2x2 prompt interventions quantify third-order
output interaction, not internal neuron steering.
A hand-coded C3 phase rotation is not evidence of a
learned causal representation.

Required independent behavioral labels `y(p)` and
targeted counterfactual pairs; two outcomes:
`TargetEffect(I)=E[m_target(L^I,y)-m_target(L,y)]`
and `OffTarget(I)=E[D_KL(Y||Y^I)]` plus independent
non-target task error. Pre-registered dose/coordinate
selection on discovery only, final on disjoint assessment.
Match controls on activation norm, displacement,
site, target token, search budget and randomization.
Evaluate **necessity** (ablation), **sufficiency**
(inject/patch), and reverse intervention, with
recurrent history and router selection recorded.

Neuron steering acceptance cannot be inferred from
a nonzero logit change alone, one LLM judge's opinion,
or a first-order gradient. Evidence must bind
`checkpoint SHA; tokenizer SHA; source doc SHA;
site/layer/time; intervention direction SHA; alpha;
labels; seeds; target+off-target metrics; router
selections; same-device replay`.
For the original LILA dense checkpoint first compare
site-specific single-coordinate/gain/ablation and
contrastive activation vectors against matched random.
Then test repaired C2/C3 source with history-aware
intervention and trained R4c delta.

The existing Leech repository has **a steering roadmap,
not a completed source-bound neuron-level steering
effect/release**. Existing
`scripts/audit_c3_causal_router_ce.py` and
`models/history_orbit_moe.py` implement router
probe actions; `scripts/eval_trialectic_intervention_cube.py`
tests independently labelled input-cube logit
interaction; `models/paired_intervention_geometry_loss.py`
implements an optional paired action regularizer,
but none alone is a neuron-site behavioral steering
result. The document
`docs/LILA_ORIGINAL_CLAIM_NEURON_STEERING_E8_369_PANTS_ROADMAP_20261111.md`
already owns N1–N5 and should be advanced, not duplicated.

## 5. Typed same-object theory to prove / test

Declare distinct carrier types:
`Input, Label, Activation(site), JointHistory,
 RoutingTrace, ObservedOutput`. For selected
`g in C3`, require independent `R_g: Input->Input`,
`rho_g: Label->Label` and a (possibly learned)
`U_{g,s}: Activation(site)->Activation(site)`.

The desired *observational* representation square is
`a_s(R_g p) approx U_{g,s}(a_s(p))`;
the desired *output* square is
`L_v(R_g p) approx rho_g^out(L_v(p))`,
with an explicit label-logit permutation/decoder.
Both defects are measured on heldout source-labelled
pairs and can be nonzero even when the finite donor
group law is exact.

For a phase readout `u`, require
`u(a+b) approx u(a)u(b)` over independent
modular task examples **and** a nonconstant
identifiable readout. A fixed Fourier character
does not count as a learned `u`.

For causal steering, specify intervention transport
`I_{s,g,alpha}` and downstream consumer `C_s`.
The *candidate steering commutator* is

`K_s(g,I,p)=
  d(C_s(I(U_g(a_s(p)))), rho_g^out(C_s(I(a_s(p)))))`.

This formula is NOT a universal theorem:
pre-route interventions can cross a top-k threshold,
post-route edits cannot retroactively change chosen
experts; any claimed commutation must include routing
margin, tie policy, histories and site conditions.
Measure or construct explicit counterexamples.
The simpler valid gauge-covariant statement uses
`U_g I = I^g U_g` with the transported intervention
`I^g=U_g I U_g^{-1}`, **conditional** on the
source consumer actually being an intertwiner.
Those extra hypotheses must be recorded, not guessed.

**Historical exact Agda donor and non-descent:**
`DASHI/Reasoning/LeechJointStateConsumerQuotientExact.agda`
proves preservation of a SELECTED consumer given
selected activation/history equality and its explicit
fibre-constancy law. A full-state consumer additionally
requires inactive-history equality.
`DASHI/Reasoning/TrialecticBoundaryFaceNonfactorabilityExact.agda`
constructs two equal-three-dyad states with
different irreducible face consumer. A steering
selector observing only the three dyads cannot,
for all cases, recover their semantic face.
This is a formal negative-control template,
not an assertion about actual trained neuron semantics.

**Exact new same-source proof/gate obligations,
reusing Agda S01–S08, C01–C08 and B01–B05:**

1. `originalX369R4SameFunctionZeroDeltaAllRecurrentSteps`:
   original C2 transition and all three residual
   transitions with zero weights agree for every
   admitted token/state, under identical routing
   kernel/tie/finite precision policy. Existing
   CPU test witnesses do not prove this universal law.
2. `originalRouterPrePostSteeringNoncommutationWitness`:
   explicit finite `TopK` state where a pre-route
   coordinate edit changes selected expert, whereas
   a post-route edit cannot. State the site and
   tie-free strict margin.
3. `sameCarrierC3FeatureDisplacementEnergy`:
   finite triad-source `||Delta T||²=2q(x)q(h)`
   transported to the EXACT original feature
   implementation (including numerical tolerance).
4. `trainedX369SiteActionEquivarianceDefectBound`:
   typed original checkpoint, input action, site
   activation, learned chart and heldout pairs.
   Evidence is a finite empirical bound, NOT
   universal exact neural-equivariance proof.
5. `trainedX369NeuronInterventionConsumerWeld`:
   exact site hook, original model state/history,
   selected neuron/direction, actual downstream
   output logits and intervention reversal.
6. `trainedX369SteeringNecessitySufficiencyVsNull`:
   independently annotated behavioral target,
   matched control distributions, paired model
   and nuisance/heldout results; no semantic
   target inferred from activation sign.
7. `trainedX369SemanticFaceNonDescentOrResidual`:
   either retain additional triadic face/context
   evidence in the actual consumer or exhibit
   original input pair colliding in the compressed
   three-dyad observable but differing in output.

Every Agda result requires actual source signature,
import-closure kernel acceptance and exact
original-execution refinement; do not mark an
interface proposal CLOSED.

## 6. One integrated maximum-cut experiment, not helper churn

**Parallel CPU:** inventory originals; native SLR R1 PG
source; build independently labelled target and
non-target counterfactual pairs; freeze steering
proposal and task `R_g,rho_g` before assessment.

**Sequential one RX580 GPU:**
(A) existing F4 frozen 12-mask R2;
(B) R4b same-source C2/coherent/mixed training
with saved R4c deltas and same-device replay;
(C) actual neuron steering N1–N5 on original
dense checkpoint and repaired C2/R4c source:
single-coordinate add/ablate, contrastive
direction, random/permuted equal-norm controls,
fixed dose/seed/target, necessity and sufficiency;
(D) source-native five-arm grammar/grokking R3.
All observations are distinct; none is an F5
confirmation.

The release decision includes paired document
effects, target/off-target/CE, top-k route changes,
effective feature rank, gradient variance,
source/target action defects, measured time/VRAM,
and artifact/provenance hashes. Freeze the
scientifically selected result before R5 on new
independently source-disjoint material.


## 7. Source correction after the 16 Nov cross-lane maximum cut

The historical three-arm account above describes
the original **fixed** mixed-orientation R4 control.
The new audit established a structural degeneracy:
with all three directed feature channels learned
independently, the original coherent and fixed-mixed
arms are equivalent under a fixed weight permutation.
The earlier matched-norm null was therefore a useful
gauge negative control but NOT an independent
phase-coherence counterfactual.

The latest branch now uses a source-dependent q-energy
gate to change +/- orientation across samples while
preserving C2 and the exact phase-displacement energy.
The new learned-score and top-k bounds show
`delta s_e=sum_j <P w_j,delta T_j>`,
with `P=I-11^T/3`, and a strict margin criterion
for unchanged selected expert set.
A rotating feature with nonzero energy but
`P w=0` is invisible to the learned router;
source history `q(h)=0` similarly blocks the
directed effect. Neither result assigns a semantic face.

The original exact polynomial already exists in
DASHI's `X369OriginalDirectedTriadC3PolynomialCandidate.agda`;
do not duplicate it. A specific new-head kernel
receipt is still required. Full precise owner,
grokking, geometric reasoning, Sophontic, dialectic
and trialectic crosswalk is
[`X369_GROKKING_GEOMETRIC_SOPHONTIC_DIALECTIC_TRIALECTIC_MAXCUT_20261116.md`](X369_GROKKING_GEOMETRIC_SOPHONTIC_DIALECTIC_TRIALECTIC_MAXCUT_20261116.md).
New null/replay semantics are NOT part of the
host-accepted `7fee764` code; the source
contract and original RX580 measurement gates
remain separate and unaccepted at this head.


## 8. Six-phase geometric learning and neuron-steering interpretation (25 November)

The controlling [canonical six-phase geometric learning hypothesis](X369_GROKKING_GEOMETRIC_SOPHONTIC_DIALECTIC_TRIALECTIC_MAXCUT_20261116.md#canonical-six-phase-geometric-learning-hypothesis-25-november-2026) is user-confirmed:

1. **Exploration** — Many competing response modes, weak organization.
2. **Harmonic discovery** — Useful frequencies and local tangent features become amplified.
3. **Partial crystallization** — Coherent but potentially incompatible characters and phase relationships dominate.
4. **Phase coordination** — Relative signs, ordering and intermode couplings become consistent.
5. **Geometric closure** — The intended action transports across previously unseen compositions.
6. **Consumer-sensitive cleanup** — Unnecessary modes are suppressed while future-required residuals remain.

Apply the existing same-source steering obligations to the actual time-indexed phases: separate response-mode power from anchored phase, relative coordination and signed action; distinguish common conjugate gauge from actual relation inversion; test causal phase correction against power/norm-matched scrambled controls; retain any mode required by future consumers. Phase labels may coexist across layers, experts and consumers. A zero-amplitude phase is undefined. The six indexed commuting faces and endpoint paths are exact mathematical controls, not yet trained-model receipts. No claim of physical neuronal oscillation, inevitable phase order, or safety from geometric sign. See the canonical owner for measurements, transition welds, source attribution and release boundary. Existing native baseline and final reserve remain unchanged.
