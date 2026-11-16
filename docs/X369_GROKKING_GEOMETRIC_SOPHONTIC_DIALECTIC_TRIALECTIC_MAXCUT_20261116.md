# X369: source-exact router projection, grokking and semantic-action maximum cut
**16 November 2026 | Owner: LeechTransformer PR #5 | Formal owners: existing DASHI Agda declarations**

This is a cross-lane research **source/claim ledger plus executable
producer/consumer code**. It does not replace accepted finite Agda
donors, create a new definition of Sophontic/dialectic semantics, or
claim an original R1/R2/R3/R4/R5 empirical result. Credit original
Leech-LILA implementation and scientific claims to Anatolii
Kornienko, DOI 10.5281/zenodo.18784424; grokking sources remain
Power et al., Nanda et al., Gromov; Sophontic public claims remain
attributed to Sophontic. No ESD sources changed.

## 0. The original proof is already in Agda source, not a new invention

The exact directed polynomial identity IS represented in the existing
`DASHI/Cognition/PNF/X369OriginalDirectedTriadC3PolynomialCandidate.agda`:
`originalDirectedTriadC3DisplacementSquaredFactors`,
`originalDirectedInverseC3DisplacementSquaredFactors`,
`originalMixedTriadC3RouterFeatureEnergyEqualsCoherent`, including
arbitrary triad lists via induction. The repository file calls itself
a *candidate* and is NOT the selected 8-module already-accepted Agda
leaf suite. PR #1513 documented a ring-solver elaboration repair,
but explicitly left **new-head Agda 2.9 kernel acceptance OPEN**.
Mathematical derivation and Python checks do not substitute for
a fresh Agda kernel receipt. Reuse this owner; do not duplicate it.

The original Leech source implementing the real coordinate feature is
`models/trialectic_wave_prior.py`, consumed by
`models/history_orbit_moe.py`.
The existing numerical C3/F3 metric audit already checks the
polynomial q identity. The missing explanatory edge was the **actual
trained router head and top-k consumer**, now implemented in
`scripts/audit_x369_projected_router_score_weld.py` and injected
into the source-bound
`scripts/audit_x369_f3_live_router_feature_metric.py`.

## 1. A fully stated exact theorem (over real triads)

For original x=(a,b,c), h=(A,B,C), let R(a,b,c)=(b,c,a)
and directed original T=(s,f,r) with
s=aA+bB+cC, f=aB+bC+cA, r=aC+bA+cB.

Then:
- T(Rx,Rh)=T(x,h) (diagonal invariance);
- T(Rx,h)=(r,s,f), T(R^-1x,h)=(f,r,s);
- delta+=T(Rx,h)-T(x,h) has sum(delta+)=0;
- ||delta+||²=||delta-||²=2q(x)q(h), where
  q(a,b,c)=a²+b²+c²-ab-bc-ca;
- q(x)=(3/2)*(re(x)²+im(x)²), when re,im are
  `wave3`'s orthonormal non-DC Fourier coordinates.
  Consequently the exact displacement energy depends on
  rotating-plane AMPLITUDES, not relative phase direction.
  The ordered T retains phase orientation; its squared norm
  discards that information.

Let P=I-(1/3)11^T act on the THREE directed channels.
For a trained directed router weight w_j on each real triad j,
the *original learned score change* at expert e is EXACTLY

    delta s_e = sum_j <P w_j, delta T(x_j,h_ej)>.

Because delta T has zero DC component, the learned score
depends ONLY on the rotating two-dimensional projection of
w. By Cauchy-Schwarz (real, Euclidean) it obeys

    |delta s_e| <=
      sum_j ||P w_j|| * sqrt(2 q(x_j) q(h_ej))

and also

    |delta s_e| <=
      sqrt(sum_j ||P w_j||²) *
      sqrt(sum_j 2 q(x_j)q(h_ej)).

These are bounds, not predictions of a nonzero effect. If
w_j=(1,1,1) for all j, then P w=0 and **delta s_e=0**,
even though feature displacement can be strictly positive.
This is a concrete source-algebra COUNTEREXAMPLE to
"nonzero C3 energy implies learned routing/steering".

The true original `HistoryOrbitMoE` shares ONE 2d
`nn.Linear` router head across experts and adds separate
expert biases. Its unary x-only feature contribution is
identical for all experts. It cannot by itself change their
relative ranking. Source top-k uses the expert-dependent
directed features, projected weights and histories.

For unperturbed scores s_e, let B_e be the certified
bound above. If S is the original top-k expert set
and

    min_{i in S} (s_i - B_i)
      > max_{j not in S} (s_j + B_j),

then the same *set* S must be selected after the relative
C3 intervention, provided deterministic source top-k
and identical source numeric/tie semantics. This is
a sufficient, not necessary condition. Even if the set
is unchanged, softmax mixture weights and expert
outputs may change; later history can amplify differences.
No classifier label or Sophontic meaning follows.

The implemented audit computes original feature deltas,
true original learned-head projections, per-expert bounds,
strict route margins and actual before/after top-k sets.
It is read-only and does not load F5. Witnesses in
`tests/test_x369_projected_router_score_weld.py` include:
nonzero energy/zero projected effect; actual selected
expert switch; strong bias/no switch; and a direct
`HistoryOrbitMoE.select_source_experts` agreement test.
New commits require host regression: do not misattribute
11/11 at prior `7fee764` to the new code.

## 2. Critical R4 IDENTIFIABILITY correction: fixed mixed null is a gauge

Original R4b C3 branch applied a fixed coherent R+, while the
old mixed branch fixed an R+/R- choice for each triad.
Since ALL THREE directed channels remain and their weights
are independently learnable, the two feature bases differ
only by a fixed per-triad coordinate permutation S.

    phi_fixed(x,h) = S phi_coherent(x,h)
    theta_fixed = S theta_coherent
    theta_fixed·phi_fixed = theta_coherent·phi_coherent.

For the minus branch, Tminus=roll(+1,Tplus), so weight
transport is exactly the SAME roll(+1). Unary invariant
channels remain fixed. Norm and trainable parameter
count are preserved. With zero initialization and ideal
permutation-equivariant optimizer updates, the trajectories
are identical after this transport, given identical
data/cost/learning conditions. Floating accumulation and
uncontrolled randomness can disrupt bitwise equality;
this does NOT restore semantic identifiability.

**OLD fixed-mixed was a gauge-equivalent control, NOT
a phase-specific independent negative control.**
`models/x369_fair_c3_residual.py` now contains
`fixed_mixed_c3_weight_gauge` as an explicit check.

### New phase null: sample-dependent C2/C3-invariant energy gate

For each source sample, use the actual q_j(x) and threshold
`q_j(x) > mean_j q_j(x)` to XOR a frozen seed41 base
orientation mask. The new `q_gated_mixed_c3_x` applies
R+ or R- per triad based on that sample-dependent mask.

This has no trainable parameters, no F5 input and no
new semantic-label oracle. Because q(-x)=q(x) and
q(Rx)=q(x), its selector preserves the ORIGINAL
signed C2 and simultaneous C3 invariance conditions
(in exact real arithmetic). Each individual +/- rotation
still satisfies 2q(x_j)q(h_ej). When two source examples
have differing masks, there is NO ONE fixed permutation
relating both samples' mixed and coherent features.
This removes the prior fixed-gauge confound, but changes
the nonlinear feature family. It does NOT match
effective feature rank, gradient distribution, neural
logit displacement or independent semantic action.

The R4 three arms remain: inherited C2 capacity,
coherent C3 and sample-varying q-gated mixed C3.
This change occurred AFTER `7fee764`, so that head's
11/11 host acceptance does NOT certify it.
`tests/test_x369_fixed_phase_gauge_and_dynamic_null.py`
checks the old gauge equivalence, source-mask variation,
C2 preservation, directed per-expert norm equality,
and malformed masks.

**Empirical gate:** verify actual source data produce
varying q masks; the R4 receipt currently records
`nonfixed_mask_source_variation_empirically_verified=false`.
Until this is measured, even the new control is only a
candidate phase-coherence discriminator.

**Artifact compatibility:** R4c source-bound residuals now
record `feature_semantics=x369_r4_q_gated_mixed_v2`
and SHA256 over the exact source bytes of the original
MoE, triadic prior and residual cloning adapter.
Restoration fails if a new model implementation silently
reinterprets old saved delta weights.

## 3. Grokking bridge: finite exact task ↔ model output action

Reuse source:
- `DASHI/Cognition/PNF/ModularFourierGrokkingC3Exact.agda`;
- `DASHI/Cognition/PNF/GrokkingEquivarianceDefectGeometryExact.agda`;
- `DASHI/Cognition/PNF/CyclotomicGrokkingActionLinearizationExact.agda`;
- `DASHI/Reasoning/DASHIgGrokkingEmpiricalBridgeExact.agda`;
- `DASHI/Biology/SpectralGrokkingLatticeExact.agda`.

The **task** carrier consists of nine explicit pairs
(a,b)∈Z3² with y=a+b mod3. The task-level generator
is R(a,b)=(a+1,b); output rho(y)=y+1.
The original finite Agda algorithmic rule has
equivariance defect ZERO; the eight-training-pair
memorizer (wrong at p22) has defect TWO. This
is exact *task* reasoning/grokking specification,
not an original language checkpoint mechanism.

`scripts/audit_x369_typed_mod3_prediction_consumer.py`
now evaluates externally captured [9,3] prediction logits
on the exact canonical nine pairs and independently supplied
task manifest. It measures:
- correct train-eight vs heldout-p22 and whole nine;
- hard model-output C3 action defect;
- mean KL between the translated output distribution
  and the output on translated input.
A perfect argmax task can still have nonzero distribution
equivariance defect: controls test this distinction.
Source/checkpoint generation and independently measured
learned hidden action remain OPEN. Do not substitute
Python fixture outputs for original trained-model logits.

A shared abstract C3 group does not mean the modular
input pair is the original 384D hidden triad. The
model-to-task/source BIDI map requires an independently
specified dataset, tokenizer, output class readout,
checkpoint and source-typed `R,rho,U`.

Grokking geometric-crystallisation mechanism remains
a *trajectory* hypothesis:
track heldout accuracy/CE, t50/t95, phase readout
defect, rotating power q, router projected power
||P w||, score margin crossing, expert distribution
and composition over saved training checkpoints.
Temporal precedence does NOT by itself prove causation;
add interventions and matched controls.

## 4. Geometric reasoning, Sophontic, dialectic, trialectic

**Geometric reasoning**
Reuse `DASHI/Reasoning/LilaWeylGrokkingFibreMaxCutExact.agda`,
`DASHI/Reasoning/GrokkingGeometricCrystallisationMemoryRoutingExact.agda`,
and `DASHI/Reasoning/E8StructuredFullWeylReceiptExact.agda`.
A useful geometric chart or neural subspace is NOT an
E8/Leech automorphism. The q energy diagnoses the
rotating plane but does not supply the learned chart,
decoder or same-action task guarantee.

**Sophontic / semantic interventions**
Reuse `DASHI/Reasoning/SemanticInterventionEquivarianceExact.agda`,
`DASHI/Cognition/Teleodynamics/TeleodynamicSemanticActionBridgeExact.agda`,
and `models/paired_intervention_geometry_loss.py`.
The existing semantics evidence ladder is
paired correctness → typed latent action → decoder
equivariance → heldout composition → future/action
separation. Independently labelled `R,rho` must be
supplied; no auto-assignment of meaning from q, sign,
C3 phase or expert ID. Proposal model and assessment
model must remain distinct, and neuron steering
needs direct intervention on true activations.

**Dialectic**
Reuse `DASHI/Reasoning/DialecticInvariantGeometry.agda`
for stance-swap, common intersection and disagreement;
`DASHI/Core/ArgumentResponseNonGeometricOppositeBidiExact.agda`
for response ≠ logical negation ≠ geometric antipode.
C2`x->-x` is NOT a semantic thesis/antithesis
involution without a separate typed map. Commuting C2×C3
on vector coordinates is NOT the S3 reflection law:
these are distinct structures and not interchanged.

**Trialectic**
Reuse `DASHI/Reasoning/Leech369TrialecticWaveLearningArchitectureBidiExact.agda`
for directed 3+6 relational geometry and dimensions,
and `DASHI/Reasoning/TrialecticBoundaryFaceNonfactorabilityExact.agda`
for a constructive NON-DESCENT theorem:
two full states can have the same three dyadic edges
and distinct triadic faces. Therefore an output
consumer that requires face information cannot factor
through edge-only geometry. Equivalently,
equal (T(x,h)) or equal q does NOT determine
an independently labelled face or response.
Retain face/context/history in a typed payload,
or provide a formal verified fibre-constancy law
for that particular consumer.

Reuse `DASHI/Reasoning/LeechJointStateConsumerQuotientExact.agda`:
selected expert projection preserves a selected
consumer only under its sufficient/fibre-constancy
hypothesis; full-state consumers additionally need
inactive-history equality. Projection need not
preserve downstream teacher/argument/semantic
observation without evidence.

## 5. Source/model/action proof gates and Pareto maxcut

| Gate | Producer → consumer | Status |
| --- | --- | --- |
| A0 directed ℤ polynomial 2q(x)q(h) | DASHI candidate ↔ Python original T | Exact derivation/source; NEW-HEAD Agda kernel still OPEN |
| A1 Fourier q = 3/2 rotated power | source wave3 ↔ q | Mathematically exact; original FP tolerance tests needed |
| A2 score DC annihilation, weighted bounds | original T + head ↔ Δrouter logits | Implemented regression; new host source run OPEN |
| A3 strict TopK margin | projected Δs ↔ selected expert set | Implemented regression; physical source margin distribution OPEN |
| A4 coherent/fixed-mixed gauge | original three-channel features ↔ parameter permutation | Exact identity and regression, reveals negative-control degeneracy |
| A5 q-gated source-varying null | energy/C2/C3 invariants ↔ nonlinear control | Implemented, host and actual source-variation OPEN |
| G0 nine-pair modular task action | Agda Grokking 0-vs-2 ↔ prediction outputs | Typed Python consumer implemented, original model logits OPEN |
| G1 grokking trajectory | q, ||P w||, route margins ↔ t50/accuracy | Original checkpoint trajectory evidence OPEN |
| S0 labelled semantic action | independent Sophontic tasks ↔ model label/decoder | OPEN on original neural source |
| D0 semantic response map | typed predicate/stance ↔ activation action | OPEN, cannot be derived from sign |
| T0 triadic face consumer | boundary/face source ↔ sufficient history payload | Finite non-descent proven in existing donor; original neural weld OPEN |
| N0 neuron causal effect | site-specific ablation/injection ↔ target/off-target | Original physical measured intervention OPEN |
| R1–R5 | original PG, frozen checkpoints, fair models, new blind test | Physical releases OPEN |

### Execution as ONE integrated campaign

CPU / exact first (parallel):
- Kernel-check the selected Agda candidate via the existing
  repository checker; do NOT import an unaccepted new theorem
  into `DASHI/Unifier.agda` or
  `DASHI/EverythingTOESymmetryResolved369.agda`.
  Existing TOE RootFourier/Root369 and trialectic donors
  already participate; source-neural weld requires a
  certified import graph, not an unsafe new root import.
- Inventory real original native SLR PG and repaired R64
  checkpoint, F4 frozen seeds and source-doc identifiers.
- Freeze independently labelled modular and Sophontic/
  dialectic/trialectic task input/output actions with
  source fingerprints. No reserved F5 token access.

RX580 GPU sequentially:
1. R2 original frozen F3-B 12 mixed-mask source probe;
2. R4b/R4c three-arm real train+source-bound delta
   replay, new q-varying mask and score/margin audit;
3. R3 real five-arm grammar + grokking trajectories;
4. N1–N5 original-neuron/contrastive-direction
   steering, necessity/sufficiency, phase and random
   controls, after original source-hook verification.

Then decide strictly by measured results:
- Positive q without P w projection: GEOMETRY but NO
  learned score effect.
- Nonzero Δscore without TopK change: possible mixture
  weight effect, no claimed route-switch.
- TopK change without output/label effect: routing
  mechanism, no semantic causal utility.
- Output-task action defect decrease without
  independently labelled heldout composition:
  correlation, not learned reusable reasoning.
- Task-equivalent and selective steering with
  intact trialectic face/context and independent
  controls: candidate semantic causal mechanism.
- Only source-disjoint independent R5 with predeclared
  analysis/resource budget can promote an empirical
  original-model advantage.

**No empirical performance improvement, new physical
R2/R3/R4 measurement, Agda kernel acceptance, or
original-neuron causal effect is asserted here.**
