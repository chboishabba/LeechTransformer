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

**Empirical gate:** the R4 runner now observes original
development-time source masks on every layer during its
already-required initial CE pass. It retains a first mask,
an accumulated change bit and number of observations,
not full activations or F5 data. The mixed arm fails
closed if no actual source mask changes; after a genuine
successful physical run the receipt can mark
`nonfixed_mask_source_variation_empirically_verified=true`.
Before such a run this is an IMPLEMENTED gate, not a
measured original-source result. The new null still
changes effective nonlinear representation, not only
a coherent phase; compare rank, gradients and compute.

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

---

## 24 November G7a — extend the existing task, do not duplicate it

Geometry-forward, zero mode, signed residual, time/model-revision and ordered guard-occurrence indices. Original Leech-LILA credit Anatolii Kornienko (DOI above); original grokking donors Power/Nanda/Gromov; DASHI task extension; BitNet MIT source pinned in G4 docs. Existing task owners verified directly on DASHI master: ModularFourierGrokkingC3Exact.agda blob 60ebb0abf6c97f40f6f00d5271350fa2a4938e19; GrokkingEquivarianceDefectGeometryExact.agda ccb2a6b8887aaaf87eb1fb527b1f73d611f1c1b9; LilaWeylGrokkingFibreMaxCutExact.agda 2414b0cca4d796e52e9ef7563bd9ad9319788471. These are existing source declarations, not new kernel receipts. The same-action learned recognition seam is explicitly open there.

Reuse this roadmap and the existing typed modular-three auditor, scripts/train_x369_mod3_codec_gpu.py (actual sequential HistoryOrbitMoE learner), scripts/run_x369_mod3_maxcut_gpu.py (four computational arms, additive and pair-MLP controls, fixed exact Fourier oracle). The historical L6 3/8 training fit at 200 steps and worsening heldout NLL are not grokking; keep F1/F3/F4 negative evidence, fixed-mixed gauge correction, dynamic q-null and independent lockbox unchanged. Do not replace the original nine-point task or treat its finite label action as learned hidden equivariance.

### G7a1 independent ordered label source and original task embedding

New companion scripts/audit_x369_ordered_signed_task.py extends the *existing* task lineage. It generates labels by applying ordered operations to two literal signed registers, not by invoking a model, its Gram codec, router or learned orientation. For initial mean M and residual D form ordered registers M+D,M-D. Translation T_k relabels coordinate i to i+k mod3 in each register. Reflection J0 relabels i to -i mod3 AND swaps the registers. N negates entries of both. Process word left-to-right. The four original shared guard labels a,b,j,n have values a,b in Z3, j,n in {0,1}; a and b select T_a,T_b, j toggles J0 and n toggles N. Repeated reads use the SAME original value. Label outputs are independently obtained by sum/difference of final registers: mean, ordered signed difference, and its six quadratic products. Initial M=e0, D ranges over all 27 signed ternary triads. M fixed makes the mean control deliberately easy, not a general learning benchmark.

Original nine-point task embeds exactly: on word ab, starting M=e0, final mean=e_(a+b mod3); the original generator increasing a shifts that coordinate by +1. All nine cases checked. Coordinate convention matters: original models/trialectic_wave_prior.py rotate3 is LEFT roll, maps basis e_i to e_(i-1); task T_1 is its inverse. Original spatial J reverses indices i->2-i, whereas J0 maps i->-i, so J0=R^2 J for the original left-roll R. Branch swap is additional, not spatial negation. This correspondence is a declared source chart, not a learned hidden action.

Separate closed-form verifier composes k,b,s with spatial index i->(-1)^b i+k mod3, global sign (-1)^s, and reflection branch swap b. Mean transforms with sign (-1)^s, residual sign (-1)^(s+b). DC sum transforms with those signs; a nonzero zero-DC phase residual is not null. Quadratic D D^T retains cross-coordinate sign, loses overall antipode. No local observer is assumed sufficient across the task.

### G7a2 ordered guards and exact collisions (P vs NP donor, not endpoint promotion)

User supplied G84–G88 report: Agda #1548 commit 213b383e110351d1e508d5aecf75ba97e3a5386f and Lean #119 commit a40a87b261310e226687dce4712bbb011eab952e. The reusable idea is to retain original shared guard labels, ordered occurrences and signed commutator, not a complexity separation. Here T_1 J0 != J0 T_1 because J0 T_1 J0=T_-1. With a=1,j=1, D=e0: chronological aj gives -e2, ja gives -e1. Repeated shared-label words ajaj and aajj have the same multiset, but the former identity and latter T_2, giving e0 vs e2. Adjacent order difference is signed operator commutator; when surrounding transfers annihilate it the consumer may not separate, so do not claim every reordering changes every state. The task is finite and solvable by constant-size group fold in linear word length; not a hardness or asymptotic neural speed claim. No external PR edits.

### G7a3 reverse-solved exact observer ceilings

Uniform finite enumeration conditioned on word, guard assignments and initial mean (given even to weak observers). For each observer key, best possible exact classification is sum of maximum label multiplicities per key. Mean observer has no D; support observer sees component squares; Gram observer sees all DD^T entries; signed observer sees D. Under fixed word and assignments all signed label maps are invertible signed permutations and all Gram class maps invertible. Thus signed target optimum counts per 27 input residuals: 1,8,14,27 respectively. Quadratic target optimum: 2,15,27,27 respectively (zero singleton; seven nonzero supports, each best Gram class size 2). Mean target 27 each. These are information-theoretic finite-task ceilings, NOT measured learning accuracy; only apply if the predictor has access solely to that key, including no token bypass, history leakage or direct label feature. Signed target needs antipodal mark, Gram needs relative signs; neither implies the hidden model discovers a unique coordinate basis. Null and nonzero DC/zero-DC controls independently tested.

### G7a4 execution and split provenance

Actually executed local pure Python exact integer checker, 11,664 rows = 12 schedules x 36 guard assignments x 27 residuals; all register/closed-action labels equal; 9/9 original modular task embeddings. Development training word set a,b,j,n,ab (4,860 rows); composition evaluation word set aj,ja,ajaj,aajj,ajb,bja,abjn (6,804 rows). Public mechanism split, NOT an untouched external test or final R5 lockbox. Canonical JSONL SHA256 cd6feb2f27519e2264d76b39e9bec460b9d431e8eb18aca6b19765c801325cb1. Each of aj/ja, ajaj/aajj, ajb/bja has 972 matched same-guard same-payload comparisons: 324 differing mean labels and 288 differing signed and 288 differing Gram labels. These are exact source label separations, not neural evaluation. No native GPU, CI, Agda kernel or training executed. No original source/reserved corpus reads or source model edits.

### G7/G8 phase/lemma/producer-consumer ledger (next large cut)

| Phase | Exact deliverable / remaining lemma | Existing producer -> consumer | Status |
|---|---|---|---|
| G7a1 | orderedLabelSource and mod3SubtaskEmbedding; signed spatial/branch actions typed separately | ModularFourierGrokkingC3Exact + existing typed mod3 audit -> ordered task producer | proved; 11,664 labels, 9 embeddings PASS |
| G7a2 | sharedGuardOrderSeparation and signed commutator; identical bag counterexamples | supplied #1548/#119 G84–88 + G5b paired geometry -> ordered consumer | proved and enumerated |
| G7a3 | exactObserverCeilings for mean/support/Gram/signed; null and DC | G5a2 residual retrieval and actual independent task labels -> audit | proved and counted |
| G7b | sourceTaskLearnerAttachment: serialize literal guard labels/read assignments/initial registers, independent per-example reset, ordered time indices; no precomputed endpoint leakage; wire original sequential Learner and paired variant with same declared task labels/readouts; retain zero/support/sign and record causal bottleneck | existing train_x369_mod3_codec_gpu.py, PairedHistoryMoE, CrossTriadBitLinear -> extended existing runner | open |
| G7c | observerBottleneckEnforcement: run mean/support/Gram/signed ablations ONLY when no alternate token/state bypass; prove actual observer receipt; matched C3/S3/full/dense ternary vs real, additive/pair-MLP/order-blind controls; count effective parameters, executed work, storage, seed, initial function and GPU time; no fixed permutation gauge falsely used as independent null | existing six-arm runner, dynamic-q control, G4/G5 -> comparable training | open |
| G8a | independentTaskOutputActionAudit: per-example signed/Gram/mean accuracy, distribution equivariance, order swaps and commutator interventions; fit train before heldout composition, fixed horizon/no composition-driven selection; separate source action, hidden intertwiner, task output and causal effect | existing typed_mod3_prediction_consumer, GrokkingEquivarianceDefectGeometryExact -> actual learned logits | open |
| G8b | learnedSameActionReceipt: learn/freeze nontrivial hidden action, unseen compositions, matched causal interventions at exact layer/time/site including routes, margins, histories and consumer; verify against independent labels; no claim of unique hidden chart modulo gauge | LilaWeylGrokkingFibreMaxCutExact, existing explanatory/steering doc -> trained representation | open |
| G5a3/G6c | fitted reachable marked quotient, sign/cap/route guard receipts, calibrated numeric envelope; recover actual source state only under certified consumer | #1550 future trace and G5a2/G6 -> fitted source | open |
| G5b5c/G9 | actual original factory test, GPU-native gfx803 packed performance and transitive formal acceptance | prior L2/L4, original HistoryOrbitMoE, GPU runner -> release | downstream; original factory tests still unexecuted |

Next G7b/c and G8a/b together: extend the existing training/consumer runner, not a new generic benchmark. The public composition split is already mathematically enumerated; any truly blind final generalization requires a separate independently frozen manifest. Preserve original attribution, task owners, source fingerprint and prior negative evidence.

---

## 24 November — G7b/G8a ordered learner attachment and exact learned generator control

Geometry-forward; retain DC, null versus inhabited zero-DC, signed residual and time/model revision/original guard index. Owner Leech #5; original Anatolii Kornienko (DOI above), DASHI extension. Reuse G7a label source and original mod3 runner, HistoryOrbitMoE, PairedHistoryMoE, CrossTriadBitLinear, original typed output audit and formal donors in preceding sections. No change to original model, original mod3 runner, source data or other PRs.

### G7b1 exact learned operator, not a hardcoded endpoint oracle

For each of ten valid elementary (guard,value) keys, fit an unrestricted real 3x3 linear residual operator from only single-token training examples. For balanced ternary residuals, sum_D DD^T=18I; repeated assignments of unused guards multiply by the same positive integer. Therefore the exact normal equation A=(sum Y_D D^T)(sum DD^T)^-1 uniquely recovers the true residual linear map, since labels Y_D=A_true D and the covariance is invertible. Use rational Fraction arithmetic. Chronological composition applies the learned A at each literal read, with the SAME guard assignment for repeated occurrences. No word endpoint lookup or composition label used in fitting. Mean reconstructed from the residual operator with the independently declared branch-swap character (-1)^j at active reflection; this character is supplied, not learned from mean-only training. Initial M=e0 is insufficient to identify arbitrary mean-generator matrices from single-token examples alone. Gram reconstructed from learned signed result. This is a learned linear generator control with a supplied composition and readout law, not a neural same-action receipt or learned latent chart. In particular it learns exact signed spatial maps in the declared task coordinates, not an internal LILA representation. All 4860 train and 6804 composition examples correct for all three targets. The observed training carrier spans R^3 for every generator; unique within linear class, not arbitrary nonlinear learner. Gauge: replacing latent coordinates by invertible U conjugates every generator by U and transports the encoder/decoder, leaving outputs unchanged; hidden same-action claims must be modulo such gauge and checked at the actual site.

### G7b2/G7c1 actual input bottleneck and ordered step attachment

scripts/train_x369_ordered_action.py imports independent label producer and uses original HistoryOrbitMoE.step or opt-in PairedHistoryMoE.step, new initial-state projections because this task supplies initial registers, not a two-token zero-state mod3 task. Initial seed contains M and a padded six-coordinate packet (mean-only/coordinate squares/Gram/signed); the only subsequent activation is a seven-coordinate one-hot literal label and its current assigned value, in order. No raw D, endpoint, full word summary, unused guard values, or skip of input into readout. A single learned projection of the allowed residual packet seeds +/- branches around projected mean; starting both branches identically without an explicit seed would not test the desired residual. Bias-free seed projections preserve zero packet. Every example resets all expert histories; group minibatches by exact word length, no padded updates; final readout consumes last expert output and all histories only. Three joint categorical heads (27 mean,14 Gram,27 signed), not coordinate-wise accuracy substituted for joint-label ceiling. Source original single branch and paired variant are distinct new task adapters, not asserted same-function or same training cost. Optional expert/transition replacement with fixed scale .25 C3/S3/full ternary CrossTriadBitLinear; source router unchanged. New weights initialized by candidate, not exact source checkpoint replacement. Paired hard signs lack original STE training semantics; this is an explicit fairness blocker. Dense real vs structured ternary not yet equal cost/parameter/initial function; dense ternary and dynamic geometry null still needed. Record parameter counts; no matched-budget claim. GPU required for training, no CPU fallback, fixed horizon and seed, only train groups enter loss; public composition metrics are reported not used to select checkpoints or optimize. No final lockbox. Exact input encoder unique keys 115/920/1610/3105 across all rows (unused guards removed), exact signed ceilings 432/3456/6048/11664 of 11664, Gram 864/6480/11664/11664, mean 11664 for all. Thus serialized input preserves exactly prior ceiling counts despite unused-guard collapse. Only deterministic per-example inference without external leakage is covered by the ceiling.

### G8a1 output consumer audit (actual learned logits still unpaid)

scripts/audit_x369_ordered_action_predictions.py derives independent class permutations by literal register execution. Compares prefix and one-token extension with same guard assignment/initial residual. Report hard action defects and mean full-distribution L1 under typed permutation, plus both endpoints correct on each source-separated order pair. 3888 available prefix-extension pairs, separated mean/Gram/signed 972/864/864; oracle label logits have 0 hard defects, L1 <1e-14 and all separated pairs correct. Uniform logits have exactly zero distribution defect yet zero separated pairs correct; hard argmax tie policy may still produce defects (0/0/486), so do not treat invariant probabilities as learned task action. This audit is wired after final fixed-horizon neural run. Full learned latent action and causal interventions remain separate; no inference from task correctness.

### Execution and remaining gates

Actually executed local /mnt/data/lila_g7 with Python Fraction and Torch 2.10.0+cpu: exact learned 10 generators, 11664/11664 all-label rows, 46656 antipodal input comparisons, exact encoder ceilings, four dense/C3/S3/full paired fixture FP64 forward/backward smoke cases, 3888 output audit pairs and uniform negative control. The fixture is from existing scripts/check_lila_paired_history.py, not actual original HistoryOrbitMoE import; original source and GPU dependencies not locally installed. No gradient claim for parameters whose gradient is absent (notably paired hard orientation), no training/optimizer parity, GPU run, fitted checkpoint, kernel or CI. No user handoff. Reproducer scripts/check_x369_ordered_action.py. Actual source mode in GPU-first runner is supplied, not executed here.

| Gate | Exact deliverable / producer -> consumer | Status |
|---|---|---|
| G7a | independent ordered label source / modular donor -> train and audit | prior proved |
| G7b1 | generatorNormalEquationIdentifies and orderedComposition / G7a -> exact learned linear control | paid restricted linear class; 10/10 and all rows |
| G7b2 | literalSequenceAttachment, per-example reset, explicit paired seed / existing HistoryOrbitMoE and PairedHistoryMoE -> new task adapter | source written; paired fixture PASS, actual original source run open |
| G7c1 | encoderBottleneckNoBypass and exact ceilings / observer key -> serialized model inputs | proved for adapter path; finite checks PASS |
| G7c2 | matchedLearningComparison: dense ternary, real and structured controls, matched initial function/effective budget/work, source STE or explicit same hard-sign training policy, ordered vs unordered baselines; native receipts | open |
| G8a1 | taskClassActionPermutation and independent output audit / G7a literal labels -> logits consumer | oracle and uniform controls PASS; neural logits open |
| G8b | learnedSameActionReceipt: fit actual hidden action on training interventions, freeze and test unseen compositions, compare actual decoder and recurrent histories including residual/zero and route margins; gauge-aware causal intervention matched null | open |
| G5a3/G6c | fitted reachable quotient and calibrated error envelope / #1550 and G5/G6 -> actual trained source | open |
| G5b5c/G9 | actual original source execution, native gfx803/packed resource comparison and formal acceptance | open |

Next close actual source acceptance and fair learning controls, not more generic task generators. Geometry-first objective remains to determine if LILA's latent actions implement the same independent ordered task action, not to equate the exact linear control with neural learning.

---

## 24 November — G7c2/G8b common-function geometric residual and frozen hidden action intervention

Geometry-forward: keep DC and nonzero zero-DC states, signed residual, time/model-revision and original literal-occurrence indices. Owner LeechTransformer #5. Original Leech-LILA Anatolii Kornienko (DOI above), DASHI extensions and BitNet donor pin in G4; existing modular task/grokking, same-action Agda and source steering owners above, #1550 future consumer, merged #1526, #1291 indexed geometry and #1548/#119 ordered guard inspiration. No external branch edits, no F5 or original corpus access. Starting head 328144bbc2043e10e139230fb897952dbc5ad738; original HistoryOrbitMoE blob a2a250462ce8e663591cd8a8f784fad9bde1b15d.

### G7c2a common-function residual arms

For each original expert or transition affine matrix W0, retain frozen dense W0 as a buffer and common original bias as trainable parameter; add lambda sum_i q(c_i) B_i per triad pair, with lambda=.25 and c=0 initially. q identity or round/clamp signed ternary STE. Thus W(0)=W0 exactly in ideal real arithmetic; all other source parameters, embeddings, readout and initial-state projections copied. Induction preserves whole initial recurrent function within EACH single/paired family (not equality of the two architectures). Use real Frobenius orthonormal frames by QR of existing C3, S3 or full geometric generators or dense-entry identity; rank 3/5/9. For matched null QR a seeded random 9 x rank matrix. Within rank, parameter counts equal and perturbation Frobenius norm equals lambda times coefficient norm. Orthonormalization changes original trit codebook: ternary coefficients in a real basis, NOT expanded signed-ternary weights or original group-generator trit coefficients; base remains dense. For rank 3/5 compare different subspaces (verify projector difference), rank 9 spans all matrices for both, real family is identical expressivity and ternary finite codebooks differ. A fixed random frame need not be independent of all possible whole-network gauges; not proof of geometry-selective benefit. Compare full vs entry at same seed as separate runner invocations, same base but different bases; no claim of optimizer-coordinate invariance (AdamW is not invariant under arbitrary rotations), same-function all training steps, whole model packed storage, equal physical work or performance. Frozen base deliberately changes optimization relative to fully trainable source. Within same frame real and STE initial derivatives coincide; quantization thereafter can differ.

### G7c2b paired sign training repair

PairedHistoryMoE.step now accepts train_sign=False by default. Opt-in true uses source formula hard.detach() + (tanh(score/source.temperature)-tanh(score/source.temperature).detach()), preserving paired chart hard forward value and restoring smooth orientation score gradient. Original branch fallback remains unchanged (paired chart deliberately different from scalar source). FairLearner uses same sign-training policy within and across source families, but paired executes two branches and is not source-identical or cost-matched to single history. No exact optimizer migration claim.

### G8b1 frozen chart and gauge-aware diagnostic

Use only independently labelled public training rows to fit an affine chart of the complete post-prefix history Z (all experts, both branches if paired), from task coordinates Y=(M,D) in R^6. Center Y,Z on training means, fit U=pinv(Yc) Zc and decoder pinv(U). Require numerical rank(Yc)=rank(U)=6 (rtol 1e-10), reject collapsed charts; do not infer latent action from output correctness or rank alone. Score held-out composition rows by recurrent index t=2,3,4 using the SAME frozen chart: task RMSE, separate mean and signed residual RMSE, orthonormal DC RMSE and relative history reconstruction error. Fit excludes composition labels. Full rank permits embedding of the task action: with column convention L=U^T, chart image action A_g=L rho_g L^-1 (inverse restricted to image). In row coordinates decoder/re-encoder gives induced image action; an arbitrary ambient extension is not identified. Under invertible hidden reparameterization, exact affine chart and on-image action transport by conjugation; Euclidean residuals, pseudoinverse off image and numeric rank thresholds are coordinate-dependent diagnostics, not GL-invariant claims. A shared affine chart across time is a deliberately strong hypothesis, failure does not exclude a nonlinear/time-dependent chart. Training task mean is fixed initially but endpoint labels span all six coordinates over training words; fit rank confirmed 6. No independent fitted hidden generator recovery or exact off-chart action claim.

### G8b2 causal consumer intervention at actual recurrent site

For each nonzero-residual public composition word with a present prefix, pair initial D and -D, same original guard assignments and word. Capture natural histories Z and Z_cf after that prefix. The independent task oracle supplies the prefix target change Delta Y, not the downstream endpoint. Inject Z_edit=Z+Delta Y U at that exact post-prefix history site (keeps residual outside fitted row space unchanged in this chart). Feed the identical next literal into the actual source or paired step and actual readout. Compare with natural counterfactual continuation, including next full history, ordered IDs, all selected signs, weights and sum of full-distribution L1 over three heads. Include unedited baseline and a seeded fixed coordinate-permuted delta (same Euclidean norm, not a proof of independent gauge or feature/score matched null). This is an oracle-defined intervention diagnostic, not predictive accuracy, and does not establish correctness of natural counterfactual by itself; use independent output audit as a separate gate. Does not retroactively change prior routing or claim causality at any other site. Full learned task-action proof still requires multiple independent interventions including translations/reflections, frozen latent composition, calibrated guard/route margins, source-specific matched null and true trained model. No selection by composition scores.

### G7/G8 actual runner and acceptance scope

scripts/run_x369_fair_ordered_gpu.py extends existing OrderedLearner, original source step and paired step, reuses labels/batch/score and output action audit. Four rank-matched arms per invocation (geometric/null x real/ternary), same seeded base and minibatch stream, signed input, fixed horizon, training only on training words; public composition evaluation not used for selection. Initial exact-equality output preflight on 32 training rows, parameter counts, buffer bytes, elapsed wall time (includes reporting/evaluation) and peak allocated CUDA bytes; not a measured equal work or matched energy claim. GPU mandatory; optional gfx803 HIP requirement. Fit frozen hidden chart and perform causal audit only after fixed horizon. Separate single/paired invocations; no GPU run here.

Actual local Python/PyTorch 2.10.0+cpu runs in /mnt/data/lila_g8:
- scripts/check_x369_fair_geometry.py: 16/16 paired source-interface fixture initial forward trace equal (outputs, entire next history, IDs/signs/weights), nonzero restored orientation gradient, finite selected residual coefficient gradients, real/STE initial parameter gradient equality within each basis, equal trainable counts in each rank; orthonormal rank/projector tests. Synthetic random full-rank affine gauge chart fit on 4860 training rows, all 6804 composition rows task RMSE and history reconstruction below 1e-12, antipodal transport and collapse rejection.
- scripts/check_x369_hidden_intervention.py: exact synthetic action carrier (not neural learning) rank 6, heldout chart RMSE <5e-14, 2808 actual post-prefix downstream edits, mean summed distribution L1 per row unedited 1.744519586968385, chart 4.616085014112304e-13, permuted null 4.068947764629847; next history L2 chart 1.5558228326392987e-13. Untrained paired source-interface fixture seed41 rank 5 of 6, intervention blocked; no same-action inference. This is a useful negative control, not an empirical result for original trained LILA.
- Direct local GitHub raw source attempt failed DNS, PyTorch CPU only; no actual HistoryOrbitMoE import/training, native AMD, Agda kernel or CI. Original source blob unchanged. No intermediate user handoff.

### Exact phase/deliverable/producer-consumer ledger

| Gate | Exact deliverable / producer -> consumer | Status |
|---|---|---|
| G7a/G7b/G7c1 | independent labels, learned generator control, ordered source interface, bottlenecks -> learner | prior scoped results |
| G7c2a | commonInitialFunction, matchedResidualRankAndMetric, quantization alphabet distinction / G4 basis + original linear -> fair residual arms | proved ideal-real; 16 fixture arms PASS |
| G7c2b | pairedHardForwardSTE, shared sign training policy / original sign score -> paired source | implemented; local gradient PASS |
| G7c2c | actualFairTrainingReceipt: native original source run, matched streams, C3/S3/full/entry/null comparisons, measured work, storage, effective gradients and route distributions; no assumption of identical cost or optimizer geometry | runner supplied; open |
| G8a | task output action and order separation / existing independent auditor -> trained logits | prior oracle checks, actual trained logits open |
| G8b1 | frozenFullRankChart, task/mean/signed/DC reconstruction by recurrent t, gauge scope / independent labels + complete histories -> fixed chart | implemented, synthetic positive/collapse negative PASS; trained model open |
| G8b2 | postPrefixAntipodalConsumerIntervention / frozen chart, original guard and actual next-step/readout -> natural counterfactual comparison | harness executed synthetic positive and untrained rank-block control; original trained neural receipt open |
| G8b3 | learnedSameActionReceipt: nontrivial generator/intertwiner composition and decoder transport under translation/reflection/negation, multi-site independent causal and matched score/feature nulls with actual route/tie margins, frozen evaluation; distinguish task accuracy and latent action | open |
| G5a3/G6c | fitted reachable quotient and calibrated FP32 signed error, guards / prior #1550 + G5/G6 -> actual trained source | open |
| G5b5c/G9 | actual original-source acceptance and native gfx803 packed resource/kernel acceptance | open |

Next actual original-source acceptance and fixed-horizon GPU learning, then independently frozen learned action/intervention and reachable quotient. Do not promote a rank-six chart or synthetic control into a LILA learning claim. Preserve geometry-forward, zero mode and signed residual.

---

## 24 November — G5b5c original-source recovery and G8b3 independently fitted residual generators

Geometry-forward: retain signed residual, actual DC and inhabited neutral phase, shared original guard/occurrence, time/model revision, and the original complete histories. Original Leech-LILA: Anatolii Kornienko (DOI above); DASHI extension. Owner Leech #5; formal donor DASHI #1550 future trace, merged #1526, #1291 indexed geometry; existing ModularFourierGrokkingC3Exact, GrokkingEquivarianceDefectGeometryExact and LilaWeylGrokkingFibreMaxCutExact source owners, #1548/#119 ordered-guard donor; BitNet source pin in G4. No external PR or original model changes.

### Original-source recovery, previously unpaid local acceptance

Git checkout/raw download remains unavailable, but manually recovered the five original modules from GitHub connector text and independently verified Git blob SHA1 over exact local bytes (all five match at starting head 4f02f9e37eae5c36467243eb99e148530a5e40b6):

- models/history_orbit_moe.py a2a250462ce8e663591cd8a8f784fad9bde1b15d
- models/trialectic_wave_prior.py 0bea25f6128c7979d95949c1dd8638541f40c15b
- models/punctured_nine_wave_prior.py d0d890c17ecaf18ea3efa14833405cd116038511
- models/e8_root_prior.py 6ee8d30d06fd20d3d7168dcd0030c6cc7290d42e
- models/c3_matched_router_control.py 2765beeaf20442280653c8b696d7a3abccf7de07

Isolated package in /mnt/data/lila_g9 with empty models/__init__.py to avoid importing unrelated full Leech application. All actual dependencies imported, no stub router/expert or source changes. Exact existing adapter/checker bytes verified against repository (check_lila_history_adapter.py c9a07fd82734dc61f76bb8e2a385acbbdb8c7d36; check_lila_paired_history.py 7e52141b727bd724bcb0e298394182c05cf59d5f; check_lila_residual_observability.py 5a8d55d18a94f2bdbec59496610b39a340467d55; original adapter and marked reflection files verified). New scripts/check_x369_original_source_acceptance.py fail-closes on five source hashes and runs original acceptance. Actual Python/PyTorch 2.10.0+cpu, SymPy 1.14.0, 90s process limit, one CPU thread:

| Original-source test | Receipt |
|---|---|
| exact witness adapter (raw and coherent, dense/low rank, cap/no cap, 3 geometric bases; step, forward, selected weighted/unweighted, rejected absent/mismatched witnesses) | 24/24 PASS; floating results atol/rtol 1e-11, exact step IDs/signs |
| paired original source (two cap choices, 12 group elements, three steps, batch 3) | 72/72 PASS, exact IDs/signs, 1e-13 FP64 continuous tolerance and inactive identity |
| centered actual original paired GELU residual | 81/81 PASS, exact histories and strict signs/IDs, 1e-14 FP64 mean/residual |
| existing independent exact paired transport/curvature controls | 2187 indexed steps, 104976 compositions, 729 collisions, 1729 curvature cases PASS |
| new original single/paired common-function fair arms, four frames x geometric/null x real/STE x two families | 32/32 PASS; exact complete one-step traces, nonzero orientation grad, selected residual grad, equal same-rank trainable counts and same-frame real/STE initial gradients |

These pay the earlier *scoped constructed CPU original-source* G4e5 and G5b5c harness obligations, and new original-source initial/gradient preflight. They do not establish a trained or fitted checkpoint, complete Leech application, arbitrary ties or all parameter families, native gfx803, GPU timing/packing, kernel verification, or matched budget training. Source-only adapter was frozen; paired/fair learning arms separate. No training or CI. No source checkout or package install required; local source identity verified via exact hashes.

### G8b3a. Learn natural residual actions independently, not by constructing a conjugacy from a decoder

For each indexed original prefix w, original shared guard assignment v, and mean M, define natural contrasts X(w,v,D)=Z(w,v,D)-Z(w,v,0). The zero baseline is the *same prefix* with D=0, not an assumed zero network history. Task labels d are the independently executed signed endpoint D_w. Every fixed prefix induces a bijection of the 27 residuals, so there is a unique natural counterfactual with endpoint rho_g d at the same w,v,M. R,J,N act on residual only here; M is fixed (not an alleged full joint task action). Use independent literal register label source, not geometric model internals. Fit U=pinv(D_train)X_train; require rank three; choose V (hidden-by-3) orthonormal basis of its row space; q=XV. Fit decoder from training q to d, then for each generator independently fit unrestricted row operator A_g=pinv(q_train)q_g,train where q_g is from natural paired histories. No group law, conjugation, or known matrix constrained in these generator fits. Train prefixes only (4860 rows); heldout composition prefixes (6804) never choose chart or maps. Keep original indices; numerical rank relative threshold 1e-10. Report all three fits, task reconstruction, full-history outside-chart residual and induced natural history edit error by t, both orders RJ/JR. Rank-three admission is not a certificate. Operator Frobenius metrics in frozen orthonormal chart are coordinate dependent, not general GL-invariant.

Exact sufficiency theorem: if X=dB with B full row rank three on entire declared domain, q=dBV and BV invertible, then A_g=(BV)^-1 rho_g^(row) BV is uniquely recovered on spanning training q. Composition follows. Conversely, satisfying group relations alone does not prove that q is the task residual (trivial/other representations possible); full-rank decoding alone does not prove natural-pair linear closure. Group defect, label/decoder action, nontriviality, order and consumer transport must all be separately tested. For any linear fitted q-action, edit Z+(q A_g-q)V^T. It preserves the outside-chart component of the original history; an independently observable outside-chart residual can prevent exact counterfactual. Thus even correct projected action need not transport the whole future consumer. This is a residual-chart diagnostic, not full learned six-dimensional same-action or global gauge uniqueness.

G8b3b: verify R^3=I, J^2=N^2=I, JRJ=R^2, NR=RN, NJ=JN; compare chronological R J vs J R on natural heldout labels. Three interventions at actual post-prefix history using independently fitted delta rather than oracle label delta; unchanged last literal, actual source step and readout; compare natural transformed prefix continuation. Measure next entire history, all ordered IDs/signs, weights and sum of full distribution L1 across three heads; unedited and fixed norm-matched coordinate-permuted delta null. Exclude fixed residual cases; retain zero baseline and fixed states for chart/relation tests. Null not proof of gauge/feature-matched independence. 2592 nonfixed R, 2592 J, 2808 N interventions = 7992 per carrier. Output-action correctness remains separate, and no fitted neural learning claim. Runner now adds this audit at final fixed horizon. No GPU execution.

### Architectural antipodal control, not learned evidence

On paired state (h0,h1), pure branch swap sends (M,D)->(M,-D) while leaving input x fixed. Router on M is unchanged. If each selected branch has nonzero orientation score (or branch fallbacks coincide for that z), both execute the same branch-local map, including cap. Thus the pure swap commutes with the step; inactive pair identities commute too. Induct over certified guards. At a zero-score opposed chart, branch-specific fallbacks can break pure swap; not a universal S3 or branch symmetry of the model. For current seeded actual untrained paired learner width6 seed41, measured all 11664 endpoint histories under initial residual negation: 11658 bitwise branch-swap matches; max absolute discrepancy 2.7755575615628914e-17. This accounts for near-exact N^2 and centrality in fitted paired residual chart: natural contrast odd component under branch swap, symmetric baseline, odd linear residual regression, induced N=-I on the fitted odd chart (ideal symmetry). No proof of learned translation/reflection or full action. This replaces interpreting antipodal-only success as generic learning.

### Executed G8b3 controls

Local scripts/check_x369_hidden_generators.py imported actual original source and existing Toy exact action carrier, no trained parameters. Each carrier evaluated on 11664 histories, train-only fit, 6804 public composition rows and 7992 nonfixed post-prefix interventions. Results:

| Metric | Exact synthetic action carrier | Untrained original | Untrained paired original |
|---|---:|---:|---:|
| residual chart rank | 3 | 3 | 3 |
| R^3 defect (Frobenius) | 2.04e-14 | 1.5511 | 1.4582 |
| J^2 defect | 1.42e-13 | 1.3574 | 1.2037 |
| N^2 defect | 1.32e-13 | 1.2287 | 3.27e-14 |
| JRJ-R^2 defect | 1.53e-13 | .46884 | 1.08431 |
| t4 residual decoding RMSE | 2.87e-14 | .79294 | .78211 |
| R fitted intervention mean L1 | 5.48e-14 | .36963 | .25714 |
| J fitted intervention mean L1 | 1.70e-13 | .36326 | .25022 |
| N fitted intervention mean L1 | 1.52e-13 | .37898 | .21118 |

All synthetic group defects <1e-12, noncommutator norm sqrt(6), all intervention errors <1e-12. Exact synthetic positive and actual untrained negative: paired improvement over unedited or norm-null does not meet same-action criterion; R/J fail relations and signed decoding is poor. Untrained architecture only, no learning or performance inference. All metrics from actual local execution (FP64, Torch CPU). scripts/audit_x369_hidden_generators.py, check_x369_hidden_generators.py and check_x369_original_source_acceptance.py committed. New runner final audit hook compiled, GPU execution unpaid. No F5, original language corpus, CI, kernel or external PR edits.

### Exact remaining phase / deliverable / producer-consumer ledger

| Gate | Exact deliverable / producer -> consumer | Status |
|---|---|---|
| G4e5/G5b5c | originalStepWitnessTransport / actual original source paired and centered GELU acceptance | 24 adapter + 72 paired + 81 centered CPU constructed checks PASS; full app/native/fitted claims separate |
| G7c2a-b | same-base residual and shared STE | prior proof, 32 actual original-source one-step forward/gradient arms PASS |
| G7c2c | actual fixed-horizon native fair learning: original and paired, C3/S3/full/entry and null, measured storage/work and matched initial streams; dense ternary whole-weight separate | runner implemented; GPU trained evidence OPEN |
| G8b1-2 | frozen full chart and oracle-delta causal transport | prior controls PASS; trained evidence OPEN |
| G8b3a | naturalResidualGeneratorFit, independentGroupRelations, orderedComposition, rank-vs-closure obstruction / task labels and actual histories -> latent action | implemented and executed synthetic positive/untrained actual negative |
| G8b3b | fittedGeneratorConsumerIntervention / frozen maps -> same source next literal, complete history and output | 7992 per carrier executed; trained source and calibrated acceptance OPEN |
| G8b3c | actual learned full mean/residual same-action: independently fit joint action, assess unseen order and time, signed task accuracy and source-specific interventions/guards; no promotion from architectural N | OPEN |
| G5a3 | fittedReachableMarkedQuotient / #1550 and actual source traces -> minimal codec | OPEN |
| G6c | calibrated route/sign/cap/precision guards and time-indexed error / actual source -> native transport | OPEN |
| G9 | native AMD packed implementation, resource and formal acceptance | OPEN |

Next mathematical cut: derive and test *consumer-sensitive residual closure* with natural contrasts and successor observers, including explicit shared mean perturbations and actual zero/guard boundaries. Native trained receipts remain critical; do not repeat fixture-only parity or present architectural negation as discovery. Maintain original attribution, scope, and independent task/lockbox boundaries.

---

## 24 November — G5a3 consumer-sensitive closure beyond a learned chart

Geometry-forward; retain actual DC, nonzero zero-DC states, signed residual, original shared guard occurrences and time/model revision indices. Owner LeechTransformer #5. Original Leech-LILA: Anatolii Kornienko (DOI above); DASHI extension. Reuse G8b3 natural generator fit, G5a2 quadratic/future visibility and DASHI #1550 future-trace and switched-affine contracts, merged #1526, #1291 indexed geometry and #1548/#119 ordered guard donor. No external PR edits. Starting head 479ab9ca5b77d9669844d5e120272c16e3c4326a; original HistoryOrbitMoE blob a2a250462ce8e663591cd8a8f784fad9bde1b15d unchanged.

### G5a3a exact factorial separation of chart and residual errors

For natural same-prefix/guard/mean histories z=B+q V^T+r, target z_g=B+q_g V^T+r_g, V orthonormal columns, r V=0, row convention, previous fitted generator edit is z00=B+q A_g V^T+r. Define z10=B+q_g V^T+r, z01=B+q A_g V^T+r_g, z11=z_g. Then z00-z11=(q A_g-q_g)V^T+(r-r_g), orthogonal squared norms add exactly. For ANY vector consumer F on a declared continuation, F11-F00=(F10-F00)+(F01-F00)+(F11-F10-F01+F00). Mixed term need not vanish. In particular z=(1,2,3,4), target=(2,4,5,7), V first two axes, A=I, F(z)=z1*z3 gives outputs 3,6,5,10, chart effect 3, residual effect 2, interaction 2. Oracle chart and residual corrections use natural counterfactual and are diagnostic, not deployable codec. A successful fitted edit alone may be cancellation, and a failed edit alone cannot identify the missing component. Retain consumer-specific observations, not just mean squared history error.

### G5a3b exact finite-horizon closure and minimal linear extension

For fixed parameters and an admissible labelled input domain, define C(z) retained chart plus any needed context. One-step chart descent iff for all equal-code states and all admissible u, observations O_u match and C(T_u z) matches (also equal admissibility if state-dependent). Necessity is immediate from a deterministic code transition/observer; sufficiency defines both on inhabited fibres, then induction gives all finite traces. Future-only equivalence may be coarser than requiring chart preservation; do not include full history as an observer unless explicitly required. Finite domain closed under admissible successors: partition initially by C, refine each block by its per-label observation, admissibility and successor block; finite termination, greatest stable refinement of chart fibres, coarsest exact refinement preserving chart. For general nonlinear continuous state no finite termination claim. Group relations/equivariance do not imply chart closure.

Exogenous affine modes T_u z=A_u z+b_u, O_u z=C_u z+d_u on full R^n: to retain Pz AND all future observations, take W0=row(P)+sum row(C_u); W_{k+1}=W_k+sum W_k A_u. Fixed space W* is minimal common right-invariant observable row space containing W0. Any exact linear extension of P sufficient for all future traces has row space containing W*: iterated P and C after every admissible word are linear observations up to affine offsets. Conversely row basis of W* gives explicit affine reduced transitions, observations and chart recovery. Additional independent coordinates required = dim W* - rank P. Under continuous Euclidean codecs with chart recovery, same dimension lower bound on a linear complement by invariance of domain, using a left inverse from finitely many observations. At most n-rank W0 strict increases. Restricted reachable domain, conditional modes, nonlinear guard cells, training dynamics and finite precision remain separate. This is the prior switched-affine theorem used as a *chart completion*, not a new universal nonlinear quotient.

### G5a3c source-legal exact delayed visibility despite perfect geometry

Construct paired coherent HistoryOrbitMoE, d_model=6, E=4, K=2, expert_width=6, no cap, fixed strict logits (2,1,0,-1) (zero baseline and C3 route weights), orientation reads sum of activation, fixed x=(1,1,1,1,1,1) so both branch scores positive. Anchor expert FFN reads first history triad h_A by paired GELU hidden coordinates (h_A,-h_A) and final subtraction GELU(t)-GELU(-t)=t, emits (h_A,0). Anchor transition ignores x,y and swaps two history triads T=[[0,I],[I,0]]. Other selected expert output zero, transitions identity, inactive unchanged. Anchor paired history branches +D,-D with D=(d,r) in R^6; all other histories zero. Mean remains zero. Selected weight alpha=e/(e+1)>0. Signed aggregate readout after first step is alpha*d (plus zero output triad), next step alpha*r. All trainable matrix entries can be signed ternary, scalar expert biases as specified. Exact ideal-real construction, not a fitted/zero-init reachability or source execution claim.

P=[I,0] chart, T swaps triads. For signed residual representation g in {R,-J,-I}, simultaneous action diag(g,g) commutes with T, P intertwines; therefore full representation and chart have exact group action. Under this residual-only transformation, fixed DC activation and positive orientation implement branch-independent maps, not a claim of full joint input/mean S3 symmetry. Yet same chart d and different r give identical current signed output and different next signed output. Explicit d=(1,1,1), r=0 vs (1,-1,0) separates an actual zero residual from an inhabited zero-DC phase residual. rank(P)=3, rank([P;PT])=6, T^2=I. On 729 signed ternary six-vectors, depth-one trace has 27 classes, depth-two 729. Minimum continuous retained dimension six on arbitrary initial domain; identity-transition control retains rank3 at all depths. Thus neither perfect group laws nor immediate consumer success justify dropping outside-chart directions. Full source step would expose other observables only if included; fixed IDs/signs/weights here add no distinctions. Current full state dimension not claimed six generally.

### G5a3d finite consumer audit and scope

New scripts/audit_x369_consumer_closure.py uses existing frozen natural-generator capture/fitting, computes four arms, evaluates all 4^h appended literal words for each selected public composition prefix, original guard values reused, h=1,2 by default (up to 3 allowed), deterministic bounded sample per generator (128 default). Each depth measures final-step full probability distribution, final history norm, ordered IDs, selected signs and mixture weights; evaluating every prefix word separately covers those terminal observations at each tested depth. Reports mixed probability effect L1, no universal pass threshold. Does not include raw expert output as separate metric or certify same action, exact closure, all possible inputs, intermediate route margins, or calibrated FP32. Natural target is the fully transformed same-prefix history, not an output label leakage into training. Optional --consumer-closure --closure-pairs flags on existing GPU-first runner; off by default to avoid multiplying audit cost. No GPU or actual original-model execution in this cut. Earlier original-source CPU receipts retained.

Actual local execution in fresh /mnt/data/lila_closure (prior isolated package not present this turn): SymPy 1.14.0, PyTorch 2.10.0+cpu. scripts/check_x369_consumer_closure.py exact rank 3->6, 729 ternary states, 27->729 trace classes, three generator commutations, identity control, factorial correction and mixed effect, FP64 paired GELU readout on 729 rows PASS. Only helper algebra tested, full model audit and runner hook not executed. No new source import, fitted model, training, GPU, CI or kernel. Preserve original attribution, all earlier negative evidence and untouched final reserve.

| Gate | Exact deliverable / producer -> consumer | Status |
|---|---|---|
| G5a3a | factorialChartResidualWeld and nonlinear mixed consumer term / G8b3 natural chart -> four-arm intervention | proved, local regression PASS |
| G5a3b | stableChartRefinement and minimalAffineChartCompletion / #1550 trace and switched row closure -> retained codec | proved under stated domains |
| G5a3c | equivariantDelayedResidualSeparation / original FFN/transition ideal-real legal construction -> depth-two signed observer | exact math and local controls PASS; actual source construction not run |
| G5a3d | boundedFutureConsumerAudit / original ordered task, natural generator fit -> four-arm indexed source runner | implemented, helper tested, end-to-end source execution open |
| G5a3e | fittedReachableConsumerClosure: actual reachable fibres, admissibility, successor closure, exact guard contracts or separating word, source-specific minimal residual; distinguish current chart vs true quotient | open |
| G8b3c | learned full joint action and actual consumer sufficiency; independently fitted mean and residual, heldout composition | open |
| G7c2c/G6c/G9 | native fair training, calibrated sign/route/cap precision, packed work and formal acceptance | open |

Next run source audit on actual trained states and derive consumer-sensitive residual closure beyond the linear model: use reachable guard cells, exact separating words, retained null and DC, and time-indexed error. No claim of compression solely from a low-rank chart.

---

## 24 November — G5a3e nonlinear chart refinement, guard separation and local rank

Geometry-forward: retain zero mode, signed residual, original guard occurrences and indexed time/model revisions. Original Leech-LILA Anatolii Kornienko (DOI above); DASHI extension. Owner Leech #5, G5a3 previous factorial and affine completion, DASHI #1550 future trace, #1526 merged, #1291 indexed geometry, #1548/#119 ordered guard inspiration. Starting head 744c2df10af73155cc5e9ece9047b6b2daf6be98. No external PR edits or original source changes.

### G5a3e1 stable nonlinear chart refinement with admissibility and shortest witnesses

Fix model revision and declared deterministic state domain S, including context/guard assignment if it determines available inputs or transition. Start partition P0 by retained chart C. For each labelled input u, record either inadmissible or (O_u(z), T_u(z)). Assume S finite and every admitted successor in S. Define P_(k+1) by old block plus ordered tuple of admissibility, observation, successor block under every label. Inductively equality in P_k iff all chart values on visited states, admissibility and observations on labelled words through length k agree (including initial chart); each strict refinement adds a block, so at most |S|-|P0| strict refinements. Stable partition is coarsest exact chart-preserving quotient, since any stable chart refinement respects every P_k. Define reduced transition and observer on inhabited blocks, decode chart only, not raw state. Retain all exposed signed observables in O; do not substitute route-only projection for entire observation. State-dependent admissibility must be observable as such; if a word is allowed only from one state it separates them even with identical numerical outputs. A breadth-first product-state search with equal observations/admissibility until first mismatch returns a shortest separating word or none on a fully specified finite graph. Revisiting a state pair cannot improve shortest path. Checker fails closed on any successor outside S. No claim of finite termination or exact partition for continuous GELU network reachable sets from a fitted seed. Arbitrary finite samples with unsampled successors do not meet hypothesis.

### G5a3e2 exact support vs signed nonlinear source-legal control

Construct original single-history coherent HistoryOrbitMoE d_model=6,E=4,K=2,width=6, fixed logits (2,1,0,-1) by zero baseline/residual matrices, fixed expert biases, no cap. Anchor history h=(d,r) consists of two triads, others zero. Input x=(s,s,s,1,1,1), s=0 (center) or 1 (shift); orientation reads last activation triad, all selected scores 6 positive. Anchor hidden FFN preactivation (d+s*ones, -(d+s*ones)), output (GELU(d+s)+GELU(-d-s),0), transition T swaps the two history triads, ignores x and output. Other selected expert emits zero, transition identity; inactive identity. All matrix entries signed ternary, routing biases as above. Fixed alpha=e/(e+1)>0 multiplies readout. E(t)=t erf(t/sqrt2)/2 even, strictly increasing for t>=0 since E'(t)=erf(t/sqrt2)/2+t phi(t)>0 for t>0. Centered output 2alpha E(d) observes coordinate magnitudes (on ternary states equivalent to d_i^2); shifted output 2alpha E(d+1) is injective on d_i in {-1,0,1}. This is a SINGLE-history even readout implemented by paired hidden GELU units, not mean of two recurrent branches with a common offset (that mean remains even in the branch difference). Strict positive orientation and constant routing keep selected IDs, signs, weights identical; choose output as exposed observation, not full raw history. Initial domain all 729 ternary h=(d,r), closed under swap, arbitrary initial seed allowed; not reachability from zero. Chart C=(d_i^2). With center only, stable classes 8->64, exact code (d^2,r^2), sign globally invisible under declared inputs. With center and shift, classes 8->216->729, exact full signed h necessary on finite domain. For a=(1,0,0,1,-1,0), b=(1,0,0,-1,1,0), same initial chart, same one-step observation and successor chart for both modes, shortest witness center,shift (also shift,shift). Hidden nonzero zero-DC phase must be retained with sign under shifted alphabet. A separate abstract graph with center/hold, swap and shift/hold has same class counts and shortest swap,shift. Nonlinear observations, affine swap transition; no claim that arbitrary nonlinear transition is solved. Finite code lower bound log2(class count) bits rounded up (64=6 bits, 729=10 bits), not parameter or native packed-memory claim. Exact source-legal math, not executed actual source model this turn.

### G5a3e3 guards and derivative blindness at null

A separate abstract guard/hold mode observes r0>0; support chart classes 8->16. An admissibility-only variant allows a constant-observation identity mode iff r0>0 and also yields 8->16. This illustrates distinct guard labels and forbidden inputs, not a claim that original HistoryOrbitMoE rejects such states. Actual source sign observer can separate same retained chart by orientation weight selecting first hidden r coordinate, score=2r0 and r0=+/-1, fixed nonzero activation, fixed strict route. For score zero, original first-nonzero fallback is separate and must be included in exact guard cells. If score margin is positive, perturbation of h by delta changes score by at most 2||w_h|| ||delta||. At score zero cannot infer stable sign merely from small Euclidean residual. Earlier cap kink and strict TopK guards retained.

For smooth finite observation map F collecting chart and all observations at chosen words, if DF(z0) has rank r in open domain, choose r input directions and r output coordinates with invertible minor, restrict to corresponding r-dimensional affine slice through z0, inverse function theorem gives locally injective F on slice. Any continuous exact codec preserving F is injective on slice, so dimension >=r by invariance of domain. Rank is a local lower bound, not global sufficiency. If a finite family of smooth future scalar observations has constant maximal rank r on a neighborhood and every other declared future observation has differential in their span there, choose r independent scalar observations as local submersion coordinates; each other scalar observation has zero derivative in fibre directions and is constant on sufficiently small connected local fibres. This gives local factorization of those observations, not automatically a global invariant chart or successor closure across guard cells. For arbitrary finite families rank can jump, and at null the derivative may vanish despite nontrivial exact observability: F=(d_i^2,r_i^2) has derivative rank0 at zero, rank6 at generic nonzero coordinates; shifted square probe H=(r_i+1)^2 together with r_i^2 gives rank3 on r subspace everywhere. GELU E'(0)=0, E'(1)>0. On a smooth convex chart fibre, exact invariance follows if every declared observer and coded successor annihilates all tangent directions throughout the fibre; for disconnected fibres or crossing piecewise guard strata must additionally compare representatives/one-sided boundary and guard values. A Jacobian at one sample cannot pay that gate. No global fitted quotient inference.

### Executed checks and remaining work

Local fresh /mnt/data/lila_nonlinear, Python/SymPy 1.14.0. Executed scripts/check_x369_nonlinear_closure.py, git blob d1d726f15b7fd05aebd6df5b1a6fcb84e0d67b20: 729 states, exact support refinement 8->64, signed 8->216->729, abstract guard/admissibility 8->16, shortest witness, nonclosed-domain rejection, source-legal label abstraction with same refinements, symbolic nonlinear Jacobian and GELU derivative checks PASS. The strict monotonic proof above justifies label equality; scalar evaluation is supplementary. No original-source reimport, actual fitted checkpoint, native GPU, Agda or CI in this cut. Earlier original-source CPU receipts remain valid. No claims about trained LILA or compression cost. Prior optional factorial audit still not executed end-to-end on a model this turn.

| Gate | Exact deliverable / producer -> consumer | Status |
|---|---|---|
| G5a3a-d | prior factorial, affine chart completion, delayed visibility and bounded auditor | prior scoped results |
| G5a3e1 | stableFiniteChartRefinement, admissibility and shortestSeparatingWord / #1550 future trace -> finite reachable graph | proved and executed |
| G5a3e2 | shiftedEvenProbeRefinement / source-legal GELU observer and swap -> support/signed quotient | proved on declared ternary seed domain; exact label checks PASS |
| G5a3e3 | localObservationRankLowerBound and fibrewise derivative sufficiency with boundary obligations / nonlinear source cells -> lower bound/conditional quotient | proved scoped; symbolic controls PASS |
| G5a3f | actualFittedReachableDomain: supply reachable invariant carrier or exact guarded fibre/successor proof, all actual declared observers, boundary/zero/DC cases, shortest separating words or constructive reduced transition; measure residual retained | OPEN |
| G5a3d/G8b3c | end-to-end factorial audit on actual trained model, full learned joint action and consumer closure | OPEN |
| G7c2c/G6c/G9 | native fair training, calibrated guards/precision, packed work and formal acceptance | OPEN |

Next: no more promotion from finite abstract charts to fitted geometry; use trained original/paired checkpoints and explicit source guards, run actual factorial audit and certify a reachable domain or produce shortest concrete separating continuations. Retain exact sign and zero mode throughout.

---

## 24 November — G5a3f checkpoint-bound bounded reachable-code witnesses

Geometry-forward; retain actual zero/DC, signed residual, original shared guard occurrences and time/model revision indices. Original Leech-LILA: Anatolii Kornienko (DOI above), DASHI extension. Owner Leech #5. Reuse G5a3a–e, G8b3 natural generator chart, DASHI #1550 future trace, merged #1526, #1291 indexed geometry, #1548/#119 ordered guard donor. Source owner starting head 478ef5af5d018bd159a0c9b4f1ab13e293454cc0. No external PR edits.

### Existing fitted owners and concrete missing inputs

Inspected scripts/replay_x369_r4_fitted_delta.py (blob 810b15c852c53e3d1490b4c5d8055bb9ce76bb5f) and scripts/capture_x369_r4_original_slr_state.py (4e64edd68915acd1ea55508f71e1953a45c37d65). These are already strict native-SLR development checkpoint restoration and complete-history capture owners, not substitutes for a trained ordered-action task checkpoint. No actual fitted weight bytes present in current local runtime; library search for checkpoint returned no hits, which is not proof of absence elsewhere. Local torch 2.10.0+cpu, no CUDA/HIP device. Do not fabricate trained state, copy unrelated R4 weights into new task architecture, run training on CPU against native policy, or read reserved data. Earlier scoped original-source receipts retained. Actual ordered runner previously emitted metrics without persisting weights.

### G5a3f1 checkpoint persistence and independent replay

New scripts/x369_ordered_checkpoint.py stores final state_dict including persistent buffers on CPU, schema, exact arm, architecture, seed, horizon, batch size, runtime and explicit selected source file SHA256 map. Exclusive create prevents overwrite; checkpoint bytes SHA256 emitted in receipt. GPU-first runner optional --checkpoint-dir writes after fixed horizon and before expensive final audits. Save only final state, no composition-based selection, not optimizer-resume checkpoint. scripts/replay_x369_ordered_checkpoint.py reads bytes once, verifies expected SHA and selected source file fingerprints, loads weights_only=True, checks schema/finite tensors, requires matching recorded torch/HIP/architecture and CUDA, constructs same single/paired fair arm, loads strict state_dict, and runs independent output-action audit plus new bounded natural-state code audit. Output receipt exclusive create. Metadata is not independently authenticated evidence of training, source manifest covers named modules rather than entire application/environment, same runtime metadata not a bitwise hardware proof. Replay does not compare historical logits to original metrics and no claim of same-device parity beyond new test; future add fixed probe receipt and validate complete provenance. No CPU fallback or F5 data. Optional persistence default off.

### G5a3f2 exact encoded collision and shortest bounded witness

Freeze train-only natural residual chart V from existing G8b3 fit; per natural prefix use actual same-context zero residual baseline B. Explicit machine codec c_delta(z)=round((z-B)V/delta), finite positive fixed delta, ties-to-even Torch rounding, finite safe range |bin|<2^53. Actual equal integer tuples are collisions, not approximate chart distances. Context includes original word, guard assignment and initial mean plus fixed checkpoint revision; contextual baseline is not a deployable standalone codec unless recoverable (can require prefix replay). Evaluate deterministic bounded sample of contexts from public composition prefixes, all 27 natural residuals in each chosen context. Default delta .125, max 8 contexts, all literal continuations over a,b,j,n through depth2 (1–3 allowed), repeated reads use same guard assignment. No heldout tuning. Compare raw expert aggregate/paired output, ordered IDs, signs, weights and consumer probabilities under identical continuation; no full raw history as mandatory observer. Observe terminal fields for each word, enumerating every shorter word separately. One identical observation-batch replay preflight, not universal deterministic hardware guarantee. Exact torch.equal on finite same-dtype tensors (not mathematical real equality or bitwise signed-zero distinction), no tolerance masquerading as equal code. For each collision pair record first separating word in increasing length, fields, original residuals/context; shortest only among declared alphabet and evaluated bounded horizon. A difference refutes exact finite-machine output sufficiency of this context-bound code on tested trace semantics (subject to deterministic repeatability); not a calibrated meaningful numerical defect or proof against unquantized projection. No separation/no collisions does not certify closure, no forward-invariant reachable domain claimed, no claim of minimal compression or global fitted quotient. Must distinguish state-space perturbations from naturally reachable pairs; this auditor compares only natural histories. Code fit can be full rank but no separating code collisions or singleton contexts, report counts rather than promotion. Future include source guard margins, exact replay of individual witness, magnitude/tolerance receipts, constructive reduced transition on invariant domain.

### Actually executed local helper checks

Local /mnt/data/lila_replay, torch 2.10.0+cpu. Executed bounded exact integer-code witness helper on 4 states, 2 code classes, 2 collision pairs; all 6 words through depth2, one pair separates only on swap,hold and one unseparated in scope; no-collision and nonfinite guards. Checkpoint helper tested on torch.nn.Linear: strict restore gives exactly equal output; overwrite, wrong SHA and changed source fingerprint rejected. Helper blobs local 25809c4c6e78f7b780bf327e48b1ca020fc3c8ea (reachable) and 786399aeed69ec0d485536e43567023d4a4d04cc (checkpoint). Combined committed checker reproduces both local checks; no original-model replay or actual learned data run this turn. New runner/replay integration source written, GPU execution open. No kernel or CI. Preserve attribution and all prior negative evidence.

| Gate | Exact remaining lemma / producer -> consumer | Status |
|---|---|---|
| G5a3a–e | prior exact closure/refinement and nonlinear dimension bounds | scoped proved |
| G5a3f1 | finalCheckpointIdentity and strict same-arm reconstruction / existing GPU runner -> independent replay | implemented; helper tests PASS, actual trained checkpoint replay open |
| G5a3f2 | exactBinCollision, boundedShortestWitness / frozen G8b3 chart and natural histories -> future observers | implemented; local helper PASS, actual source end-to-end open |
| G5a3f3 | fittedInvariantDomain or concrete certified separating continuation / actual trained states, guard margins, reachable context and successor closure -> exact or approximate deployable quotient | OPEN |
| G8b3c | full learned joint action, causal consumer closure | OPEN |
| G7c2c | native fair training and actual trained checkpoint | OPEN |
| G6c/G9 | calibrated precision, native packed cost, formal acceptance | OPEN |

Next native fixed-horizon run with final checkpoint, independent replay and concrete witness receipts; do not substitute more abstract tests for the missing fitted source. Reuse existing strict original R4 development restore/capture path for original language-model states if its actual artifacts become available, retaining distinct task/source indices.

---

## 24 November — consolidated native acceptance cut (G7c2c / G8b3c / G5a3f / G6c)

User direction: stop piecemealing; geometry-forward large cuts, original source, zero/DC, signed residual, indexed time/revision/guard geometry. Owner LeechTransformer #5, original Leech-LILA Anatolii Kornienko (DOI above), DASHI extension. Reuse existing typed ordered task and modular C3 owner, G5/G6/G7/G8, #1550 future quotient, merged #1526, #1291 indexed geometry, original topk observer and rational interval owner, existing native R4 fitted-delta restoration and source capture. No external PR edits, no F5 access, no CPU training or GitHub CI.

### Joint task and actual exact learned action control

The old initial mean e0 is fixed within each original context. Although its endpoint labels can span three mean coordinates across contexts, it cannot identify natural mean perturbations within one prefix/guard fibre. Extend to the invariant carrier M in {0,+/-e0,+/-e1,+/-e2}, D in {-1,0,1}^3. This retains actual null, signed means, all residual DC/zero-DC strata; not all possible mean vectors (e.g. phase-only mean is outside this particular orbit). Every fixed context now has 189 states. Existing literal register source gives 81,648 rows (34,020 train, 47,628 public composition), exactly seven times previous task; old fixed-mean task is a restriction. Independent closed-form label checks all PASS. Digest 186a5a7e234ca7c744919b8a40ddf9aaf48b35c2ab04a5ccfface7865b94b43a. On each elementary context the joint carrier covariance is diag(54I3,126I3); cross blocks vanish. Repeated unused guard values multiply by the same positive integer. Exact normal equations recover unrestricted 6x6 generator maps uniquely from elementary train labels. Ten maps fitted, both mean and residual independently; no supplied reflection character. R=diag(T,T), J=diag(J0,-J0), N=-I6 emerge. Chronological composition law supplied, not learned. All 81,648 rows recovered exactly, including all unseen word compositions. This pays the exact linear positive control, not neural learning or latent identification.

Joint natural chart subtracts actual same-prefix/guard (M,D)=(0,0) baseline, fit from predetermined train contexts, freeze rank-six subspace, fit R,J,N independently on natural joint pairs (no imposed group relations). Measure task/mean/signed/DC reconstruction, outside-chart residual, six relations, order defects and nontriviality on separate composition contexts. Actual downstream four-arm factorial edits of natural jointly transformed histories, identical next literal with original guard assignment. Both chart and residual correction diagnostics; oracle arms not deployable codecs. Existing residual-only and full affine audits retained on e0 restriction for comparability. Fixed joint chart is a strong hypothesis, not global latent uniqueness. Joint integer-bin closure audit tests natural states across different initial M,D in same context; default eight train/evaluation contexts and all 4^h appended words through h=2. Reuse bounded witness engine now vectorized over candidate pairs, replay each separating observation batch exactly, report shortest word and absolute field deltas. No collision/no separation not certification; context baseline recovery and invariant-domain proof unpaid.

### One integrated native campaign (not a series of handoffs)

scripts/run_x369_ordered_campaign.py drives existing GPU-first runner and independent replay per arm. Default 3 seeds (41,42,43), four frames (C3/S3/full/entry), two source families (single/paired), geometric/null and real/ternary = 96 arms, fixed 2000 steps, width12, batch128, configurable. Joint mean training enabled; same source base, minibatch stream, sign STE and no composition-based selection. Plan mode no execution. Actual mode requires gfx803 HIP; no CPU fallback. Immutable campaign manifest binds source fingerprints, runtime, arm set, horizon and audit budgets; validates source map each arm and checkpoint payload against manifest and training receipt; exclusive logs and checkpoint/replay receipts, skip only completed valid arms. Resume at completed-arm boundary, not interrupted optimizer step. Failed or unreceipted checkpoint fails closed; no overwriting. Output summary lists checkpoint and replay receipt hashes plus train/composition counts; does not admit scientific release. No native packed kernel or equal-work assertion: residual coefficient ternarity in real frames and dense base retained. Training runner records synchronized train-block wall time (incl host overhead, excludes public evaluation) and training peak allocated memory, parameter counts and buffer bytes; no hardware energy or isolated kernel timing claim.

Final checkpoint saved before deferred post-training audits; now binds task digest and exact fixed probe SHA over typed initial/full recurrent state, per-step expert output, IDs, signs, weights, and final logits. Strict reload on matching recorded native runtime and source manifests; reject if probe differs. This is a finite trace identity check, not whole-network equivalence or authentication of training metadata. Old checkpoint schema with older source map requires its original source revision. Replay consolidates full joint task accuracy, independent output action, e0 affine and residual charts, bounded factorial consumer closure, residual and joint code witnesses, exact joint-generator control, natural joint action, and precision guards. No new task or lockbox selection by evaluation results.

Precision guards reuse actual original source topk dispatcher observer (observe_x369_r4_original_topk.py, original model bytes unchanged); source selected IDs/weights must match actual step. Same selected branch-local orientation score arithmetic and nonzero sign replay; explicit zero score count, cap preprojection distance and boundary when cap active. Separate independently advanced FP32 and FP64 paths on same native device, compare every visited step, including logits, ordered selection, signs, history and output probability. Measured reference gap >2*observed max score delta suffices for those recorded two vectors; strict score sign comparison analogous. FP64 is NOT exact real reference, no rigorous rounding enclosure or calibrated perturbation transport; reuse existing interval certificate only after certified errors available. Cap absent in ordered default architecture; no cap-boundary numerical claim without cap cases. Record native differences and guard deficits, do not infer global stability.

### Executed local acceptance and exact blocker

Local /mnt/data/lila_rollup Python, PyTorch 2.10.0+cpu. Byte-verified actual original label source c0d42b018663f53aeb1abed42b505cc4de84972d; local new joint, guard, reachable and campaign files match Git blobs. Executed check_x369_native_campaign_contracts.py: 81,648 independent label checks and ten exact joint learned maps, rank-six synthetic affine chart over 2268 train and 2268 composition rows (max relation defect 4.923960562915293e-15; collapse rejected), 17,766 joint-sized exact code pair software checks, shortest delayed witness and repeat rejection, 6561 measured ordered-margin test cases, 96 unique campaign arms, unreceipted checkpoint and incomplete replay rejection. These are exact/task/software controls, not fitted neural or native GPU training evidence. New full source-dependent training/replay/guard/causal audit integration not executed in this runtime. Prior original-source CPU acceptance retained; no fresh original model import. GitHub full PR tree (nontruncated) at afa3de6e had historical JSON receipts but no tracked fitted pt/pth/safetensors/ckpt/npz weights; release list empty. Previous Library search no matching artifact; current runtime has no checkpoint bytes, no GPU device, torch.cuda.is_available=False, HIP None. No access to user's workstation execution. Existing original R4 development replay/capture remain separate from ordered-task model. No scientific release, native packed performance, kernel or universal quotient claim.

### Consolidated completion gates (do not subdivide into more abstract demonstrations)

| Integrated gate | Current disposition |
|---|---|
| Independent joint task, full linear generator identification and nontrivial order controls | exact label/control PASS |
| Native campaign, final state/task/probe identity, complete independent replay | implemented, local contracts PASS; actual native execution blocked |
| G7c2c fitted fair learning across matched arms and seeds | OPEN: actual trained receipts |
| G8b3c full learned joint action and causal transport | implemented; OPEN: fitted histories/relations/consumer evidence |
| G5a3f fitted reachable quotient | implemented bounded repeatable joint/residual witnesses; OPEN: actual receipts, invariant carrier or guarded fibre/successor proof, contextual baseline cost |
| G6c calibrated guard/precision transport | empirical native diagnostic implemented; OPEN: measured and rigorous error enclosures |
| G9 native packed execution, cost and formal acceptance | OPEN: actual native implementation/measurements, no packed claim |

Single native entrypoint: python -m scripts.run_x369_ordered_campaign --output <new-output-directory> ; --plan prints default 96-arm manifest, optional reduced scope must be declared in immutable manifest before running. Do not execute on CI or CPU. No intermediate user handoff. Next substantive cut must consume native checkpoint/receipt set or actual existing R4 development artifacts, not more isolated abstract toy examples. Original source attribution, negative evidence and independent reserve unchanged.
