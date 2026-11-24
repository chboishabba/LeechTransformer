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
