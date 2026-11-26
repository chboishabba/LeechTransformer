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

---

## 25 November — latest PR donor audit; native-first acceptance unchanged

Live Leech #5 starting head 22960e221ace98846315f69cabc99cc9cdd20dc2. Verified actual campaign line 63 requires torch.cuda.is_available(), torch.version.hip and gfx803; every training child passes --require-gfx803, uses CUDA namespace on ROCm, FP32 device-resident model/batches/backward/AdamW, and independent replay uses matching recorded HIP/device. No CPU training fallback. Host label production, batch selection, basis construction, checkpoint serialization, latent chart fitting and several audit reductions remain CPU; current fair residual expands dense effective matrices, not a packed native trit kernel. No native job launched or observed on the user's workstation in this turn. Prior actual CPU controls and native execution blocker unchanged.

Latest source audit (25 Nov, Brisbane time); read pinned donor roadmaps or checker code, not just PR titles. Keep original authors and source scopes; proposed adaptations below are DASHI/LILA research targets, not implemented new neural receipts.

| Donor / pinned head | Reusable exact result | LILA target / boundary |
|---|---|---|
| DASHI #1291 65950c7fe941a817464ea647cbfc49c58f7db6ab, IndexedResidualCube369NineDOFMaxCut20261120.md | typed Singer two-spoke path (416 paths + tagged origin); 312 moving singleton fibres, 26 returning four-way fibres, two extra bits on returns; all six H05 faces/routes checked by finite arithmetic; 176/416 translated paths change signed per-step trace | after native receipts, test endpoint-only versus independently labelled history-sensitive consumers, indexed chart/actual execution/transport faces. Nonempty zero-net command is not null; retain ordered marks only if declared future consumer demands them. No claim that finite-field wheel is a learned real LILA symmetry. |
| DASHI #1552 4e9049ab899a9beee76b493698cb647cda116c48, NullTriangleModularJOrderedUltrametricCube20261125.md N2c | source carry identity g L_r(z)=L_red(gr)(gz+c_g(r)); carry cocycle, C12->C4 with C3 kernel; finite controls and algebra separate from analytic J | for quantized latent codecs record chart index, quantization residual and transported carry; test encode/action/decode and both action orders under actual consumer. Real floating chart is not a p-adic field; apply only an independently justified rounding identity, including tie rule. Keep zero vs inhabited null and scale/refinement indices. |
| DASHI #1548 87c46ab1c0a4ce7910ddcbf16bcce9aa62d37d8a, signed isolation checker; #1549 e8bd35e56c6f0291f5f0d9a4bf1788b79cc14962 R226–228; Lean #119 a2d22ac35698bd31c8a3e6dd200086999d7fc742 / #120 31ce4fe5099eb261bc6b24a3f7d943346cda0414 mirrors | signed Singer lift Q and I Q=24 I_26, exact dyadic recovery with three guard bits; arbitrary sparse source isolation conditional; 4^k returning histories when complete history is the observer | compare exact precision-costed signed codec and matched endpoint-only negative control; distinguish storage, executed work and side-information. No SAT lower bound, no universal three-bit bound for real GELU/ROCm. |
| DASHI #1545 35a2ac7de201df469591139925d4ad9a4bdbecae, monster3b_actual_projective_sym2_intertwiner.py | simultaneous operator equations on actual labelled D/E blocks, invertibility and explicit rejection/inconclusive boundary; modular and ordinary claims distinct | independently solve common intertwiner for all fitted R/J/N on identified joint chart (not separately conjugating each generator); check full rank, conditioning, heldout natural pairs, causal consumer and outside-chart residual. Projective Sym2/Monster source remains distinct. |
| DASHI #1547 8ccbd62de11d114306a95d80e3eaa5aa32c31964, KSW668F8GeometryWeylRoadmapMaxCut20261117.md | source-native spectral lift, receiver-sensitive signed window, stage-clocked conditional energy/Gronwall argument, numerical acceptance separated from analytic sufficiency | use per-step measured/soundly bounded error and actual route/orientation guards to propagate consumer error through original recurrent times; no borrowing F8 stability constants or PDE norm as LILA certificate. |
| DASHI #1550 0cc1079f3e20075bc9fb15930b0e7767b98d8072 | same immediate observer but identity vs erasing transition rank9 vs12, original tagged restricted source | controlling future-trace theorem and G5a3f fitted quotient; use actual checkpoint and declared consumers, not another abstract rank example. |

Concrete next donor-driven welds AFTER actual native checkpoint receipt, not more task expansion before the run: (i) common gauge equation U rho_g=A_g U on the identified joint chart for all three labelled generators, with rank/conditioning and actual natural pairs; (ii) retain signed quantizer residual r=z-delta*c and evaluate c'=round((A(delta*c+r)+b)/delta) under actual tie policy; do not assume c' depends only on c unless fibre descent is paid; (iii) endpoint/history observer choice and indexed commuting-face replay, with no automatic history storage lower bound for endpoint-only task. Existing 96-arm manifest stays unchanged; no additional empirical or formal acceptance from this donor audit. Prior native execution, fitted action/quotient and calibrated error gates remain open. No external branch edits, CI, new source/label changes or reserved data reads. Original Anatolii Kornienko credit; Singer/Griess/Sah/Conway/JMD source contributions per #1291; classical local-field/representation theory prior art; DASHI extension only.

---

## 25 November — user-reported native L6 result and next joint learning decision (CURRENT)

Evidence origin: user-supplied workstation-agent report, native Radeon RX580 gfx803, Schaka101 Python, FP32, detached base 6ef7d59a2e1cf968fc98b1bd67d817e14d209c98, scripts.run_x369_mod3_maxcut_gpu, 3 seeds (41,42,43), six arms, 2000 fixed steps, report every 100. Output reported at x369_mod3_maxcut_3seed.jsonl on workstation. Agent report explicitly describes a local runner edit before execution; exact local diff, raw JSONL, model state hashes, environment receipt and complete trajectory bytes not supplied here. Base commit is not proof of exact executed runner bytes. Evidence below transcribes the supplied final table, NOT an independent rerun or verification of raw artifacts. Default task is the older nine-point modular addition (a,b)->a+b mod3, p22 held out; not the expanded signed ordered-action task.

| Arm | seed41 train / nine / heldout NLL | seed42 train / nine / heldout NLL | seed43 train / nine / heldout NLL |
|---|---|---|---|
| original | .375 / .333 / 10.8833 | 1 / .889 / 24.5387 | 1 / .889 / 28.8453 |
| shared_head | .375 / .333 / 10.9493 | 1 / .889 / 24.5615 | 1 / .889 / 28.8573 |
| two_dc | .375 / .333 / 10.5930 | 1 / .889 / 24.3807 | 1 / .889 / 28.9122 |
| differential_vjp | .375 / .333 / 10.8177 | 1 / .889 / 24.6243 | 1 / .889 / 28.8385 |
| additive_linear | .500 / .444 / 18.5223 | .500 / .444 / 18.9443 | .375 / .333 / 19.8255 |
| pair_mlp | 1 / .889 / 20.5684 | 1 / .889 / 19.2860 | 1 / .889 / 20.7738 |

All 18 reportedly complete; 8/12 R4 arms train-fit at first sampled step 100 (two of three matched seeds, not eight independent draws), 3/3 pair MLP train-fit (first sampled steps 1,100,1), 0/3 additive train-fit. 11/18 training fits, 0/18 sampled joint fits. No heldout p22 correct at final horizon. Nonlearned Fourier positive control reported full correct, zero hard action defects, heldout NLL ~3.05e-8. R4 matched training classifications, heldout NLL spread seed41 .3563, seed42 .2436, seed43 .0737 nats. Earlier L2/L4 parity and performance evidence remains separately scoped. This supports similar observed learning behaviour under these configurations, not identity of optimizer trajectories, universal drop-in equivalence, intrinsic impossibility of unconstrained learning, or proof that inductive bias alone explains failure. Seed41 plateau mechanism not identified; 2000 steps/3 seeds not a universal grokking no-go. Final high NLL and train fit are memorization-consistent, not proof of a particular internal representation.

### Actual source readback and exact discriminators

Original committed run_x369_mod3_maxcut_gpu.py blob 96e3656bf65a087fc800cfd981ac8a3c2bf116d1: original/shared/two_dc/VJP use copied initialization with fingerprint, full-batch CE on eight labels, seed-matched selectors, fixed horizon, additive/MLP controls separate initialization, heldout only evaluation, no checkpoint save. Original writer emitted literal backslash-n; repaired in 012294aca5b02674b28f0a817720de24535d3308, also records runner SHA256 and actual architecture. This is a source-confirmed bug repair, not a claim that the local edit was identical or that the amended runner was executed.

Exact additive control obstruction: logits L_c(a,b)=u_c(a)+v_c(b)+k_c. For a=0,1 and all b, strict correct preference for c=b in row0 versus c=b+1, and opposite preference in row1, imply d_(b+1)>d_b where d_c=u_c(1)-u_c(0). Summing the three strict inequalities is impossible. Even with deterministic lowest-index argmax ties, all three weak inequalities would force equality and at least one pair of opposing strict preferences is required by tie ordering. Hence additive logits cannot fit all eight, not just an empirical optimization observation; this does not exclude nonlinear recurrent/MLP capacity.

Exact task-action conditional: L(a+1,b)=P L(a,b) (or corresponding probability action) transports class shifts; training correctness at (0,b) plus exact equivariance and strict unique top class implies all nine correct, since all b have a training representative. The law/constraint is supplied in this statement; no claim that unconstrained learner necessarily discovers it. Train-only action edges can measure/regularize geometry without heldout labels; heldout edges are evaluation only. Hard orbit canonicalization is a distinct, explicitly imposed geometry control, not evidence of emergent latent recognition. C3/S3 fair residual frames in the joint campaign modify trainable updates around a frozen arbitrary dense base; they do NOT guarantee full layer or model equivariance, since base, embedding, router, biases, nonlinear orientation and readout need their own intertwining constraints. Therefore treat the existing 96-arm as a matched inductive-bias test, not as a theorem of task generalization. If joint native results fail, compare train-only consistency, true equivariant architecture with charged geometry, and independently fitted common intertwiner (DASHI #1545); do not select hyperparameters by public heldout success then call it independent.

### Consolidated next native acceptance (do not repeat L6 or widen first manifest)

Run existing scripts.run_x369_ordered_campaign on gfx803 Schaka101 runtime: 96 arms, default width12, steps2000, batch128; initial joint mean orbit plus signed residual, ordered recurrence, final checkpoint and independent joint/causal/closure/precision receipts. Plan inspection is not execution. User report confirms old mod3 experiment only; 0/96 joint runs evidenced at this point. Preserve raw L6 JSONL and exact local patch/hash separately, do not overwrite. Training on GPU required; symbolic CPU reference allowed. No CI. Existing campaign is independent of the legacy writer repair; no change to its experiment configuration. Prior donors #1291 conditional history, #1552 carry, #1545 simultaneous Hom, #1550 future consumer retained. Future L6 structured arms require separately frozen configuration and train-only label access, not another substitute for actual joint checkpoints. Native packed kernel and rigorous error enclosures still unpaid.

Original Leech-LILA Anatolii Kornienko attribution, DASHI extensions, independent final reserve, signed residual, actual null/DC and time/revision/guard indices unchanged.

---

## 25 November — endpoint-first steelman / bidi / reverse-solve geometric maximum cut (CURRENT)

Standing invariant: original source, actual null distinct from inhabited zero observation, signed residual, indexed time/model/guard/consumer hyperfabric, GPU native for learning, local SymPy in lieu of CI/kernel where appropriate. Owner Leech #5. Original Anatolii Kornienko observation: [March 10 issue #3](https://github.com/SPUTNIKAI/sovereign-lila-e8/issues/3#issuecomment-4029035489) reports E8 horror first ~50k and normal stories ~100k, no Leech horror phase; proposed stronger frozen geometry suppression. [March 10 stable rank report](https://github.com/SPUTNIKAI/sovereign-lila-e8/issues/3#issuecomment-4031821578) and [March 21 research status](https://github.com/SPUTNIKAI/sovereign-lila-e8/issues/3#issuecomment-4103466389) distinguish preliminary observation from a proved safety law. Original grokking Power et al arXiv:2201.02177, Nanda et al 2301.05217, Gromov 2301.02679; finite-group projection standard representation theory. DASHI extension is the explicit joint consumer weld and scoped exact controls, not original discovery. User-reported native L6 0/18 joint fit and 11/18 train fit remain distinct from 96-arm joint campaign. No new training receipt. Do not identify geometric negative sign with moral badness.

### Exact forward and reverse mathematics

A. For C3 shift P on class logits, de-rotate F(a,b)=P^(-a)L(a,b), and set Fbar(b)=sum_a F(a,b)/3. Nearest equivariant logit table in squared Euclidean norm is Lstar(a,b)=P^a Fbar(b), by orthogonal averaging. Three-cycle identity sum_a ||F(a+1,b)-F(a,b)||^2=3 sum_a||F(a,b)-Fbar(b)||^2. Thus exact defect zero iff equivariant; nonzero defect has exact squared distance interpretation. Full table audit only: averaging all nine logits uses heldout input predictions and is not a train-only discovery result. Train-only two-operand shift graph on 8 observed nodes has 14 directed edges, incidence rank7, missing p22 column zero; full 9-node graph rank8. In gauge P^(-(a+b)), training edge consistency alone does not constrain missing node. Structural extension (e.g. hard first-operand orbit architecture L(a,b)=P^a l(b), train-only representatives) and correct representative labels suffice on entire declared task, but supplied action is not learned discovery. Approximate path: if every edge along word has max-norm logit defect <=epsilon_i and representative correct margin m>2 sum epsilon_i, transported label correct, using permutation isometry and triangle inequality. Conversely wrong unique top with wrong-vs-target gap k forces sum epsilon_i >=(m+k)/2; target NLL n implies k>=n-log 3 when k denotes max logit minus target. No claim of training convergence or unseen semantic transfer.

B. Joint task representation rho(R)=diag(T,T), rho(J)=diag(J0,-J0), rho(N)=-I6 (row/column convention explicitly transpose for row fits). Group 12 signed permutations S3 x C2, M=trivial+standard, D=sign+standard, N acts as minus identity on both. Real commutant dimension 1+1+2^2=6, not unrestricted 36 coordinates. Exact orthogonal projection Reynolds Wstar=(1/12)sum_g rho(g) W rho(g)^(-1) is nearest commuting map, idempotent, and preserves already commuting W. For distinct representations average rho_out(g) W rho_in(g)^(-1), produces intertwiners. In an actual fitted chart, independently fit generators then solve common Hom(A,rho) via stacked vectorized equations A_g U-U rho_g=0, require invertible well-conditioned U and independent decoder, natural-pair, unseen word and causal checks. Exact Hom may contain only singular matrices or no nonzero solution; nonzero nullspace and group relations alone not enough. If exact representations known, averaging an arbitrary candidate U into Hom is a constructor, not invertibility proof. Approximate generators require residual and conditioning bound; do not average over assumed group closure when laws fail. Distinguish group-generated frame from commutant. True equivariant source architecture must project/tie base AND updates, seeds, route scores, orientation branch chart, nonlinear maps, biases, cap, readout, including inactive expert identities. Existing common arbitrary dense W0 residual arms cannot guarantee this. Reuse paired sign/reflection fibre, geometric adapter and original source; signed permutations with negative components do not commute with ordinary GELU unless branch canonicalization or odd/even lift pays it. Geometry-supplied architectural positive control vs independently learned natural action are distinct.

C. Quantizer c=round(z/delta) ties-to-even. On signed-permutation group rho, Q(rho z)=rho Q(z) exactly for coordinatewise common scale and symmetric rounding; no carry needed for code transport of these generators alone. General affine mixing does not descend: (0.4,0.4) and (0,0) same unit code, shear [[1,1],[0,1]] separates next code. Retain signed residual r=z-delta c only as required by consumer, carry equation Q(A(delta c+r)+b) and residual update exact, #1552 cocycle donor not p-adic identification. For fixed A,b and error box E on r, code stable iff every admissible image lies in one rounding cell with exact tie policy. Route/logit/sign/cap boundaries separately paid. #1291 4-way return fibres require 2 bits only for history-sensitive consumer, zero net not null; no endpoint-only lower bound.

D. Safety boundary: exact group equivariance and even perfect task accuracy do not determine intended semantic acceptability. If acceptable set S(x) is independently defined and action compatible (S(gx)=rho_g S(x)), verify membership on orbit reps and exact equivariance implies membership throughout orbits. If sets are nonempty convex in an orthogonal output carrier, pointwise nearest projection commutes with compatible orthogonal actions by uniqueness; existence, admissibility, semantic coverage and safety of future transitions still separately required. If action does not preserve intended constraints, do not enforce it blindly. For sequence safety, independent invariant K with initial state in K and every allowed transition staying K, with decoder safety on K, gives induction; otherwise counterexample or unresolved. No general guarantee of moral correctness from lattice geometry; competing causal accounts include data/order, initialization, optimization, decoding, frozen capacity, calibration, and chance. Stable rank ||W||_F^2/||W||_2^2 (zero case separate) and condition number on specified support are diagnostics not safety proof; rank reduction may discard required directions. Track time-indexed same-action, margins, confidence, geometry-specific intervention and independently defined semantic consumer, with matched null and frozen priors, not anecdotal sample selection. Author reports and own mod3 results are not same causal mechanism evidence.

### Constructor-to-consumer weld ledger, priority by endpoint

| Gate | Existing actual owner/constructor | Exact remaining weld or falsifier |
|---|---|---|
| M0 intended task and semantic action | ModularFourierGrokkingC3Exact: algorithmicRule, TrainingEquivalent, FullTaskEquivalent; SemanticInterventionEquivarianceExact: semantic-intervention, representation-equivariance, model-equivariance | independently define admissible semantic/task domain, action composition, acceptability, orbit coverage; no inference from textual coherence |
| M1 generalization | GrokkingEquivarianceDefectGeometryExact: equivarianceDefectAt, algorithmicEquivarianceDefectIsZero, memorizerEquivarianceDefectIsTwo; typed_mod3_prediction_consumer | C3 orbit extension / strict margins proved above; train-only missing-node nonidentifiability exact; implement hard equivariant positive control and matched train-only consistency with heldout audit |
| M2 geometric source | geometric_history_adapter.adapt_history_experts, c3_orientation; geometric_cross_triads, geometric_fair_residual; geometric_paired_history.PairedMark.decode/transport, encode_pair, indexed_pair_step, PairedHistoryMoE.step | common equivariant initialization and full-source step/decoder naturality including opposed zero chart, signs, routing, all inactive state; actual same-function null vs full symmetry distinction, native gradients |
| M3 recognition | audit_x369_joint_action.fit_joint/audit_joint, joint_generator_control; LilaWeylGrokkingFibreMaxCutExact; #1545 Hom donor | simultaneous invertible common intertwiner, independent task and latent generator fits, condition/error bounds, causal interventions across time/site, null controls; actual trained checkpoint |
| M4 recurrent sufficiency | #1550 X369InputIndexedFutureConsumerExact: TraceEquivalent, inputIndexedTraceDescent, everyStableCodecFibreIsTraceEquivalent, futureSeparationRefutesCodec; QueryFactorisationSufficiency.FactorsThrough/staticSufficient; audit_consumer_closure, bounded_witnesses, audit_joint_reachable_bins | reachable invariant fibre, observation and admissibility/successor descent or separating word; minimal required outside-chart residual and baseline/side-information cost; no global proof from finite samples |
| M5 carry and numerical | #1552 carry donor, #1291 conditional return, geometric_bitlinear, x369_r4_interval_certificate.Interval/pairwise_difference/selection_certificate/selected_softmax/response, audit_x369_precision_guards | exact signed permutation Q descent paid; general mixing carry interval, certify actual floating roundoff/GELU/cap and indexed propagation, guard/tie stability or explicit split; FP64 comparator not enclosure |
| M6 learning dynamics | LLMGrokkingLearningFutureExact: ExactLearningPost, learningSystem, sameTrainingFitDoesNotImplyLearningFutureEquivalence; DASHIgGrokkingEmpiricalBridgeExact; run_x369_ordered_campaign, checkpoint trace | execute 96-arm native baseline, freeze actual checkpoints at declared steps for future learning law including optimizer/STE state if comparing continuation; measure mode suppression and temporal order, matched seeds, no proof of convergence from geometry |
| M7 semantic transport and safety | SemanticInterventionEquivarianceExact: representationEquivarianceImpliesModelEquivariance, modelEquivarianceComposes, geometryCannotCreateCausalMechanism; TeleodynamicSemanticActionBridgeExact: teleodynamic-semantic-action-experiment, semantic-action-evidence-ledger, exceptional-action-recognition-promotion; OrientationCompassExperimentalInterpretationBidiExact: orientationReversalDoesNotNegateConclusion | independently sourced safe/unsafe task consumers, compatible actions, invariant or risk bound, matched data/compute/frozen kernel controls, causal ablations, heldout semantic composition; safety hypothesis remains unverified |
| M8 native packing/release | existing shared head/two+DC/differential VJP receipts, FairResidualLinear, original source, checkpoint/manifest | actual packed signed representation including residual/carry storage, forward/backward/optimizer parity and measured physical cost, kernel receipt separate from local SymPy. No release until empirical scope and formal premises paid |

Executed local SymPy 1.14.0 scripts/check_x369_geometry_maxcut.py (blob 3cd6e34d84eefc7338b567e23333f98feaaa6ff7): group closure 12x12, exact 36x36 Reynolds idempotence/symmetry/rank6, C3 defect identity, train graph rank7/full rank8, 300 signed-permutation rounding cases, exact shear collision. No GPU, new model run, kernel check, or new semantic data. Existing formal source inspected, not recompiled. Mathematical completion is scoped; no guarantee all empirical obligations are solvable by algebra alone. Keep native campaign baseline immutable and use geometry-forward construction for subsequent separate arms, not endless prerequisite expansion.

---

## 25 November — M1/M2/M3/M6 signed phase coordination and indexed cube (user oscillator hypothesis)

User hypothesis: early learning finds locally useful harmonics, tangents, conjugate or mixed characters; these can intersect the intended behaviour without transporting its ordering. Later learning coordinates their relative phase and suppresses incompatible consumer-visible modes. Treat neurons/latent features as response modes on a declared input/action domain, not literal physical oscillators or periodic SGD trajectories. Prior art Nanda et al arXiv:2301.05217 (memorization/circuit formation/cleanup and Fourier circuits), Gromov arXiv:2301.02679 (learned task-dependent feature maps), Kornienko March 10 [original E8/Leech observation](https://github.com/SPUTNIKAI/sovereign-lila-e8/issues/3#issuecomment-4029035489). New hypothesis credited to user, exact indexed phase attachment DASHI extension; no new semantic safety claim. Keep actual null, signed residual, DC, time/revision/consumer index.

**Six-face cube:** on existing joint rho(R),rho(J),rho(N), define vertices H_(r,j,n)=N^n J^j R^r H, bits r,j,n in {0,1}; forward edges R^((-1)^j), J, N. JR=R^-1 J and centrality of N imply all six squares and six ordered paths; ordinary unindexed R/J square fails. Reuse actual H05 owner GovernedIndexedResidualCubeExact and #1291 at 65950c7fe941a817464ea647cbfc49c58f7db6ab. Eight vertices are an indexed diagram, not all 12 elements. For complex triad phase p(v)=v0+w*v1+w^2*v2, rotation gives w p, J0 conjugates, joint reflection gives (conj p(M), -conj p(D)), N negates both. DC separate (mean reflection even, residual odd). A conjugate chart is not inherently wrong: if decoder and transport also conjugate it is a gauge. In the anchored task chart with fixed decoder, a wrong relative character or phase is observable. Learning-time transport must separately satisfy U_(t+1) rho_g = A_(g,t+1) U_(t+1), and common gauge transport B_(t+1,t) A_(g,t)=A_(g,t+1) B_(t+1,t) on identified image if it exists. No claim SGD itself commutes with task actions.

**Exact harmonic discrimination:** For logits L_c(a,b), use z(a,b)=sum_c L_c(a,b) w^c (class DC separate; w^3=1,w!=1). Intended task character chi_(1,1)=w^(a+b). Full 2D DFT zhat_(p,q)=mean z*conj(chi_(p,q)). Parseval sum |zhat|^2=mean |z|^2. Orthogonality distinguishes wrong relative frequency (1,2),(2,1) and fixed-decoder conjugate (2,2); all three match target on exactly 3/9 points, equal unit amplitudes, zero target coefficient. Global phase offset w*chi_(1,1) has all spectral energy in intended frequency and 0/9 anchored correct labels. Real two-plane power alone cannot distinguish conjugation; signed complex phase and decoder compatibility required. Existing hard memorizer changes only p22 from w to 1; target energy 19/27, other eight Fourier modes each 1/27, 8/9 correct. Thus a mostly correct function can have distributed residual harmonics, and growing raw spectral power/confidence does not imply progress. Same-frequency phase lock can still be globally misanchored. Measure absolute/relative energy, complex anchored coefficient, zero/amplitude masks, per-mode action and intermode composition. A phase tangent i*z is locally informative but not finite transport; on nonlinear task manifolds distinguish input action derivatives from parameter-space loss Hessian. Local matching or low gradient/Hessian rank cannot pay global closure. Use full table only for evaluation, no heldout fitting or selection. Exact checker scripts/check_x369_phase_cube.py local SymPy 1.14 PASS, Git blob 67d4c0809f55b8c793e9c30b5d3434b88e07c18a: six faces/six paths, 81 character inner products, three wrong-character controls, 8/9 spectrum, wrong global phase. No native training, kernel, or CI.

**Next integrated receipt, not more prerequisite expansion:** (1) M1 spectral task audit on each declared training checkpoint: centered logits DC, signed Fourier coefficient, target and other energy, phase angle/anchor, calibration and exact action defects; distinguish real conjugate gauge from wrong anchored output. (2) M2/M3 identify common generator and decoder maps from independently typed natural pairs; six indexed cube faces and endpoint paths on actual histories, fitted phase and outside-chart residual, compare to matched norm/phase-scrambled controls. (3) M6 track amplitudes, complex phase and consumer-specific projection by checkpoint; need actual time-indexed weights, activations and optimizer/STE state for learning continuation, not just final accuracy. Intervene by removing wrong modes or rephasing in the independently identified chart, compare matched spectral-power null and natural consumer target; separate output correction from causal source evidence. (4) M4/M5 retain residuals that carry future observers, actual precision/carry guards; #1291 history and #1552 carry donors. (5) M7 language: independently define semantic relation/action pairs, counterbalanced lexical roles, controls for token frequencies, story genre, initialization/data/decoding and frozen capacity; test relation inversion and phase-coordination hypotheses with causal intervention and unseen composition. Horror content is not itself a mathematical negative character, and no evidence yet that the same mechanism caused Kornienko's E8 observation. Native joint campaign baseline unchanged, actual checkpoint receipts still required. Attribution and final reserve unchanged.


---

## Canonical six-phase geometric learning hypothesis (25 November 2026)

User-confirmed framework for LILA grokking: learning as progressive discovery and coordination of distributed geometric response modes, rather than simply loss minimization or a binary memorization/generalization distinction. This is the controlling interpretation for M1/M2/M3/M4/M6/M7 and the signed phase cube above. The six names and descriptions are intentionally retained verbatim.

### The six phases

1. **Exploration** — Many competing response modes, weak organization.
2. **Harmonic discovery** — Useful frequencies and local tangent features become amplified.
3. **Partial crystallization** — Coherent but potentially incompatible characters and phase relationships dominate.
4. **Phase coordination** — Relative signs, ordering and intermode couplings become consistent.
5. **Geometric closure** — The intended action transports across previously unseen compositions.
6. **Consumer-sensitive cleanup** — Unnecessary modes are suppressed while future-required residuals remain.

These are diagnostic regimes, not a proved universal sequence, an exclusive six-state automaton, or six fixed training-step intervals. Different neurons, latent subspaces, layers, experts, time/depth indices and consumers may occupy different regimes simultaneously, overlap, regress, or skip a regime. Track the actual null state separately from inhabited zero-DC/zero-observation states. A zero-amplitude mode has no defined phase. Response-mode oscillation refers to variation over declared inputs/actions (or independently measured recurrent time); it is not evidence that neurons are physical oscillators or that SGD has periodic trajectories.

### Reverse-solved evidence and transition welds

| Phase | Observable evidence to collect | Exact remaining transition / falsifier |
|---|---|---|
| 1 Exploration | Absolute and relative spectral energy, amplitude masks, dispersion, route/sign/gradient distributions, independently recorded initial state | Demonstrate growth of a reproducible task-associated response mode beyond matched initialization/null; noise or weak amplitude alone is not a phase certificate. |
| 2 Harmonic discovery | Signed complex task coefficients, mode power, DC, local action/tangent response | Distinguish correct character from conjugate, mixed character, global phase offset and tangent-only fit under fixed decoder. Power alone cannot certify phase or law. |
| 3 Partial crystallization | Training fit/coherence, confidence, residual spectral energy, incompatible relative phase or ordering, held-out action defects | Independently identify candidate competing modes and show their consumer effect; do not infer internal inversion from incorrect confidence or horror content alone. Distinguish coherent wrong law from a harmless common gauge. |
| 4 Phase coordination | Common train-fitted chart, signed relative phase, cross-mode coupling, simultaneous R/J/N intertwiner rank/conditioning, anchored decoder | Show coordinated action in an independent gauge-aligned chart; compare phase-restoration versus power/norm-matched scrambled/null interventions on actual recurrent consumers. High coherence alone admits a coherent wrong global phase. |
| 5 Geometric closure | All six indexed cube faces, six endpoint paths, unseen word composition, independently defined output action, guard/roundoff margins | Establish actual same-action transport and decoder compatibility; prove descent on declared domain or explicitly bound transport error. Correct sampled endpoints alone do not certify universal closure. |
| 6 Consumer-sensitive cleanup | Mode ablation/reinsertion, future trace comparisons, exact codec collisions, shortest separating words, retained residual and cost | Prove observation/admissibility/successor stability for removed modes or retain a separating residual. Stable-rank reduction, lower power or compression alone cannot certify dispensability. |

### Indexed measurement contract

Use fixed predeclared training checkpoints and the same independently labelled task/action probes, with checkpoint/source/task fingerprints, training step, recurrence depth, site/layer/expert, original guard assignment and consumer. Preserve actual signed complex coefficients and DC, not only spectral magnitude. Fit charts on training data, freeze them for public composition evaluation, and use explicit gauge transport between checkpoints where identification permits it; otherwise report unmatched or rank-deficient sectors. Log raw magnitudes and normalized mode energies separately. Record null/undefined phase explicitly. Retain matched nulls, counterexamples, negative results and uncertainty. A single aggregate phase label must not erase local phase heterogeneity. For claims about future learning dynamics, retain optimizer, STE and random-state provenance separately from evaluation-only checkpoints.

Causal discrimination: compare natural counterfactuals, phase-restored interventions, equal-power relative-phase-scrambled controls and mode removal/reinsertion, including downstream routing, signs, weights, histories, decoder probabilities and future consumers. Keep phase-correction of outputs distinct from demonstration of the original latent mechanism. Temporal precedence, correlation and causal necessity/sufficiency are separate. No use of the final reserve for training, chart fitting or model selection.

### Source attribution and empirical boundary

Nanda et al. (arXiv:2301.05217) established their memorization/circuit-formation/cleanup account and Fourier-circuit measurements; Gromov (arXiv:2301.02679) studied learned Fourier feature maps. Anatolii Kornienko's [10 March 2026 E8/Leech report](https://github.com/SPUTNIKAI/sovereign-lila-e8/issues/3#issuecomment-4029035489) motivates the hypothesis of unwanted-mode suppression; it reports E8 horror narratives early and no Leech horror phase. The six-phase refinement and oscillator/partial-crystallization hypothesis are user-originated here, with DASHI geometric formalization and proposed tests. Do not label geometric sign/conjugation as moral good/evil or claim the language-model mechanism was established by arithmetic controls.

Existing exact phase/cube controls in `scripts/check_x369_phase_cube.py` support the distinctions, not a measured six-phase neural trajectory. The native 96-arm baseline remains unchanged; actual intermediate checkpoint acquisition, spectral/causal audits, fitted geometric closure and semantic tests remain open. The six-phase framework governs interpretation of those receipts, rather than imposing new prerequisites before the baseline GPU run.


---

## 25 November — six-phase implementation maximum cut: P1–P6 / M1–M6

User-confirmed six named phases remain verbatim in canonical section above. This cut implements measurement and checkpoint infrastructure, not a classifier that assigns a model a phase solely from one statistic. Original Leech-LILA Anatolii Kornienko, user oscillator/partial-crystallization hypothesis, DASHI extension; Nanda/Gromov prior art and semantic orientation firewall retained. Source starting head d9ea6ed5cb69706aa6bf36eb57f632383bdfe86e. Existing 96-arm final-only baseline and its fingerprints unchanged.

### Implemented producers and consumers

- scripts/x369_phase_geometry.py: signed triad complex phase and full nine-point public mod3 DFT, absolute/relative power and undefined phase at zero, natural rotation orbit spectra, anchored feature coefficients, DC and phase errors, row-convention common Hom stacked equations, indexed six faces and six paths with natural endpoint comparison, train-fitted cross-checkpoint chart transport.
- scripts/audit_x369_learning_phases.py: actual source-dependent natural histories from existing capture and fit_joint, predetermined full contexts and separate training/ composition, raw orbit spectra even when rank-six chart rejected, recurrence-length stratification (only sampled lengths), layer effective weight stable rank and numerical nonzero-support condition, train-fit decoder and common Hom candidate, six-face transport on public composition, natural versus fitted/chart-removed/relative-phase-null/unedited arms through every literal continuation through declared depth. Null rotates residual triad in decoded chart; preserves decoded DC and per-triad phase power, not ambient history norm or route guard. All terminal consumer fields evaluated at every enumerated depth; full history difference reported separately, not declared mandatory observer. Removal with natural target outside residual restored by comparison to natural arm is a bounded causal test, not proof of universal dispensability. Joint chart artifact retains V, U, decoder, A, evaluation coordinates and exact row keys.
- scripts/run_x369_phase_trajectory_gpu.py: separate native gfx803-only campaign reusing FairLearner, fair_arm, existing source labels and matched minibatch sequence, fixed 96-arm default. Snapshot schedule step0,1 and interval multiples plus final (default interval250, horizon2000 =>10 checkpoints per arm). New manifest binds sources (baseline plus phase modules), task digest, scope, runtime and fixed arm identities. Persist model, optimizer, Python minibatch RNG, Torch CPU and GPU RNG; atomic exclusive checkpoint publication, digest and fixed full recurrent trace receipt, no overwrite; resume only from contiguous verified scheduled snapshot. Independent audit after entire arm training, no heldout selection. Plan only previews; stage train/audit/both independent of manifest. Restore exact trace required; actual continuation parity across interruption not yet measured. A crash between checkpoint and receipt leaves orphan explicitly rejected; manual recovery not automated. No actual native job launched here. Full public composition accuracy remains existing baseline full replay; phase trajectory accuracy is on the declared sampled contexts only.
- scripts/check_x369_phase_geometry.py: local FP64 numerical controls on 729 signed states, exact intended row action with invertible synthetic charts; six Hom dimensions, six cube faces and paths, wrong-J negative control, correct/wrong-global-phase/conjugate/memorizer/zero Fourier fixtures, anchored phase and orbit Parseval, cross-time chart transport. Actual local PyTorch 2.10.0+cpu executed PASS; max path RMSE 4.1145137624745983e-16, cross-time RMSE 2.1567695735732703e-16. Source module Git blob 4536a5fef2ddc8d8d82f18116ad6a81a3bda667a; checker ba3105b64b686717b3d52d35ae9a442f989518c9. Numerical tolerance used for floating reductions, not exact equality. Prior exact SymPy phase/cube checker remains separate. Source-dependent phase audit, snapshot runtime and native campaign not executed in this cut; no Agda/Lean kernel or CI.

### Exact remaining lemmas, producer -> consumer and phase deliverables

| Phase / gate | Producer -> consumer | Exact remaining lemma / acceptance |
|---|---|---|
| P1 Exploration / M6 | native initial snapshot, raw orbit spectrum, layer spectra -> learning observer | initialStateTraceIdentity; zero/undefined-phase mask and actual zero-vs-inhabited residual, gradient/route/sign dispersion over fixed probes; establish baseline and matched null, no inference from random power alone. Raw spectra implemented, actual receipt and gradient dispersion open. |
| P2 Harmonic discovery / M1 | triad_phase, orbit_spectrum, mod3_spectrum, existing C3 exact owners -> anchored task/latent character | anchoredCharacterIdentification on independently typed probe orbit; distinguish conjugate gauge, mixed characters and tangents, retain signed complex coefficients and DC; derivative/finite action comparison and full per-site/depth probe coverage open. |
| P3 Partial crystallization / M1/M6 | anchored_phase, training/evaluation fit, spectra -> competing-mode witness | coherentWrongActionWitness: training fit plus actual heldout defect and independently identified mode with measurable consumer effect; compare confidence amplification vs spectral change; actual trajectories and counterbalanced semantic relation experiment open. |
| P4 Phase coordination / M2/M3 | fit_joint, common_hom, cross_time, paired orientation -> common action/decoder | invertibleCommonIntertwiner (exact or bounded on identified image), relativePhaseConsumerWeld, timeIndexedChartNaturality. Row convention rho_g U=U A_g, B=U_t^-1 U_(t+1), A_(g,t) B=B A_(g,t+1); report B condition, natural heldout chart difference and projector drift. Numerical near-null dimension or projected rank not invertible Hom proof. Relative-phase null matched in decoded power only; need ambient norm/guard matched independent controls, cross-mode coupling and phase correction dose/necessity. No claim SGD itself commutes. |
| P5 Geometric closure / M2/M3/M5 | cube_metrics, original paired/single source step, existing prediction/guard replay -> actual ordered consumer | sixIndexedNaturalFaces and allSixEndpointPaths on learned chart with independently observed natural targets (implemented diagnostic); actual source-step naturality, signed orientation including zero ties, branch/cap/TopK/decoder and inactive state, unseen compositions and sound error bounds unpaid. Chart cube agreement alone not full neural recurrence certificate. |
| P6 Consumer-sensitive cleanup / M4/M5 | natural/removed/reinserted phase arms, #1550 trace descent, exact code witness auditor -> minimal future residual | futureConsumerFibreDescent: equal codec => equal admissibility, observers, successor code on invariant domain; or shortest witness with charged residual. Bounded ablation diagnostics implemented, actual reachable closure, carry/guard residual, cost and minimality open. |
| All P1–P6 / M6 | native snapshot and optimizer/RNG -> learning continuation | sameRevisionSnapshotReplay and interruptedVsUninterruptedOptimizerParity; prove no effect of extra instrumentation on matched training, identify time order with actual checkpoints, permit overlap/regression/undefined phase, do not impose sequential universal stage labels. |
| Semantic M0/M7 | independent labels/action and phase receipts -> original author hypothesis | counterbalanced lexical/relational action, compatibility and causal safety control, matched data/order/architecture/decoding, actual E8 vs Leech checkpoint evidence. Arithmetic controls do not prove semantic safety or relation inversion. |
| Native M5/M8 | actual phase/closure/guard receipts, BitLinear and differential VJP -> native signed codec | rigorous numeric interval/guard transport, packed kernel backward/optimizer parity and physical cost, separate formal kernel acceptance. |

**Next integrated execution**: existing baseline remains valid; new optional time-indexed cohort uses the same training objective and fixed seed/arm definitions with a separately frozen manifest. Run on workstation:
```bash
python -m scripts.run_x369_phase_trajectory_gpu --output runs/x369-six-phase --stage plan
python -m scripts.run_x369_phase_trajectory_gpu --output runs/x369-six-phase --stage both
```
Default 96 arms × 10 checkpoints; explicit reduced declared scope is allowed but never reported as full cohort. Run on gfx803 Schaka ROCm environment, not CI or CPU. First actual source-dependent audit and native restore/continuation smoke must precede scientific promotion. No new prerequisite expansion of previous baseline. Keep exact source/consumer indices, signed residual, null/DC, attribution, independent reserve, and negative results.


---

## 25 November — native four-arm receipt boundary; reverse-solve action, temporal and carry closure (CURRENT)

User direction: steelman/bidi/reverse-solve/geometry, big cuts, existing constructors, actual null, signed residual, indexed time/site/guard/consumer. Source head e165eff996268575a29611937c59cb35dbde277e. User-supplied workstation report: gfx803 HIP 7.16.26385 / Torch 2.14.0+gitd70fd61; historical OpInfo 183/183 reconciled (131 upstream skips,52 xfails), prior L2 FP64 parity (max grad 4.68e-16, param 2.07e-11), L4 routing saved bytes -57.5%, peak -20.5%, backward -15.5%, L6 18 runs, 11 train fits, 0 sampled joint fits. This is scoped reported evidence; not proof all GPU kernels universally correct, identical learning trajectories, or necessity of geometric bias in every optimization regime.

New user-reported native smoke: runs/x369-six-phase-test, seeds41/mode c3/single family, 4 matched null/ternary arms, 10 scheduled steps each, 40 snapshots plus 40 audits, fixed probe replay and no GPU faults. Raw manifest/JSON/PT and exact quantitative audit fields are not accessible in current CPU-only environment. Report says clean cross-time transport but supplies no numerical learned trajectory values; reported synthetic regression 3.28e-16 distinct from prior local 2.16e-16. Do not infer phase labels, generalization, full learned Hom, cube closure, resume parity or cleanup from successful audit generation. release_admitted=False is an intentional scientific release boundary, not an execution failure. 4/96 configuration coverage, not a completed broad cohort. No rerun or raw receipt commit claimed.

### A. Exact common anchored gauge reverse solve (M3/P3–P5)

Existing fit_joint supplies training-only anchor U, separately fitted natural generators A_g and public composition coordinates q; shared data does not make estimates statistically independent. Use row convention q=Y U+e_chart. Require invertible U, finite condition; define z=q U^-1=Y+e, G_g=U A_g U^-1, F_g=G_g-rho_g. For natural target j=g(i), Y_j=Y_i rho_g. Then exactly

z_i G_g-z_j = z_i F_g + e_i rho_g-e_j.

First term is wrong operator in common task gauge; second is natural chart residual transport. The unique operator satisfying rho_g U=U A*_g for this fixed invertible anchor is A*_g=U^-1 rho_g U. It satisfies all group relations, but actual natural pair compatibility after this constructed correction still needs e_i rho_g=e_j. In task gauge, ||raw|| <= ||z|| ||F_g|| + ||e_i||+||e_j|| for Euclidean rows/orthogonal rho. Conversely, reverse-solve needed operator precision and chart residual from downstream margin; no cancellation-based claim both components small. Report each term, not just total. This proves exact linear algebra condition, not learned recognition or whole-source equivariance. A conjugate latent gauge is not inherently a wrong character. Rank-deficient U requires partial image/quotient identification, not pseudo-inverse promoted to invertible Hom.

Temporal transport B=U_t^-1 U_(t+1) implies exact identity A_(g,t) B - B A_(g,t+1)=U_t^-1 (G_(g,t)-G_(g,t+1)) U_(t+1). Also q_t B-q_(t+1)=(e_t-e_(t+1))U_(t+1) for same independently labelled state. Hence a small raw temporal square may be chart scaling, or two checkpoints carrying the SAME WRONG action. A correct action needs anchored G_g=rho_g (or bounded defect) AND natural chart/consumer compatibility. Cross-time agreement is not independent proof of geometric learning or commuting SGD. Existing six indexed R/J/N faces, decoder and actual natural endpoint measurements remain required.

### B. Exact carry-fibre descent, necessity and constructive separating witness (M4/M5/P6)

For rational affine column map y=A x+b and shared positive rational scale delta, input code c=round(x/delta) ties-to-even, each normalized coordinate u_i=x_i/delta is in [c_i-1/2,c_i+1/2] for even c_i, open interval for odd c_i. For output row a, normalized center m=a c+b/delta, radius r=sum |a_i|/2. Image interval endpoints m+-r attained iff every nonzero coefficient uses an even input code; zero row attained. Interior attained by convexity. Candidate output code k=round(m). Necessary AND sufficient for exact code descent on the COMPLETE fibre: [m-r,m+r] is inside k cell, allowing endpoint equality for even k, and for odd k only when the touching input image endpoint is not attained. All rows must pass. Exact rational calculation uses Fraction including sign/zero/tie parity; no floating enclosure assumption.

On failure, center x=delta c and second point along signed row extremizer (interior rational point if strict crossing, attained extremizer for exact boundary tie) share identical input code but differ in that output coordinate. Constructive witness included in implementation. Thus affine carry is precisely consumer/coordinate-dependent, not always necessary, and signed permutation generators descend even on odd cells. General shear fails. This theorem covers full rectangular quantization fibre and exact rational affine maps, not a fitted reachable subset, nonlinear GELU/TopK/cap, actual FP32 rounding, or global minimal carry cost. A restricted reachable subset may eliminate a witness; need actual source reachability or sound outer enclosure. Combine with #1552 carry cocycle, #1291 marked return, #1550 TraceEquivalent / inputIndexedTraceDescent / everyStableCodecFibreIsTraceEquivalent.

### C. Exact phase-null feasibility and full remaining causal weld

Current phase null rotates decoded residual triad, matching decoded phase power/DC but not ambient norm or guard. For fixed invertible decoder D, decoded row y maps to q=y D^-1; norm metric K=D^-1 D^-T. A candidate null P preserving decoded power must satisfy y P K P^T y^T = y K y^T to match latent norm, plus actual route/sign/weight/guard constraints separately. No guarantee of a nontrivial simultaneously matched null. Counterexample in a 3-coordinate slice with fixed mean m=1 and unit residual phase (cos theta,sin theta), positive definite metric K=[[1,a,0],[a,1,0],[0,0,1]], 0<a<1: squared norm 2+2a cos theta; at theta=0 unique phase on circle with maximal norm. Matching both power and norm then forces original phase. Report feasibility, use separately matched controls or nuisance stratification, do not manufacture a universal phase-only intervention. Need independent natural pairs, dose/reversal and actual consumer tests; no phase assignment from horror text or confidence.

### D. Implemented offline receipt consumption and local checks

scripts/x369_phase_reverse_solve.py adds gauge_audit, affine_cell_transport and read_campaign on existing snapshot/audit manifests. Checks original source SHA256 map against exact checkout, snapshot byte digest, manifest/arm/step identities; reopens stored chart arrays and original task labels for natural target lookup; preserves all existing measured phase/cube/Hom/consumer/confidence fields and computes task-gauge defects. No checkpoint deserialization, training, mutation of original manifests, authentication of audit JSON, or rerun of actual source; new summary hashes are integrity receipts only. Uses CPU for offline analysis and exact Fraction/SymPy, native training remains GPU-only. Output must be new file. Existing baseline and phase runner source fingerprint unchanged (new standalone module only).

Local /mnt/data/lila_gauge, torch 2.10.0+cpu / SymPy1.14: scripts/check_x369_phase_reverse_solve.py PASS 729 synthetic signed states, invertible correct gauge, bad J correction, nonzero residual after corrected action, rank collapse reject, symbolic unconstrained matrix decomposition, 625 rational fibre cases (167 constant, 458 constructed separating witnesses), rational-grid acceptance including odd/even ties, receipt fixture and altered snapshot rejection. Local blob e60ae7cd227c224310affa590739ac9a9aff1850 and checker 242abb1d6425b128ccdfa9d23ffe26b65c23d255 verified committed. Source-dependent real receipt ingestion / natural label lookup not executed here; raw 40 audit records unavailable. No GPU training, kernel or CI. Prior exact and native reported results stay distinct.

### E. Consolidated remaining gates, constructors, deliverables

| Gate | Existing constructor / producer | Exact remaining obligation and deliverable |
|---|---|---|
| P1/M6 initialization and trajectory | run_x369_phase_trajectory_gpu snapshot/reopen/trace_probe, orbit_spectrum, weight_spectra | read 40 actual receipts, identify sampled step/site/guard and seed/arm; initialStateTraceIdentity reported smoke, gradient/route dispersion and full per-step (not just prefix-length) activations still open. |
| P2/P3/M1 | GrokkingTaskCharacterPhaseExact, ModularFourierGrokkingC3Exact, mod3_spectrum, anchored_phase, orbit_spectrum | actual signed harmonic and phase trajectories, confidence/defect and phase-aware causal comparison. mod3_spectrum exists but is not wired into historical L6 training receipts; no retroactive recovery of unavailable L6 checkpoints. |
| P4/M3 | fit_joint, common_hom, new gauge_audit, #1545 simultaneous Hom, cross_time | anchor full rank/condition, common action vs chart residual, independent natural and consumer pair, exact/approximate invertible Hom; gauge correction is supplied positive control. Quantify real trajectories before phase classification. |
| P4/M2 | FairResidualLinear, geometric adapter, paired mark/orientation | full-source naturality on seed/router/selection/branches/nonlinear/transition/cap/readout/inactive expert; arbitrary dense base means existing residual arms are not hard equivariant. Separate architecture positive control from learned geometry. |
| P5/M3/M5 | H05 indexed cube, cube_metrics, output_action, audit_precision_guards | actual natural six faces and endpoint paths, all ordered guards and unseen source recurrence; bound errors via e_(k+1)<=L_k e_k+delta_k with actual Lipschitz/guard estimates; strict route gap >2 score error, orientation margin, explicit tie split, cap/GELU roundoff. FP64 comparisons are not certified enclosures. |
| P6/M4/M5 | new affine_cell_transport, bounded_witnesses, audit_consumer_closure, FactorsThrough, #1550 trace descent | exact affine full-cell weld paid; general nonlinear fitted invariant domain, source reachability of witnesses, admissibility/observer/successor code closure, carry/baseline/side-info cost and minimality unpaid. |
| P1–P6/M6 continuation | actual snapshot optimizer/RNG and source hashes | interrupted versus uninterrupted native optimizer parity and instrumentation-neutrality; 40 reported replays do not prove resume. Then broaden matched cohort on GPU, fixed manifest and untouched reserve. |
| M0/M7 | SemanticInterventionEquivarianceExact, TeleodynamicSemanticActionBridgeExact, orientation firewall | independent semantic action/acceptability labels and compatible invariant/risk bound; matched lexical/order/genre/init/architecture/decoding controls, causal semantic interventions. Kornienko E8 observation preliminary, no moral sign claim. |
| M8 | L2/L4 native VJP receipts, BitLinear, exact interval owner | actual packed code/carry/residual kernel, forward/backward/optimizer parity and resource measurement, rigorous floating certificate and formal kernel/release separate. |

Immediate no-retraining command on exact source checkout with new standalone analyser present:
```bash
python -m scripts.x369_phase_reverse_solve \
  --input runs/x369-six-phase-test \
  --output runs/x369-six-phase-test-summary.json
```
Use native environment Python if needed for imports; this is offline CPU analysis, not training. Preserve raw snapshot/audit files and actual source manifest; share summary or commit selected non-reserved evidence with provenance only when actual files accessible. Do not modify prior manifest to bypass hash mismatch. Prioritize actual 40 receipt metrics and native resume smoke, then broader cohort and structured hard-action control, not additional abstract demos. Original Leech-LILA Anatolii Kornienko; Nanda/Gromov prior art; user six-phase oscillator hypothesis; DASHI new gauge/carry attachment, classical linear algebra and interval arithmetic attribution. No external branch changes.


---

## 25 November — L7 supplied-action versus emergent phase geometry (CURRENT)

The repeated user report is the same e165eff four-arm native smoke (40 snapshots/40 audits), not a second run. Existing reported L2/L4/L6 and gfx803 scopes unchanged. Raw audit values still not supplied. Preserve the old manifest and use the previous standalone x369_phase_reverse_solve receipt analyser first; no re-training required for that analysis. The new L7 is a separate native control, not a replacement for the joint signed trajectory or evidence of new native runs.

### Reverse-solved structural extension and wrong-character obstruction

Original typed task producer: TASK_PAIRS, TASK_LABELS, TRAIN_INDICES, HELDOUT_INDEX from audit_x369_typed_mod3_prediction_consumer. Original recurrent Learner from train_x369_mod3_codec_gpu. Let P be positive cyclic class permutation. Compute exactly one model output for each observed orbit representative (0,b), b in Z3. Define L_sigma(a,b)=P^(sigma*a) l_b, sigma in {+1,-1}; l_b are actual trainable recurrent model logits, NOT supplied labels. This is a hard output orbit extension, not a proof of internal LILA recurrence equivariance; first input is canonicalized, raw network may remain wrong. Correct + orientation: strict representative fit implies all nine strict labels correct, hence p22; no heldout label or heldout input used in training. Wrong - orientation: strict correct class at (a,b) requires representative argmax b+2a mod3. Within each b the requirements for distinct a are distinct. At most one strict fit per b, giving exact 3/8 strict training ceiling, attained for appropriate representatives. This is a wrong-character control with zero defect relative to its OWN action, not evidence of semantic or moral inversion. Tied argmax is a separate chart, do not apply strict ceiling to arbitrary tie policy. Prior task-gauge decomposition and six-phase framework give the interpretation: local coherent harmonic/character structure is insufficient until anchored to intended action. The earlier 14 train-only edges still leave p22 unconstrained: a memorizer has zero train edge loss but can fail there.

For raw logits, bound first-operand edge defects along (0,2)->(1,2)->(2,2) in max norm; if margin m at (0,2) > 2 sum edge defects, strict p22 correctness follows. Exact full-table evaluation, not train-only predictor or rigorous floating rounding certificate. Hard extension supplies zero edge defect by index gather from same shared representative outputs in same evaluation batch. Both operand action defects measured independently. Full nine-point signed spectrum (target coefficient/energy, wrong conjugate/mixed energy, global phase, DC, null mask) and actual raw/unwrapped logits retained; a wrapper making logits correct cannot pay learned action recognition.

### Implementation and execution

- scripts/x369_mod3_geometric_control.py: rotate_rows, orbit_forward, training_edges, edge_loss, transport_metrics. Input canonicalization + gather, correct and reversed orientation; centered train-only consistency over 14 internal edges; evaluation strict margins and both action defects. Classical group action and finite orbit extension, DASHI application to original LILA, user six-phase hypothesis and Kornienko/Nanda/Gromov attribution unchanged.
- scripts/run_x369_mod3_geometric_gpu.py: separate GPU-only gfx803 L7, default 3 seeds x 4 arms (unconstrained, train_edge_consistency, hard_c3, hard_wrong_orientation), 2000 steps, width384/expert4, original recurrent source and common initial parameter hash; no equal initial function or equal compute claim for wrapper arms. CE on eight training labels; positive fixed default edge coefficient 1 in consistency arm only; hard orbit reps all observed; fixed schedule 0,1,100,...,2000; each snapshot model/optimizer/RNG, audit raw and wrapped logits, exact action diagnostics, signed spectrum and independent original consumer; source/task/manifest/hash provenance; new output directory only. No checkpoint selection from heldout, no CPU training fallback, no F5, no claim of exact resume or independent replay in this new runner. L7 reports supplied action separately from emergent neural geometry. Full source-dependent runner not executed here.
- scripts/check_x369_mod3_geometric_control.py: actual local FP64 CPU fixture, 27 representative tables enumerated, one strict correct extension, 3/8 reversed-orientation strict ceiling, 14 edges and heldout-independent edge-loss counterexample, finite backward gradients through gather. Local blobs 4c7d6d347e345037b8c296d926ce4efdfae2089d and af0195c60bff8c16b18aae1f3e936ec5d91a1c03. No actual neural training, native GPU, CI or kernel check.

```bash
python -m scripts.run_x369_mod3_geometric_gpu --output runs/x369-l7-geometric --plan
python -m scripts.run_x369_mod3_geometric_gpu --output runs/x369-l7-geometric
```
Use same gfx803 ROCm Python as previous native report. No need to repeat L6. Scope may be explicitly narrowed before execution; do not compare reduced cohort as full. New L7 runner and helpers are not part of old trajectory manifest fingerprint.

### Consolidated exact welds and priorities

| Gate | New disposition | Remaining deliverable |
|---|---|---|
| P1-P3 actual phase trajectories | 40 reported native audits; offline reader implemented | Consume actual JSON/snapshot hashes, extract anchored spectrum, confidence and task-gauge residual by step; no phase sequence claim yet. |
| M1/P2-P5 structural C3 extension | exact + orientation extension and reversed strict ceiling paid; native L7 implemented | Actual native 12-arm data: supplied law vs train-only bias vs unconstrained; raw model logits and latent mechanism separately. |
| M2/P4-P5 complete recurrent equivariance | existing paired orientation, indexed cube, Reynolds/Hom | seed/embedding/router/branch/transition/GELU/cap/readout naturality, inactive histories, actual common action and guard margins; L7 output wrapper does not pay. |
| M3/P4 | fit_joint, common_hom, gauge_audit and temporal decomposition | actual learned anchored action versus chart defect, rank/condition and causal intervention on independent natural pairs. |
| M4/P6 | affine_cell_transport exact rational, bounded witnesses, #1550 | source-reachable nonlinear admissibility/observer/successor closure or witness, carry and residual storage cost. |
| M5/M8 | actual native L2/L4 and precision guards | certified rounding/guard enclosures, packed GPU forward/backward/optimizer equivalence, physical measurements, formal kernel. |
| M6 | intermediate native snapshots reported | interrupted/uninterrupted continuation parity; broaden fixed 96-arm cohort after quantifying pilot, preserve signed/indexed measurements. |
| M7 | independent semantic-action bridge | actual E8/Leech matched language experiments, independent relation labels, phase intervention and safety invariant; no semantic conclusion from L7. |

No raw workstation receipts committed in this cut (files unavailable). Do not treat 183 historical annotations as universal GPU qualification or 4/96 arms as full learned closure. Preserve original source and independent reserve. 


---

## 25 November — proof-search-governed constructive P0; stop redundant training (CURRENT)

User explicitly requested reuse of repository experimental-design/proof-search content and construction before testing. This supersedes the earlier priority to execute full L7 and 96-arm campaigns. Six-phase hypothesis P1–P6 unchanged; M0–M8 remain dependency gates. Original Leech-LILA: Anatolii Kornienko; user phase/oscillator hypothesis; DASHI application and extension; standard orbit/stabilizer and Reynolds mathematics retains classical attribution.

### Reopened actual formal owners (read-only donors, no fresh Agda kernel acceptance)

DASHI master file blobs at inspection:

- DASHI/Core/ProofSearchLeastPrivilegeAdmissionExact.agda 4f0865a64e4f12670e996c60ac4582a6032ab790: RouteAdmission exact target, same-object, prerequisites, no-go, no circularity/hidden strengthening, authority, novelty and frontier improvement; duplicateTheoremRegression = redirectedReuse; promote requires ExactConsumerReceipt, not LocalLemma. Existing theorem statements and source constructors reused, not new claimed kernel receipts.
- DASHI/Core/ProofSearchExperimentalParetoCrossPollinationExact.agda 5839ea47497e70e8f7f30245c5e1fd168937f7d4: FrontierRefinementProposal, producerAuthority, no numerical experiment directly closes a formal leaf; only residual-relevant coordinates opened.
- DASHI/Core/FibreRefinementExperimentSelectionBidiExact.agda 23f91d7ba336cd1a0ac5b0650aa757a6d800abd1: ExperimentRefinementReceipt/chooseStrictRefinement require prior-compatible eliminated witness, not invented identification percentages.
- DASHI/Core/SequentialConsumerExperimentPlannerExact.agda 24520c8ccc1ff4751b77ef58b9b5b51af5387b0c: closeConsumer when all compatible worlds agree on declared consumer; askThen only on possible outcome, worst-case cost separately certified. No need to identify entire latent world.
- DASHI/Core/DiscriminatorSynthesisExact.agda dfdb3c5edab9430b18ce0e5b825988f33d70cf49: CurrentObserverCollision, BundleSeparates, ProspectivelyClosesConsumer, MinimalDiscriminator, NuisanceRobustSeparator. Pair separation is not global closure.
- DASHI/Core/ExperimentalCoordinateDesignExact.agda 409c6698d1a7dbb7461eee00d919f1f5d8f2ae09: reference invariant, controlled input, measured observable, nuisance, derived discriminator; require calibrated collision separation.
- DASHI/Core/ActiveBidiDiscriminatorLoopExact.agda 46f55a3dddd733e06fcc9c2dcd70e24fbdadd9f1: ActiveBidiCycleReceipt, consumer closure may precede world identification; next move conditioned on outcome; selectively reopen affected dependencies, frozen consumer cannot be weakened.
- DASHI/Core/AdaptiveMechanismDiscriminationPlannerExact.agda 28d3991a31e2f65b82faebb72537c8594c4b3f4c: residual-conditioned experiments and decision closure without world identity.
- DASHI/Core/ProofSearchFrontierObserverTrustExact.agda 814c425035245206916b06f5ea77d673a82e0fc9: actual/reported/believed frontier separate; stale open leaf routes to reuse, self-report cannot manufacture exact consumer receipt.

Apply these to the real LILA consumer: construct the signed geometric recurrence first. Existing exact cyclic extension, strict wrong-orientation ceiling, commutant, indexed six faces, fixed-anchor correction, affine fibre criterion are REUSE, not candidates for another training sweep. Native qualification, snapshot persistence and reported smoke do not close learned-action or numerical-enclosure leaves. Full L7 and 96-arm comparative execution deferred. The existing 40 audit records (reported, raw bytes still unavailable here) are the first no-retraining evidence source for actual learning mechanisms. No new training launched. A future new experiment must name the live collision, consumer, possible outcomes that alter a decision, nuisance/authority, stopping rule and incremental cost; if no such discriminator remains, close for that consumer. Preserve the independently reserved F5 material.

### P0 constructive alternative to individually projecting every source weight: whole-source orbit descent

Reopened models/history_orbit_moe.py a2a250462ce8e663591cd8a8f784fad9bde1b15d, geometric_paired_history.py 5579a772df6a2778b4834bbc00976fe42c2a1236, geometric_reflection_fibre.py b6daec81d85357455436c918f7abe5908c1a315b, geometric_cross_triads.py 0db9da720ff1317a4099b31a9484f288a239ec12, original signed task source and OrderedLearner. Existing indexed_pair_step explicitly assumes a jointly equivariant transition; arbitrary source weights, directed router, zero fallback, GELU and decoder do not establish that assumption. The group-algebra frame is not the commutant. Do not use it as proof of full equivariance. The new construction works at the whole-step boundary and is a new opt-in function, not checkpoint-preserving surgery or a proof that the original unmarked step was equivariant.

Let s be the complete joint activation/history or typed operation/history; choose a in G with a.s=c the lex-max canonical representative. K=Stab(c). Let f(c) be ONE call of the original source in that local frame, including original routing, orientation, nonlinear FFN, history transitions and cap; paired variant reuses actual PairedHistoryMoE.step. On continuous result representation define P_K v=|K|^-1 sum_(k in K) k.v, and F(s)=a^-1.P_K f(c). If a' sends g.s to c then a' g a^-1 belongs to K; P_K f(c) is fixed by K, hence F(g.s)=g.F(s). Independent of coset representative, including all nonfree/zero cases. At free orbit K={1}, raw local result transported without projection. At stabilizer fixed point, projection is necessary if arbitrary f(c) is not already fixed; for an orthogonal action it is the nearest K-fixed continuous result. Inactive expert histories are already fixed by K, so remain identity; implemented explicit restoration in world frame. Discrete ordered ids, weights and local orientation signs are original canonical source observations; retain raw/projected local tensors, mark and K in receipt, never pretend local signs are raw world scores. Source may depend on all parameters freely. Actual world recurrence now uses projected result; removed local components are not claimed irrelevant to ORIGINAL source consumers.

Implementation models/geometric_orbit_descent.py: act_tensor, canonical_joint, project_stabilizer, decode_frame, OrbitDescentHistoryMoE.step and raw/projected indexed OrbitStep. Reuse existing GROUP, inv, mul and paired source rather than invent new group; original donor R rolls left and J reverses triad, while task positive R is donor R^-1 and original J0 (fixes coordinate0) is donor R^2 J. Do not identify conventions silently. One original source step per batch, not 12 full source calls; group orbit comparisons/projections still cost work, and canonicalization is nondifferentiable at boundaries. Exact ideal-real theorem; no universal bitwise/STE/optimizer parity, performance, migration or packed claim. Signed zero floating, overflow, reduction order and guard precision require actual implementation acceptance.

### P0 control, seed and typed decoder weld

Existing seven-coordinate op encodes literal tag plus guard. Conjugation by central N fixes EVERY group operation, whereas polar activation changes sign. Any equivariant embedding of these controls alone into a polar carrier with rho(N)=-I is identically zero. This is a no-go for naive projected polar operation embeddings. Repair by keeping operation in separate conjugation carrier (g,u) -> g u g^-1 and invariant literal tag; canonicalize joint (operation,history) before embedding operation LOCALLY. Conjugated reflections may require a reflection axis not expressible in original literal alphabet, so explicitly extend admitted control carrier to the 12 group elements, with original literal/tag mapping. step_controlled and canonical_controlled implement this; local embedding must read canonical control and invariant tag only.

scripts/x369_source_geometry.py reuses original encode_inputs signed seed layout (M,D,zero padding), actual task TRIADS/GRAMS, and HistoryOrbitMoE. signed_seed injects M+D,M-D into a retained expert branch pair, so simultaneous signed action (including reflection swap) exactly induces mean rho_g and residual (-1)^reflection rho_g. Other histories initially zero. literal_controls maps original a/b translation, j reflection, n antipode to same source group indices; tag retained, invalid guards rejected. SourceGeometryOrderedLearner calls actual paired step under controlled orbit descent, with learned local control embedding and local readout. On mean/signed class labels transport the corresponding triad; on Gram classes use original gram of transformed D (check representative independence); class-permutation stabilizer average then frame decode. Thus ideal-real seed, all controlled recurrent steps and typed readout form end-to-end equivariant sequence on explicitly conjugation-closed task carrier by induction. Learned maps may still implement an incorrect equivariant function. This is NOT proof of correct task prediction from equivariance alone.

known_action_register applies the already-known operation law directly to signed M,D, emits (M,Gram,D), with original Gram order (d0^2,d1^2,d2^2,d1*d2,d2*d0,d0*d1). Use for declared synthetic task consumer whose action is supplied; no need to train the known answer. The learned neural consumer is separate, not presented as evidence of independently discovered geometry. Future general LILA integration should attach unknown residual functions to this known geometric structure, not relearn supplied actions. Actual language embedding/token/semantic action still separate M0/M7.

### Exact remaining frontier and stopping rule

| Gate | Disposition / existing owner | Next exact consumer deliverable |
|---|---|---|
| M1 supplied C3 and signed ordered task law | reuse structural extension, original endpoint and generator controls, known_action_register | no new SGD sweep; verify original literal/Gram convention at native integration gate only |
| M2 full source ideal-real geometry | constructed whole-step stabilizer descent + controlled operation + paired seed + typed readout; proof above | actual original-source native implementation acceptance; formal Agda same-source refinement/kernel still open. Scope is new typed adapter, not production language model or parity with unmarked original. |
| M3 learned recognition / P1–P5 | existing fit_joint, gauge_audit, cross_time | consume 40 existing audits; quantify live operator vs chart residual/phase collisions. No fresh experiment unless consumer-relevant alternatives remain. Supplied equivariance is not learned recognition. |
| M4/M5 P6 cleanup | existing TraceEquivalent / FactorsThrough / affine_cell_transport | actual nonlinear reachable fibre, stabilizer-projected residual consumer distinction, observation/admissibility/successor descent or shortest separating word; preserve required residual and cost. No general inference from stabilizer invariance. |
| M5/M8 native precision/packing | existing L2/L4 and interval/guard owners | certified finite rounding/guard enclosures, actual physical cost of canonicalization/projection, optimize only identified bottleneck, packed forward/backward parity. Whole-step descent proof is ideal-real; native numerical/refinement not yet accepted. |
| M6 learning phases | canonical six-phase hypothesis; snapshot provenance | offline actual receipt metrics first; only next minimal strict discriminator if uncertainty changes causal learning conclusion; resume parity only if continuation consumer required. |
| M0/M7 semantic | existing semantic intervention and orientation firewalls | independent real semantic/task action and consumer labels, admissibility, causal evidence; no moral sign/semantic safety claim from supplied geometry. |

New source acceptance command (one small declared actual-source check, no training):

    python -m scripts.check_x369_source_geometry_native

Three original signed input cases including actual null and nonfree stabilizer, four literal steps; compare original endpoint with supplied register, actual original source outputs/history and canonical route/sign/weights under generators and typed decoder, one backward plumbing call. No optimizer step, fixed finite sample not a universal numeric proof. Script added but NOT executed here: current container Torch CPU-only, no gfx803 and raw GitHub DNS unavailable; original source files inspected through connector. No new local fixture campaign, no GitHub CI/kernel or workstation receipt access. Original runner fingerprints unchanged, raw workstation receipts not committed. Stop after the stated source implementation acceptance unless a concrete failure reopens the affected gate; no 12/96-arm prerequisite for construction.


## 26 November — CURRENT controlling steelman and endpoint-first delivery recut (supersedes training-first cuts)

User instruction: keep implementing while outcomes can be determined from formalism or existing source/weights; ask for a training experiment only for a genuinely unidentified learned outcome. Do not delegate deterministic materialization, algebra, checkpoint analysis or native qualification as training. Geometry-forward, bidi and reverse solve retain zero/DC, signed complementary state and time/model/guard/consumer indexing. Existing source/model/lockbox provenance remains binding. Original Leech-LILA attribution Anatolii Kornienko (DOI 10.5281/zenodo.18784424); Power/Nanda/Gromov prior work and user six-phase hypothesis retain credit; DASHI contributes scoped source algebra, constructions and falsifiers. No new empirical original-author claim replication.

### What the constructive campaign actually closed

See [source geometry roadmap §§22–27](X369_LILA_STABILIZER_CODEC_20261125.md) and [integrated compact source](../scripts/x369_dense_compact.py), through 33d0bb511f3ac6acc8febd19f99724f187c80b08. Known finite signed ordered action, entire 312-atom dictionary optimum (9 rank-one terms), exact-real original dense recurrence with state12/hidden15, hidden12 conditional exact FP value realization and rigorously correct exact-real typed labels up to 2^2390 controls. Common-root parity dictionary (-t*,0,+t*) needs >=13 hidden units; unrestricted hidden12 exact-real state equality still open, not a training question and not required for declared finite typed label endpoint. Existing actual-history typed polynomial consumer solves affine Gram obstruction; actual source forward now explicitly opts into it, with separate original affine diagnostic and mode-bound checkpoint/source fingerprint/strict replay. Do not call hard-supplied action independently learned geometry. Exact constructed future-consumer quotient: 729 classes, six trits (M,D) sufficient/minimal for the declared typed trace; reversible 2185 integer-value states retain seed/parity mark, zero has single physical state. Actual fitted model raw trace/future residual not implied. Local reconstructed active-source CPU equation replay 26244 labelled edges and 78732 strict head comparisons per mode/dtype, dense12 value error zero, dense15 errors ~5.33e-15/1.43e-6 FP64/FP32. Actual committed full module import/native backend/checkpoint replay not executed in this environment, no kernel verification, no SGD, no release. New content hashes do not rewrite historical manifests.

### Strongest defensible research claim and reverse-solved residual architecture

Do not reduce the original claim to merely knowing a group action or terminal modular accuracy. The steelman is: an independently justified geometric action and typed consumer remove already-known degrees of freedom from the learning problem, allow an exact transport/codec/positive-control path, and may constrain optimization to learn only genuinely unknown residual structure; this **might** improve time/data/active-compute/bit cost to held-out consumer quality relative to source-matched controls. Whether it does so for an actual language model is an empirical claim. A frozen shared orthogonal Q/K transform alone is an exact attention gauge; any genuine advantage must come through separately activated regularization, history, noncommuting mechanisms, optimization, data or other source-authorized effects. Do not conflate preliminary negative/positive author reports, finite mechanism tests and original TinyStories BPC.

For a declared semantic projection q with section S, qS=I, keep indexed h=S(z)+r, q(r)=0. On the constructed carrier z=(M,D), z'=A_u z exact; reversible mark is a finite residual. For a future learned extension, a semantic-base-plus-residual family can have z'=A_u z and r'=F_theta(z,r,u) on an explicitly admitted product/fibre, or a residual projected to ker q when the underlying transport and next admissible domain have been proved. This is a constructor/interface, NOT a theorem that adding arbitrary residuals to dense12/dense15 preserves their parity carrier, or that an existing learned LILA history admits this section. Use existing indexed_residual_closure, FactorsThrough, fair zero-init residual and common-Hom owners. Before removing a residual, reverse-solve actual future consumer and admissibility; retain it or give separating word. Retain original source function/optimizer semantics when claiming migration, otherwise explicit new opt-in arm. Source-target matching and same-object costs are required.

### M0–M8 current decisions and exact next producers

| Gate | Current disposition | Next concrete endpoint |
|---|---|---|
| M0 task/domain | Known finite synthetic signed task paid; original language/source semantic action OPEN | Original SLR PostgreSQL source-revision/parser runs + original tokenizer/checkpoint, disjoint document/split lineage; independent prefix-visible typed labels and acceptability, no F5 exposure. Existing R1 source owner, not another fixture. |
| M1 generalization | Supplied-action positive control paid (incl exact wrong-orientation and finite classifier) | No known-action training; test heldout truly unspecified task/action/semantic residual only, with correct vs reversed vs matched null distinction. |
| M2 source geometry | Exact opt-in signed recurrent transport on declared carrier; original unmarked and fitted equivalence OPEN | Source-mode/fingerprint/strict reload, deterministic native finite-carrier replay, exact producer/consumer and actual same-source formal kernel weld. Don't expand the 12/15 construction into a language-claim prerequisite. |
| M3 learned recognition | Not implied by constructive transport | Read existing 40 reported actual audit snapshots (raw files unavailable here); actual common invertible Hom from independent natural pair data, anchored phase/condition, unseen words, causal intervention and matched null. No repeat of existing L6. |
| M4 future sufficiency | Constructed typed quotient closed 729; raw value mark retained | Existing fitted checkpoint decoder/kernel witness/bounded closure on original raw O=(output,ids,signs,weights) with indexed T/admissibility. Require all-future stable fibre or shortest actual separating word. Don't equate known constructed 2185 states with arbitrary learned states. |
| M5 numeric/guard | Finite constructive proof and conditional FP contract; general source OPEN | Existing intervals/guard and same_bits/stabilizer codec, native backend rounding/route/sign/tie/cap/optimizer as appropriate; actual packed recurrent value/byte consumer must be declared. FP64 comparator not certified interval. |
| M6 learning | P1-P6 six-phase hypothesis remains unconfirmed; prior 40/40 audit smoke reported only | Consume prior trajectory receipts before deciding any fresh training; independently identified anchored harmonic/phase/coordination/closure/consumer-cleanup evidence, preserve null and failed/regressed phases. Optimizer/RNG continuation parity for actual learning-future consumer, not proof from final weights. |
| M7 semantic/causal | No source-independent language/safety implication | Original paired neuron steering (N1–N5), predeclared discovery/test, necessity/sufficiency and independent semantic labels, nuisance/geometry/order controls, no moral interpretation of sign or anecdotes. |
| M8 cost/release | Narrower constructive widths mathematically; actual native physical improvement OPEN | Only measure identified physical bottleneck at same declared consumer and error contract: active experts/parameters, latency, peak allocation, bytes including mark/residual, tokens/compute/energy if available. Keep prior L2/L4 native results scoped; no GPU claim from local CPU proof. |

### Single next genuine learning decision (conditional, not a request to run now)

First recover original task/checkpoint/tokenizer and actual existing R2/R4/L6/L7/40 snapshot evidence; no raw F5/lockbox and no repeat training. If original provenance or independent labels absent, stop at blocker rather than manufacture inputs. If existing data settles a declared consumer decision, close it without a new run. If causal learning advantage remains unidentified after source and deterministic audits, the minimum discriminating matched campaign is **same-starting-function C2 bypass vs zero-init coherent C3 residual vs rank/capacity/compute-matched null residual** on an independent task requiring UNKNOWN structure, with a hard constructive action positive control (reported separately, not an equal-capacity learning claim), and wrong-orientation control where meaningful. Existing FairResidualLinear, original task/data and runner owners are preferred over a new training harness. Freeze tokenizer, document split/order, checkpoint hash, optimizer/RNG, steps/resource budget, and target heldout quality/CE/BPC before run; use exact source-bound fingerprints. Compare time/updates/tokens/active compute to target quality, censored failures, matched physical cost, independent task action/causal contrast, and six-phase diagnostic trajectories. A coherent effect without independent action does not establish mechanism; a hard-control win is supplied knowledge; a null/negative matched effect is a valid stopping decision. Existing 96-arm and L7 broad campaigns deferred until this pilot/receipt residual warrants them, not discarded. Original R1–R5 and grammar feedback decision (predicted vs aux-only/token-only, prefix-causality) remain distinct; no synthetic signed task substitutes for original TinyStories/FineWeb evidence.

Formal donor #1550 currently open master-based at 0cc1079f3e20075bc9fb15930b0e7767b98d8072, reported not mergeable; reuse future-trace and factorization source, do not assume fresh kernel or edit unrelated Agda branches. Historical #1442 is closed; use its explicit formal contracts, not its PR status as original neural acceptance. Keep controlling ledger, original claim and six-phase owner cross-referenced. No user training handoff for construction or validation; current missing original fitted files/native backend are access/measurement blockers, not learned degrees of freedom.


## 26 November — CURRENT source-recovered geometry maximum cut, exact remaining welds

See [constructive source roadmap §28](X369_LILA_STABILIZER_CODEC_20261125.md#28-26-november--recover-existing-future-constructors-close-raw-value-quotient-reverse-solve-residual-correction). Master/PR re-audit at DASHI master f55ec6217181, #1550 0cc1079f3e20, #1554 bb8fbf96757a, #1291 e131b156974d, #1552 6d2260fbd45f and #1548 1b17f2f60e76; #1526/#1094/#1095/#1265/#1463 merged, #1442 closed unmerged. Actual source declarations and blobs listed in §28. Do not create another general future quotient, observer factorization, learning-state reopening, semantic intervention or group-action theorem. Reuse existing compiler and consumer-relative source contracts. User requires geometric steelman/bidi/reverse-solve first; no deterministic training handoff.

New concrete closures: constructed dense12 saturated numeric-value carrier has 2185 reversible states, 1457 raw future classes (also indexed OUTPUT observer classes), 729 typed future classes. Single positive reflection separates all raw classes; negative reflection emits zero raw output without erasing transported history. Exact 10/11/12-bit consumer-relative indices and successor maps; raw numeric-value audit integrated into existing compact acceptance, not IEEE-byte or fitted quotient. Existing orthonormal six-row semantic decoder q, S=q^T, ranks 6/90 and all 12 intertwiners give unique minimum-change repair y*=S A_u qh+(I-Sq)T_u(h), preserving entire complement and retained raw signed defect. Requires uncapped admitted source and expert0 selected when defect nonzero; otherwise active-only repair impossible. Explicit modified positive control, not unchanged-source/fitted equivalence. Known typed task gradient w.r.t. raw next history vanishes under fixed-target repair; unknown consumer must actually observe residual to learn it. Local exact reference and reconstructed active CPU source: 26220 distinct value-state edges per dtype, all raw/typed/indexed class laws, exact dense12 history, raw mixture output differences up to 1.42e-14/7.63e-6 FP64/32; SymPy projection and 729 dyadic repair/gradient cases per dtype. Full committed package/native/loader/kernel unexecuted. No training, CI, migration or release.

| Gate | Existing constructor / closed part | Exact remaining weld or authority (no duplicate theorem) |
|---|---|---|
| M0 / R1 | original literal/guard/endpoint, independent signed/Gram consumer, SemanticInterventionEquivarianceExact | Real source/tokenizer/checkpoint/parser authority and document-disjoint prefix-visible independent semantic action, admissibility and labels. Can formalize a given action, cannot infer authentic source rows or intended labels from geometry. |
| M1 / P2–P5 | ModularFourierGrokkingC3Exact, GrokkingTaskCharacterPhaseExact, known_action_register, finite signed recurrence, wrong orientation controls | Known task all-word closure exact-real hidden15, conditional FP hidden12; hidden12 exact-real unbounded state equality open beyond restricted obstruction, not needed for the finite consumer. Unknown task requires independently declared action/target; geometry may solve it if specification determines it. |
| M2 / G8 | actual controlled orbit descent, paired seed, compact dense18/15/12, semantic_adjoint/corrected_source_step | For modified uncapped source with expert0 active or zero defect, exact semantic transport by construction. Original-source equality iff zero correction (ideal real), prove qT_u=A_u q on actual fitted admitted carrier or exhibit defect. Do not assume arbitrary residuals preserve compact parity/cap/domain. Concrete source interpretation and formal kernel/native refinement separate. |
| M3 / P3–P5 | existing fit_joint, independent natural generator fitting, common Hom/gauge_audit, raw singleton reflection discriminator | Actual fitted state q/section from natural contrasts; same-object common invertible H, anchor rank/condition, independent typed readout/action and causal separation. On given exact generators solve linear equations and determinant nonvanishing; no SGD needed to solve the weld. Missing fitted weights and receipts prevent source-specific determination; do not substitute constructed law. |
| M4 / P6 | master FutureQuotientInvariantRealizationCompilerExact and FutureSufficientInvariantSubspaceExact, #1550 trace descent, constructed 729/1457/2185 hierarchy, reversible residual split | For actual fitted O,T/admissibility prove same-code observations and successor stability, or produce separating word. Correct pre/post observation indexing. Finite closed actual graph -> canonical refinement, not merely bounded samples. For arbitrary full-domain affine quotient use existing kernel iteration; nonlinear fitted domain not inferred from compact finite carrier. |
| M5 / G6 | signed permutation rounding, affine full-cell criterion, existing interval/guard/carry, stabilizer codec, minimum-norm q correction | Actual native arithmetic and signed zero byte replay, local guard/cap/input reachability, certified error on actual fitted matrices. Active and capped repair conditions exact at ideal-real scope (§§30–31); native rounded cap, reachable guard and fitted residual transition remain; no reduced precision claim from dyadic fixture. Reopen raw tensor rather than assuming FP subtraction is byte inverse. |
| M6 / P1–P6 | FullLearningStateFutureQuotientExact, LLMGrokkingLearningFutureExact, LearningUpdateMechanismSeparationExact, phase_reverse_solve, existing snapshot/RNG/AdamW owners | Reopen prior 40 actual audit receipts/optimizer state (unavailable here), determine anchored harmonic/phase/circuit residual, future learning observation. Exact optimizer map if state/data/RNG/backend specified; actual learned trajectory and time-to-quality cannot be obtained from abstract phase theorem. No rerun until existing evidence consumed. |
| M7 / N1–N5 | SemanticInterventionEquivarianceExact, WeightSpaceProgrammingLilaE8InterventionExact, prior paired intervention geometry and orientation firewall | Real independently defined semantic action and compatible acceptable-set invariant or causal paired evidence. Explicit necessity/sufficiency and matched controls; source geometry never itself creates semantic authority. |
| M8 / R2–R5 | exact consumer indices, existing packed codec/precision and native L2/L4 receipts, original release owners | Native runtime validation, same-consumer bit/byte/latency/VRAM/compute accounting incl residual/correction/marks, independent original-task heldout evidence. Original orthogonal Q/K gauge, regularizer, grammar feedback, true BPC and physical efficiency remain distinct. |

Stopping rule: mathematical premise already supplied -> construct; finite declared source -> close exact graph or separate; fixed weights available -> deterministic algebra/causal audit; missing source -> blocker, not fabricated or train again; only unidentified unknown-structure learning advantage -> matched source learning experiment. Explicit residual projection is a useful supplied-law control, not evidence for emergent C3. Preserve original R1–R5, F5 reserve and author attribution. Actual raw/provenance data, hardware receipts and kernel acceptance are not paid by these exact finite source contracts.


### 26 November — existing 729 carrier reuse, DC pointed action (M1/M4/M5)

[Source geometry §29](X369_LILA_STABILIZER_CODEC_20261125.md), commit 797975002b5d22454832d4f2fdd628fa2b4aa475. Actual master appraisal/X6 BIDI, pointed 8*90+9, affine-plane address, monomial Heisenberg, 27x27 pair, extended ternary Golay and hyperfabric refinement owners recovered; #1291 original appraisal ladder and #1545 729 Weil reuse checked. Exact chosen DC-adapted chart C(M,D)=(m0,d0,m1-m0,m2-m0,d1-d0,d2-d0) mod3 identifies central nine with spatially constant semantic pairs; 80 outer nine-state cosets, reuse actual pointed address order. Direct signed skew-product action and inverse implemented in existing x369_constructed_raw_quotient.py 18833d595826ddc0e845d780374c41517fd1f5b0, compact self-test 65f26b1050622932fb2471f2aa9eedcc902a7fe0. Local extracted exact checks: 729 roundtrips, 8748 control edges, 104976 composition checks, 12 SymPy orthogonal and block-triangular identities; 76 signed orbits (not typed quotient). Rotation fixes nine quotient cosets; eight noncentral fixed cosets have no fixed lift, hence no equivariant section, retain inner nine-state shift. Typed/raw/reversible class counts unchanged 729/1457/2185. Existing Heisenberg model normalizer applies to original signed X6 action, not selected Monster operator/phase/gauge or original appraisal relation-field authority. Golay protected-wire consumer remains optional distinct from hidden width; preserve arithmetic vs semantic-order trit labels. No same-number identification of X369 residual dimension90 with Monster multiplicity90. Local exact extracted equations, not full import/native/kernel; zero training, no other branch changes.


### 26 November — sharp bounded-history cut (M2/M5)

[Source geometry §30](X369_LILA_STABILIZER_CODEC_20261125.md); executable `scripts/x369_capped_semantic_geometry.py`, commit f6d346afd890cdd4531cae01f204db3d4843b618, local/GitHub identical blob c403a0704043102d85d9dfd732b3b21f42dd61ca. Original selected-expert per-branch cap source inspected. Fixed complement: unique h*=q^T z+Rh, exact selected branch cap and inactive-expert iff guard. Zero-complement uniform cap sqrt(9/2); 61/729 targets fit cap1. Free complement: balanced right-inverse pair with sharp uniform cap sqrt(24-12sqrt(3)), attained dual lower bound. Different consumer contracts; no radial clipping or silent residual change. Exact rational/SymPy run 729 targets and Gram/section/negative controls PASS; zero training. M5 cap mathematics paid, native rounding/guard, fitted reachability and unknown residual preservation still open; M0/M1/M3/M4/M6/M7/M8 unchanged. No source model mutation, CI, GPU/kernel or release.


### 26 November — CURRENT cap-profile BIDI and uniform residual admission (M2/M4/M5)

[Source geometry §31](X369_LILA_STABILIZER_CODEC_20261125.md), existing cap script commits 4746ed3108167ae8f090a42d645f4fc08804418f / ce1e13c3214e7703aa51dfa4c6dfee40cddde96f. Recovered actual master ConsumerFibreRepairExact, FutureSufficientInvariantSubspaceExact, SpectralResidualFutureDistortionExact; existing fitted separator, consumer closure, interval and origin-ray source owners inspected. #1550/#1291/#1545 heads unchanged; #1554 now 7914ec131be9, #1548 4c99285ef35b (metadata only on new heads). No duplicate generic theorem. Exact 14-scalar cap-energy profile k=(q0r0, eight branch energies) has iff equality with entire target-indexed energy observer (zero + six basis probes invert); signed action v->(-1)^b A_g v with branch swap. Attained all-target cap maxima in 24 local corner evaluations, independent of 729 enumeration; witness target for infeasibility. Exact real profile-image inequalities and symbolic section, generic 76-dimensional fibre; retained raw collision proves section is not inverse on original residual. Local committed blob 633093c8e32f53a06b7897a3d3256aaef60bdca9 exact run: 8748 target checks, 24 maximizers, 12 BIDI recoveries, 144 signed checks, 12 positive/12 negative image checks; previous cap certificate passes. No training, CI, GPU/kernel, fitted claim or model mutation.

| Gate | Current next exact weld / deliverable |
|---|---|
| M0 | Existing source admission / typed task authority: original source/tokenizer/checkpoint, independent labels and disjoint lineage; no geometry can authenticate absent rows. |
| M1 | Constructed known action paid; unrestricted original hidden12 exact-real state equality separate; unknown task action first specify independently, then solve deterministic geometry. |
| M2 | Original unchanged source qT=Aq iff zero defect on admitted source; modified source fixed-complement cap candidate iff selected branch and inactive guards. Actual fitted weights/source and reachable residual needed. |
| M3 | Existing common-Hom owner; solve exact linear equations and invertibility on supplied original fitted generators; anchor and unseen-word receipt, no SGD for algebra. |
| M4 | Constructed 729/1457/2185 hierarchy and cap-energy profile observer paid. Exact next weld: same (z,k) and admissibility -> same actual next profile + all declared observations; otherwise replay separating word; use existing future compiler / fitted witness, retain full residual until closure. |
| M5 | Ideal-real fixed/free complement cap, all-target admissible residual domain and profile image paid. Exact remaining weld: native cap/rounding guard, reachable domain and selected mask, calibrated error; actual source cap applies only selected branches. Existing interval owner, not a new interval system. |
| M6 | Reopen original 40 audit snapshots, optimizer/RNG/source cursor; determine actual phase/learning future from supplied receipts. |
| M7 | Existing paired intervention owner; independent residual-sensitive consumer, necessity/sufficiency and matched control. |
| M8 | Same-consumer native physical cost including profile/residual, marks and corrections; retain original release evidence. |

Only unidentified unknown-structure learning advantage may require matched training; deterministic source audit and native execution are distinct. No user training handoff. No local artifacts in chat; code/certificates remain committed here.


### 26 November — CURRENT M4 cap-profile descent cut

[Constructive geometry §32](X369_LILA_STABILIZER_CODEC_20261125.md), source fixture and affine certificate commit 348774426d66929e8dd8dc21bb3355e29383c9c9. Two exact full-domain cap-admitted counterexamples: same z/profile gives strict ordered routes (0,2) vs (0,1), and separately identical route (0,1) gives next expert1 energies (1,1) vs (0,0) under a triad projector. No universal cap-profile future sufficiency, no assertion of fitted/seed reachability. Exact affine entire-sphere product criterion: vector descent iff all active blocks vanish; energy descent iff offset and cross-block Gram terms vanish and each block Gram is scalar identity. Zero-radius blocks removed. Guarded subsets, nonlinear/capped source and actual fitted reachability remain separate. Local independent exact rational/source-equation reconstruction and symbolic sign-flip/Gram controls PASS; full committed CPU fixture and native GPU/kernel not run, no training. Reuse existing #1550 trace, master fibre repair/compiler, actual fitted separator, consumer closure, and interval/guard owners; retain original residual until actual future closure.

M4 exact next deliverable: (1) reopen actual fitted checkpoint, source/task and existing receipts; (2) for a declared admitted domain partition by actual frame/stabilizer/ordered route/orientation/cap guards, evaluate same-profile observation and next-profile descent; full-sphere affine criterion only where its premises hold; (3) if a collision separates, use existing ConsumerFibreRepairExact to refine via its signed witness, then recheck closure, not automatically discard all 76 fibre dimensions; (4) finite closed actual graph -> existing canonical compiler, otherwise record bounded negative/conditional result. Original byte identity separate. M0–M3/M6–M8 source authority, fitted Hom, learning reopen, independent causal consumer and physical cost unchanged. M5 ideal-real cap admission paid; native numerical cap and source guards open. No training handoff, no new generic theorem, no original model changes.


### 26 November — CURRENT exact affine future residual realization (M4)

[Constructive geometry §33](X369_LILA_STABILIZER_CODEC_20261125.md), executable `scripts/x369_affine_future_profile.py` commits d23f1e8c43f486dcb86231c8fc867918a963b24b and 46ff36764b015489f6b6dd01dd391a429ef9ab61. Recovered actual master CanonicalFutureObservableAlgebraExact (cb3740461be531fc473f3028e44b7ad70427d42c), actionPullback and pointwise canonical observable factorization; reuse existing #1550 trace/consumer repair. Exact affine homogeneous quadratic pullback closure with independent basis, literal generating words, decoder and latent transition matrices. Equal code iff all future observations for declared all-word affine system; minimal linear span only. Existing q recovered by exact_repair standard-basis defects; 20 initial source forms on 96 histories (6 semantic+6 signed cross+8 energies), ambient4753 quadratic dimension is ceiling not runtime size. Source-legal restricted four-triad contraction (1/2,1/4,1/8,1/16) energy rank4; exact unit-ball collision depths0–2, separates depth3; independent signed probe rank8, switched affine rank6, zero rank0. Local SymPy exact executed script matches Git blob 72f761f96042a5f18a4568e46e99c4f5b5702845; prior extracted identical exact_repair function used for 2 source-shape checks, not full package/native replay. No training, CI, fitted or Agda kernel claim. Actual fitted guard/frame/route/orientation/GELU/cap and legal switching must be proved or separately audited; arbitrary affine overapprox words not necessarily reachable. Retain signed residual and original source until actual future closure.

M4 next source weld: consume actual checkpoint and existing native receipts; for affine admitted cells use this exact compiler, for nonlinear guards use existing actual fitted separator/finite canonical refinement. Do not promote bounded samples, source-legal synthetic matrices, or a minimal form span to global fitted-state compression. M0/M1/M2/M3/M5 numerical/M6/M7/M8 source, independent action, recognition, precision, learning, causal and physical gates remain as previously scoped. No new training request, no files in chat.


### 26 November — CURRENT nonlinear GELU / cap stage weld (M4/M5)

[Constructive geometry §34](X369_LILA_STABILIZER_CODEC_20261125.md). Existing original PairedHistoryMoE, affine residual, indexed residual and future-form compiler inspected; #1550 unchanged, #1554 9a3959667309 and #1291 5f5c0e307d56 metadata rechecked. Exact selected-branch formula pre=s^2 Lh+s^2 Mx+s(Nb2+bT)+sNW2 GELU(p) for fixed admitted orientation/activation, including s=0. Homogeneous coefficient-dependent stage pullbacks interpolated via 0,+/-e_i,e_i+e_j using existing compile_future. Algebraic overapproximation closure is sufficient with factored guards/coefficients; not actual nonlinear minimal quotient or legal word witnesses. Separate actual pre-cap energy forms, independent branch scale, subsequent frame/stabilizer, inactive identity. Full96 assembly implemented on exact source layout, not full96 closure rank. Final script commits d15ff7e21894e1a0abeac30c98d639765f7878d5 / f6420f7d917ef1774b6e7c729674d146e795e91b / cce689f52959e0683f49b8e69cb5dece1b711d70; local SymPy 1.14 final Git blob 03f2fb333ff6ce946c0754fb0f909192879b1a89 matches remote, actual affine helper bodies copied. Three sign cases, rank3 symbolic GELU/active cap1/4 from admitted branch, rank6 mixed coefficients, 96-coordinate synthetic assembly four active/four inactive and four preactivation/energy/scale checks PASS. Assembly uses rational test coefficients, not fitted GELU/scale values; no full import/native/kernel or training.

Remaining M4/M5 weld: actual fitted source/receipt authority, exact frame/stabilizer/route/orientation/fallback guard factorization or retained original source receipt, actual admitted stage sequence, source-bound coefficients and independent consumers. Pre-cap norm must not be confused with post-Reynolds or residual energy. Full guard value retention can cost rank96; do not assert guard minimality from sufficient rows. Actual fitted minimal quotient and native cap/rounding remain open, use existing guard/interval, fitted separator and canonical finite compiler. M0/M1/M2/M3/M6/M7/M8 unchanged. No original model changes, no CI, no user training handoff, no files in chat.


### 26 November — CURRENT exact controlled guard/frame cut (M4)

[Constructive geometry §35](X369_LILA_STABILIZER_CODEC_20261125.md), scripts/x369_guard_geometry.py commit 39674868d4edc4670b7fc397f631a3522029d639. Existing controlled canonical, signed group, origin-ray and nonlinear stages inspected; #1550 unchanged, #1554 06ce2150d0ce, #1291 03067d08a673 metadata rechecked. One maximizing tie set J determines first mark m and local stabilizer K=J m^-1, with 12/6/4 candidate frames after control conjugacy max. At most12 membership bits for current discrete guard, not autonomous recurrence or bit-minimal claim. Exact signed paired G and Reynolds P matrices; selected D yields h'=G^T D P h_raw_local+(I-D)h, preserve native inactive restoration. Full-domain cap profile + exact raw mark needs additional affine linear rank84 on ker[Q;V] (rank12 signed initial rows); any remaining direction yields cap-admitted antipodal collision with identical semantic/profile/extra values and different marks. Not a nonlinear codec or consumer-relative/reachable-domain lower bound; internal mark may cancel in downstream equivariant consumer. Local SymPy script exact blob 5b6bcc19ee6462351c01bdf7ae550e7c3acedfde matches remote, after dense timeout sparse PASS: 8748 frame/tie checks, 144 group products, 36 full96 projection cases, 84 antipodal basis checks. Extracted original group and decoder helper bodies used, no full package/native/fitted/kernel/training. 

Next exact M4 guard weld: current discrete J sufficient for frame/stabilizer only; show next J and source-selected relative logits, ordered ids, tie policy, orientation/fallback, preactivation and pre-cap scalars factor through actual fitted consumer code, or provide actual separating word and retain witness. Do not force raw mark as independent observation unless declared; use downstream world consumer invariance to eliminate unnecessary guards, and existing future quotient/interval owners. Source-valid guard switching, fitted checkpoints and numerical replay remain; no artificial training or generic theorem duplication. M0/M1/M2/M3/M5 native/M6/M7/M8 remain as scoped.


### 26 November — CURRENT relative routing / orientation / gauge weld (M4)

[Constructive geometry §36](X369_LILA_STABILIZER_CODEC_20261125.md), scripts/x369_route_orientation_weld.py c899c99dfeb97d96f121bcd88de955f0ed0dcff7. Actual original c3_orientation, directed features, TopK capture, interval and precision guard owners inspected; #1550 unchanged, #1554 c46293ab4fab, #1291 03067d08a673 metadata checked. Exact first-nonzero-triad orientation: if local embedding x nonzero, both reflection zero-score fallbacks depend only on x, not history; x=0 retains first history triad, all-zero sign0. Learned dot guard distinct. Ideal R4 score relative form b_e+w_abs dot |mean_e|+fused_activation(Rx,w_C3) dot mean_e; adjacent selected and last-vs-outsider comparisons suffice (3 for E4 top2); exact mixture weights require only selected relative gap; weak ties require authentic original source receipt. Three source-legal independent collisions preserve z/profile/J: distinct weights with same ordered IDs/signs, distinct learned orientation with same logits, distinct fallback at zero embedding. Exact controlled frame covariance m' g m^-1 in K: local guards shared under simultaneous input/control conjugation; invariant world consumer can omit mark under that transformation, not arbitrary fixed-control recurrence or signed world output. Local exact Fraction checks 1458 fallback,729 directed,972 order (228 weak nonstrict),3888 covariance,3 separators PASS, exact blob d901766ad921b43d625dfb3b5f84adf8c27d7e1b verified; local extracted cap import repaired with original body, not full Torch/native/kernel replay. No training.

Remaining exact weld is actual fitted next-guard and consumer factorization under source-valid action words, using existing affine/nonlinear stage and canonical future compiler. No generic guard minimization or training needed to settle deterministic geometry; actual original fitted checkpoint and source authority still necessary. Native tie/precision remains separate. M0/M1/M2/M3/M5 numerical/M6/M7/M8 as scoped. No files in chat or source mutation.


### 26 November — CURRENT composed recurrence / delayed guard cut (M4)

[Constructive geometry §37](X369_LILA_STABILIZER_CODEC_20261125.md), scripts/x369_guarded_recurrence.py commit af69e7c30aa2a1ffcfba83e5dfb2497cd6df9b27. Existing original fitted discriminator, checkpoint integrity, finite refine/witness and source guard/stage owners inspected; #1550/#1554/#1291 metadata unchanged. Composed exact symbolic controlled reference (rational guard inputs, strict top2 only, nonrational successors need separate guard evaluation) executes frame, route, orientation, GELU, cap, Reynolds, inverse frame and inactive restoration without semantic target. Rich current code includes semantic/profile/J/ids/gaps/signs/preactivations/precap energies/world output, not raw history or learned label readout. Two-control same-code separator and stronger same-control delayed guard separator: selected expert1 swaps triads 3/4, exposes +/-4/5 coordinate to next route; same code now, same next semantic/profile, next route differs. Four-state one-control closed finite graph, existing refine/witness gives 1->3->4 classes, shortest distinguishing word depth2, 4 edges. Separate exact nonlinear cap/Reynolds fixture 4 pre-cap +4 active cap norms and projected zero PASS. Local final executed blob 9293468bcf7cdb039279ccdb9e8de013616d4f10 verified against remote; initial rational-helper integration fixed, extracted source helpers only. No fitted/native/full-package/kernel/training/CI claims.

M4 remaining actual source/consumer authority: original checkpoint and native receipts, source-valid complete controls and readout, then actual same-code observation/successor factorization or separating word with retained residual. Finite quotient only on actual closed graph; conditional staged compiler only with proved guard descent. No further isolated generic guard framework, no original model changes, no user training handoff. M0/M1/M2/M3/M5 numerical/M6/M7/M8 remain scoped.


### 26 November — CURRENT source-bound future consumer admission (M4)

[Constructive geometry §38](X369_LILA_STABILIZER_CODEC_20261125.md). Actual new-source acceptance and native integration owners inspected; historical R4 delta cannot silently migrate. No accessible authenticated new-architecture checkpoint located in changed-path/default code search; actual local Torch2.10 CPU-only, HIP=None. New scripts/x369_source_future_acceptance.py uses original native_source and actual step_controlled/consume, byte states and actual named byte outputs/IDs/weights/signs/logits, repeats each visited edge, rational z/profile candidate; existing refine/witness on exhaustively closed graph, direct negative witness retained even at budget. Native acceptance opt-in --future-budget N --future-scope all-literals (10 actual literal/tag controls) or fixed (4), immutable original checkpoint/source/runtime admission unchanged. Commits 375623720167ed25823ec58e1846d97c342d12a6, f58629734d50828b356daa3b790a1a6937c01589, f84fe4f6ae167fcff197dfad3b798e1ab5c1e9a0. Final local executed Git blob 828c4304d405209f7da8b3422c95ed4a5384f246 exact matches remote; actual extracted refine/witness used, four adapter controls PASS (delayed pair, positive finite closed, budget nonpromotion, budget with negative witness). Native adapter/CLI not executed here. Scope finite graph only; no deployed codec or general precision proof, discovery timeout not whole-run hard deadline, no intra-stage resume. Need frozen new-source original checkpoint and actual gfx803 receipt to settle remaining fitted result; no more generic guard machinery or training to pay deterministic obligations. No CI, F5, original model mutation, historical migration or files in chat. M0–M3/M5 numerical/M6–M8 remain as scoped.


### 26 November — CURRENT concrete finite future realization and BIDI replay (M4)

[Constructive geometry §39](X369_LILA_STABILIZER_CODEC_20261125.md). Actual master FutureQuotientInvariantRealizationCompilerExact edabf6c1ed6a421cab69e12aa07fbec82b2bd639 and StablePartitionCanonicalFutureBridgeExact 2cbe6f31562defdea84099d9247b3f48f8db8670 recovered, #1550 unchanged / #1554 7624d68f9863 / #1291 553a63e5f000 metadata. Concrete stable-class quotient table, original representative section and exact JSON byte/rational source receipt; verifier checks all original edges, consumer and chart descent, reachability and minimality using existing refine, not duplicate framework. Closed graph only; independent original callbacks required to replay, compile verifies recorded graph not a second GPU run. Negative replay reopens original seed paths, legal word and exact differing observations; JSON native tuples restored and validated. Existing acceptance --future-export optional with --future-budget, native_future_contract shared; host chart avoids redundant device copy, actual source still gfx803-only. Final local executed blobs e747a729a95b7eedd7756f2f8a034b08de405ab6 / c6e0f28b98006ba82841d602ff028b7d23960452 match committed remote: 4 states ->2 classes, 8/8 edges, 2/2 sections, lossless JSON, 5 invalid certificates rejected, delayed depth2/forged shorter rejection/tuple control replay, four existing graph controls PASS. Original new-architecture checkpoint and GPU runtime not available here; actual fitted closure and native execution remain outstanding. No full package/Agda kernel, training, CI, F5, historical migration or original model change. No physical cost claim; receipt may be large, export optional, discovery deadline not whole-run bound. M0–M3/M5 numerical/M6–M8 unchanged.


### 26 November — CURRENT exact source admission / decoder identifiability cut

[Constructive geometry §40](X369_LILA_STABILIZER_CODEC_20261125.md). Actual previous source admission had parse_args before future flags; repaired single parse and validated options, commits c5f1c2b34642 / 9856b06e6575. Existing exact decoder Hom/echelon solver retains full TRAIN affine solution family c0+N alpha (ac6163919419); original MEANS x TRIADS 189 signed seed states fix first paired-triad six coefficients (1,1,1,0,1,0), E4 W12 parameter rank6/nullity90, not recurrent-state compression or actual fitted rank. Held-out rows classify forced match, forced mismatch or adjustable; exact joint reduced-system contradiction checks all training-compatible equivariant linear decoders, never emit held-out-selected decoder. Source acceptance integrated one heldout stream with canonical + family receipt. Local extracted exact function-body checks via scripts/check_x369_decoder_family.py, commits a390b68dcbe1 / aff3394df685, matching regression blob a105aafbd772a490105b59c32fc9b7a0003c73a9: 189 seeds x90 kernel direction vanish checks, individually adjustable/forced/joint incompatible, free-pivot/offset, legacy/2 future scopes/3 invalid parser cases PASS. No fitted actual source, native GPU, complete package, Agda kernel, CI or training. DASHI prior master fibre repair reopened; #1550/#1554/#1291 metadata unchanged. Actual checkpoint missing in local inventory; original historical source cannot silently migrate. M3 actual common-Hom still requires independently measured fitted maps, M4 actual graph audit via repaired CLI, M5-M8 numerical/learning/steering/cost as scoped. No F5, original model mutation or training handoff.


### 26 November — CURRENT signed common-Hom and reconciled M0–M8 terminal cut

[Constructive geometry §41](X369_LILA_STABILIZER_CODEC_20261125.md) is the exact remaining-weld ledger. Reopened actual DASHI master cyclotomic, semantic-equivariance, fibre repair and signed-isotypic donor; relevant PR metadata #1550/#1554/#1291 unchanged. Exact common-Hom of arbitrary rational stored generator matrices: invariant closure of relator row images W gives largest exact relation quotient V/W; six signed sectors from DC/phase, reflection and central sign. Hom dimension sum multiplicity products, maximum rank sum dimension*min multiplicities; constructive forward/partial reverse, true inverse iff equivalent. Failed relator excludes invertible similarity but may leave smaller exact factor, reverse then only on quotient. Original independent endpoint target; joint self Hom6, residual self Hom2. New scripts/x369_common_hom.py commits 33cec08d1d93 / c8c9fea08131; actual local SymPy1.14 final blob 204e40bf670532c4f4ba48f127204e7612a51f7a equals remote, exact conjugate/sector/relator/quotient controls + four Kronecker Hom dimension checks + exact dyadic JSON + cache adapter PASS. Optional --hom-contexts uses actual native_source histories and existing natural fit_joint (complete contexts, no label input to transition, no partial fit on timeout); source acceptance ba7961aaf652; extracted parser default/combined/negative PASS. Native adapter not executed, local CPU-only and no authenticated fitted checkpoint; exact stored-matrix verdict not approximate-real exclusion. Existing phase, precision, factorial intervention and cost owners rechecked, exact source-specific welds listed for each M0–M8 in §41. No generic duplicate framework, CI, training, F5, migration, original-model change or kernel acceptance. Actual frozen new-source checkpoint, native receipt, independent causal labels, optimizer trajectory and matched physical cost remain distinct evidence dependencies.


### 26 November — CURRENT simultaneous source/action/consumer graph (M1/M2/M4)

[Constructive geometry §42](X369_LILA_STABILIZER_CODEC_20261125.md). Actual source, decoder, native graph/refiner/realization, independent task endpoint and label vocabularies reopened. DASHI #1550 actual X369InputIndexedFutureConsumerExact 8b8c9d4c447dba49c8c2029494188753ff038d7b and master future compiler/fibre repair/semantic intervention reused; #1554 7268c6ae5df2 / #1291 553a63e5f000 changed paths checked, not exhaustive. Exact seed equations B(h_s)c=z_s and edge equations (B(n)-A_u B(h))c=0 iff decoder on closed reachable graph; equivalent independent path label propagation plus existing fit. Same actual byte source edge/consumer graph, separate labels; equal source state/different label collision excludes any deterministic state decoder; linear conflict distinct from nonlinear table possibility. Budget negative retained, positive only closed. Constructed signed defect q(n)-A_u q(h), actual selected IDs and cap fixed-complement repair admission (no state mutation); raw logits unique correctness/ties via original label vocabularies, separate from decoder. Joint finite realization preserves chart, semantic label and source consumer, independent verifier replays seeds/edges and existing reachability/minimality. New module commits 1357459b81f1/e2ccc30978e9/507aa8c3f20b, source CLI d468a91b69f4 --simultaneous-weld requires --future-budget, optional export/ten-literal envelope or fixed assignment. Final exact local regression blob 7734f0007ddba4300e08da92d4c02b47ae67ee8f matches remote; five graph cases, two signed defect checks, five cap cases, JSON joint replay/forged label rejection, 7290 task comparisons and extracted CLI checks PASS. No full native source package, GPU or Agda kernel execution; CPU-only local runtime and original checkpoint absent. M0 actual source, M1/M2/M4 actual result, M3 independent measured generator correspondence, M5 calibrated backend bounds, M6–M8 physical/causal/trajectory receipts as §41 remain distinct. No training, CI, F5, migration or model mutation.


### 26 November — CURRENT M5 controlled-source numerical refinement

[Constructive geometry §43](X369_LILA_STABILIZER_CODEC_20261125.md). Actual original interval/precision/frame/paired source/guard/route/readout owners reopened; master ApproximateIntertwinerCompositionExact 0272cb209752f and #1550 input-indexed theorem reused, #1554 c498219ed8e1 / #1291 859fe570b65c metadata. New scripts/x369_controlled_interval.py composes exact correlated signed-frame comparisons, relative TopK and zero-aware orientation (unresolved guard stops), rational sqrt/normal-CDF GELU enclosure, existing softmax, affine/cap/mixture/Reynolds/inverse frame/inactive identity and actual local readout/label action. Cap crossing continuous, ideal cap nonexpansive even below norm floor via scaled projection. Optional explicit product/sum-tree affine rounding bound, all 12 external stage budgets required for conditional backend mode; common unary rounding not canceled, actual HIP kernels not authenticated. Native one-edge adapter reuses original gfx803 byte-source/consumer callbacks, compares actual observations only, no implicit zero budget or CPU fallback; not integrated into existing admission CLI. Commits 4e789e165610 / bf3f6b595450. Local final executed blob 9fbcbd7174ac1367ddc2358a6072138df227dd1b matches remote; extracted actual helper bodies, 8748 frames, 4 sqrt, 41 GELU high-precision sanity, 54 exact emulated rounding, cap crossing/projection/guard controls and independent nonzero SymPy source-shaped point/corner enclosure PASS; first inverse-frame Interval type failure fixed. No native/fitted/full package/kernel result. M5 conditional stage propagation paid; remaining actual source-bound per-stage calibration and replay; M0–M4 actual source, M6–M8 original indexed/causal/physical evidence as §§41–42. No training, CI, F5, migration or model mutation.
