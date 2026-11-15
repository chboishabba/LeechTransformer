# X369 C2 / C3 / C3²: consolidated maximum-cut roadmap, exact obligations, phases and receipts
Date: 2026-11-14. Owners: LeechTransformer PR #4 (neural code and measurements); DASHI Agda PR #1442 (formal counterpart). This is the operational owner for the X369 neural lane. Preserve prior roadmaps and their historical evidence.

## 0. Evidence taxonomy and provenance

- EXACT-SOURCE: declaration inspected in Agda or Python; does not imply a fresh kernel execution.
- HOST-PASS: test/run reported from the user's machine; preserve the command, exact commit, data and output hashes to make independently auditable.
- IMPLEMENTED: runnable code exists; no full physical run or scientific result implied.
- OPEN: the stated proposition or measured condition is not paid.
- COUNTEREXAMPLE: an actual failing original-object witness can close an existential "or counterexample" target, but must not be promoted to a positive universal result.

Remote GitHub PR #4 reached 9aa6ad0 on 2026-11-14. Subsequent host-only device-argument and test-boundary fixes were reported after checking out that detached head; 55/55 across 11 suites was reported on the host. Until the exact new commits are pushed and their SHAs reconciled, do not claim the remote 9aa6ad0 itself passed those tests. Its original tests/test_frozen_c2c3_lockbox.py uses an invalid 14-token near-overlap example (210..223), which should instead use 208..223 (16 tokens). Wrapper supplies --device cuda to the finite modular Fourier audit; the script must accept it even when CPU-only. These two source corrections are prerequisites for a clean PR.

Empirical immutable reference:
- Original source: repaired C2 checkpoint, R64 retained history, early even gains, prior heldout CE 4.5966; original 4.7630 and frozen teacher 2.5834 are DIFFERENT models. That heldout was previously viewed and MUST NOT be re-used to choose architecture.
- 9 development windows / 576 correlated tokens, teacher pre-FFN input local residuals, initial expert history zero. Layer 0 raw rotating fraction 0.6688, bootstrap interval [0.6666,0.6711], coordinate signed-permutation p=0.850, centered isotropy defect 0.0302, C3 covariance commutator 0.0518. Layer 1 0.6748, interval [0.6712,0.6784], p=0.742, isotropy defect 0.1243, C3 commutator 0.2125. 1999 permutations/bootstraps on each layer.
- The signed-axis null checks privileged coordinate grouping, NOT multivariate isotropy, population C3 symmetry, or a learned algorithm. Pooled 3x3 triad covariance is NOT complete hidden dxd covariance. Cluster bootstrap interval for rotating energy and the coordinate-permutation p-value answer different questions.
- Prior C2 x/-x paired odd/even cross cancellation on that sampled source is compatible with source C2 parity, NOT proof of all-state recurrent C2 equivariance.
- Host-reported 55/55 are source contracts only; NOT a GPU F4 experiment or F5 lockbox receipt.

## 1. Carrier identity: do not collapse three distinct C3 objects

1. Source model c2_invariant: absolute-value routes on concatenated activation and per-expert retained history. No intrinsic C3.
2. Source model c2c3_wave: cyclic three-coordinate wave3 Fourier rechart and ordered self/cyclic/anti-cyclic relational invariants in models/trialectic_wave_prior.py and models/history_orbit_moe.py. Its router has input width 2d and samples C3-invariant functions, not a demonstrated learned phase representation.
3. Source model punctured9_wave: pointed-eight -> zero-mean nine -> four conjugate character pairs of F3², with retained independent centre, from models/punctured_nine_wave_prior.py. Its compact router features have width 3d/2, and are NOT capacity-matched by the existing two-arm screen. Translation on punctured eight re-gauges the centre; neither ambient Leech lattice automorphism nor semantic triality follows.
4. Exact modular task: Z/3Z addition, eight-of-nine memoriser versus algorithmic nine-pair answer; finite mathematical Fourier factorization lives in DASHI Cognition/PNF, not automatically in this language model.
5. Physical NS integer Z³ Fourier modes, YM 3x3 block indices, quantum qutrits, orbital labels and Monster character sectors each have different carriers and separate same-object map obligations.

Source algebra that SHOULD be paid before any causal claim: I(Rx,Rh)=I(x,h) for cyclic R and directed trialectic channels I; however generically I(Rx,h)!=I(x,h). Witness x=h=(1,0,0): original (self,cyclic,anti)=(1,0,0); rotate one argument only to shift which relational channel is active. A simultaneous 120-degree rotation is a NEGATIVE control (intended invariance), not a phase-destroying ablation. Scramble relative phase at a specified routing site and compare equal-norm random controls. For punctured9_wave the diagonal translation-invariant pair features similarly should be distinguished from one-sided changes.

## 2. Named exact obligations: existing declarations versus proposals

Existing inspected Agda source declarations (not a new kernel receipt):
- ModularFourierGrokkingC3Exact: algorithmicRuleIsCharacterComposition, memorizerAndAlgorithmTrainingEquivalent, memorizerAndAlgorithmNotFullTaskEquivalent, heldOutPointSeparatesMemorizationFromAlgorithm.
- GrokkingEquivarianceDefectGeometryExact: algorithmicEquivarianceDefectIsZero, memorizerEquivarianceDefectIsTwo, characterGeometryStrictlyImprovesTaskActionDefect.
- CyclotomicGrokkingActionLinearizationExact: algorithmicPairLinearises, phaseOneActsByOmegaOnEveryState, heldOutCyclotomicFactorization.
- OggSSPP2CenteredTernaryGroupExact: centerSheet-assoc, centerSheet-inverse, sheetPhaseRoundTrip, sheetToPhase-preserves-addition; punctured plane owner includes puncturedNineSheetCountIsThreeSquaredMinusOne.
- C3FourierConjugationExact: zetaCubedIsOne, chi1ConjugateIsChi2, chi2ConjugateIsChi1 and related finite rows.
These are finite carrier results: not source checkpoint action theorems.

ALL identifiers below are PROPOSED proof/acceptance obligations. They are not claimed to be implemented declarations or proved lemmas. For experimental entries the exact statement is a source-bound executable assertion with hashes and a falsification alternative; do not confuse with Agda theorem.

### A — finite C3² same-carrier formal closure
A1 c3x3SameCarrierGroupLaw — original FiniteThreeCycleTorusExact carrier equals the F3² indexed nine-site group with identity, inverse and 81 compositions.
A2 c3x3CharacterOrthogonalityOnOriginalCarrier — Z[omega] character pairing sum and all nine character orthogonality on the same group.
A3 c3x3FourierConvolutionSameCarrier — pointwise product/convolution and inverse Fourier operator commute as stated on that carrier.
A4 c3x3ConjugationParsevalAndReality — inversion, real conjugate pair, full Parseval/plancherel and centre mode.
A5 c3x3PunctureCentreGaugeBidi — source pointed-eight embedding, Fourier inversion and independent centre/re-gauge commute.
A6 c3x3AgdaKernelSelectedSourceAccepted — repository-pin, exact Agda checker invocation, no metas/postulates for selected proof path.
CURRENT: host 4/4 Python integer cyclotomic Fourier contracts and source finite Agda donors; the same-carrier cross-module Agda bridge and fresh kernel receipt remain OPEN.

### S / F0 / F1 — original source parity and statistics
S1 originalLeechC2PairedOddEvenCrossCancellationAllRelevantStates — upgrade the finite teacher-input mean cancellation to named source states and recurrent histories, or construct counterexample.
S2 originalLeechSignedHistoryRecurrentActionBidi — selected actual student source signed expert-state update, top-k ties, capacity, and history cap match formal semantics over >=32-token trajectories.
S3 originalLeechSameQKGaugeVersusGeometry — distinguish cancelling shared Q/K orthogonal gauge from actual geometry-dependent computation at identical checkpoint.
F0 originalResidualC3PlaneFractionNullCalibrated — DONE empirically for the named nine-row sample; do not upgrade to full isotropy.
F1a originalResidualTriadPooledCovarianceEqualsSourceReduction — raw M, mu, Sigma, cyclic commutator and eigenvalue reduction agree with original source data.
F1b originalResidualC3CommutatorClusterUncertainty — row-cluster uncertainty for D_C3 itself; predeclare null/equivalence margin and familywise control.
F1c originalHiddenFullCovarianceVsPooledTriads — measure full dxd covariance, individual triad heterogeneity, non-stationarity and cross-triad blocks.
F1d originalResidualPopulationC3SymmetryOrCounterexample — conclude within predeclared tolerance only with independent rows and power; never infer symmetry from p>0.05.
CURRENT: F0 audit measured; F1 raw pooled defects measured; population and full hidden tests OPEN.

### F2 — actual task and learned representation
F2a originalC3InputLabelActionTyped — nontrivial R on genuine examples with an independently verified output label action rho; existing arbitrary language tokens do not come with one.
F2b originalC3HiddenLayerSiteAndGaugeWeld — tensor layer, token, input/hidden action rho_h, gauge, source checkpoint, recurrent history and consumer are pinned.
F2c trainedC3TaskActionEquivarianceDefectBound — measured normed h(Rx)-rho_h(R)h(x) on unseen source-disjoint examples, across seeds/layers and source states.
F2d trainedC3LearnedPhaseReadoutIdentifiable — learned decoder/readout u, not hard-coded cos/sin, with held-out phase labels and gauge/rotation controls.
F2e originalC3TaskActionVsCoordinatePermutationNoGo — counterexample demonstrating that arbitrary hidden triad rotation does not define a semantic input action.
F2f originalC3OutputConsumerEquivarianceWeld — logits, next-token or task labels actually respect the proposed action, with off-target/nuisance preservation.
CURRENT: all OPEN on trained original LM; finite modular-algorithm task control exists separately.

### F3 — composition, mechanism and causal intervention
F3a learnedC3CharacterPhaseNontrivial — learned u(a) is distinguishable from collapsed constant/label-leaking memoriser and carries a nontrivial character.
F3b learnedC3PhaseCompositionHeldout — u(a)u(b) approx u(a+b mod3) on never-trained combinations, with preregistered error/equivalence thresholds.
F3c originalC3SimultaneousRouteRotationInvariant — same-source proof I(Rx,Rh)=I(x,h) for selected source route; detect failure/ties if action changes dispatch.
F3d originalC3RelativePhaseChangesDirectedChannels — exhibit x,h and R with I(Rx,h)!=I(x,h), preserving x norm, history and explicit intervention site.
F3e originalC3PhaseInterventionReversibleAndScoped — selected intervention exact inverse, same scaling, no inadvertent kernel/hidden-state changes.
F3f learnedC3PhaseCausalExcessLoss — paired CE deterioration after relative-phase scrambling exceeds equal-norm random/gauge controls with seed/row-cluster interval.
F3g learnedC3OffTargetAndMemorizerControls — unrelated tasks maintained and eight-of-nine memoriser shows expected false phase-composition failure.
F3h originalC3InterventionSourceConsumerBidi — actual source router/expert history trace matches formal/intervention observation, including sign C2 and order of sites.
CURRENT: F3c/F3d finite identities available from code formulas but no dedicated selected-source kernel receipt; remaining original learned mechanism OPEN.

### F4 — paired two-arm original-model screen
F4a originalC2C3RouterOnlySourceIdentity — original repaired source hashes, strict shared state, R64/early gains and non-router weights frozen.
F4b originalC2C3MatchedParameterSchedule — count, seeds, fit rows, 24 steps, optimiser, batch/device and top-k/precision effects paired.
F4c freshC2C3DevelopmentDocumentDisjoint — original tokenizer revision, unique record IDs/text hashes and no exact >=16-gram overlap vs both previous sources; reject data leaks.
F4d matchedBoundedRouterPredictiveGainOrCounterexample — per-seed delta = CE(C2C3)-CE(C2) on fresh development, with predefined meaningful effect and error intervals.
F4e matchedRouterBestStepSelectionExposed — acknowledge best development step optimistically selected; do not call it independent final evidence.
F4f originalC2C3RoutePhaseActionMeasured — record actual router logits, top-k IDs/weights and phase-specific controls alongside CE; raw CE alone is mechanism-agnostic.
CURRENT: source and code implemented, candidate models can be checkpointed; true fresh validated dataset and native two-seed run OPEN.

### F5 — one-time independent locked confirmation
F5a frozenSelectionIndependentOfLockbox — winning geometry precommitted by development mean before reading lockbox; C2 wins ties.
F5b frozenLockboxCandidateSHAAndDataDisjoint — source, both checkpoints, tokenizer, original train, prior heldout, F4 development and final test SHA/record IDs, exact ngram checks.
F5c frozenC2C3PairedCEOrCounterexample — on SAME untouched final dataset, paired across seeds, compare both geometries and original repaired C2 source; do NOT contrast against 4.5966 from a different split.
F5d originalC2C3History32PlusBoundAndReplay — >=32 token recurrent provenance, cap R64, exact selected state/outputs with independent replay and numerical errors.
F5e originalC2C3FinalStudentAgdaBidi — source-to-Agda-to-source test of selected states/consumers, not just abstract invariant route algebra.
F5f originalC2C3MeasuredQualityComputePareto — actual throughput, peak memory, energy if measured, selection overhead, active and total parameters; matched hardware.
F5g originalC2C3LockboxProtocolNonReuse — official test manifest used only for one preregistered comparison; non-overwrite is NOT a technical guarantee against peeking.
CURRENT: first-pass scorer implemented, host 55/55 contracts reported after local fix; no lockbox receipt or full F5 admission.

### F6 — separate genuine C3² nine-sheet comparison
F6a originalPunctured9AndC3TriadCarrierDistinction — explicit 9-sheet/8 puncture to LM tensor chart with centre state, not a 3-triad alias.
F6b originalPunctured9CapacityMatchedRouterAdapter — control 2d versus 3d/2 input widths via a preregistered common parameter/compute budget; preserve all non-router source weights.
F6c originalPunctured9TranslationAndRelativePhaseSourceAction — source-specific F3² action (nine translations), diagonal invariance and one-sided counterexamples.
F6d matchedC2C3C3SquareDevelopmentGainOrCounterexample — three-arm, paired-seed, source/fit/validation matched, separately chosen development set.
F6e frozenC3SquareIndependentLockboxAndMechanism — one untouched test after selecting among three architectures, plus functional phase and causal checks. Never retroactively add F6 to an already-opened F5 lockbox.
CURRENT: exact mathematical chart and model implementation available; matched three-arm evidence and trained same-object theorem OPEN.

### X369 cross-lane source welds — parallel, not prerequisites for narrow neural comparison
X369-A c3x3ExactCyclotomicCharacterConvolution — A1-A6 on original finite torus; OPEN Agda same-carrier join.
X369-B atomic369PhysicalOrbitalRepresentationIntertwiner — original 369 labels and genuine antisymmetric Hamiltonian eigenfunctions/spectral filling; OPEN.
X369-C nsPeriodicGalerkinNonlinearPhaseConvolutionBidi — actual integer Z³ periodic Fourier/Leray nonlinear trajectory, not quotient Z3²; OPEN.
X369-D ymOriginalWilsonC3BlockFourierTransport — side-4 SU(N) source Wilson and gauge block index, Haar/Ward/RG physical estimates; OPEN.
X369-E quantumC3TritQutritFourierIntertwiner — same Hilbert register and unitary Fourier action, not classical trit code only; OPEN.
X369-F lilaGrokkingC3PhaseVsC2RepairedRepresentations — F2-F6 original real neural evidence; OPEN.
X369-G monster369RegularTracePhaseVsRiemannZetaTypeBoundary — type-safe separation cyclotomic omega, Monster traces, Epstein and analytic zeta; selected same-object map OPEN.
Any NS/YM/atomic/quantum/Monster cross-lane relation requires the corresponding original carrier and physical map; a shared label 3/6/9 is not such a map. E8/Leech lattice theta/zeta all-index and source-model lattice action stay separately OPEN.

## 3. Phases, deliverables and exit gates

| Phase | Parallel owner | Deliverable | Exit (do not skip) |
|---|---|---|---|
| P0 Source/revision | Leech+Agda | SHA ledger, 55/55 host receipt and cherry-picked device/test fixes on PR, provenance hashes | source + host and remote heads reconciled |
| P1 Finite/negative formal | Agda A1-A6, F3c/d | same-carrier F3² character proof, centre gauge, relative-phase counterexample; local Agda kernel output | no postulates/metas for selected closures, finite vs neural boundary explicit |
| P2 Source action and statistics | Leech F1a-d, S1-S3 | covariance cluster uncertainty, full dxd and recurrent C2 history probes | reproducible source hashes, correlations correctly clustered, counterexamples preserved |
| P3 Functional mechanism | Leech+Agda F2a-f and F3a-h | typed task actions, learned readouts, unseen composition and controlled relative-phase interventions | causal/functional result OR exact negative evidence, no reliance on 67% |
| P4 Matched 2-arm screen | Leech F4a-f | real disjoint development corpus, two-seed/24-step full original matched CE + frozen checkpoints | SHA-bound fit/validation and per-seed paired receipt |
| P5 Frozen confirmation | Leech F5a-g | one independent final lockbox score, source-control CE, 32+ replay, BIDI/resource profile | preregistered positive/negative verdict; no reuse of lockbox |
| P6 Full C3² hypothesis | Leech+Agda F6a-e | capacity-controlled three-arm analysis, new development and genuinely new final test | cannot promote F4 two-arm result to C3² |
| P7 Cross-lane physical welds | Agda X369-A-G | exact same-object maps or counterexamples, per-domain physical input and independently checked proof receipt | no semantic, analytic, physical or lattice claim without source map |

## 4. Dependency DAG and maximum cut

P0 can be closed independently and urgently. P1 formal finite source and P2 full statistical/recurrent source audits can proceed concurrently. P3 (task action + phase readout + causal manipulation) does NOT depend on F4/F5 availability if an actual modular-task network/labelled paired dataset is selected, but MUST NOT be silently transferred to Leech original-language source. P4 requires trustworthy fresh development data. P5 depends on completed P4 models and a distinct NEVER-SEEN lockbox. P6 requires a brand-new preregistered comparison to prevent lockbox reuse. P7 has its own original physics and proof frontier; its completion is not needed to answer the narrower neural hypothesis.

Highest information cuts:
1. Integrate/re-run host 55/55 and receipt; preserve user's detached commits.
2. Formalise same-source simultaneous/relative triad and F3² group-action control with a real selected model state.
3. Run true learned-phase F2/F3 probe with hidden-layer and task actions, not the hand-coded Fourier reference alone.
4. Obtain exact document-level disjoint development + reserved final manifests (no selection on final), then F4/F5.
5. Keep F6 as separate three-arm protocol and cross-lane A-G as parallel proof stream.

## 5. Success/falsification and release policy

- F0/F1 only certify finite measurement. Neither high coordinate permutation p nor 2/3 energy supports "learned C3".
- F2/F3 can fail: output valid counterexample and null/memorizer differences; success requires task semantics, low defects on unseen examples and selective causal change.
- F4 negative delta in development is suggestive, not unbiased; F5 negative paired delta on locked test supports useful C3-feature routing conditional on named corpus/model/budget. No single two-seed result establishes broad superiority or phase computation.
- If F4/5 show predictive improvement but F2/3 fail, attribute to generic route-feature change, not C3 circuit. If F2/3 succeed without CE gain, report learned mechanism without claiming useful LM improvement.
- The 4.5966 control cannot be compared numerically with a CE measured on a different locked dataset. Both frozen arms and original repaired source must score the same test data.
- Explicit no-promotion: no Monster/J invariant, D4/S3 triality, true Leech automorphism, atomic physical orbital, C3² quantum mechanics, NS/YM regularity or semantic argument-face success without independently paid same-object evidence.

## 6. Concrete deliverables

D0: source commit + canonical SHA/receipt reconciliation; D1: Agda finite/relative-phase proofs with kernel log; D2: F1 full-covariance/cluster-uncertainty receipt; D3: F2 learned equivariance/readout dataset and diagnostics; D4: F3 norm-preserving selective ablation + memorizer/gauge controls; D5: F4 document-disjoint validation manifest, per-seed checkpoint hash and CE curves; D6: F5 untouched official lockbox manifest, original C2 control scores and 32+ replay; D7: F6 independent matched C3² architecture test; D8: X369-A-G same-source separate formal proofs/negative controls.

Owner of each deliverable must publish exact commit SHA, source paths, executable command, data/tokenizer hash, seed, expected/fail-closed outputs, mathematical meaning, counterexample route and original-source attribution. Do not mark OPEN as DONE simply because a script has been written.

## 2026-11-14 wide implementation tranche — A/B/C, not small sequential handoffs

**Evidence cut:** The previous 55/55 host acceptance remains credited ONLY to the prior source head. This tranche adds code, not a fresh RX580 run. Independent local CPU checks exercised the F1 covariance algorithm (4/4) and source-equivalent F3 C3/C3² feature equations (4/4); the comprehensive repo suite on this new head is NOT yet accepted, and an Agda kernel receipt is NOT reported.

### A: selected finite/source action
- DASHI PR #1442 now includes `DASHI/Cognition/PNF/X369FiniteTorusAndRouterActionWeldExact.agda`: paired character eigenactions on the EXISTING `FiniteThreeCycleTorusExact.Torus3x3`, commuting order-three generators, directed triad polynomial simultaneous-rotation identity and a nontrivial one-sided exact witness. **Source-written, Agda checker OPEN.** Its simple nonnegative integer semiring witness does NOT yet instantiate real signed activations.
- Leech `scripts/audit_c3_source_actions_maxcut.py` explicitly evaluates diagonal C3 and one-sided C3 on the source `directed_trialectic_invariants`, plus all nine diagonal and one-sided F3² character-pair translations on the source punctured-nine wave chart. The latter preserves **Fourier pair norm**, not necessarily naive punctured-eight norm. Includes separate 4-test regression owner; neural learned action remains OPEN.

### B: F1 covariance and F3 causal source-probe
- `scripts/audit_c3_covariance_f1_maxcut.py`: per-layer pooled 3×3 and COMPLETE hidden d×d centered covariance, full cyclic action defect, per-triad heterogeneity and off-triad covariance fraction, matched-row cluster bootstraps across layers, Bonferroni two-sided intervals, exact residual-file SHA. New `tests/test_c3_covariance_f1_maxcut.py`. This upgrades F1 *instrumentation*, not population equivalence or independence of nine dev documents.
- `HistoryOrbitMoE.select_source_experts` now has default-OFF, non-state-dict `route_probe_action` hook. Acts only on source-owned ROUTING FEATURE INPUT, not FFN/expert activation or persistent history: `c3_relative_shift`, `c3_diagonal_shift`, `c2_diagonal_negation`, and fixed-seed `signed_axis_permutation_x`. The latter is equal-norm, **not equal intervention displacement**.
- `scripts/audit_c3_causal_router_ce.py`: frozen F4 C3 checkpoint, SHA-bound F4/source/training/previous heldout/development inputs, mandatory separate reserved F5 lockbox exclusion, disjoint F3 probe rows, row-level next-token CE for all four actions plus no-action control, paired row-cluster intervals. Its output is mechanism-localisation evidence **only when run**; it does NOT establish a task input/label action, learned u(a), phase composition or causal C3-specificity by itself. `tests/test_c3_causal_router_ce.py` adds source-probe/reset and paired-bootstrap checks.
- F2a-f and F3a/b/f/g/h remain *OPEN* despite this executable instrumentation. No original LM learned-phase readout or meaningful symbolic task generator has yet been welded.

### C: separate F6 original-model three-arm screen
- Existing F4 2-arm defaults are preserved. `scripts/screen_bounded_c2_vs_c2c3_router_fresh.py --include-c3-square` adds a clearly DIFFERENT, opt-in F6 three-arm DEVELOPMENT screen: original C2, C3 triple features, and `punctured9_wave_matched`. This is NOT retroactive promotion of the old two-arm F4 results.
- `HistoryOrbitMoE` has a distinct `punctured9_wave_matched` fixed nonlinear feature lift: compact 3d/2 Fourier character pair features plus d/2 sign-log magnitude features. This yields exactly 2d router input width and strict same-shape initial source weights under `clone_with_geometry`. It is an **experimental adapter**, not the untouched original compact punctured9 architecture, nor proof of equal effective feature rank, compute or mathematical information.
- F6 requires original official F4 screen and F5 frozen-score receipts, matching SHA of both excluded data sets, original source checkpoint hash and newly disjoint F6 development data. It writes a distinct receipt kind, per-seed 3-arm CE, and rejects capacity-count differences. `tests/test_c3_square_three_arm_f6.py` checks equal router tensor count/initial weights and exclusion of old screens.
- F6 independent final lockbox, source F3² learned composition and three-arm generalisation remain **OPEN**. An F6 development winner is not an independent locked inference.

### Minimal host acceptance and execution order

1. Re-run `bash scripts/check_c2_quality_maxcut.sh` under the pinned native Schaka Python; compare results against the former 55/55 only after auditing the NEW test count. Our local CPU source-only checks do not replace this.
2. Run `scripts.audit_c3_source_actions_maxcut` under the existing launcher, verifying the new source-action report is finite and diagonal controls have tiny defects while one-sided controls are nonzero. Then run `scripts.audit_c3_covariance_f1_maxcut --residuals <layer0.npy> <layer1.npy> --output <new.json>` on pinned existing residuals; 1,999 row bootstraps, 0.05 familywise alpha.
3. Provide original-tokenizer, document-ID-hash audited fresh development and reserved lockbox data; run F4 and store both models. Run F3 router-only causal CE on a THIRD, disjoint, document-audited probe set, with mandatory reservation of F5 lockbox.
4. Score official F5 ONCE, with locked original-source C2 control. Only after those receipts exist run the F6 3-arm mode with a FOURTH disjoint development set, and reserve a DIFFERENT F6 final test. No train/eval on the original previously viewed heldout for architecture selection.

**Remaining exact gate truth after this tranche:** A1-A6 substantive convolution/orthogonality, centre gauge, Agda kernel still OPEN; B1/B2 finite polynomial version SOURCE-WRITTEN not kernel accepted or trained-source signed-real welded; F1b/c scripts IMPLEMENTED but measured new data OPEN; S1–S3 OPEN; F2 all OPEN; F3 trained task/phase all OPEN except source function maths; F4/F5 real data, paired CE and BIDI/resource OPEN; F6 three-arm implementation READY but separate fresh development/final evidence OPEN; X369 physical A-G source claims OPEN. No false "all completed" assertion.

### F2 original-student task-defined C3 orbit probe — source implementation

`scripts/audit_trained_c3_task_orbits.py` now consumes explicit C3 orbit input triples `[document_groups,3,context]` on the original tokenizer, plus external input/output action manifest, unique document IDs and SHA-bound exclusions from original fit/previous heldout/F4 development/reserved F5 final sets. It freezes the original student checkpoint, hooks the selected original hidden layer, trains a **separate supervised 2D linear phase readout**, and measures heldout orbit equivariance and nine phase-product/composition defects. `tests/test_trained_c3_task_orbits.py` covers exact Fourier control, noncompositional counterexample and document-ID fail-closed behavior.

Scientific firewall: labels and orbit semantics are supplied externally, not inferred or independently verified by script; the readout was **trained with explicit phase supervision**. Even near-zero heldout defects would demonstrate task-action decodability for this probe, NOT that original LM learned an autonomous modular addition circuit, NOT causal necessity, and NOT universal semantic triality. Independent task/label source attribution, a frozen test, unlabelled representation discovery and selective downstream intervention remain OPEN.

**Required operator inputs for F2:** original repaired checkpoint and teacher, C3 orbit train/eval token arrays, task action manifest, original training/heldout/development token sets, a reserved untouched final set. Never pass the original already-viewed 4.5966 heldout as architecture-selection or newly published final test evidence.

### F6 separately frozen independent three-arm confirmation — implemented, NOT run

New `scripts/score_frozen_three_arm_c3square_f6.py` closes the PROGRAMMING gap between a three-arm development screen and a legitimate three-arm final comparison:
- Require source-identical F6 screen receipt and three frozen original-model checkpoints per seed.
- Preselect among `c2_invariant`, `c2c3_wave`, `punctured9_wave_matched` on **F6 development means only**; tie-break C2 then C3 then C3².
- SHA-bind original repaired C2 checkpoint, original fit/previous heldout, prior official F4 development and F5 final with both receipt chains, new F6 development, new F6 final, and optional additional previously viewed inputs.
- Reject exact 16-token overlap and duplicate new final rows; score all three frozen arms and original repaired C2 control on the SAME never-tuned dataset, by seed.
- Record nominal matched router parameter count, but DO NOT assert effective feature-rank/compute equality, document-level independence, an actual learned C3² circuit or unmeasured resources.

`tests/test_frozen_three_arm_c3square_f6.py` covers development-only architecture precommit, conservative ties and missing/duplicate checkpoint exclusion. Full new host suite execution and actual F6 final dataset remain OPEN. The old F5 final cannot be reused as F6 final.

**Current exact remaining high-alpha gates after full source implementation:** P1: pinned Agda kernel for A1/A2 and remaining A3/A4/A5; P2: actual full F1 data/commutator CI and full recurrent signed-history BIDI; P3: task-labelled, independently reviewed C3 orbit source and original learned readout/causal CE observations with phase-specific controls; P4: actual new document-disjoint original F4 two-arm run; P5: genuine reserved lockbox and 32+ state replay; P6: brand-new three-arm dev and final data and original RX580 run; P7: original NS/YM/atomic/quantum/Monster physical same-object maps. None can be promoted on implementation alone.

**Authoritative local acceptance commands:**
```bash
bash scripts/check_c2_quality_maxcut.sh
# Source algebra (CPU code via GPU wrapper's pinned environment):
bash scripts/run_native_schaka101_lila.sh --module scripts.audit_c3_source_actions_maxcut -- --output /path/to/unique_source_controls.json
# F1 on actual prior source residual arrays:
bash scripts/run_native_schaka101_lila.sh --module scripts.audit_c3_covariance_f1_maxcut -- --residuals /path/to/layer0.npy /path/to/layer1.npy --output /path/to/unique_f1_report.json
# F2, F3, F4, F5 and F6 additionally require original checkpoint and
# independently sourced input manifests listed in each module --help.
```

**Evidence cutoff:** existing 55/55 host run applies to a prior Leech commit. New checker includes new test suites, but no full new-head source or GPU execution is claimed here. All new source files are committed to PR #4.


## 2026-11-14 additive A03/A04/A05 + F2 absolute-label + F6 independent max-cut

**Scope/evidence distinction.** The earlier 55/55 host report is an historical
contract suite receipt, not acceptance of these later commits. The following
source additions are on the **same** LeechTransformer PR #4. The exact finite
certificate was exercised locally in a source-matched nine-site test harness
(3 finite tests passed; 6 task-action fail-closed tests passed). Neither that
preflight nor any GitHub connector source write executes the current full
Schaka native suite, a current frozen checkpoint, the RX 580, or an Agda kernel.

### A. Original-carrier exact spectral and pointed-centre algebra

- `scripts/certify_x369_original_torus_maxcut.py` uses the actual
  `models.punctured_nine_wave_prior.SITES` ordering and exact integer pairs
  for `Z[omega]`, with `omega^2 + omega + 1 = 0`. No floating threshold or
  random-vector argument is used.
- A03 `c3x3FourierConvolutionSameCarrier`: 81 inverse Fourier matrix
  coefficients, 729 bilinear convolution coefficients, and 729 independent
  pointwise-product/frequency-convolution coefficients. Thus the finite
  identities are verified for arbitrary functions by linearity/bilinearity
  of their nine basis coefficients. **Exact Python certificate; Agda same-carrier
  theorem and selected kernel receipt OPEN.**
- A04 `c3x3ConjugationParsevalAndReality`: 81 sesquilinear Parseval
  coefficients and 81 character-reality coefficients, plus **729 complete
  centred Gram coefficients**, not merely the nine diagonal basis vectors.
  Real and complex field claims are confined to this exact finite algebra;
  **Agda all-coefficient proof OPEN**.
- A05 `c3x3PunctureCentreGaugeBidi`: all nine independent centre/offset basis
  recoveries; 648 one-translation re-gauge coefficients; 5,832 ordered
  two-translation induced-action coefficients. True invariant norm is the
  centred nine-sheet norm, with the independently retained origin. Explicit
  no-go: a delta at the original centre has naive punctured-eight offset norm
  squared **8**, whereas translation by (1,0) re-gauges it to **1**.
  **Source-defined finite pointed-chart certificate; Agda wave/centre BIDI
  and original trained-model mapping OPEN**.
- `tests/test_x369_original_torus_maxcut.py` requires all above counts and
  checks Eisenstein conjugation, inverse translation, and the false-norm
  counterexample. Run: `python -m scripts.certify_x369_original_torus_maxcut`
  and `python -m unittest discover -s tests -p 'test_x369_original_torus_maxcut.py' -v`.
- A01 group law and A02 exact orthogonality remain source-written on DASHI
  PR #1442; A06 requires the **actual** pinned Agda checker. Independent
  symbolic coefficient verification does not silently close A03/A04/A05
  inside Agda.

### B. Typed F2 action, orbit gauge, and learned readout boundary

- New `scripts/certify_c3_task_orbits_original_tokens.py` checks an
  *optional* executable `machine_action_certificate` on the **original
  token-ID arrays**, rather than accepting a prose `input_action_rule`.
  Version 1 defines an exact right cycle of three named token positions and
  external labels transformed by `+1 mod 3`; each fit/evaluation orbit must
  close after three actions, be nontrivial, have exactly one of each label,
  and preserve all untouched positions. Forged positions, forged source
  triples, corrupted labels and collapsed orbits fail closed.
- The certified machine action is **structural only**. Those task labels were
  supplied externally and are not independently established natural-language
  semantics. No automatically inferred C3 transformation on arbitrary
  text, Leech lattice symmetry or output-logit intertwiner is claimed.
- Critical correction in `scripts/audit_trained_c3_task_orbits.py`:
  absolute phase labels are now used when training the readout and when
  scoring held-out character composition. A valid C3 orbit may start at
  label 1 or 2; using orbit index as absolute label gives a false
  composition defect. Numerical negative control with exact shifted
  characters: mean defect about **1.1547** before alignment versus
  roundoff zero after alignment. `tests/test_trained_c3_task_orbits.py`
  now includes shifted-base and nonbijective-label regressions.
- Without an executable machine certificate, the old exploratory readout
  remains accessible, but the output records structural admission **false**.
  With a certificate, the exact input and claimed finite label actions
  are checked, but the result still records semantic correctness and an
  autonomous trained C3 circuit as **false/unverified**.
- Gate F2a is now **PARTIAL (machine structural witness implemented)**.
  F2b/F2c/F2d/F2e/F2f and F3a/F3b/F3e/F3f/F3g/F3h still require
  actual checkpoint activations, independently meaningful labels,
  identifiability/null controls, source-consumer action and causal evidence.
  Existing F3c/d engineered router negative controls remain mathematical
  source tests, not learned-mechanism results.

### C. F6 frozen three-arm guard and operational tests

- `freeze_three_arm_winner` now denies the F6 final lockbox unless there
  are at least **two preregistered paired development seeds**. Conservative
  C2/C3/C3² tie-breaking, complete candidate path/SHA sets and distinct
  F4/F5/F6 data still apply. The regression expressly rejects a one-seed
  F6 final-score attempt.
- The original `scripts/check_c2_quality_maxcut.sh` adds the two new suites:
  `test_x369_original_torus_maxcut.py` and
  `test_certify_c3_task_orbits_original_tokens.py`; the shifted-gauge and
  one-seed tests extend preexisting suites. **No new aggregate suite count
  is claimed until the pinned Schaka interpreter runs this new head**.
- F4/F5/F6 actual data/checkpoint training, independent held-out scoring,
  32+ recurrent history replay, effective compute-capacity matching,
  cluster intervals and actual energy/throughput Pareto remain **OPEN**.
  The source-code F6 adapter only matches router tensor dimensions, not
  functional rank or training cost.

### Exact remaining lemma and deliverable matrix, current cut

| Gate | Concrete acceptance producer | What is still unpaid |
|---|---|---|
| A01/A02 | DASHI original-torus group/81 orthogonality leaves | Fresh Agda 2.9 kernel receipt, selected safe/transitive source audit |
| A03 | Exact 81+729+729 integer-pair coefficient witness | Agda theorem on original carrier using the source Fourier conventions |
| A04 | Exact 81 Parseval + 81 reality + 729 centred Gram checks | Agda conjugation, Parseval, nine-mode reality, centre treatment |
| A05 | 9 centre, 648 gauge, 5,832 composition checks + 8-to-1 counterexample | Agda original pointed-eight/centre ↔ nine ↔ real Fourier BIDI |
| A06 | `scripts/check_x369_finite_original_torus_maxcut.sh` exists | Run pinned Agda and authoritative Track A, no postulates/metas |
| S1–S3 / F1 | Full covariance source and cluster/bootstrap runner exists | Original independent document-row sources, recurrent original state trace, source QK distinction |
| F2 | Executable exact token group + claimed labels, corrected readout orientation | Independent semantic task and output action; actual trained h/rho and unseen defect |
| F3 | Directed-router diagonal/relative probes and causal runner exist | Source-bound site-specific learned phase, matched controls, paired CE and mechanism verdict |
| F4 | Two-arm checkpointed development runner exists | Real source-disjoint new data, two-seed/24-step run, same-source CE |
| F5 | One-time frozen scorer exists | Preregistered untouched final, same-test original C2, ≥32 step BIDI, compute Pareto |
| F6 | Three-arm candidate and final scorer with minimum-two-seed guard | Truly independent final corpus, effective cost match, F3² learned mechanism |
| X369 A–G | Exact finite donors and domain-specific mappings identified | Physics/quantum/Monster same-object source maps; no promotion by shared labels |

**Execution order.** P1 Agda theorem/kernel proof and P2 real source-data
provenance can run in parallel. P3 should first bind genuine task semantics,
then rerun phase diagnostics with the corrected absolute-label alignment and
perform controlled relative-phase interventions. P4 uses genuinely new
document-disjoint development. P5 is one precommitted independent final
comparison, never an optimisation split. P6 gets its own independent F6
final; never reuse F5. P7 physical welds continue on their existing owners.
Keep exactly one active Leech neural rollup PR (#4), and one active DASHI
neural-formal rollup PR (#1442); do not create an issue/PR per lemma.


## 2026-11-14 continuation — A00 source crosswalk, F3 matched perturbation, P0 portable audit

**Commit lineage:** same Leech PR #4; no new standalone PR. Accompanies DASHI PR #1442 same-source formal leaves `X369OriginalSignedSiteOrderBidiExact`, `X369OriginalTorusSpectralBasisExact` and `X369OriginalPointedEightSlotBidiExact`.

### Accepted source crosswalk / remaining proof distinction

- A00 `x369PythonSignedSitesToOriginalTorusBidi`: GitHub-connector live-source comparison inspected the actual branch blobs `models/punctured_nine_wave_prior.py` (`d0d890c17ecaf18ea3efa14833405cd116038511`) and DASHI `X369OriginalSignedSiteOrderBidiExact.agda` (`1007f95db7f5fea41bf57dfacda57111a3971893`), and independently verified **9/9** exact signed site/centre table entries. This is **SOURCE-TABLE VERIFIED**, not an Agda kernel acceptance.
- New `scripts/audit_x369_agda_original_crosswalk.py` requires a REAL Agda worktree module path, validates all nine source positions/coordinate signs and emits SHA provenance; fixture-based regressions reject swapped slots and sign flips. Without `--agda-source`, no claim of live machine crosswalk is made.
- Agda A00 now provides explicit original-site slot coding, inverse original torus mapping, first/second original translation transport and phase-sign witnesses, all **SOURCE-WRITTEN, KERNEL OPEN**.
- Agda A03/A04 now has `X369OriginalTorusSpectralBasisExact.agda` with original-residue encoding under addition; original character multiplicativity; the ring's 9 phase multiplication and conjugation cases; original delta Fourier convolution coefficient; **81 inverse**, **81 Parseval**, **81 reality** basis proofs and **81** phase-middle-swap cases, all **SOURCE-WRITTEN; KERNEL OPEN**. Universal arbitrary-function convolution/Parseval, exact coefficient-ring ring laws and source Python Fourier wave chart remain separate signatures. The earlier Python integer coefficient witness is an independent oracle, not the kernel proof.
- Agda A05 now adds `X369OriginalPointedEightSlotBidiExact.agda`: the eight concrete source-ordered offset slots + independent centre; full nine-slot decode/encode and encode/decode *pointwise* under explicit cancellative algebra hypotheses. Four real Fourier character-pair roundtrip and original signed-real weight/consumer transport remain **OPEN**.

### F3 stronger source-bound causal comparator

- `models/c3_matched_router_control.py`: randomized, shared-source activation control preserving **both** ||x|| and ||x'−x|| relative to the actual `rotate3(x)` action, with deterministic seed and zero-row handling. The matched perturbation is router-input-only; expert input/history remains the source state. Four focused CPU mathematical tests passed in the isolated source-equivalent harness.
- `models/history_orbit_moe.py` accepts opt-in `c3_matched_displacement_random_x` at the existing `select_source_experts` hook; all experts receive the SAME perturbed x, preserving shared activation semantics.
- `scripts/audit_c3_causal_router_ce.py` includes this matched action and reports `c3_relative_vs_equal_norm_equal_displacement_random`, a paired cluster-bootstrap contrast beyond the older norm-only signed permutation. A causal effect here is **not** an independently meaningful C3 task/label action nor a proof of learned composition.
- Original-model integration regression in `tests/test_c3_causal_router_ce.py` checks determinism, shapes, finite weights and restoration. **Native full Leech suite, actual original-checkpoint CE intervention and cluster provenance remain UNRUN/OPEN.**

### Acceptance and remaining exact cuts

The local isolated, partial-source CPU harness passed **17/17** focused cases (prior nine plus four A00 fixture and four matched-displacement mathematical checks), but is NOT a complete checkout/native suite receipt. To reproduce on an actual Leech checkout, run `bash scripts/check_x369_portable_cpu_source_contracts.sh`; optionally set `X369_AGDA_A00_SOURCE=/absolute/path/to/DASHI/Cognition/PNF/X369OriginalSignedSiteOrderBidiExact.agda`. The pinned Schaka `scripts/check_c2_quality_maxcut.sh` registers the new suites separately. Agda selected check is `DASHI_NO_TMUX=1 bash scripts/check_x369_finite_original_torus_maxcut.sh` in DASHI and still lacks a real Agda 2.9 receipt.

| Cut | New evidence | Exact gate still open |
|---|---|---|
| A00 | 9/9 live GitHub source signed mapping | Agda kernel proof + audited source-to-actual-checkpoint refinement |
| A03 | Original torus delta coefficient source and Python full bilinear oracle | Kernel check and arbitrary original-field convolution theorem |
| A04 | Original torus 81-basis Parseval/reality source and full Python Gram oracle | Kernel check, arbitrary-field Parseval and real four-pair descent |
| A05 | Original explicit eight+centre conditional pointwise BIDI | Fourier pair chart, instantiated ring cancellation, kernel and trained-source weld |
| F2 | Typed source action + corrected absolute phase readout | Independent meaningful label action and actual neural representation/consumer |
| F3 | Equal-*displacement* causal null and paired reporter | Original-model run, causal specificity/cluster confidence, off-target and memorizer controls |
| F4/F5/F6 | Existing reproducible source pipelines with frozen guards | True fresh corpus and checkpoint outputs, independent lockboxes, native RX580, effective compute matching |
| P0/A06 | Source code and selected check commands | Fresh full repo tests, pinned Agda checker and CI/source revisions |

**Dependency release:** A00 selected-source/kernel and A03/A04/A05 original finite formal completion can proceed in parallel with F2/F3 original checkpoint probes and fresh-data F4 preparation. Do not open either F5 or F6 final set without fully independent data and preselected frozen models. Independent physical X369-B–G frontiers remain under existing original domain PR owners.

## 2026-11-14 23:00 Melbourne — host acceptance reconciliation and next integrated A/B/C cut

**Host report reconciled (source revision as explicitly identified by host):**
- DASHI eight selected original-torus modules on PR #1442 at `7c549c9892f5`: reported Agda 2.9 checker **0 errors, 0 warnings**; 8 leaf modules, module checks (64, 157, 194, 158, 196, 159, 65, 18); total 1011 reported individual checks. The fixes were made on host's local `test/pr1442`; this branch **was not on the GitHub remote**. Corresponding `field` → `fld` (excluding record declarations), `Set` → `Set₁` and unsupported Equality `cong` import cleanups have now been pushed directly on the existing DASHI PR branch. A new remote-head Agda replay is mandatory; the old host success cannot silently be assigned to the later PR head.
- Leech PR #4 at `e63229d2887f`: host reported **92/92** tests, 21 suites, plus 17/17 portable and live A00 signed map crosswalk. Those receipts belong to the **specified earlier commit**. This continuation adds F1 and F4/F6 new tests, so **the new branch head's full local suite is not yet accepted**.
- Host's F1 received residual data: **9 purported document rows, 1999 cluster bootstrap resamples**. Layer 0 **pooled** C3 defect 0.0518, CI [0.0403, 0.0686], off-triad Frobenius energy 93.54%; layer 1 **pooled** defect 0.2125, CI [0.1921, 0.2352], off-triad 99.26%. **Do not relabel these pooled CIs as full d×d commutator CIs.** Layer 1's pooled defect is inconsistent with tolerance 0.05 under the supplied source-row resampling protocol; layer 0 is inconclusive. Off-triad energy is an observed ratio, not its own independent population CI. Original source-document independence remains unverified by the provided numbers.
- Host's F3 engineered source-action results: diagonal C2 0, diagonal C3 ~2.22e−16; C3² diagonal ~5.55e−17; one-sided source feature difference ~0.2063. They demonstrate architecture/source invariants and negative-control separation, **not** a task-defined learned input/output action nor original checkpoint causal CE.

### New implemented F1: actual full-hidden bootstrap and source-row provenance

`scripts/audit_c3_covariance_f1_maxcut.py` now performs exact **full d×d centred covariance** recomputation per *source-row bootstrap* using sufficient statistics of each original row. It uses the SAME 1999 row resamples as the historical pooled audit and outputs:
- `cluster_bootstrap_full_hidden_commutator_interval`
- `empirical_bootstrap_full_hidden_defect_median`
- `full_hidden_against_predeclared_tolerance`
- `full_hidden_bootstrap_includes_off_triad_blocks: true`.

No bootstrap at token or triad level; no substitution of pooled invariance for full covariance. Source-equivalent numerical replay at hidden d=12 and d=384 against explicit full-matrix recomputation had roundoff-level maximum discrepancy (2.22e−16 / 4.44e−16), with zero invalid-draw admissions. Added regression tests including scaling invariance, exact comparison to brute-force replay, and nonidentification of pooled vs full interval.

Optional `--document-row-manifest` enforces a SHA-bound original row identity list with distinct source document IDs/text hashes and original residual artifact SHA256. A self-asserted manifest is explicitly **not** independent documentary validation. Without it, F1 receipt records `document_row_manifest_status: not_provided`.

Re-audit on the host, reusing existing residual arrays and not the original model:
```bash
python -m scripts.audit_c3_covariance_f1_maxcut \
  --residuals /path/to/original/layer0_residual.npy /path/to/original/layer1_residual.npy \
  --bootstraps 1999 --familywise-alpha 0.05 --relative-tolerance 0.05 \
  --output /path/to/new/x369_f1_full_hidden_bootstrap_v2.json
# Optional: --document-row-manifest /path/to/x369_f1_original_rows.json
```
Reaudit outcome is a **new empirical gate**, not available as a fabricated result here.

### New implemented F4/F6: real-document-bound fresh source generator

`scripts/prepare_x369_document_disjoint_fresh.py` accepts original source document **text and immutable IDs** as JSONL with `split="development"` or `split="reserved"`; uses the checked-in 2048-vocab original SentencePiece tokenizer; rejects 16-token window collisions against supplied historical token arrays AND across new development/reserved rows; rejects repeated document IDs and text SHA256; emits source ID, document hash, token offset and array/tokenizer/input SHA256. Refuses writing into nonempty output directories. It reserves the second set without training or scoring it.

A strict attempt requires `--historical-doc-inventory` with `kind=x369_historical_document_inventory_v1`, `complete=true`, list of historical source document IDs + text SHA256, and exact SHA256 of ALL supplied historical token arrays. An incomplete original historical document inventory can be used **only with `--provisional`** for development; no independent document-disjointness or release is claimed. Even an internally consistent user-supplied inventory cannot independently prove its own completeness.

```bash
python -m scripts.prepare_x369_document_disjoint_fresh \
  --documents-jsonl /path/to/independent_original_documents.jsonl \
  --tokenizer vocab/e8_morpheme_.model \
  --prior-ids /path/to/original_train_ids.npy /path/to/original_heldout_ids.npy \
       /path/to/all_previously_viewed_development_ids.npy \
  --historical-doc-inventory /path/to/historical_original_document_inventory.json \
  --length 128 --window 16 --development-rows 32 --reserved-rows 32 \
  --output-dir /path/to/new/untouched_x369_fresh_split
```
**For F6** additionally exclude the full F4 development and consumed F5 lockbox, plus all previously viewed test/probe splits; create a truly new independent F6 final set. Never use the reserved set to select a model.

Six regression scenarios added for exact prior-window and cross-split non-overlap, historical document omission, SHA-bound manifest, historical ID/text duplicates, source exhaustion, and token-width mismatch. These are source-written; the new native suite must run before acceptance.

### Exact post-host max cuts

| Lane | Now closed or substantially advanced | Still decisive |
|---|---|---|
| Agda P1/A00–A05 | Host 8/8 leaf acceptance at old SHA, repairs synchronized into current PR branch | Fresh current-branch Agda 2.9 kernel and Track A; A03/A04 arbitrary-field linear lift; A05 real Fourier four-pair ↔ pointed-eight |
| F1/S1 | Full hidden point covariance observed; pooled document-row CI and new full-matrix bootstrap code | Run **actual full d×d** 1999-row bootstrap, bind historical document identities, quantify off-triad uncertainty; reroute hypotheses if source hidden action fails |
| F2 | Absolute phase readout labels and structural original token group law | Independently meaningful task/label action and original frozen model consumer alignment |
| F3 | Source diagonal invariants, one-sided action, equal-displacement random route-null | Original source checkpoint row-CE against paired geometric controls with honest clustered uncertainty and an independently labelled task |
| F4/F6 | New strict original tokenizer/docs/16-window split producer; prior guard suites | Acquire genuinely new documents plus historical IDs, run same-source actual 2/3-arm screens, confirm effective compute/rank |
| F5 | Existing scorer, untouched final heldout protocol | One-time final after F4 selection only, no prior peeking |
| P0 | Host historical 92/92 and selected Agda 8/8 receipts | Fresh full native suite on new commit, fresh selected Agda replay, source/dataset provenance and receipts |

**Negative finding must remain visible:** a pooled covariance C3 defect at layer 1 with CI strictly above tolerance is evidence AGAINST native hidden-triad C3 symmetry under the measured protocol. Do not silently rebrand the hidden representation as equivariant. A learned restricted subspace, task-dependent action or explicit trained gauge may still be tested, but each requires a fresh separately measured witness, not merely source-engineered invariant features.

## 14 Nov 2026 late host replay — F1 full covariance observed, C3 PSD ceiling, and current-source admission

### Authoritative host-reported observations (not a new raw replay in this connector session)

- Host executed 1999 whole-original-row bootstrap draws over its TrueNAS student residual arrays (9 claimed original source-document rows; **independent document provenance still open**) and reported the raw JSON path `/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113/x369_c3_f1_full_hidden_bootstrap_audit.json`. This raw host file is not part of this GitHub PR and was not directly reloaded here.
- Original hidden width `d=384`, 128 consecutive triads; **full** commutator `D_0=1.33227`, CI `[1.33199,1.35126]`, and `D_1=1.41233`, CI `[1.41198,1.41259]`. Both lower bounds exceed predeclared `tau=0.05`; these are the full covariance statistics, not pooled surrogates.
- Observed historical pooled `D_0=0.05181`, CI `[0.04029,0.06859]`, `D_1=0.21247`, CI `[0.19213,0.23522]`. Reporting both prevents substituting the weak pooled L0 interval for the strong full-matrix result.
- Off-triad `||C - blockdiag(C_iii)||_F / ||C||_F = [0.9354,0.9926]`. This is a **covariance Frobenius fraction** (not the trace-based percentage of stochastic variance/energy).
- Host reports **8/8 selected Agda modules** exit 0, zero errors/warnings at synchronized remote `81542ade028f`, and **103/103 native Leech tests, 22 suites** in a host checkout referencing `4761459c7b30` with a local syntax fix in `prepare_x369_document_disjoint_fresh.py`. The syntax fix is now independently pushed into the PR's newer head; the reported 103/103 result must be attributed to the patched host working tree, not to unchanged remote `4761459`. Previous 92/92 is retained as historical. A fresh host run on the **current later PR head** is still required.

### Exact algebraic consequence of the F1 result — no unjustified learned-symmetry promotion

Let `C` denote observed symmetric positive-semidefinite original hidden covariance, and `P` the **specific globally coherent** consecutive-triad cyclic permutation. Define
```text
D  = ||C - P C P^T||_F / ||C||_F
E  = (C + P C P^T + P² C (P²)^T)/3
overlap = <C, P C P^T>_F / ||C||_F² = 1 - D²/2
noninvariant = ||C - E||_F / ||C||_F = D/sqrt(3)
invariant = ||E||_F / ||C||_F = sqrt(1 - D²/3)
0 <= D <= sqrt(2) for C >= 0.
```
At host point estimates, `overlap=[0.112528,0.002662]`, `noninvariant=[0.769186,0.815409]`, `invariant=[0.639024,0.578885]`. Layer 1 nearly reaches the PSD ceiling `sqrt(2)=1.414214`, so its **fixed-action covariance and translated covariance have almost orthogonal Frobenius geometry**. This does NOT establish absence of *all* possible nontrivial group representations after a learned basis change, nor a general population statement unless original document independence and stationarity are independently justified.

**New source:** `scripts/audit_x369_f1_rank_and_orientation_null.py`. It takes the **same existing residual arrays** and returns the exact PSD overlap identity, exact Reynolds-invariant fraction, full-covariance effective rank `(tr C)^2 / ||C||_F²`, and a source-calibrated random orientation null. The null randomizes the independently chosen +1/-1 nonidentity three-cycle direction **per triad**: same cycle eigenphases and order 3, but no imposed common orientation. It compares the particular original global orientation against a precisely specified structurally matched class; it is not a learned action, predictive CE or semantic symmetry witness.

```bash
python -m scripts.audit_x369_f1_rank_and_orientation_null \
  --residuals /mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113/layer0_residual.npy \
              /mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113/layer1_residual.npy \
  --orientation-null-draws 199 --seed 41 \
  --output /path/to/new/x369_f1_rank_orientation_20261114.json
```
**Check those raw filenames before execution**: the exact TrueNAS residual filenames were not provided, only the F1 JSON receipt path. Never infer that the example `layer0_residual.npy` files exist at the stated location.

Five new focused locally run structural tests passed: PSD `sqrt(2)` sharp extremum, coherent C3 preservation beating mixed orientations, deterministic seeded null, one-triad unavailability, and invalid source refusal. New tests registered in both portable and full native suites; their current-branch full native replay remains OPEN.

### Reprioritized maximum cut after negative F1 finding

| Priority | Exact work and outcome needed |
|---|---|
| **A/Agda universal** | A03 full original field Fourier convolution with explicit ring distributivity and 1/9 scaling; A04 arbitrary sesquilinear Parseval and real descent; A05 four-pair+centre source-to-original typed chart. Host selected leaves accepted; Track A open |
| **F1 model-discriminating** | Run new rank and coherent-vs-mixed C3 orientation audit on original residuals; bind original-document IDs/hashes and compare effective rank, trace/Frobenius diagnostic, layerwise covariance |
| **F2 action** | Independently define an actual task-level C3 action/label generator; source checkpoint's learned hidden/consumer response (not manually designed invariants) must be tested on unseen documents |
| **F3 causal** | Frozen same-source router-site relative C3, matched-norm+displacement null, untouched model baseline; paired original-document CE and adjusted intervals, noncollapsed phase and output-consumer agreement |
| **F4/F5/F6** | Fix now pushed Python 3.14 split-preparation syntax; obtain genuine new identified documents and original history inventory, then source-locked matched arms and separate final; never reuse old CE baseline for final promotion |
| **Cross-lane** | Static finite C3 indices do not validate physical NS/YM/Monster/VOA fields without domain owners' original same-object maps and checks |

The original finite action invariants still hold as **architectural identities**, while the newly quantified F1 full-hidden finding is a **negative result for covariance equivariance in the selected native hidden basis**. This updates, rather than erases, the hypothesis ledger.

### Post-rejection decision rule: trivial DC invariance is **not** a neural phase rescue

The rank/orientation diagnostic also computes
```text
Q_DC = blockdiag(J3/3, ..., J3/3)
s_DC = Tr(Q_DC C)/Tr(C)
s_rot = 1-s_DC
```
with 128 original three-coordinate blocks at `d=384`. The `DC` coordinate is the trivial C3 fixed vector on each triad; the remaining two coordinates are the real two-dimensional rotating character. `s_DC` and `s_rot` are *trace-based* variance shares, **distinct** from the Frobenius covariance ratios `0.9354,0.9926`. The values must be **measured on the original stored residuals**; no values are assumed here.

Any automatic projection to `Q_DC` has tautological exact C3 invariance but throws away the nontrivial rotating representation. It cannot satisfy `originalC3NoncollapsedRotatingPhase`, task-orbit composition or original output-consumer covariance by itself. The F2/F3 admission chain must therefore test rotating phase variance, learnable noncollapsed phase *and* predictive consumer effect, not just invariant router statistics. The new diagnostic emits `trivial_dc_fixed_vector_variance_share`, `rotating_two_character_variance_share`, and the explicit non-promotion warning. Five regression cases include both the sharp PSD rank-one counterexample and an invariant, noncollapsed rotating-sector synthetic control (synthetic **test data only**).

## 15 Nov 2026 local host handoff — F1 rank 1.67, F4 running, spectral and pre-F5 cut

### Host reports after exact-head replay

- On the reported synchronized source heads, **DASHI #1442 selected 8/8 Agda 2.9 leaves**, and Leech native suite at the then-current head **103/103** across 22 suites. These are host-supplied receipts; they are not new CI runs on commits added below.
- Physical F1 `audit_x369_f1_rank_and_orientation_null.py` executed against actual TrueNAS source residual arrays (receipt `/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113/x369_c3_f1_rank_orientation_audit.json`). Layer 0: `D=1.33227`, effective covariance *participation* rank 44.30, rotating two-character trace variance share 66.84%, mixed order-three orientation null mean 1.3304, quantiles [1.3270,1.3332], observed fraction below null 0.0955. Layer 1: `D=1.41233`, participation rank **1.67**, rotating trace share 64.78%, orientation null mean 1.4121, quantiles [1.4114,1.4124], observed fraction below null 0.1658. These null fractions are **not formal calibrated p-values**, particularly with only 199 null draws and a potentially adaptive selection.
- High layer-1 defect with effective rank 1.67 is consistent with near-rank-one source covariance being generically noninvariant under most fixed nontrivial rotations. This is still strong counterevidence for **the predefined coherent consecutive-triad generator**, NOT a basis-independent proof that no nontrivial order-three representation exists. Rotating coordinate variance near 2/3 is not independent evidence for a learned C3 circuit.
- Host reports collection of 600 real TinyStories source documents, with a new 32 × 64 development array and 32 × 64 **reserved, unscored** array in `/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113/x369_fresh_splits/`. Development array SHA prefix `881e8649539653b1`, reserved SHA prefix `7d13b31c73a7bc9d`. The split is **16-token-window disjoint with the supplied train and previously-viewed heldout arrays**; this does NOT establish independent document-level separation against the historical training corpus if source-document inventory is incomplete. The generator's receipt intentionally records `release_document_disjointness_admitted=false`.
- Host says an RX 580 HIP 7.16 2-arm F4 run was **launched**: C2 invariant vs C2×C3 wave, seeds 41/42, 24 matched router/expert-bias-only steps, output target `/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113/x369_f4_c2_vs_c2c3_fresh.json`. **No completed paired F4 CE values were supplied.** A launched background task is not a passed training or lockbox gate; do not invent a completion receipt or consume F5.

### New A/B spectral identifiability cut — one-pair exact order-three theorem

Source: `scripts/audit_x369_f1_spectral_pair_identifiability.py` and `tests/test_x369_f1_spectral_pair_identifiability.py`, registered in native + portable suites.

For PSD empirical covariance `C`, diagonalized with eigenvalues `lambda_i`, an orthogonal 120° rotation on ANY chosen pair of eigenvectors `(i,j)`, identity elsewhere, obeys `R³=I` and satisfies the **exact finite identity**
```text
|| C - R C R^T ||_F / ||C||_F
    = sqrt(3/2) * |lambda_i - lambda_j| / ||C||_F.
rotating variance share = (lambda_i + lambda_j) / Tr(C).
```
The unconstrained smallest-gap pair may consist entirely of zero/near-zero eigenvalues, thereby creating an exact or nearly exact covariance-preserving C3 *without any trained mechanism*. This is the **spectral gauge identifiability trap**. We enumerate all pairs, then separately constrain rotating TRACE variance share to 0%, 1%, 5%, 10%, 25%; each returned pair is *achievable within the single-rotating-pair class*, not proof of the global optimum across multiple pairs. The auditor also reports original top-PC row energy concentration and `1/sum(row_fraction²)` effective contributing row count. Its latter quantity is NOT the number of independent source documents.

Exact regression concepts: rank-one nullspace gives vacuous zero-defect C3 but fails high-variance nontriviality; equal high-energy eigenvalue pair supports an exact spectral rotation even if the **original coherent triad** fails; dominant document detection; invalid threshold refusal; global amplitude invariance. Test source is written and registered; **new-head native execution remains open**.

```bash
python -m scripts.audit_x369_f1_spectral_pair_identifiability \
  --residuals /path/to/ACTUAL/layer0_original_residual.npy \
              /path/to/ACTUAL/layer1_original_residual.npy \
  --output /path/to/new/x369_f1_spectral_pair_identifiability.json
```
Exact residual filenames must be taken from the original F1 JSON receipt; do not invent paths.

### New F4 development-only, read-only decision gate before F5

Source: `scripts/audit_x369_f4_development_fairness.py` and `tests/test_x369_f4_development_fairness.py`, registered in native + portable suites.

For each actual seed, let `I2,I3` be **initial** development CEs and `B2,B3` be each arm's **best** development CEs; `G2=I2-B2`, `G3=I3-B3`. Prove/recheck purely from original receipts:
```text
B3-B2 = (I3-I2) - (G3-G2).
```
A C3 source geometry can already be better at step zero, before learning, because changing the router feature geometry changes the initial computed function despite identical learned router weight tensors. Report both the initial geometry advantage and the excess learned development improvement; do not conflate them. Two seeds do not supply robust uncertainty on model-seed variability.

The new adjudicator checks source F4 `kind`, explicit nonrelease, full SHA256 development/reserved split hashes (without **opening** the reserved array), original source-document ID and hash lists without duplicate selected documents, exact two seed-matched arms, trainable parameter counts, actual 24-step budget and fit result, frozen candidate checkpoint hash format, finite nonnegative CE and exact paired-value consistency. It chooses between C2 and C3 using development-only best CE with conservative C2 tie-break, but records `f5_document_independent_release_admitted=false`. Missing/NaN paired CE must fail closed.

Run **only after** physical F4 output exists and is complete, without touching the actual F5 reserved `.npy`:
```bash
python -m scripts.audit_x369_f4_development_fairness \
  --screen-receipt /mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113/x369_f4_c2_vs_c2c3_fresh.json \
  --fresh-provenance /mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113/x369_fresh_splits/provenance.json \
  --output /path/to/new/x369_f4_initial_vs_training_gain.json
```
This does not run model inference or access the reserved F5 tokens. If the background F4 run is incomplete or failed, no adjudication is permitted.

### Exact remaining evidence gates

| Gate | Decision required |
|---|---|
| F1-S | Source original spectral pair threshold curve; original row concentration; verify whether near-invariant rotations have **meaningful rotating variance** rather than only rank-deficient nullspace |
| F2 | Define independently meaningful C3 input/label action and test frozen source hidden-state response on document-disjoint orbit triples; high-energy pair rotation alone does not close this |
| F3 | Measure frozen original-model causal predictive CE relative C3 vs equal-norm/displacement null on **separate fresh probe docs**; not F4 development and never F5 lockbox |
| F4 | Finish physical two-seed RX 580 screen and obtain hash-bound per-arm *initial*, best, best-step CEs; perform new read-only F4 decomposition |
| F5 | Remain locked and unscored until development-only model choice/seed rule and independent historical document/source lineage are accepted |
| A03/A04/A05 | Arbitrary-function finite original Fourier algebra; conjugate wave-pair real descent; original model consumer weld; selected 8-leaf Agda kernel success is not these universal proofs |

Attribution: finite group representation identities are mathematics on the original selected torus carrier; source spectral tests are computed at **original trained neural residuals** when physical arrays supplied. Neither implies learned semantic phase, Monster/VOA, NS, YM, lattice physical dynamics, or original model benchmark release.

### F1 algebraic rank bound independent of running eigensolver

For positive-semidefinite covariance, participation rank `r_eff=(Tr C)^2/Tr(C²)` implies
```text
1/r_eff <= lambda_max / Tr(C) <= 1/sqrt(r_eff).
```
Therefore the host-reported effective ranks alone imply, approximately, **L0 leading-PC trace variance share between 2.26% and 15.03%**, and **L1 between 59.88% and 77.39%**. This is a rigorous eigenvalue inequality conditional on the rounded reported participation ranks, not a measured top eigenvalue or a claim that only two dimensions have nonzero variance. It explains why L1's almost-maximum covariance defect can be dominated by one high-energy direction. It also makes the new F1 source-row concentration and rotating-pair threshold audits more decisive than a search for arbitrary low-energy covariance symmetries.

The single-pair rotation identity was separately sanity-checked on 160 randomly constructed PSD matrices of dimensions 3,6,12,24, with maximum direct-vs-formula discrepancy about `1.33e-15`. That is a numerical check of the algebraic identity, **not** execution of the new GitHub-hosted X369 test suites or physical TrueNAS experiment.

## 15 November 2026 — completed RX 580 F4/F3 receipts, original-router causal confound and revised exact maximum cut

### Host-provided physical acceptance

Host reports native source contract suite **118/118 tests across 25 suites**, portable **37/37**, selected Agda 2.9 Track B original-torus leaves **8/8 exit 0** at heads Leech `9d06fd85d7ad`, DASHI `a812825058b5`. These receipts belong to those exact source heads; later PR changes below are **not** yet native-suite accepted.

**F1 spectral pair:** Original TrueNAS receipt `x369_c3_f1_spectral_pair_audit.json`, physical d=384.
- L1: top-PC trace share 77.01% (lambda0 24.204 / trace 31.428), top-PC Frobenius 99.43%, nine-row top-PC energy range 8.86%–13.72% and row-energy participation 8.72/9. This makes a **single dominant source row** an unlikely explanation for the sampled PC concentration, but does **not prove** architecture independent of data or independent source documents.
- L1 near-nullspace pair (372,373): achievable C3 covariance defect 2.22e−9 but rotating trace share only 2.97e−7. L1 pair (0,1) with >=10% trace: D=1.09832. Low-energy near-perfect covariance symmetries are **vacuous** as neural phase explanations.
- **Crucial positive L0 counterexample to an overly broad no-go:** L0 pair (1,2) with >=10% original trace has achievable D=0.0281 (<0.05). This is a **non-null high-variance covariance-compatible C3 candidate**, not yet an identified model/task intertwiner. Do not state "all nontrivial spectral rotations have high defect" or "the only good rotations are in nullspace" for both layers. L0 leading trace share 7.93%, participation rank 44.30; no pair has >=25% trace in the tested covariance.
- All spectral pair rotations are selected **from the same observed covariance**; they need a separately frozen action and heldout task/orbit/consumer evaluation. One-pair covariance algebra alone is not independent learned mechanism evidence.

**F4 original RX 580 two-arm source-matched screen:** `x369_f4_c2_vs_c2c3_fresh.json`, seeds 41/42, 24 optimizer steps, original 55 fit rows, fresh 32 x 64 token development rows. C2 best = initial 4.9903 (best step 0) for both seeds. C3 initial = 6.8026 both seeds; seed41 best 6.8026 at step0; seed42 best 6.7438 at step24 (gain 0.0588). Original mean best C3-minus-C2 CE **+1.7829 nats**, initial geometry difference **+1.8122**, incremental training gain C3 vs C2 **+0.0294**. **C2 wins under this exact original checkpoint/fresh-development comparison.** Repeating the identical initial checkpoint across two training seeds does NOT establish two independent initial geometry effects; the initial +1.8122 is the same deterministic model mismatch. Twenty-four steps do not constitute a converged fair-training result.

**F3 physical source-router intervention:** `x369_f3_causal_probe_receipt.json`, 16 new 64-token rows with exact 16-gram disjointness from supplied viewed rows, frozen C3 seed41 checkpoint **best step 0 and unselected by F4**. Baseline CE=6.7296. C2 diagonal delta 0.0000, C3 diagonal 1.31e−7 (engineered routing invariance, not an independently learned task action). Relative C3 delta -0.0644, Bonferroni CI [-0.2349,+0.1148]; matched-activation-norm/displacement random delta -0.1823, reported CI [-0.3215,-0.0310]. The **relative-minus-random** contrast is **+0.1179** with a separately generated *unadjusted 95%* paired-row bootstrap interval [+0.0504,+0.1898], favouring the random control on these rows. Do NOT label that interval familywise-adjusted 99% or invent an exact p-value from its exclusion of 0. This evidence shows **no C3-selective advantage in the tested unselected, best-step-zero arm**, not that all trained C3 or all semantic task rotations have been refuted.

### NEW decisive original router mismatch: why F4/F3 cannot yet identify mechanism

Exact original code `HistoryOrbitMoE.select_source_experts` in `models/history_orbit_moe.py` has one **shared** `nn.Linear(2*d,1)` scoring head across all experts. Expert-dependent differences come from their *individual retained histories*, expert biases and top-k selection. Two original feature paths:

```python
# Original C2
F2(x, h_e) = (abs(x), abs(h_e))

# Original C2 x C3 source
F3(x, h_e) = (trialectic_router_invariants(x),
              directed_trialectic_invariants(x, h_e))
```

The unary x score `w_x·F_unary(x)` is common to all experts under **either** geometry and therefore cancels in relative expert scores, except for numerical roundoff / top-k ties. With all expert histories zero and identical biases, BOTH routers make equal routing selections at the first step. As histories evolve, `abs(h_e)` and bilinear direction-sensitive `directed_trialectic_invariants(x,h_e)` are not interchangeable: **C2 is even under isolated history sign change, while the directed C3 contribution is odd under isolated history sign change.** Therefore copying trained weights with equal tensor width is not a function-preserving initialization. This is an exact ORIGINAL source distinction, not a hypothetical separate expert-head theorem.

**F3 null correction:** `matched_displacement_random_x` matches `||x||` and `||x'-x||` with `x'=R_C3 x`, but the actual router processes nonlinear `F3(x,h_e)`. It generally does **not** match `||F3(x',h_e)-F3(x,h_e)||`, nor the centered per-expert logit perturbation. Moreover relative C3 rotation leaves unary invariant features unchanged while a randomized x generally changes them. Hence the reported F3 contrast is useful negative source-arm evidence, **not a feature-displacement-calibrated C3 causal specificity test**. The source F3 receipt generator now emits this limitation, records actual F4 best step, and explicitly distinguishes unselected C3 candidate from the development winner.

**New implemented tools in existing PR #4:**
- `scripts/audit_x369_f3_f4_router_feature_metric.py`: exact source feature construction using the original `trialectic_router_invariants` and `directed_trialectic_invariants`; reports common unary cancellation, original C2-vs-C3 expert-centred score divergence, actual feature-space C3-relative/random displacement ratios, and fraction of source samples matched within 5% in **feature space**.
- `scripts/audit_x369_f3_live_router_feature_metric.py`: reads the SHA-bound frozen F4 C3 checkpoint + **already-viewed F3 probe** (not the reserved F5), safely captures a bounded deterministic reservoir of actual `[x,h_e]` tuples on each original layer via temporary restoration-safe method hooks, runs the feature audit on real original states, and writes new SHA-bound, nonoverwriting receipt. No original checkpoint updates or additional learning.
- `tests/test_x369_f3_f4_router_feature_metric.py`: seven source-written regression tests (including genuine source feature invariance, nonlinear mismatch, zero-history agreement, history sign, bounded observer restoration), registered in portable/native suites; these new tests need a **new current-head** native run.

**Exact live measurement command** — use the **already-viewed F3 probe data**, not F4 selection and never F5 token files:
```bash
ROOT=/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113
bash scripts/run_native_schaka101_lila.sh --module \
    scripts.audit_x369_f3_live_router_feature_metric -- \
  --screen-receipt "$ROOT/x369_f4_c2_vs_c2c3_fresh.json" \
  --source-candidate "$ROOT/c2_bounded_repaired.pt" \
  --original-train-ids "$TRAIN_IDS" --original-heldout-ids "$HELDOUT_IDS" \
  --selection-validation-ids "$ROOT/x369_fresh_splits/development_ids.npy" \
  --probe-ids "$ROOT/x369_f3_probe_splits/development_ids.npy" \
  --reserved-lockbox-sha256 7d13b31c73a7bc9d...FULL_ACTUAL_SHA256 \
  --seed 41 --max-states-per-layer 32 \
  --output "$ROOT/x369_f3_live_router_feature_metric.json"
```
The full original SHA256 is needed if passing the optional `--reserved-lockbox-sha256`; otherwise **omit that option**, rather than using the illustrative prefix above. Actual `TRAIN_IDS` and `HELDOUT_IDS` paths should come from original receipt hashes, not guessed names.

### Post-replay decision ledger

| Gate | Revised status |
|---|---|
| P0 | 118/118 + 37/37 + 8/8 reported at old identified heads; new 7 regressions and new source need current-head replay |
| F0 | Isotropic null not rejected by rotating-energy ~two-thirds: do not market as trained C3 phase |
| F1-L0 | Exact constructive candidate 10%+ rotating eigenpair, covariance D=.0281; **task action and causal consumer OPEN** |
| F1-L1 | 77.01% trace in leading PC; high-variance spectral pair D=1.09832; near-zero nullspace pair not meaningful |
| F3 | Frozen seed41 C3 best step0, unselected; source-space relative/random contrast disfavors C3; **nonlinear ROUTER-feature matching OPEN** |
| F4 | 24-step two-arm **C2 development win**, mean gap +1.7829, initial gap +1.8122; same-step initial model not function matched |
| F5 | Reserved 32 rows reportedly unscored; document lineage against original corpus not independently verified: **KEEP LOCKED** |
| F6 | Three-arm `C3²` comparison still needs predeclared comparable initialization, effective rank/FLOPs, truly fresh provenance |
| A03–A06 | Agda selected 8/8 finite leaves accepted; arbitrary-field and original model consumer extensions still open |

**Interpretive boundary:** F4 C2 is the winning **development selection under the tested checkpoint and budget**; F3 tested an unselected step-zero C3 arm. Neither is a basis-independent theorem that C3 representation learning cannot work. Re-preregister any calibrated F4 redesign on a NEW independent development sample. Do not retune against or consume reserved F5 to rescue the failed arm.

## 15 November 2026 late — F3 exact feature-distance-matched mixed C3 null, F5 read firewall

### Source-equivalent mathematical identity (new source implementation; native current-head acceptance open)

The RX 580 F3 live audit on seeds 41/42 reports the old arbitrary matched-activation perturbation had roughly **2.0× to 2.2×** the original C3 intervention's nonlinear router-feature displacement. Note the SIGN: the old F3 random control yielded **LOWER** CE (−0.1823 vs C3 relative −0.0644), so do not report greater **CE degradation**. A larger perturbation happened to improve the source CE more on those probe rows; it does not imply a causal direction generally.

The structural solution is exact and does not require tuning a random vector to match a scalar. For the original 384 coordinates partitioned into 128 triads `x_j=(a_j,b_j,c_j)`, define the same coherent `R_+` as the original `rotate3`: `R_+(a,b,c)=(b,c,a)`. The reverse `R_-(a,b,c)=(c,a,b)` is its inverse. A **mixed orientation null** chooses an independent seeded sign `s_j in {+,-}` for every triad, explicitly enforcing at least one of each in each original input row. Define `M_s x` by applying `R_{s_j}` on its respective triad.

For every real triad:
```text
||R_+x_j||² = ||R_-x_j||² = ||x_j||²
||R_+x_j - x_j||² = ||R_-x_j - x_j||²
trialectic_router_invariants(R_±x_j) = trialectic_router_invariants(x_j).
```
For fixed original expert history `h_{e,j}`, let `T(x_j,h_{e,j})` be the *actual source three-channel directed correlation* (self/cyclic/anti). Crucial exact identity:
```text
T(R_+x_j,h_{e,j}) = cycle(T(x_j,h_{e,j}))
T(R_-x_j,h_{e,j}) = cycle²(T(x_j,h_{e,j})).
||T(R_+x_j,h)-T(x_j,h)||²
  = ||T(R_-x_j,h)-T(x_j,h)||².
```
Sum these per-block identities and concatenate unary and directed channels: for **every** original source row and expert `e`,
```text
||M_s x|| = ||R_+x|| = ||x||
||M_s x - x|| = ||R_+x - x||
||F_C3(M_s x,h_e)-F_C3(x,h_e)||
  = ||F_C3(R_+x,h_e)-F_C3(x,h_e)|| .
```
Unlike the old null, this matches **BOTH** original activation- and nonlinear feature-displacement norms, not just average norm across the batch. It breaks globally coherent phase orientation without removing local order-three source structure. The **direction** of the change, and its projection through the learned router weights, are deliberately **not matched**: do not claim equal centered-logit displacement or equal CE.

New implementation: `models/c3_matched_router_control.py:matched_mixed_orientation_c3_x`; original `HistoryOrbitMoE.select_source_experts` action `c3_feature_exact_mixed_orientation_x`; original live state `audit_x369_f3_f4_router_feature_metric.py` and `audit_x369_f3_live_router_feature_metric.py` now report the exact per-expert feature metric error alongside the old unmatched random. Four added targeted tests in `test_c3_matched_router_control.py`; original source F3 admission and action regressions extended. Local independent float64 numerical checks at d=6/24/384 showed maximum absolute new-null feature-distance error <= 7.11e−15, while centered-logit perturbations were different. **This is a source-algebra check**, not a new physical trained model score or current-head full native test.

### F3 causal protocol revision: two specificity controls, corrected inference

`scripts/audit_c3_causal_router_ce.py` now runs: (1) coherent relative C3, (2) historical generic x-norm/displacement-matched random, (3) **new exact feature-distance-matched mixed C3**. It preserves C2/C3 diagonal and signed-axis structural controls. The next exploratory receipt reports **relative-C3 minus mixed-C3** paired row CE, alongside historical relative-C3 minus generic random. The two contrast intervals are computed with `alpha=.025` each (Bonferroni family for two descriptive comparisons; not a joint adjustment with the six action-vs-baseline questions). Since the **same 16-row F3 probe has been inspected** and the new null was designed after seeing results, these are **exploratory sensitivity tests, NOT independent confirmatory significance or F5 results**. Original source-document independence is not independently verified.

**F5 correction:** The previous F3 scorer's own code loaded `--reserved-lockbox-ids` through `rows_from_file` to check token overlap, so the historical shorthand **"unopened" was literally inaccurate** even if **the reserved set was never scored, used for model selection or used to compute loss**. The new F3 scorer rejects that old raw-array argument and instead requires `--reserved-lockbox-sha256` (full 64 hex) plus `--probe-provenance`. It checks that the previously produced F3 token split's SHA-bound **prior array list includes the reserved F5 SHA** and that the F3 `development` artifact SHA and 16-token window checks match. No F5 token array is loaded or hashed by the revised scorer. This **audits the producer receipt**; it does NOT independently prove corpus-level source-document independence or certify absence of a duplicate missing from the supplied training history.

Run on original already-viewed F3 probe, **one new output per seed**, never F5 token bytes:
```bash
ROOT=/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113
F5_SHA256="<FULL_SHA_FROM_ORIGINAL_F5_SPLIT_PROVENANCE>" # 64 lowercase hex
# TRAIN_IDS, HELDOUT_IDS must be the original receipt-bound paths, not guesses.
for SEED in 41 42; do
  bash scripts/run_native_schaka101_lila.sh --module scripts.audit_c3_causal_router_ce -- \
    --screen-receipt "$ROOT/x369_f4_c2_vs_c2c3_fresh.json" \
    --source-candidate "$ROOT/c2_bounded_repaired.pt" \
    --original-train-ids "$TRAIN_IDS" --original-heldout-ids "$HELDOUT_IDS" \
    --selection-validation-ids "$ROOT/x369_fresh_splits/development_ids.npy" \
    --probe-ids "$ROOT/x369_f3_probe_splits/development_ids.npy" \
    --probe-provenance "$ROOT/x369_f3_probe_splits/provenance.json" \
    --reserved-lockbox-sha256 "$F5_SHA256" \
    --seed "$SEED" --device cuda:0 --bootstraps 1999 \
    --output "$ROOT/x369_f3_causal_feature_exact_seed${SEED}.json"
done
```
The `provenance.json` must record reserved F5 SHA under `prior_token_array_sha256`; **if it does not, fail closed**—do not inspect raw reserved F5 to fix the missing proof. Only seed 42 had an F4-selected training step 24, though **C2**, not C3, still won geometry selection. Neither seed demonstrates the learned task-phase action.

### Updated maximum-cut decision

- **F3-CAL:** Run the exact-feature mixed-C3 control on actual original frozen seed41/seed42 states. If coherent C3 does not outperform the exact-feature mixed-orientation null on a genuinely NEW independently labelled task, no evidence supports a global phase-specific predictive mechanism. If exploratory reused F3 produces a difference, treat it as candidate-only.
- **F4-FAIR:** The original C2 and C3 arms start as **different functions** even with identical router tensors (original source C2 history `abs(h_e)` versus C3 directed bilinear history channels). Never reinterpret the 1.8122 step-zero gap as a learned C3 weakness. A genuinely function-preserving comparison needs an original C2 bypass plus a zero-initialized, separately matched C3 residual feature branch, and a capacity-matched null branch; evaluate on genuinely NEW development rows. Do not tune using the existing F5 reserved set.
- **F2-L0:** Physical layer0 eigenpair candidate (1,2), >=11.71% trace and covariance defect 0.0281, remains task-unidentified: obtain verified source token-action triples, freeze the source-only spectral pair before assessment, and test actual model output consumer/phase composition on new documents. It cannot be promoted from covariance fit alone.
- **A03–A06:** Formal original all-field Fourier welds and original trained consumer still open; selected Agda 8/8 finite leaves remain accredited on previously checked head.

Attribution: original model/architecture ownership remains Leech-LILA; DASHI Agda finite C3/C3² algebra is distinct; original RX 580 data are user-host-reported observations and only new future source metric executions can establish the new exact-null physical result.

### Stronger analytic F3 certificate: factored exact two-triad product

SymPy independently reduced the actual source directed-three-channel squared displacement (for BOTH cyclic directions) to the following **closed-form exact polynomial**, with all six symbols `a,b,c,A,B,C` unconstrained real variables:
```text
q(a,b,c) = a²+b²+c²-ab-bc-ca
          = 1/2[(a-b)²+(b-c)²+(c-a)²]

||T(R_±(a,b,c),(A,B,C))-T((a,b,c),(A,B,C))||²
        = 2 q(a,b,c) q(A,B,C).

||R_±u-u||² = 2q(u).
```
The factorization is stronger than comparing two simulation paths. Summed over all triads, its right-hand side is the complete squared displacement of the **original nonlinear C3 router feature vector** for a fixed expert history (the unary invariants do not move). This gives a third independent analytic audit on physical original states:

```text
direct observed coherent displacement squared
    = direct observed mixed-orientation displacement squared
    = Σ_triads [2 q(x_j) q(h_e,j)].
```

New source function `models/trialectic_wave_prior.py:directed_c3_displacement_energy_exact` evaluates the formula using stable pairwise difference squares, checks nonfinite source tensors, and returns per-row/per-original-expert squared energies. `scripts/audit_x369_f3_f4_router_feature_metric.py` now validates its observed coherent source route-feature energy against this independently computed polynomial (and checks mixed-orientation equality). Regression tests compare both source actions to the polynomial and include the q(1,0,0) and q(0,0,0) exact-control cases. SymPy symbolic cancellation yielded **zero residual** for both rotations and unary invariants; local floating tests did not access real TrueNAS source arrays.

If the exact-source polynomial check fails materially in the native F3 live receipt, **do not interpret CE or symmetry** until the implementation/environment mismatch is found. Agda companion target: `originalDirectedTriadC3DisplacementSquaredFactors` plus finite-many-block summed equality. This is a proposed formal theorem, not yet Agda kernel proof.

## 15 November 2026, 09:20 Melbourne host handoff — F3 ensemble and formal ring-solver candidate

### Verified original-host receipts at prior heads

Host independently checked **Leech `33a34876bf13`**, native `scripts/check_c2_quality_maxcut.sh` **125/125 tests across 26 suites**, portable `scripts/check_x369_portable_cpu_source_contracts.sh` **37/37**, and original targeted `test_c3_matched_router_control` + `test_c3_causal_router_ce` **12/12**. At **DASHI `272bb1be33ee`** the 8 selected original-torus Agda 2.9 leaves passed. The exact source polynomial `2 q(x) q(h)` was checked in float64 over (B=4,D=384,E=8): observed coherent squared-feature energy minus analytic polynomial max absolute error `9.09e-13`; mixed minus analytic `4.55e-13`; coherent vs mixed `9.09e-13`. These are host-provided execution receipts; do **not** transfer their pass status to later heads after this new work.

The host's live feature-logging synthesis reports the old generic activation-matched null changed actual original nonlinear route features by ~2.0x–2.2x more than coherent C3 on original states. The historical random null **improved** CE more than coherent relative C3 (it did not degrade CE). That old contrast cannot be used to infer global phase specificity.

### New F3-B ensemble source implementation and decisive statistical distinction

New `scripts/audit_x369_f3_mixed_orientation_ensemble_ce.py` and `tests/test_x369_f3_mixed_orientation_ensemble_ce.py`, registered in both the full native and portable source-contract checkers.

- The source original frozen `c2c3_wave` seed 41 (F4 best step **0**) and seed 42 (F4 best step **24**) checkpoints are tested *separately*. Both were **unselected** because the original F4 C2-invariant arm won development.
- For each frozen seed, compute original per-document baseline CE and coherent `c3_relative_shift` CE **once**, then compute **12 preselected** exact-router-feature-displacement-matched mixed ±C3 mask CE draws over the **same** already-viewed F3 source rows. No gradient updates, dropout, original model/FFN execution changes, new tokenizer, F5 CE or heldout peeking.
- The generator's seeded mixed mask is coherent across original experts for a given invocation and varies orientation independently by triad; source `matched_mixed_orientation_c3_x` enforces at least one + and one − original triad per row.
- Record the entire `[null_draw,original_source_row]` matrix and the exact seed schedule; compare the **mean mixed CE per source row** to coherent C3. The reported mean `CE(coherent)-CE(mean_mixed)` is negative if globally coherent C3 wins. Report document-cluster bootstrap 95% **descriptive** interval and fraction of null draws where coherent wins; DO NOT call that fraction a calibrated randomisation-test p-value. The 12 masks are NOT twelve independent documents/training seeds, and the original 16 F3 rows have already been viewed. Conditional document independence is not independently established.
- Source SHA binds F4 original fit, previously viewed heldout, fresh development, checkpoint, F3 probe and original F3 producer manifest. The `--reserved-lockbox-sha256` input is a **64-char hash only**. The scorer does not accept an F5 token file path, open F5 bytes or score F5 rows. The source-producer receipt is verified to include the F5 SHA among excluded original arrays, but this is not an independently certified historical corpus-ID crosswalk.
- Optional `--original-f3-receipt` checks the **same seed/ckpt/probe** and the replayed baseline+coherent CE against original recorded CE within `1e-4`, fail-closed on discrepancy. This is useful for seed 41; optional for seed 42, which did not have the original F3 receipt.
- Reusable pure paired-row test coverage checks correct CE-gap sign, row clustering, invalid/null data refusal, bounded mask budget, seeded source schedule, and restoring all original router hook states on errors. **These five new tests need fresh native acceptance.**

One predeclared physical launch per seed on the RX 580, **new output paths**, no sealed F5 token input:
```bash
ROOT=/mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113
F5_SHA256="<FULL_64_HEX_HASH_FROM_F4_SPLIT_PROVENANCE>"
# TRAIN_IDS and HELDOUT_IDS: source original physical receipt-bound paths.
for SEED in 41 42; do
  bash scripts/run_native_schaka101_lila.sh \
    --module scripts.audit_x369_f3_mixed_orientation_ensemble_ce -- \
    --screen-receipt "$ROOT/x369_f4_c2_vs_c2c3_fresh.json" \
    --source-candidate "$ROOT/c2_bounded_repaired.pt" \
    --original-train-ids "$TRAIN_IDS" \
    --original-heldout-ids "$HELDOUT_IDS" \
    --selection-validation-ids "$ROOT/x369_fresh_splits/development_ids.npy" \
    --probe-ids "$ROOT/x369_f3_probe_splits/development_ids.npy" \
    --probe-provenance "$ROOT/x369_f3_probe_splits/provenance.json" \
    --reserved-lockbox-sha256 "$F5_SHA256" \
    --seed "$SEED" --null-seed-base "$((40100+1000*SEED))" \
    --null-draws 12 --device cuda:0 --bootstraps 1999 \
    --output "$ROOT/x369_f3_mixed_ensemble_seed${SEED}.json"
done
```
`--original-f3-receipt "$ROOT/x369_f3_causal_probe_receipt.json"` may be added **only for seed41** after confirming it binds the same original checkpoint/probe. The masks are sampled automatically; no post-hoc selection of a favourable mask or repeated F5 re-evaluation is permitted.

### Agda exact original-directed triad candidate

Within the **same** DASHI PR #1442, `DASHI/Cognition/PNF/X369OriginalDirectedTriadC3PolynomialCandidate.agda` now contains *proposed* Agda ring-solver proofs over original six integer triad/history coordinates of `||R_±x-x||²=2q(x)`, `||T(R_±x,h)-T(x,h)||²=2q(x)q(h)`, and a list induction proving all mixed-orientation block energies equal coherent block energy. This is not a toy enumeration of `2^128` assignments. The formal source is **not kernel checked yet**, so the exact lemma names are source candidates rather than accepted theorem receipts. Isolated checker: `scripts/check_x369_original_directed_c3_polynomial_candidate.sh`; keep the historical 8-leaf Agda suite intact. Integer polynomial formalization alone does NOT close the corresponding abstract-real ring weld, the Agda A03/A04 arbitrary-torus-field FFT/Parseval gates, or A05/A06 original neural consumer.

### Exact gate state after current source changes

| Gate | Evidence required |
|---|---|
| F3-SOURCE | Re-run new-head native and portable contracts including five new ensemble tests |
| F3-PHYSICAL | Execute seed41/seed42 frozen original 12-mask ensemble and hash-bind their receipts; **no CE claimed until then** |
| F3-CONFIRM | A NEW document-independent, task-labelled F3 sample; reused 16-row rows cannot make a post-hoc control selection confirmatory |
| F2-L0 | Freeze 10%+ trace L0 eigenpair with D=0.0281 and test explicit original task token/label C3 orbit/consumer action; no same-source spectral-only promotion |
| F4-FAIR | New true function-preserving C2 residual-init C3 feature study on NEW development documents; original C2 win unchanged |
| Agda 3Q | Kernel check new integer ring-solver factorization and arbitrary-list mixed-orientation induction separately from selected 8/8 Track B |
| Agda A03–A06 | Arbitrary field Fourier, same-object Parseval, real wave/centre BIDI, original trained consumer transport still open |
| F5 | Reserved 32-row CE lockbox reportedly still unscored; do NOT open or use for architectural selection |

Attribution boundary: Original Leech-LILA neural model and its frozen physical checkpoints retain their original owner; DASHI finite torus mathematics and proposed Agda ring identities are separate; all new F3 ensembles are source implementation until actual RX 580 receipts are checked.


## 15 Nov — TinyStories × SLR-spaCy × PNF/sophontic hyperfabric and long-grokking execution cut

This is the **original neural execution owner** for the same existing Leech PR #4. The detailed mathematical lemma and donor census is on Agda PR #1442, Docs/roadmaps/X369C3SquaredNeuralSameObjectFullMaxCut20261114.md, new 15 Nov integrated section. Do not fork model or mathematical rollup PRs.

### Host acceptance and current inference ceiling

- Host reported **130/130 native across 27 suites** and **37/37 portable** at source head 383ad3f35350, including the five exact-feature-mixed-orientation ensemble tests. Native result from the host, not my independently-run RX 580 test. Agda previous 8/8 original-torus leaves accepted; **new** six-variable integer polynomial checker failed Nat/ℤ type check and needs rerun after fixing (+ 2) and (+ 0).
- F3-B actual frozen original 12-mask CE on seed41 step0 and seed42 step24 **OPEN**. The original probe was inspected; only exploratory inference. Original F4 C2 result +1.7829 nats over C3, dominated by unequal step-zero *functions*; does not disprove longer genuine C3 training.
- F5 reserved 32×64-token lockbox **reportedly unscored**. Old F3 overlap producer did read raw F5 to rule out duplicated ngrams; do not claim never physically opened. New F3 scorer is **F5-SHA-only**.
- Original pretrained dense Leech context/attention KV and recurrent per-expert histories already exist; no source-bound permanent PNF/world-state neural consumer has yet been accepted. Original one-epoch 64-row TinyStories students and 24-step F4 router-only adaptation are not a grokking-duration test.

### Phase T0 — source corpus and parser truth-source

**Deliverable:** one stable original TinyStories document identity, tokenizer revision/offsets, original training/dev/heldout and full SLR spaCy numeric parser execution receipt; exact lossless character-span join to SentencePiece. SensibLaw owns src/storage/postgres/spacy_numeric_projection.py. Never create an alternative Leech-owned spaCy parser. Record parser model/version/capabilities (tagger, dependency parser, lemmatizer, morphology, entity), head links, missing fields, Unicode source bytes and execution time. The join must roundtrip source text spans; missing parser outputs must be masked, never mapped to grammatical zero. Ensure doc-level split identity, not only 16-token-window disjointness.

**Acceptance contracts:** originalTinyStoriesSLRSourceRevisionSameObject; originalSubwordSpacyCharacterSpanTransport; missingParserFeatureNotZeroTag; parserCandidateNotAdmittedWorldFact; noTrainHeldoutDocumentOverlap. All OPEN physically.

### Phase T1 — early-epoch grammar prediction and parallel surface

A small model **should first predict SLR spaCy labels from original token sequence**: POS one-hot/categorical, detailed TAG, multi-hot MORPH, dependency relation and graph HEAD (not just independent per-token category). Predict at matched SentencePiece/source word spans. Report POS accuracy and macro F1, morphology micro F1, dependency UAS/LAS, per-category calibration, training-token/time/FLOP cost and original LM CE.

Comparators: raw tokens only, tokens + source spaCy feature embedding with *zero-initialized gate*, shuffled source spaCy labels, head-only auxiliary grammar supervision without syntax at inference, then combined lexical+dependency features. Prepopulated grammar can serve as rich early scaffold; one-hot full feature vector is a legitimate target but can be implemented efficiently as class-index embedding and multi-label BCE. Test both parser-oracle and *predicted* parser input, so the learner cannot win only by consuming source labels unavailable during production.

Source precedent: Google SyntaxNet / Parsey McParseface (2016) is a dedicated neural POS+dependency parser; Gemma 3 270M (2025) a separately released compact, broadly fine-tunable language model. Neither implies this repo has passed the new task.

**Acceptance:** originalLeechGrammarHeadPredictsSLROneHotSourceTags, sourceSpacyHeadGraphArcValidity, worldParserZeroGatePreservesDenseLogits, grammarBenefitSurvivesPredictedFeatures, parserLabelShuffleDropsGain — all new physical gates OPEN.

### Phase T2 — explicit full-context and prefix-visible supervision

Two non-confusable objectives, both valid:
- FULL-CONTEXT: complete-story input, full spaCy parse, bidirectional masked infilling or iterative PNF-hypothesis denoising. If a masked target is also exposed as a label derived from the complete text, report a *teacher-oracle upper bound* and an independently available masked-input parser branch; do not call it target-blind inference.
- PREFIX-CAUSAL: autoregressive next-token prediction with prefix-visible parser annotations; full-story spaCy supervision can teach a student to infer missing future structure but cannot be passed as future-visible input to an ordinary causal decoding CE evaluation.

**Acceptance:** futureParserAnnotationVisibilityAtSourcePosition, maskedTargetAndTeacherOracleSeparated, fullTextInfillingConditionedOnAllowedContext, prefixStudentDistillationWithoutSuffixIO.

### Phase T3 — sign/C3 versus S3 and zero semantics

Do not impose arbitrary geometry before determining original source action:
- C2 support-only/absolute value is even and loses orientation; a **two-bit support+sign fibre** can roundtrip all balanced-trit values (existing Agda codec). "Binary forgets sign" applies to a chosen quotient, **not every binary encoding**.
- One trit's sign involution conjugates a three-cycle to its inverse: **S3**. Simultaneous trit-history inversion commuting with rotation of a three-component history triple: **C6**. Test separately with original model weights and representation/action hooks.
- Zero must be typed: actual zero amplitude, neutral grammatical role, unknown/missing parser observation, conflicting PNF evidence, neutral stance, orientation-specific −0/+0, punctured centre absent, zero-frequency wave/DC component, and zero route score/tie are distinct. No loss function should collapse them by accident.

The new Agda-PR source checkers provide actual finite 196830-state marked-sheet C3 and 196884-state S3 actions with 54 fixed coarse dimensions and 26+26+1 reduced punctured-27 chart. This is a **candidate geometric donor**; not yet the trained Leech action, nor a selected Monster 3B Griess module. Compare real original trained source action consistency, regular/character sector projection, sign-equivariant and oriented-zero controls.

**Acceptance:** originalLeechC2SymmetryVersusSelectedS3Action, originalSourceRouterCharacterSectorIntertwiner, orientedZeroFutureCounterfactualNonFactorability, selectedPuncturedNineWaveCentreReconstruction, originalSourceC3PhaseCommutationCompositionHeldout.

### Phase T4 — long-horizon grokking and checkpoint instrumentation

Use the **DASHIg mod-97 reproduction** and existing Agda GrokkingInvariantSubspaceSelectionExact / GrokkingMeasureStrataExact rather than expecting every short-run dip to indicate failure. Keep the ORIGINAL dense controls and original Leech tokenizer/optimizer/precision/architecture reproducible. First validate true initial-function preservation for gated syntax and C3 residual arms. Do not restart with a 12-layer brand-new MoE, identical nominal parameters and unlike initial functions, then call its immediate CE a meaningful symmetry verdict.

Instrumentation:
- **Each optimizer step, cheap:** global/auxiliary train CE, separate grammar and PNF losses, optimizer and gradient norms, LR, data cursor, active routing entropy/load, signed-history energy, numerical alarms.
- **At registered geometric checkpoints** (128,256,512,1024,2048,... and longer, subject to pinned token budget): frozen LM/dev CE, document-independent semantic minimal-pair QA, mod-97 task-character C3 composition, C2/S3 action defects, layer-wise covariance/MP empirical-null/spectral projector stability, MDL residual coding rate and actual consumer reconstruction, checkpoint full AdamW/RNG/data provenance, GPU energy/runtime.
- **Transition microsteps:** only triggered by preregistered change-point criteria; avoid full covariance/SVD on every gradient update.
- **Stop** on source misidentity, nonfinite/unstable training, broken task-action contract, unsafe GPU reset, proven non-factorability for a required semantic consumer; **continue** within a predeclared extended budget if heldout CE has plateaued but the task-character circuit, gradient health and counterfactual distinction improve.

**Acceptance:** originalLeechAdamWDataCursorRngReplay; originalStepIndexedCharacterDefectReceipt; originalSignC3S3SourceActivationAudit; originalGrokkingMechanismBeforeVisibleTaskGain; originalMPNullCalibratedForCorrelatedDocs; originalMDLConsumerRateDistortionImprovement; originalLongRunQualityParetoNoTeacherLeak.

### Phase T5 — sophontic PNF world, k/t hierarchy and diffusion

Data structure is NOT one giant dense 196884-dimensional allocation. Reuse the existing DASHI MDL/codec, pointed-sheet, punctured-wave, observer and gluing donors. The declared source hierarchy is **D(9)=(1+9)3^9+6×9=196884** and a chosen general n-site extension D(n)=(n+1)3^n+6n. Its depth/k, actual time t and optimizer step s are independent. Genuine self-indexed nine-valued-function tower A_(k+1)=9^A_k is far faster than D(n) and not an ordinary tensor power. Memory should store active address, witness, history, residual and recoverable projections, not expand the tower.

For narrator, Lily and Max maintain distinct event-time, evidence-availability-time, observer belief and PNF status fibres. A reported character false belief must not become a narrator fact. PNF dialectical candidate/objection/unknown and signed-neutral zero need typed tags; tentative PNF revisions keep source evidence and prior states. Perform FULL-CONTEXT masked PNF refinement and PREFIX-CAUSAL world prediction as separate tasks, with genuine minimal-pair counterfactuals (witness/no witness, move/imagined, negation, later observation). Compare no parser, parser only, PNF only, world+S3, and world+diffusion.

**Acceptance:** originalSLRPnfAdmissibilityAndRevision, observerSpecificBeliefFutureNonDescent, timeIndexedWorldProjectionReopenOrResidual, admissiblePantsEdgeStalkTransport, originalLeechKVExpertHistoryWorldReplay, maskedPNFRevisionVisibleEvidenceOnly, learnedConsumerCausalImportanceUnderShuffledWorldAblation.

### Release acceptance

R0: restore Agda polynomial candidate via (+ 2)/(+ 0) and check kernel, run new C3/S3/growth finite source checkers; do not relabel original 8/8.

R1: SLR original TinyStories parser receipt and early-epoch grammatical prediction, with actual macro F1/UAS and resource receipt.

R2: longer matched grokking run (same original training source) with mechanism-aware every-step logging, log-scale snapshots, conditional continuation, quality Pareto.

R3: real observer-specific PNF world and time-indexed masked revision / KV replay, heldout semantic pairs and causal ablations.

R4: published FLM 3B actual selected generator/representation basis and Griess-product intertwiner via Monster-owner PR #1472; source action and full learned Leech consumer remain **OPEN** until physically and formally welded.

**No current claim:** the original Leech neural network has learned S3/Monster, parsed future-token labels causally, benefited from PNF, or passed a long-running TinyStories world-learning test. The roadmap specifies those next discriminators.


## 15 Nov 2026 T0/T1 implementation: numeric SLR parser bridge and Leech frozen grammar probe

**Actual code added in same PR #4, not roadmap-only.** models/slr_spacy_parallel_surface.py provides source SHA-bound Unicode/UTF8 character spans, train-only compact SLR BIGINT symbol mappings, POS/dep/morph-set class prediction plus dependency head attachments, and an optional parser input adapter with gate initialized to zero. It aggregates original Leech SentencePiece hidden vectors into SLR words and broadcasts word-tag embeddings to the original subwords by exact overlap (including cross-word mixed spans). It never runs a second spaCy parser or adds model parameters to the ORIGINAL Leech checkpoint.

scripts/audit_x369_slr_spacy_grammar_early_epoch.py is a runnable source-receipt producer. One SLR numeric export JSONL record per source contains document_id, split (train or development only), source_text, source_sha256, parser_contract_sha256, parser_receipt_sha256, visibility, and slr_tokens array of numeric token_id, start_char, end_char, pos_symbol_id, dependency_id, morph_set_id and head_token_id. Prefix-mode records ALSO require prefix_visible_end_char and SHA256 of the actual visible prefix supplied as parser_input_sha256. The original 2048-piece SentencePiece model must supply immutable-proto exact offsets; optional expected_token_ids may pin previously saved original IDs. The CLI rejects duplicate document IDs, train/development source-hash cross-over, foreign source/dependency IDs and reserved/test/F5 split.

Mode A source-only validation needs --documents-jsonl, --tokenizer and --result. Mode B additionally needs --original-checkpoint plus --original-checkpoint-sha256, and optionally --epochs, --device and --seed. It loads the original model using a strict state dict, freezes it, computes real original hidden states, trains ONLY a POS/DEP/MORPH/arc probe, and records per-development-epoch CE, accuracy/macro-F1, head UAS and dependency LAS. These are future physical receipts; no original TinyStories/SLR input was available here, so no grammar improvement is claimed.

New test owners are tests/test_slr_spacy_parallel_surface.py (9 contracts, 8 of which were independently exercised on controlled fixture in CPU; ninth requires original LeechGPT host run) and tests/test_audit_x369_slr_spacy_grammar_early_epoch.py (4 controlled-fixture parser/training-source/metric contracts locally passed). Both test suites are registered in scripts/check_c2_quality_maxcut.sh and scripts/check_x369_portable_cpu_source_contracts.sh; new-head suite acceptance is OPEN. Prior 130 native and 37 portable passes apply ONLY to their earlier exact head. Estimate new native inventory 143 and portable 50, to be confirmed on host.

This is a diagnostic of syntactic information ALREADY available to frozen Leech hidden states; it is not yet a fair test of whether parser prepopulation accelerates training/grokking. That requires active nonzero gated-fusion learning, matched data/steps, shuffled-label negative control, time-to-quality, independent parser-export authority, and no F5 reuse. Full-context masked/diffusion input and prefix-only causal input remain different declared source capabilities.
