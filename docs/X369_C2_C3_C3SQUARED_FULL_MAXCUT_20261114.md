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
