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
