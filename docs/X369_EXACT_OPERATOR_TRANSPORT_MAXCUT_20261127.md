# X369 exact operator transport — maximum cut, 2026-11-27

Original Leech-LILA: Anatolii Kornienko. DASHI extension. Supersedes fitted six-arm controller as primary transport route, not its diagnostic evidence.

## Owners and obligations
- Historical admission: x369_translate_paired_checkpoint.restore, source_identity, original trace probe; source remains unchanged.
- Seed geometry: historical rank-six basis B and rational right inverse L, BL=I; canonical signed_seed basis A, forward L A. This is a seed-image map, not a global recurrent decoder.
- Nonlinear transport: original literal and canonical frame, encode_history/decode_history/decode_frame, then original history.step and consume. Preserve routing, signs, caps and floating bytes; no assumed GELU commutation.
- Exact stored-value carrier: h=S(z+e)+r, qS=I, qr=0; x369_indexed_residual_closure split/join/transport. The retained raw bytes distinguish signed zero and roundoff. e'=A_u e+delta_u. No target label feeds the historical operator.
- Independent target action: x369_simultaneous_weld.literal_matrix for all ten original literal controls. Compare R^3, J^2, N^2, JRJ=R^2, NR=RN, NJ=JN; report historical differences rather than assume equality.
- Decoder/descent: simultaneous uses actual historical byte states, exact equivariant decoder equations and independent target labels. A collision refutes deterministic semantic decoding on this domain; a linear conflict refutes the declared equivariant linear family only. Budget exhaustion is not positive closure.
- Cap: exact_repair and cap_profile owners, selected-expert guard; candidate repair not applied to original model.
- Future quotient: compile_realization/verify_joint_realization, preserve chart, independent labels and exact observations. Independent replay required on closed graph. Physical compression not inferred from class counts.
- Formal donors to reconcile at theorem-body level: FutureSufficientInvariantSubspaceExact, NeuralBottleneckResidualFutureSafetyExact, FullLearningStateFutureQuotientExact, GovernedIndexedResidualCubeExact, SemanticInterventionEquivarianceExact. No new kernel proof asserted.
- Remaining: actual native receipt, closure or separating witnesses, global numerical envelope, optimizer carrier if available, physical resource reduction only after justified descent. No retraining, CI, or F5 access.

## Run
From repository root on original gfx803 runtime:
```sh
python -m unittest tests.test_x369_exact_operator_transport
python -m scripts.x369_exact_operator_transport --checkpoint "$DONOR" --sha256 "$DONOR_SHA" --source-head "$(git rev-parse HEAD)" --receipt runs/exact_operator_transport.json --budget 1000 --seconds 90
```
For a phase checkpoint also supply --phase-manifest, --phase-manifest-sha256 and --phase-receipt from original authority. Do not substitute translated or repaired readout checkpoint for donor. The 90-second parameter bounds graph discovery, not admission, rational solves, relation audit or independent replay. No positive universal claim. Source-written regression and native run not yet executed.

## Native receipt and transported seed chart follow-up
Native receipt docs/receipts/x369_exact_operator_transport_20261127.json at 406249e; SHA256 cf3818fab1a2df2c620d504f146a11a76714cebd492636808188aac5b12acaf4. Reported native 3/3 regression pass. Six target relations pass on tested paths, zero historical byte equalities; 1000 admitted states/811 edges, 1001 observed labels/812 edges, budget exhaustion. 704 feasible cap repairs, 108 inactive-expert obstructions. No collision or direct separator observed; no closed quotient. The exact linear contradiction uses nine INITIAL seeds, not recurrent endpoints. Rank-six basis does not imply exact native linear evaluation on arbitrary seeds.

New scripts/x369_transported_seed_action.py (35c8901, 4cd504a) uses the existing rational seed dictionary. For row coordinates L=B^T(BB^T)^-1, P=LB, G=L M B+I-P, where M is transpose of the independent column-action matrix. Identity on the complement is a chosen extension, not a discovered learned action. BIDI map to (x A,r) and reverse with target right inverse. Verify the original contradiction by weighted exact row cancellation; audit actual native seed-coordinate error and separate its transported contribution from recurrent defect on native paths. Extended quotient uses actual historical bytes plus independent target register and preserves observations and cap profile. It is not a claim that raw historical states alone determine labels. No source repair or retraining. Remaining: native run of new implementation, complete graph or separating witnesses, formal kernel integration, universal rounding bounds, optimizer/resource closure.

Run from repo root with original donor arguments as above, adding:
```sh
python -m unittest tests.test_x369_transported_seed_action
python -m scripts.x369_transported_seed_action \
 --checkpoint "$DONOR" --sha256 "$DONOR_SHA" \
 --phase-manifest "$MANIFEST" --phase-manifest-sha256 "$MANIFEST_SHA" --phase-receipt "$PHASE_RECEIPT" \
 --source-head "$(git rev-parse HEAD)" \
 --prior-receipt docs/receipts/x369_exact_operator_transport_20261127.json \
 --prior-receipt-sha256 cf3818fab1a2df2c620d504f146a11a76714cebd492636808188aac5b12acaf4 \
 --receipt runs/transported_seed_action.json --budget 1000 --seconds 90
```
90 seconds bounds graph discovery only. New regression and native campaign not executed by author.


## Priority pivot — native geometry learning (supersedes historical closure priority)

Accepted native transported-seed receipt: docs/receipts/x369_transported_seed_action_20261127.json at 3470b8c, SHA256 355dc296d039920eced8fc9a3b9f4cf1cd3b79312e824aae4d38bb17aeeecda2. 189 seeds, 176 nonzero native errors, max ~3.925e-8; rank six and complement dimension 90. Sampled recurrent edge defects up to 2.6294; seed-only nine-witness contradiction replayed. Both graph campaigns exhausted their state budget; no closed quotient.

Decision: historical-function transport is a reusable diagnostic, not a release dependency for new learning. Prioritize new architecture training, then warm-start comparison, then historical maintenance. Do not expand historical graph enumeration merely to exhaust the budget. Keep formal residual/future sufficiency and optimizer translation as separate research lanes.

### P0 matched learning campaign
New scripts/run_x369_geometry_training_gpu.py reuses original ordered task, canonical source, checkpoint restore and phase owners. Three arms: fresh signed source; geometric warm start via existing effective_dense/build_candidate with representative control policy (not exact historical-function parity); original FairLearner with fresh initialization. Seeds 41/42/43, 2000 updates per arm, batch 128, AdamW lr .001 weight decay .1. Warm uses the same authenticated step-2000 donor for every seed, so its prior training is not matched; report incremental learning separately from end-to-end cost. All arms use identical per-seed minibatch draws, not necessarily identical parameter counts or FLOPs. No optimizer migration. Warm canonical seed discards the old seed map as an initialization choice, not a certified historical quotient; old checkpoint retained unchanged. No new fitted controller.

Default objective is matched endpoint CE; optional --semantic-weight adds actual-history prefix loss for new arms only, separate intervention not a matched-objective comparison. Target register is supervision only, not model input. Evaluate predeclared public contexts only after training, including every scheduled snapshot. Six operational relation paths over 189 seeds plus common continuation; distinct from coordinate covariance. Record per-head unique accuracy, ties, CE, actual semantic error (new arms), time, allocation and parameter counts. No F5, no held-out-based selection, no release claim.

### Runbook
```sh
python -m unittest tests.test_x369_geometry_training
python -m scripts.check_x369_source_geometry_native
ARM=runs/x369-paired-c3-donor/ordered_c3_paired1_seed41_null0_ternary0
python -m scripts.run_x369_geometry_training_gpu \
 --checkpoint "$ARM/step_00002000.pt" \
 --sha256 6b824ca6806daeb49066bfb1b0b31bf54cc94a75b9d86121eedbbb53da0de277 \
 --phase-manifest runs/x369-paired-c3-donor/manifest.json \
 --phase-manifest-sha256 8165a5eb5551dd43a726af63bfdf1716ac3b2a019404bd84a192b6abe2b78b3d \
 --phase-receipt "$ARM/step_00002000.json" \
 --source-head "$(git rev-parse HEAD)" \
 --output runs/x369-geometry-training-20261127 \
 --seeds 41 42 43 --steps 2000 --interval 250 --batch-size 128 --contexts 8
```
Resume with identical command and unchanged source head. Snapshot and receipt orphans fail closed. Campaign is not subject to a total 90s deadline; checkpoint schedule provides restart boundaries. Commit summary and evaluation receipts with exact hashes, retain model snapshots on native storage. No execution of new training campaign claimed by author.

P1: compare full trajectories and common continuation, diagnose actual semantic action vs classification. If needed predeclare a separate prefix-supervision campaign. P2: verify packed execution, native resource/performance and rounding against the trained model, then consider quotient/compression. Historical universal quotient/optimizer transport deferred, not deleted.


## 2026-11-28 consolidated closure cut (supersedes earlier priorities)

Original Leech-LILA: Anatolii Kornienko; DASHI extensions. PR #5 is based on PR #4 experimental branch, not default main; merge conflict remains. Formal repo default master inspected at 45e5bb9bd02fff06b75f46af8c8ea85d3aed76b0 and PR #1550 at 0cc1079f3e20075bc9fb15930b0e7767b98d8072. No F5, CI or fabricated source evidence.

### Accepted evidence
- Stabilizer fixed-slot projection native reported defects 1.8626e-9, previous maximum 1.75, sampled gate only.
- Frozen typed supervised receipt docs/receipts/x369_frozen_typed_supervised_20261128.json: all six original probes reproduce; supplied split reference 1512/1512 each composition head; max chart reconstruction 3.815e-6. Supplied action is not learned action.
- Typed supervised and endpoint controls: docs/receipts/x369_typed_supervised_summary_20261128.json and x369_typed_endpoint_control_summary_20261128.json. 60 snapshots each, supervision improves aggregate composition, no universal closure. Do not select from previously viewed composition metrics.
- Exact elementary generator fit reuses joint_generator_control geometry, now x369_learned_generator_weld.py (7372096): fit only elementary training rows, verify rank and consistency, then independent action comparison, 6 relations, 144 group products, 10 literals. Signed gather/sign split implementation 2428efb. Linear compositional hypothesis supplied; coefficients identified, not unrestricted neural discovery.

### Formal owners and exact remaining welds
1. P0 semantic execution: check_x369_learned_split_native.py (ef8fa54) checks full finite public task and sampled sparse integration. Existing FutureSufficientInvariantSubspaceExact (runEncodingIntertwines, latentKernelImpliesTraceObservationEquality) and X369InputIndexedFutureConsumerExact PR #1550 (inputIndexedTraceDescent, futureSeparationRefutesCodec) already prove arbitrary-word descent from concrete one-step premises. For split state (z,h), pi(z,h)=z; step_u=(A_u z,T_u h); semantic observe_u=typed(A_u z). Both premises by projection/congruence; no theorem duplication. Need kernel-checked concrete attachment; distinguish ideal action from native bytes. 
2. P0/P1 residual memory: run_x369_residual_memory_gpu.py (4c00775). Equal semantic endpoint for ab and ba with same guards, initial payload and common suffix, differing first literal. Paired labels imply exact 50% semantic-endpoint-only ceiling, not a bound for observers given the word. Train distinct sparse-history consumer, disjoint train/composition suffixes. No action labels injected into residual classifier. Native results pending. Compare multiple seeds, original source capacity control and invariant versus unrestricted readout before promotion.
3. P1 BIDI: h=S(z+e)+r, e=Qh-z, r=(I-P)h. Preserve all inactive histories; original native bytes authoritative. LeechJointStateConsumerQuotientExact.reopenInactiveHistory/fullHistoryReopenPointwise supplies exact owner contract. GovernedIndexedResidualCubeExact supplies indexed six-face obligations, not proof of arbitrary neural commuting faces. Need concrete split/step/consumer instances.
4. P2 residual quotient: current semantic observer is safe by structure; memory observer is not. For affine fixed guarded family reuse exact 9+rank D result in X369_LILA_AFFINE_SIGNED_RESIDUAL_QUOTIENT_20261124.md. For nonlinear trained source reuse x369_future_realization, x369_source_future_acceptance, x369_common_hom and separating witnesses; no universal sufficiency from sampled local Jacobian or graph budget exhaustion. FullLearningStateFutureQuotientExact only needed for optimizer migration; deferred.
5. P3 numerical/resource: existing x369_controlled_interval requires externally authenticated kernel budgets and currently raw affine consumer; adapt to fixed-slot projection and typed/split consumers, preserve route/orientation ties and signed zero. Existing stabilizer codec and source resource acceptance require trained-model native replay; no speed claims from rank. Measure actual end-to-end time, peak allocation, host memory and native packed parity.
6. P4 original LILA: original source admission, original development task/SLR and steering owner, release ledger. Synthetic task performance does not establish original language quality. No reserved lockbox access. Resolve PR base conflict before merge.

### Acceptance commands (native gfx803, no CI)
```sh
python -m scripts.x369_learned_generator_weld
python -m scripts.check_x369_learned_split_native --source-head "$(git rev-parse HEAD)" --receipt runs/learned_split_native.json
python -m scripts.run_x369_residual_memory_gpu --source-head "$(git rev-parse HEAD)" --output runs/x369-residual-memory-41 --seed 41 --steps 2000 --interval 250 --progress-seconds 10
```
Repeat residual seeds 42,43 with separate output directories. Resume identical command at pinned source head; never delete valid receipts to bypass identity checks. Current additions source-written, native execution and formal kernel acceptance outstanding. Report both positives and negatives; residual task ceiling comparison is not arbitrary-future compression or original LILA release.


## FracDASH / PNP constructive residual correction (28 November)

Sources: FRACDASH master README (exponent-vector execution, reversible trace codecs); DASHI PR #1548 head f80df18ee92dc19d826a3b73d738dbedbc9922df, PNPStandardStackNumericExact (value, length, popShape, moveRightExact, moveLeftExact, stayExact), PNPStackFiniteDigitOwnerExact (digitBound, decodeDigit, blankZero), check_pnp_multicut_crossing_campaign.py and scripts/receipts/pnp_stack_multicut_cubic_descent_local_20261128.json. FRACDASHPairedPrimeActivityExact on master supplies activity-to-priority adapter, not unconditional concrete numeric divisibility proof.

The ab/ba first-literal task is already constructively solved: unset/a/b latch, one bit after initialization, three states including unset. Exact 50% endpoint-only ceiling, exact 100% latch consumer. This supersedes P1 neural training as mandatory. Full history can be encoded by base-B push/pop with extent, preserving zero digits; do not equate commutative prime valuations with order. Keep signed semantic coordinates separate from ordered literal tags and retained extent. No free arbitrary-precision arithmetic: ordered history needs up to n log2 B bits, integer arithmetic and access charged; first-literal latch constant state, no claim of general history compression. If a consumer queries all n independent binary prefix choices, 2^n distinguishable states force n bits; no unrestricted DAG/time lower bound follows. PR #1548 multicut proof applies to specific unit-head crossing model, not GPU runtime or P vs NP resolution. Its huge raw-step/block jump receipt distinguishes compressed macro cost from literal steps. 

Priority: construct known finite residual automata directly; identify smallest sufficient consumer-specific residual; use learned residual only for unknown source-derived consumers. Existing residual memory training remains optional neural acquisition ablation. No user GPU run requested for analytically determined latch or stack results. Actual fitted-source quality, numerical parity and physical resources are genuinely empirical. Agda kernel attachment and unrestricted consumer closure remain open. 


## Consumer compiler and formal attachment — consolidated next cut

PR #5 latest metadata at 574727e reports mergeable=true (supersedes previous conflict report). LeechTransformer default main; base remains PR #4 experiment. DASHI master checked at fad22a838ab8324d863bc1d683c8135c3be1bef1. Existing GenericFuturePartitionRefinementExact, StablePartitionCanonicalFutureBridgeExact.stableRefinementExactlyCanonicalFuture, CanonicalFutureMinimalDynamicalRealizationExact.compileCanonicalQuotientDynamics/canonicalEncodingCommutesWithTrace, ConsumerRelativeMinimalFidelityExact and FullLearningStateFutureQuotientExact are authoritative donors. Do not duplicate abstract refinement or claim minimum physical cost from minimum class count.

Implemented:
- a089f48 extent-retaining literal history and minimal first-literal latch; 4a6424c/574727e constructive semantic/first/history execution; ten indexed literal guards encoded in 4-bit digits, extent separate.
- DASHI PR #1550 71244d3 X369ConstructiveConsumerSplitExact: split semantic future theorem, arbitrary suffix latch, same semantic memory separator. Formal total latch ignores j/n before a/b; restricted Python first-literal mode rejects these initial tokens. Do not equate outside declared domain.
- 20e14a8 x369_consumer_automata.py: reachable finite product, refusal observations with identity successor, existing refine/compile_realization/verify_realization, independently replayed table, minimal class identifier and transition-index ledger. No positive closure on budget exhaustion. Joint product means synchronous all-or-nothing admission; not independent partial-observer execution.
- DASHI PR #1550 5591843 X369SectionedConsumerRealizationExact: reducedStep/Observe, transitionDescends, observationDescends, traceDescends, sectionPreservesFuture, Joint.jointFuture. Concrete proof bodies written; kernel pending. Section is representative, not raw inverse.
- Remaining P0 acceptance: local exact producer checks, negative controls and formal kernel admission. No training for known finite consumers.
- Remaining P1: bind actual unknown consumer/source dynamics to finite admitted domain or provide symbolic invariant. For unbounded history, use radix/extent law rather than graph exhaustion. If arbitrary future query reveals any prefix bit, exact carrier needs n bits for n independent choices. Use PNP #1548 charged stack/macro owner only with its actual premises; no unrestricted constant-time claim.
- Remaining P2: native interval typed readout + shared fixed-slot projection, separate exact signed gather from learned GELU/cap/TopK. Actual backend error budgets are missing and cannot be inferred from finite samples.
- Remaining P3: packed native execution and physical cost measured only after source/observer scope established; account class IDs, table index bits, observation payload, headers, codebook and preprocessing. P4 actual LILA source integration retains original release gates. Historical optimizer migration deferred unless required.


## 2026-11-28 P0/P2 constructive maximum-cut attachment — authoritative latest

LeechTransformer PR #5 tracked head at 9af8a14; DASHI Agda PR #1550 signed-six backend at ceda7fb and radix BIDI at 5c44645. Formal PNP donor PR #1548 at 2545673 includes numeric stack quotient, remainder, unpack/pack, source run decoding. No duplicate PNP theorem or unmerged dependency. Original Leech-LILA attribution: Anatolii Kornienko; DASHI extensions.

### Exact producers now written and consumer welds
- Chronological BIDI: X369RadixHistoryExact.reverseAppend/reverseInvolutive/chronologicalBidi reconstruct oldest-first X369 literals from the newest-first length-aware positional value; zero-digit/extent policy retained. Written against local list append, not assumed reversal.
- Six indexed signed slots: X369SignedSixBlockBackendExact.SignedSix.Code carries exactly six source indices and sign bits; the shared collapsed binary compiler uses index/sign composition and an involution. WithAtom.binaryExact specializes X369BinaryBlockExecutionExact.Collapsed.correct. IMPORTANT: sixExt is an explicit finite-index function equality premise and flipInvolutive is an explicit value-carrier premise. Instantiate both for actual six-coordinate vector representation before unconditional closure. No unrestricted function extensionality silently assumed.
- Cost: x369_costed_constructive.py counts six index reads and six sign products per signed composition, extent bits, history payload and copy volume. Physical hardware operations and arbitrary-precision costs are not inferred from algebraic counts. Discarded history has retention=false and cannot be reopened.
- Consumers: x369_constructive_consumer.py semantic/latch/history and x369_consumer_automata.py existing stable refinement to x369_future_realization.py. Section reopening yields future-equivalent representative, not necessarily identical original raw state. Actual source bytes/inactive histories retained by their respective owners.
- Numeric source: x369_controlled_interval.py shared signed orbit projection and typed polynomial readout are conditional enclosures; native backend budgets, arbitrary-input ties, signed-zero policy and precision remain P2 open.
- Acceptance owner: tests/test_x369_constructive_maxcut.py checks ten training-identified literals, 144 group products, matrix corruption rejection, chronology/zero extent, semantic collisions/latch separation, inverse blocks, powers 0..79, refusal to reopen discarded history, minimal future quotient, budget refusal, signed orbit fixed point and typed margin. Source-written, NOT locally executed in author environment.

### Actual remaining terminal welds
1. Agda kernel typecheck of X369InputIndexedFutureConsumerExact, X369ConstructiveConsumerSplitExact, X369SectionedConsumerRealizationExact, X369OrderedBlockCompositionExact, X369RadixHistoryExact, X369BinaryBlockExecutionExact and X369SignedSixBlockBackendExact. Existing proof bodies have not yet passed kernel; repair type errors without weakening premises.
2. Unconditional concrete six carrier: discharge sixExt for exact six-coordinate vectors and flipInvolutive for signed integers/reals; ten literal generator inverse receipts. Arbitrary signed index maps are NOT automatically permutations; the input ten need explicit bijection.
3. Charged PNP bound: fixed-slot action compositions vs binary digits/ones vs physical bit-complexity of arbitrary integer history. Minimality is consumer-relative; prove n independent queries require at least n bits only for that observer class.
4. Actual unknown consumers: source-owned one-step observation and transition stability, or counterexample. No finite public-task-to-original LILA inference; reserved data and F5 excluded.
5. Numerical/native original source: calibrated HIP per-kernel stage budgets for routing/projection/readout or fail-closed conditional enclosure, physical GPU resource/latency under authenticated checkpoint. No redundant GPU work for algebra already proven.
6. Receipt/roadmap reconciliation only after actual kernel checks and native admission. No current kernel or new Python regression acceptance claimed.

### Local verification owner (not a user task for deductively solved facts)

    python -m unittest tests.test_x369_constructive_maxcut
    python -m scripts.x369_ordered_residual_codec
    python -m scripts.x369_constructive_consumer
    python -m scripts.x369_consumer_automata
    python -m scripts.x369_costed_constructive

Source-written but unexecuted here. Formal modules require actual local Agda kernel and supported stdlib, no GitHub CI credits. Native learned-source receipts remain a separate gate.


### Shared block charge proof attached to DASHI #1550

At formal commit 84868ee, X369BinaryBlockExecutionExact now includes ones, compositionsExact, onesAtMostDigits and twiceDigitsBound. The shared collapsed schedule obeys exactly compositions = digits + ones <= 2*digits, conditional on constant-size backend composition. This is NOT an assertion about arbitrary-precision history allocation, bit-complexity or HIP kernel wall time.

The six-slot code at ceda7fb satisfies conditional signed action composition using explicit finite-index equality and sign involution. BIDI chronology at 5c44645 uses reversal/involution and length-aware quotient/remainder. Cross-cut regression at 9af8a14 covers public finite geometry and negative controls. These bodies have not been kernel-run or locally unit-tested; do not mark accepted. No additional user-run requirements for analytically derived facts.
