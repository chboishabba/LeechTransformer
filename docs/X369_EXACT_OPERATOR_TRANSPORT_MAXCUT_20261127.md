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
