# E8-first Leech-LILA max-cut: failed student quality → guarded learning → genuine shared-orbit reuse

**1 November 2026.** Joint roadmap/implementation owner: LeechTransformer PR #4; DASHI Agda PR #1442.

## O — Source authority and actual mathematical hierarchy

The original *current* LeechTransformer model (Anatolii Kornienko, DOI
10.5281/zenodo.18784424) is a 384D 12-layer, 8-head transformer. Its
`W_leech` is a **QR-generated orthogonal 24×24 matrix** applied identically
to Q and K, so its contribution cancels in attention dot products. The
saved dense teacher uses `lambda_geo=0`. Its E8-named SentencePiece vocab
does NOT certify an E8 embedding or a Leech lattice automorphism.

The user correctly challenged the previous overemphasis on the
`3²−1=8` punctured nine-sheet. The proper distinctions are:

- **E8 lattice (rank 8):** 240 roots (112 integer two-sparse + 128
  half-integer even-parity). Every root defines a Weyl reflection. This
  is a valid local geometric PRIOR for an 8D learned representation.
- **Leech (rank 24):** 24D with its own lattice membership and gluing.
  `24=3×8` and `384=48×8=16×24` are exact compatible chart counts,
  NOT a proof that E8³ IS the Leech lattice, or that a neural Q/K gauge
  is a valid Leech group action.
- **DASHI 369/27/face/wave:** finite ternary carriers, C3 phase, C2
  orientation, cyclic/anti-cyclic directed channels, irreducible
  triadic face, observer/consumer fibres. The eight nonzero points
  of a punctured T² nine-sheet may index an 8D chart; that index set
  is NOT the eight-dimensional E8 lattice or its dual Fourier modes.
- **Sophontic/argument opposition:** typed semantic intervention,
  not geometric sign or site antipode.
- **Modular j/Monster:** separate sourced analytic/VOA objects; a
  matching 8, 9, 24 or 27 is not a same-object representation theorem.

DASHI source owners:
`Algebra/Trit/E8RootEnumeration.agda`,
`Biology/ExceptionalLatticeGrokkingProtocolExact.agda`,
`Moonshine/MonsterThreeLocalE8LeechBridgeExact.agda`,
`Physics/Closure/LilaE8InitialisationPriorNote.agda` (explicitly
engineering prior **only**, not proof of the external LILA-E8 claims).
New `Reasoning/LeechE8PriorStrataBidiExact.agda` cross-links
these owners and disambiguates the punctured nine-sheet.
Original E8–Leech–Moonshine authorship remains Shimakura,
Chen–Lam–Shimakura, Conway–Sloane and earlier lattice authors as
credited in existing source. No Moonshine/ESD same-object promotion.

## R — Real run and the obstacle it exposed

User's *executed* TinyStories study, original tokenizer, 64 training
and 32 held-out sequences, one epoch:
- Dense trained teacher held-out CE **2.58**.
- Independent-history C2 signed MoE CE **8.63**.
- Independent-history punctured-nine signed MoE CE **17.30**.
- 2,048 frames per candidate, all exact reopened in JQ1, ~9.25–9.26 MB
  vs 10.07 MB raw framed; **all frames chose ordinary joint zlib**.
- Modeled 1Gb/s ~11.42ms JQ1 vs 5.16ms raw (measured codec time and
  assumed link+RTT). Both **quality and latency promotion FAIL**.

The current teacher-student procedure trains each FFN separately on
frozen teacher activations and then swaps all twelve. This does not
preserve full-model logits/CE as errors compound, nor cause separate
expert histories to occupy common exact geometric orbits.

## C/S — Concrete E8 prior, guarded learner, and sharing architecture

1. `models/e8_root_prior.py`: actual enumerable 240-root E8
   (112 two-sparse, 128 even-parity half-integer), Weyl root
   reflections, eight joint Weyl-invariant router features per 8D
   chart (stabilized norms/dots and genuine angular root extrema).
   Uses eighth angular moment, **not redundant fourth moment**;
   the E8 root kissing set is a spherical 7-design. The router
   computes only invariants; the expert outputs are **not thereby
   E8-equivariant**. New opt-in `e8_root_invariant` variant. Test
   `tests/test_e8_root_prior.py` checks root counts, reflection
   closure, sign and Weyl invariance, 384D/24D arrangement.
2. `models/model.py` adds independent opt-in
   `moe_guarded_ffn`. Each layer retains its frozen original trained
   FFN; it mixes `teacherFFN+alpha*(studentFFN-teacherFFN)`.
   For `alpha=0`, the original dense computation is exactly
   recovered (absent FP invalids); a separate local teacher FFN MSE
   still trains the student when `alpha=0`.
3. `scripts/train_guarded_moe_continuation.py`: loads the already
   trained independent-history student and original dense model,
   verifies every shared Q/K/attention/embedding weight is identical,
   freezes the original teacher, and trains the signed/odd and
   even students with local MSE plus next-token CE and teacher KL.
   Reuses TRAIN samples for a separate calibration subset; final
   HELDOUT data are never used to choose the gate. A per-layer
   progressive gate can open from zero only if calibration CE stays
   within declared budget (default 0.02 nats). Failing final HELDOUT
   leaves a nonpromotable receipt; **zero-gate correctness does not
   count as a successful student or a parameter reduction**.
   Tests `tests/test_guarded_moe_continuation.py`.
4. `models/e8_orbit_actions.py`: small explicit E8 Weyl subactions
   (identity, swap 0↔1, two-coordinate sign flip, all-coordinate
   sign flip), with tensor differentiability and **byte-exact**
   binary16/32 signed permutations including NaN payload.
5. `models/e8_shared_orbit_history_moe.py`: independently trainable
   alternative with ONE canonical 8D-chart recurrent expert history
   and up to four exact derived per-expert action histories. The
   selected expert updates are inverted to canonical coordinates,
   route-weighted, and the entire family regenerated, preserving the
   orbit relation by construction. It can perform SOURCE-side
   post-remote merging from only selected updates and known weights;
   all-E histories need not cross the wire. It is a deliberately
   different architecture, NOT lossless conversion of the bad
   independent-history students.
6. `scripts/federated_joint_state_quotient.py`: in addition to
   old raw, joint-zlib, punctured-site and sign/orbit candidates,
   adds opt-in E8 byte-orbit predictors and the stronger
   `codec=shared_e8`. For ANY selected expert IDs (even if the
   identity expert is absent), it can transmit only **one canonical
   base history**, approved action labels, and exact per-expert XOR
   corrections. If the asserted orbit relation is exact, all
   correction bytes are absent; otherwise they are transmitted.
   Receivers must supply `expected_orbit_actions` from a trusted
   model/session: an attacker-controlled wire action is not
   accepted as authoritative.
   Test `tests/test_e8_canonical_history_jq1.py` exercises arbitrary
   selected IDs and the nonzero correction path.
7. `scripts/capture_joint_moe_state.py` emits source-model action
   labels only when `cfg.moe_shared_e8_history` is true; the
   actual-capture benchmark checks them and compares entire framed
   bytes + measured CPU costs, 1/10Gb serialization, and
   source-owned vs peer-retained reply payloads.
   `scripts/benchmark_joint_state_quotient.py --e8-orbit-actions`
   optionally tests the additional E8 predictor search. It
   must not claim a WAN win without counting encoder overhead.

**Proof owners:** DASHI
`LeechSharedCanonicalOrbitHistoryExact.agda` (generic arbitrary-E
reopening and inverse chart law, source-written `--safe`) and
`LeechE8PriorStrataBidiExact.agda` (E8/Leech/punctured-nine
roles and 384D alignment). Agda kernel NOT run. The real Python
E8 root enumeration and signed-permutation closure tests do not
substitute for an Agda proof that every implemented operation
is a concrete Leech automorphism.

## L/P — Controlled acceptance and experiment order

**First run the diagnosis on both SAVED real candidates**, with
`scripts/diagnose_moe_quality_collapse.py`: single-layer and prefix
replacement CE, same-input FFN MSE, expert load/entropy, history
amplification. This identifies whether local training, recurrence,
layer accumulation or router scale dominates.

**Then rescue teacher quality** with guarded continuation on the
original same checkpoint and existing token IDs. The teacher remains
active if no safe gate opens; this is a quality-control *implementation*,
not proof that the MoE replacement is successful or smaller.

**Then compare source-grounded independent E8 routing and explicit
shared E8 histories** on the SAME train/heldout split, number of
updates, expert widths and seeds, with matched C2 and punctured-nine
baselines. The new shared-orbit prior is expected to make canonical
reconstruction POSSIBLE, but may worsen task quality due to reduced
historical capacity; evaluate both quality and compression before
claiming a win.

**Then measure actual federated transport**, including model/optimizer
migration and ACK cost, CPU encode/decode, packet bytes, full return
of selected outputs/histories, and 1/10Gb link effects. Refuse
sign-only, E8-only, or synthetically shared histories as proof of
semantic/Sophontic performance.

**Finally add typed paired reasoning tasks** with independent
correctness and action annotations; TinyStories CE does not measure
dialectic/trialectic reasoning.

## Exact local execution, reuse the VERIFIED dataset files

```bash
git fetch origin experiment/consumer-preserving-multiscale-codec-20261031
git switch experiment/consumer-preserving-multiscale-codec-20261031

python -m unittest \
  tests.test_e8_root_prior \
  tests.test_e8_canonical_history_jq1 \
  tests.test_guarded_moe_continuation \
  tests.test_moe_quality_collapse_diagnostics -v

# Diagnose the already-saved one-epoch students; adapt artifact paths.
python -m scripts.diagnose_moe_quality_collapse \
  --teacher data/best_model.pt \
  --config /tmp/leech_dense_config.json \
  --student /tmp/real_text_moe_jq1_smoke/punctured9_wave/student_model.pt \
  --heldout /tmp/real_text_moe_jq1_smoke/dataset/heldout_ids.npy \
  --output /tmp/quality_punctured9.json

# Preserve teacher quality while the student improves.
python -m scripts.train_guarded_moe_continuation \
  --teacher data/best_model.pt \
  --student /tmp/real_text_moe_jq1_smoke/c2_invariant/student_model.pt \
  --config /tmp/leech_dense_config.json \
  --train-ids /tmp/real_text_moe_jq1_smoke/dataset/train_ids.npy \
  --heldout-ids /tmp/real_text_moe_jq1_smoke/dataset/heldout_ids.npy \
  --epochs 3 --calibration-budget-nats 0.02 \
  --output /tmp/c2_guarded_candidate.pt

# ITIR-venc crashed AFTER real dataset preparation: reuse files
# and verify the original saved SHA-256 provenance instead of downloading
# the dataset again in a Python subprocess.
python -m scripts.run_real_moe_jq1_experiment \
  --checkpoint data/best_model.pt \
  --config /tmp/leech_dense_config.json \
  --reuse-splits-dir /tmp/real_text_moe_jq1_smoke/dataset \
  --train-rows 64 --heldout-rows 32 --token-length 128 \
  --route-geometries e8_root_invariant \
  --include-shared-e8 --guarded-continuation \
  --epochs 2 --guarded-epochs 3 --batch 2 \
  --device cpu --output-dir /tmp/real_text_e8_moe_jq1
```

Exact user artifact paths are illustrative and MUST be checked locally.
The source files/tests were pushed through GitHub; the user-confirmed
82 passing tests on prior commit `9dcaec6` do not certify these
later commits. Root enumeration/reflection closure was independently
checked in a local CPU Torch mathematical calculation, but all
new branch integration tests, real guarded continuation and E8
trained ablations remain unexecuted against the user's source data.

## Remaining mathematical/empirical max-cut

M0. Kernel-check Agda E8 and canonical-history proof import closure
(no admission of conditional bridges as certified Leech same-object).
M1. Run isolated-layer CE and history-norm diagnostics on the real
bad students, then strengthen the actual weak local losses.
M2. Demonstrate >0 quality-approved gates on disjoint calibration
and independent heldout sets; eventual standalone student CE near
the dense CE=2.58 with teacher FFNs fully removed.
M3. Verify real shared histories satisfy an actual versioned E8
group action and yield zero/substantial corrective residuals for
selected top-k, not only mathematically fabricated state.
M4. Measure complete send+return network Pareto; raw fallback must
remain admissible when encoding cost dominates.
M5. Run typed paired SemanticIntervention correctness, causal
and trialectic face evaluation, rather than interpreting sign as
semantic opposition.

None is promoted by exact numerical dimension coincidences or
by the mere presence of a named E8 tokenizer.

## 2 November max-cut: revalidated gates and strict teacher erasure (source-written)

**Problem boundary.** A teacher-retaining hybrid at gate 0 matches the
original dense inference, but it has not replaced a dense FFN. A partially
opened gate also retains all dense weights. Reusing gate decisions after
an optimizer update is unsafe without rechecking the changed student.

**Implementation receipts (LeechTransformer PR #4):**

- `scripts/train_guarded_moe_continuation.py` now revokes every former
  gate before recalibrating each optimizer revision. Its zero-gate
  dense fallback is explicitly rechecked on train-only calibration data.
  Exact training/heldout token-window overlap is rejected for direct
  callers that bypass the dataset preparer.
- `scripts/promote_guarded_moe.py` refuses any gate not **exactly 1**.
  It strips all original `.ffn.` and `.quality_gate` parameters,
  strictly reloads the independent teacher-free model, and compares
  its logits against the full-gate hybrid. It evaluates full-model CE
  independently against the pinned original dense checkpoint. The
  CLI requires the guarded source SHA and the prior independent
  heldout-pass receipt. It refuses to write a promoted checkpoint
  when these conditions fail.
- `scripts/run_real_moe_jq1_experiment.py` invokes that export only
  when the full-replacement gate passes. Otherwise its consolidated
  report explicitly records why no standalone model exists.
  Capture and JQ1 continue to diagnose candidate histories
  without implying a quality or network win.
- `E8SharedOrbitHistoryMoE.reconcile_selected` additionally refuses
  negative/non-normalised/nonfinite routing weights and nonfinite
  selected updates before the authoritative shared history is merged.
  This is local data validation, NOT authenticated distributed commit.
- Source-written regression suites:
  `tests/test_guarded_moe_continuation.py`,
  `tests/test_teacher_free_promotion.py`,
  `tests/test_e8_canonical_history_jq1.py`.
  They have **not** been executed via this GitHub-only editing session.

**Agda contract (DASHI PR #1442):**
`Reasoning/LeechQualityGateTeacherErasureExact.agda` gives a
`--safe`, source-written finite-layer **conditional** theorem:
the hybrid and teacher-free traces are equal for every finite
execution schedule *if every gate has a proof of full admission*.
Gate-zero teacher identity is a separate lemma. Numeric intermediate
gate values, PyTorch floating-point parity, source authentication,
training CE and real E8 action adequacy are NOT derived. The module
has **not been Agda kernel-checked**.

**Exact remaining gates:**

1. **M0 (formal):** Agda kernel-accept the new gate/trace lemma and
   existing canonical-history proof with the pinned DASHI toolchain.
   Import them into the intended golden ownership graph if accepted.
2. **M1 (diagnostic):** Execute both existing failed-student
   isolated-layer/prefix CE and history-norm diagnostics on the saved
   original real-text split, without fabricating student data.
3. **M2 (quality):** Refit and recalibrate under a disjoint training
   calibration partition. Obtain positive nonzero gates. Then satisfy
   the **all-gates-one**, final heldout CE and strict standalone
   export contract. Zero-gate teacher parity is NOT sufficient.
4. **M3 (geometric reuse):** Run the independent E8-route and actual
   shared-canonical-history ablations with matched split, seed,
   capacity and updates. Inspect exact orbit corrections from real
   captured states rather than theoretical constructed actions.
5. **M4 (network):** Compare full request + response + ACK + versioned
   state ownership and measured encoder/decoder time. Choose raw on
   links where compression overhead dominates; no real WAN claim.
6. **M5 (semantics):** Only paired, independently scored intervention
   and trialectic-face tasks can establish reasoning quality, not
   signed-orbit matching or language cross-entropy alone.

**Source/attribution boundaries.** Original Leech-LILA author
Anatolii Kornienko (DOI 10.5281/zenodo.18784424).
Existing E8/Leech origins and Ibrahim Monster 3B work retain their
separate authorship. This DASHI theorem and the LeechTransformer
source implementation are repository-authored reconstructions, not
results attributed to those external authors. E8 coordinate actions
are not Leech lattice automorphisms without a separate proof.
No ESD modules were edited. **Previous measured quality gate still
fails**: dense CE 2.58; C2 CE 8.63; punctured-nine CE 17.30 on the
reported 64/32 TinyStories sample. No new quality numbers are claimed.

### Additional 2 November transport Pareto cut: policy before encoding

`scripts/benchmark_joint_state_quotient.py` now uses the first
half of captured frames to select **raw or JQ1** for each 1/10 Gbit/s
assumed link. It then reports the unchanged policy's whole-message
cost on the subsequent half. The full estimated round trip includes
measured local encode/decode CPU time, both serialisations and the
stated RTT/ACK/response-size assumptions. Small packets no longer
count as a gain if the predicted total latency is worse. Fewer than
four captured frames default to raw (insufficient calibration).
`tests/test_joint_transport_latency_policy.py` gives deterministic
slow-encoder and heldout-distribution-shift counterexamples.
This is a **prior-profile decision simulation**, not a real remote
deployment, authenticated state-owner protocol or measured WAN.
The actual request API already supports `permit_compression=False`;
deployment must select that before encoding, from a revision-bound
profile, rather than running both encoders on every message.
New tests are source-written and have not been executed here.

Focused local checks (optional; no CI dependency):
```bash
python -m unittest \\
  tests.test_guarded_moe_continuation \\
  tests.test_teacher_free_promotion \\
  tests.test_e8_canonical_history_jq1 \\
  tests.test_joint_transport_latency_policy -v
```


## 2 November continuation max-cut: end-to-end student, source-bound exchange, typed cubes

This tranche is **source-written on LeechTransformer PR #4**. No new
training, local unit test, GitHub CI, Agda kernel verification, remote
execution, or real WAN benchmark has run. A prior 2.58 dense CE / 8.63
C2 CE / 17.30 punctured-nine CE study is the LAST executed quality
evidence. It must not be relabelled as an E8-student score.

### M1–M2 — Close the zero-gate training deadlock by construction

A frozen alpha=0 quality gate blocks the final-model CE/KL gradient
to every student FFN even if local per-layer MSE flows. That makes
local distillation the only effective objective until at least one gate
opens. The original all-twelve FFN swap had already failed badly.

`models/model.py` now has an explicit `student_only=True` forward
mode **only for guarded MoE models**. The teacher FFN is bypassed in
this second rollout without modifying a quality-gate tensor, avoiding
in-place autograd invalidation. `train_guarded_moe_continuation.py`
trains with **both** the normal guarded hybrid path and a full
all-layer student-only forward. Its loss contains local FFN parity,
hybrid CE, hybrid teacher KL, independent global student CE and
independent global student teacher KL. Both global student weights
are positive by default, and disabling both is rejected. A student
rollout at alpha=0 can therefore receive gradients across EVERY
student block. This removes the *gradient-path obstruction*, not the
trained quality obstruction. Gate admission still occurs only on a
train-derived calibration subset after each optimizer revision.
The separate final heldout receipt now explicitly includes
`student_only_heldout_ce` and `student_only_heldout_pass`.
Full replacement additionally requires ALL gates exactly one,
independent heldout CE within budget, and strict teacher-free extraction.
Full-gate eval and standalone outputs follow the same teacher-erased
branch, avoiding needless floating-point subtract/re-add cancellation.

`run_real_moe_jq1_experiment.py` invokes the preexisting
`diagnose_moe_quality_collapse.py` on **TRAIN rows**, not the final
heldout slice, before continuation; its isolated/prefix CE, scale,
router entropy and history diagnostics are retained. It carries
`--student-ce-weight` and `--student-kl-weight` to continuation.

### M3–M4 — Profile/packet/capture same-object closure

When and only when strict standalone promotion succeeds, the
experiment runner captures and benchmarks that SAME promoted
checkpoint. Otherwise it captures the initial full student as an
explicit non-promoted candidate. The comparison receipt records
`capture_checkpoint_stage` and
`same_model_quality_transport`. The capture NPZ includes SHA-256
of the exact model file and token-ID split. The benchmark rejects
the wrong `sha256:<model>` or `token-split-sha256:<ids>`
revision instead of accepting a caller's ungrounded version label.
Legacy captures remain usable but are clearly labelled
source-unbound.

`scripts/federated_joint_admitted_session.py` composes existing
JQ1 selected execution, the measured-local-codec/ideal-link
pre-encoding raw/JQ1 decision, and the existing HMAC-authenticated
monotonic session envelope. It DOES NOT evaluate multiple encoders
per real packet. It verifies selected IDs, model/history revisions,
and exact reopening on receive. It does NOT advance state until an
external application executes, persists and calls `commit`.
HMAC is NOT AEAD encryption; this adapter is an in-process packet
interface, not tested socket transport, remote compute, durable
atomic commit, fault-tolerant delivery, or measured WAN latency.
Modelled policy still includes explicit RTT, return and ACK
assumptions. No claimed full-system speedup follows.

### M5 — Source-labelled semantic intervention parity

The existing `eval_paired_semantic_geometry.py` remains the
source-owned typed two-item intervention evaluation.
`eval_trialectic_intervention_cube.py` now supports an optional
independently sourced NPZ with `tokens[N,8,T]`,
`targets[N,8]`, and `fixed_probe[N]`. The eight distinct
interventions are indexed 000,001,010,011,100,101,110,111. It
computes separate all-corner target correctness and the third-order
finite difference of **log P(the SAME probe token)**:
`f111 - f110 - f101 - f011 + f100 + f010 + f001 - f000`.
Using different targets in that subtraction would be ill-typed as
a single response channel. This is a measurable third-order *model
response interaction*, NOT a theorem about argument meaning,
causal intervention independence or an actual Leech/E8 action.
The main runner accepts `--semantic-pairs` and
`--semantic-cubes`; it compares the pinned dense checkpoint
and the *same evaluated student checkpoint*. When source labels are
not supplied it records NOT EVALUATED, rather than fabricating them.

### Agda golden theorem and source ownership

`LeechQualityGateTeacherErasureExact.agda` adds
`studentRolloutIndependentOfGates`: if two layer programs share the
student step (pointwise), their complete student-only finite traces
are equal even when gate/teacher functions differ. The golden
`LeechQualityGuardTOEBridgeExact.agda` imports the theorem into
the existing TOE graph. This source-written finite state theorem
does not prove PyTorch autograd derivatives, CE improvement,
IEEE numeric parity or E8 lattice-equivariance. Kernel execution is
still outstanding.

### Exact remaining obligations and reproduction

1. **M0 kernel:** Accept complete Agda source/import roots with the
   pinned compiler and record exact revisions/receipts.
2. **M1 empirical:** Run original saved-student train-only diagnostics
   and compare vulnerable layers/route/history scales.
3. **M2 empirical:** Run global student-only continuation on the
   pinned real data; demonstrate CE restoration and full alpha=1
   on disjoint calibration/heldout, strict standalone checkpoint
   parity, plus newly unseen final data for any model-selection claim.
4. **M3 empirical:** Run matched E8 root-router and shared canonical
   E8 student ablations with actual captured signed-permutation chart
   actions and corrective residuals; no synthetic orbit evidence.
5. **M4 deployment:** Measure full request+response+ACK including
   authentication, codec CPU, remote expert execution, memory
   placement, retransmission and real network, against the best
   local or raw baseline. Add durable session and state commit.
6. **M5 semantic:** Acquire independent correct labels for pairs and
   8-corner cubes; score usefulness, calibration, third-order response,
   and semantic correctness separately.

Source-written regression suites added/extended:
`tests/test_guarded_moe_continuation.py`,
`tests/test_joint_transport_latency_policy.py`,
`tests/test_federated_joint_admitted_session.py`,
`tests/test_trialectic_intervention_cube.py`.
They are NOT executed by the GitHub-only edit.

```bash
python -m unittest \\
  tests.test_guarded_moe_continuation \\
  tests.test_teacher_free_promotion \\
  tests.test_joint_transport_latency_policy \\
  tests.test_federated_joint_admitted_session \\
  tests.test_trialectic_intervention_cube -v

python -m scripts.run_real_moe_jq1_experiment \\
  --checkpoint data/best_model.pt \\
  --config /path/to/actual_dense_config.json \\
  --reuse-splits-dir /path/to/original_provenance_checked_real_splits \\
  --train-rows 64 --heldout-rows 32 --token-length 128 \\
  --include-shared-e8 --guarded-continuation \\
  --route-geometries c2_invariant e8_root_invariant punctured9_wave \\
  --epochs 2 --guarded-epochs 3 --device cpu \\
  --output-dir /path/to/new_scored_results
# Optional only when independent labelled NPZ files really exist:
# --semantic-pairs /path/to/typed_pairs.npz
# --semantic-cubes /path/to/axis_cubes.npz
```

**Attribution:** Original Leech-LILA architecture and model:
Anatolii Kornienko, DOI 10.5281/zenodo.18784424.
The E8 root, Leech lattice, C3/369, Sophontic and source AGDA owner
chains remain distinct as credited elsewhere in this roadmap.
Preserve Ibrahim's Monster 3B ownership. No ESD modules edited.


## 2 November 2026 — completed C2 diagnosis: revised P0, steelman and exact cuts

**Evidence authority.** User executed the train-only quality diagnostic
on the saved original C2 candidate; artifact:
`/tmp/real_text_moe_jq1_smoke/c2_invariant/train_only_quality_diagnosis.json`
(LOCAL USER WORKSTATION, not a repository-committed result). The user
also reports full PR #4 tests **125 passing plus 7 subtests**,
focused new tests **34/34**, zero-correction E8 shared-history decoder
repair at `264d9f2`, and diagnostic progress printing at `77ae9ef`.
These test reports apply to user-verified revisions, NOT Agda kernel
verification or new training outcomes. Do not claim the local absolute
path is remotely executable.

**Numerical C2 receipt (training split; 4032 next-token targets):**
dense CE=2.5456663616; complete student CE=8.6084148468;
uniform 2048-class CE=7.6246189862.
Single-layer replacement CE deltas: layer 0 +1.9003832681;
layer 1 **+3.4202431119**; layer 2 +0.7150547315;
later layers from +0.676 down to +0.0919.
First two layer replacements CE=6.5523705861
(delta **+4.0067042245**). All twelve CE=8.6084148468
(delta +6.0627484851). All logits finite.
These intervention effects are NOT additive: the layer-1
isolated effect and a cumulative prefix contrast use different
network states and cannot be summed as causal independent terms.

Teacher-forced FFN probe (first two sequences, distinct from the
4032-token whole-model CE scope): layer 1
teacher RMS=0.0372745655, student RMS=0.3456580639,
same-input RMSE=0.3469125208, relative MSE=86.61932.
Derived uncentered teacher/student cosine by the norm identity
is about **0.0202**; student/teacher RMS is about **9.2733**.
Layer 1 routing counts were [120,6,126,4] across 256 top-k
assignments, sign-positive fraction 0.0234375,
final history RMS 1.39248 and max trace history RMS 1.45353.
This indicates large amplitude error AND near-zero alignment.
The small teacher-forced probe cannot establish population route
collapse, sign causation or that clipping alone can restore the
teacher's computation.

### Revised minimal Pareto/steelman experiment ladder

**P0-A sanity and negative controls:** train a non-geometric dense-FFN
copy and an ungated parity even+odd FFN at matched data, optimizer
steps and capacity. Require teacher-input FFN RMSE and full-model
CE improvements. Include a frozen dense teacher and zero-gate
hybrid (which is quality-safe but not compressed). Do not run
full 12-layer replacement as an initial success condition.

**P0-B isolate layer 1:** ablate (a) frozen/zero history,
(b) fixed balanced expert routing versus hard top-k,
(c) fixed/direct sign versus learned orientation,
(d) odd-only versus even-only versus combined parity,
(e) full-rank versus rank-24 history transition,
(f) output amplitude scaling versus learned direction correction.
Include layer 0 as a second priority. Run same-input normalized
teacher layer tests AND actual student-rollout tests.
Assess router load, entropy, sign fraction, expert gradient norms,
history amplification, aligned output cosine and CE. Explicitly
separate signed-chart algebraic correctness from training adequacy.

**P0-C graduated retraining:** reuse `student_only=True` global
CE+teacher KL, retained local even/odd target parity loss,
and train-only calibration. Revalidate every gate after each
optimizer revision. Adopt a layer admission only after its
calibration improvement. Report disjoint final heldout without
using it to select epoch/architecture; reserve a newly unseen
final test partition if multiple experiments are compared.

**P1 model-wide:** compare fixed/learned C2 and `e8_root_invariant`,
shared E8 canonical histories and the punctured-nine ablation
with matched training budgets, seeds, capacity and tokenizer.
Only after standalone teacher-free CE meets its declared budget
may real-history byte/latency promotion be attempted.

**P2 original historical steelman:** compare no-geometry, QR only
(which cancels in ideal Q/K attention), geometric-loss-only,
QR+geometric-loss, unconstrained router, sign-conjugate router,
and history-ablated variants. Keep human-labelled safety and
over-refusal separate from C2 signs or GELU parity. More epochs,
more numerology or stronger compression cannot repair a student
whose FFN approximation fails.

### Exact remaining lemma ledger (target or status; NOT kernel claims)

| Key | Exact proposition and assumptions | Owner / evidence | Status |
| --- | --- | --- | --- |
| F01 | A named involution I on input/history, additive output codomain, valid half operation imply even/odd projections, their parities and F=E+O. Include paired output AND updated state for recurrent version. | Agda `PhysicalInvolutionParityExact`, `LeechSignedHistoryExpertBidiExact`, GELU steelman | Formal real/additive bridge open |
| F02 | If a student is globally odd and target F has nonzero even component on (z,Iz), mean squared pair-error is at least norm² of target even component. Makes even channel necessary, not sufficient. | New exact no-go target, source-written Python even path | Open |
| F03 | Same named signed action on activation+history and output+next-history yields commuting expert conjugation square; finite identity is not trained equivariance. | `conjugateIntertwinesInverse` | Source-written; kernel acceptance open |
| F04 | A routing map with action-compatible selection, weights, tie-breaking and history owners commutes with that action; raw learned top-k need not. | `LeechGenericTopKActionTransportExact` and `LeechWeightedHistoryMoEBidiExact` | Concrete implementation refinement open |
| F05 | Exact gate-zero teacher behaviour and a gate-independent full student-only trace; neither establishes CE success. | `zeroGatePreservesTeacher`, `studentRolloutIndependentOfGates`; user-verified Python gradient tests | Finite source-written; Agda kernel open |
| F06 | Pointwise proof every gate is full implies teacher-free trace equality; actual numeric export must also eliminate all teacher FFN parameters and match heldout logits. | `fullAdmissionPreservesEntireTrace` and `promote_guarded_moe.py` | Finite source-written, regression passed; real promotion open |
| F07 | For per-layer student S_i and teacher T_i, e_(i+1)<=Lip(S_i)*e_i+delta_i^T with delta_i^T=norm(S_i(h_i^T)-T_i(h_i^T)); derive telescoping bound through layers and final-logit CE bound. No numerical constants supplied without source evidence. | New norm/forward-refinement lemma target, reuse analysis owners | Open |
| F08 | Log-softmax NLL difference is <=2*infinity-norm logit difference for same target; combined with F07 yields a sufficient, not necessary, quality ceiling. | Existing real-analysis owners to search before proof | Open |
| F09 | A shared canonical E8-chart history plus inverse action witnesses reconstructs selected expert histories; zero corrections admit the shorter representation and nonzero residuals must be fully transmitted. | `everyExpertReopensSharedCanonical`, JQ1 and user-verified zero-residual repair | Conditional finite source-written; Python regression passed; kernel open |
| F10 | A selected-only consumer factorizes through the selected-history quotient iff its output is constant on omitted-history fibres or exact reconstruction+retained inactive histories is available. | `LeechJointStateConsumerQuotientExact`, prior `ConsumerQuotientBaseChangeExact` | Conditional source-written; concrete consumer/kernel bridge open |
| F11 | MAC-authenticated model/history/sequence, selected IDs and exact decode permit receive-side verification, not durable exactly-once remote commit. | `FederatedExpertTransportBidiExact`; Python session+JQ1 | Packet regression passed; persistent remote state proof open |
| F12 | Independent semantic target labels and a common fixed probe across 8 cube corners permit a third-order output interaction statistic; nonzero interaction does not imply semantic validity or causation. | Existing Sophontic owner, `eval_trialectic_intervention_cube.py` | Instrumented; independently labelled run open |

**Immediate deliverables:** D1 immutable machine-readable C2 diagnosis
with source/model/split digests and small-sample scope; D2 layer-1
ablation matrix plus negative controls; D3 matched student-training
trajectories, tensor-level diagnostics, calibration and heldout
receipts; D4 strict all-gates-one teacher-free artifact or explicit
nonpromotion; D5 kernel-checked Agda import/proof receipt plus any
typed cross-language refinement; D6 source-bound whole network
request/response/ACK Pareto; D7 independently annotated semantics.
**D1 is user-executed but the local artifact is not yet committed
to the PR. D2–D7 remain open** except for their already implemented
interfaces, conditional theorems and Python regression checks.

Attribution remains unchanged: original Leech-LILA model by
Anatolii Kornienko (DOI 10.5281/zenodo.18784424);
classical E8/Leech mathematics credited to original authors,
repository theorem-owner proofs kept distinct, Ibrahim Monster 3B
ownership preserved; no ESD surface changed.


## 2 November further implementation: F02a/F07 source completion and D2 executable owner

The user-supplied **original full 13,137-byte** train-only C2
diagnosis now resides at
`docs/receipts/c2_train_only_quality_diagnosis_20261102.json`
(source attachment SHA-256
`e31dec9a6135d1e9b0a819e93947d8ebfb610491473c83e118fcbe3d58aff4fc`).
Its source JSON does NOT include teacher checkpoint hashes; it is
therefore an immutable measurement receipt, not sufficient
same-checkpoint provenance independently. D1 source handoff CLOSED;
full cross-file provenance remains an experimental concern.

The exact runnable protocol and acceptance grid are maintained in
`docs/early_layer_quality_rescue_maxcut_20261102.md`:
`scripts/ablate_early_layer_quality.py` with seven **frozen
source-model** variants and separate pointwise F08 numeric
logit/NLL bound; matching regression
`tests/test_early_layer_quality_ablations.py`.
`models/model.py` now accepts a focused
`student_only_layers=(1,)` rollout, and
`scripts/train_guarded_moe_continuation.py` can train only
`--focus-layers 1`, `0` or `0 1`, keeping all other experts
frozen with dense fallback and *forbidding full-model promotion*
on any focused receipt. This is genuine source implementation,
NOT executed training. Use a TRAIN-derived development partition
for comparing focus configurations; the final independent heldout
set must not choose the layer.

Formal new source owners on DASHI PR #1442:
- `LeechOddOnlyApproximationObstructionExact.agda` (F02a)
  proves exact nonfit at two mirrored historical states when a
  target is not odd. The quantitative squared-error floor F02b
  requires a real inner product and is **still open**.
- `LeechLayerErrorTransportExact.agda` (F07)
  proves the finite telescoping bound under explicitly supplied
  metric, monotonicity and Lipschitz hypotheses. Actual learned
  coefficients, source↔PyTorch refinement and F08 real analytic
  proof are **still open**.
Both are source-imported into the existing TOE bridge but have
not been Agda kernel checked.

**Do not over-promote:** the user-verified 125 tests + 7 subtests
predate these latest D2/focus code edits. No new regression results,
new teacher-free checkpoint, trained E8 ablation, or real WAN speedup
have been demonstrated in this GitHub-only tranche. Attribution
and ESD/Monster non-edit boundaries remain unchanged.


## 3 November 2026: negative matched layer-one quality comparison

The user's **executed, train-only development** comparison
(three epochs, same reported seed/CPU, nearly identical trainable
parameter count) gave dense teacher CE **2.513**, guarded
focused-C2 hybrid **2.486**, strict C2 layer-1 student **2.833**,
and ordinary dense single-layer GELU control **2.191**.
C2 odd+even has **1,185,606** trainable parameters versus
**1,185,413** in the dense control (193 difference).
The dense control wins by **0.642 nats** against strict focused
C2. The guarded 0.5 gate retains original dense FFN weights
and compute; its 2.486 score does NOT certify teacher erasure,
standalone MoE quality or network Pareto.

**Status correction:** D2 source/ablation and D3 focused training
have user-executed measurements, but D3 *quality-superiority gate
FAILS* for this trained candidate. D4 full 12-layer teacher-free
promotion stays OPEN; D6 remote codec/transport and D7 semantic
quality should not inherit any quality-win assertion. Existing
Golden F02a and F07 are conditional/algebraic properties, not
certificates of learned quality or a measured E8 advantage.

**Steelman caveat from exact source review:** The historical
C2 focused trainer internally withheld a calibration partition,
while the ordinary dense control trained over all supplied fit rows.
It also optimized extra hybrid CE/KL terms and averaged local
loss over all layers, unlike the dense control's single-layer
local MSE + direct CE/KL. Thus matched parameters, seed, epochs
and CPU are not identical optimizer data/loss contracts. The
observed negative ranking stands for the tested procedures,
but should not be construed as a causal falsification of
all history/signed/E8 student architectures.

The original code path remains untouched by default.
LeechTransformer PR #4 adds an opt-in
`--matched-control-objective` with one focused layer and
`--export-optimizer-fit-ids`: directly match the focused
student-only local MSE + next-token CE + logit KL, revoke
previous calibration gates before each training epoch and
export the actual C2 optimizer rows for the dense control.
This source was written **after** the reported negative
experiment. It has not been newly executed or verified in
the GitHub connector session. The concrete rerun instructions
are at `LeechTransformer/docs/early_layer_quality_rescue_maxcut_20261102.md`.
The comparison remains limited by different initializations,
FLOPs, memory, routing and one-seed uncertainty; measure those
instead of asserting a universal prior advantage.

Original model provenance and attribution: Anatolii Kornienko
(Leech-LILA; DOI 10.5281/zenodo.18784424), independent
DASHI/C2/E8 owners as already cited, Ibrahim's Monster
ownership unchanged. No ESD edits, postulates or theorem
reclassification. The local JSON files are not available
through this connector, so these scores are the user's
reported results rather than independently parsed artifacts.


## 3 November 2026 — uploaded source-backed D3 result and GFX803 Nix admission

The two **uploaded** (not guessed) matched-objective CPU training
receipts are committed unchanged in LeechTransformer PR #4:
`docs/receipts/c2_focused_matched_loss_20261103.json` and
`docs/receipts/c2_dense_matched_loss_control_20261103.json`.
Their teacher/student checkpoint identities and *exact exported
optimizer training-window file hash* agree. Development scores
for 45 fit rows and 13 dev rows (819 next-token targets) are:

| Source model / execution | CE, nats |
| --- | ---: |
| Frozen dense | 2.5132008088 |
| C2 guarded hybrid, alpha=.5 | 2.5158496153 |
| Strict focused layer-1 C2 | 2.8918613345 |
| Strict all-layer student-only C2 | 11.5599299653 |
| Matched dense GELU control | **2.2160529764** |

Difference strict C2 minus matched dense control =
**+0.6758083581 CE nats**. The hybrid is *worse*
than frozen dense by 0.0026488 CE nats on this train-derived
development partition, yet remains calibration-admitted with
its original dense FFN retained. No teacher erasure or
C2 quality/efficiency win. Original model: Anatolii Kornienko
(Leech-LILA). Attribution and theorem owner boundaries unchanged.

The user correctly recalled a separate existing Nix runtime:
`chboishabba/gfx803_compat_graph/gfx803_flake_v1`.
It contains frozen ROCm 6.4 Python/PyTorch old-HSA/HIP control
and a diagnostic upgrade shell with Cachix artifacts. Its
historical Leech GPU numerical correctness verdict is
**negative** despite successful GPU enumeration; some
runs also trigger long async host resets. LeechTransformer
now adds `scripts/gfx803_moe_gpu_gate.py` and
`scripts/run_gfx803_moe_nix.sh` (plus CPU-only source
tests), reusing that Nix+host-docker-python runtime with
a fail-closed check of the REAL saved Leech teacher/student
checkpoints, full actual training token window, intra-GPU
repeatability, matched-objective gradients and one AdamW
update before any `--device cuda` training run.
Exact instructions/limitations:
`docs/gfx803_gpu_training_admission_20261103.md`.

**Evidence boundaries:** Nix shell source exists; physical
GPU parity pass does NOT exist in this conversation.
This new source has not been run on the user's hardware.
`torch.cuda.is_available` alone cannot promote
numerical correctness. A single-step pass cannot
prove long-duration reset safety, speedup, token-quality
success or the full student D4 criterion. The model's
GPU optimization remains contingent on per-workload
admission; CPU remains the verified quality reference.
The existing Golden Agda `GFX803CompatibilityReceiptExact`
already describes the query-indexed visibility/correctness
separation; no new Agda machine-check acceptance claimed.


## 9 November — LILA-only dense-first lane; ROCm worker split

The relocated Schaka 10.1 physical candidate has a **reported and
repository-preserved PASS** for original dense LILA GPU forward
at batch 2 × sequence 64, max CPU/HIP logit difference
`2.288818359375e-05`, 3/3 forward executions, zero
observed GPU repeat drift. This does **not** prove dense
backward or training and does **not** fix the signed/history
C2's separate hard `warptopk` crash.

The separate gfx803 worker owns the TopK dispatch patch,
pinned PyTorch build/wheel, ROCm/HIP/Nix and Cachix;
this LeechTransformer lane owns **model-level** original
LILA dense gradients, AdamW trajectories, training quality,
C2 student architecture and eventual empirical controls.
We will not edit the gfx803 worker's compiler or kernel code
here. Schaka retains attribution for WGM/FP32 patches.

**New P0 standalone owner:**
`scripts/lila_dense_training_parity.py`, using ONLY
the original dense teacher and real saved tokenizer rows.
It compares full CPU/HIP dense autoregressive CE,
all parameters' gradients, matched AdamW updates,
post-update logits and repeated GPU forwards for an
opt-in 20-step bounded trajectory. **No C2 checkpoint,
router, or TopK** is loaded. It is diagnostic-only,
never persists a trained checkpoint and does not assert
a speedup. `tests/test_lila_dense_training_parity.py`
is a CPU-only correctness contract.
See `docs/LILA_DENSE_FIRST_GPU_MAXCUT_20261109.md`
for exact host run commands and formal/empirical gates.

Separate C2 quality still **FAILS** the matched ordinary
dense FFN control: strict focused C2 CE `2.89186`
versus dense control `2.21605` on train-only
development (0.67581 nats penalty). The 0.5 hybrid
gate remains teacher-dependent. The GPU worker's
reported source patch commit `5080c1c` and 88
software tests are not a physical C2 admission; await
the separate wheel result before advancing the
original C2 forward/backward/optimizer trajectory.

**Status:** source implemented and committed. No
new CPU regression execution, real HIP dense training
trajectory, model training checkpoint, or Agda kernel
receipt was created in this GitHub-only session.


## 11 November 2026 — original LILA claim / neuron steering / E8→369 pants roadmap correction

**Correction of an earlier assistant misstatement:** original Leech-LILA
(Kornienko) **did claim** that a frozen Leech-inspired
orthogonal kernel/geometric regularisation improves model
quality/efficiency. Do NOT recast this claim as if geometry
were merely our proposed downstream codec prior. Preserve
the original source's claimed TinyStories BPC/compression
comparisons as **author claims** pending independent
reproduction, with matched architecture and metric.
The shared orthogonal Q/K action has a concrete
mathematical cancellation in dot-product attention
`(QW)(KW)^T = QK^T`, but this does NOT dismiss
separate geometric resonance/training/initialisation
hypotheses or the original author's distinct experiments.
Our actual dense checkpoint and just-completed 200-step
GPU smoke both have `lambda_geo=0`, so that smoke
did NOT test an active geometric regulariser.

**First-class new P0: LLM-guided neuron/direction
steering** on the source-pinned actual LILA checkpoint.
Keep proposal LLM, real intervention and independently
labelled evaluator distinct. Assay single-unit/direction
ablation and injection, causal necessity/sufficiency,
dose-response, E8/Leech/punctured-nine-aligned versus
random/equal-norm controls, held-out effects and
off-target quality. No semantic label is licensed by
C2 activation sign alone.

**Formal cross-owner reuse already present in Agda:**
`LeechE8PriorStrataBidiExact` (rank-8/240 roots),
`LeechPuncturedNineLearningBridgeExact` (8 sites,
C2 vs coefficient sign and Fourier),
`Leech369TrialecticWaveLearningArchitectureBidiExact`
(C3/face/chiral and 384D chart),
`DialecticInvariantGeometry` (typed predicate stance),
`TrialecticBoundaryFaceNonfactorabilityExact`
(three dyads do NOT determine the triadic face),
`TernaryCylinderPantsGeometryExact` and
`TernaryPantsSynthesisS3BridgeExact`
(recursive 1→3 finite pants/S3), and
`TypedHyperfabricPantsGluingBridgeExact`
(typed `InterfaceMatch` on realized edge stalks;
NOT new topology). `WavePantsBundleGluingAdapterExact`
proves canonical zero residual and a phase-changed
nonzero residual but leaves arbitrary sheaf gluing
UNPAID. No unverified neural action intertwiner,
smooth-pants claim, source-to-Agda weld or
Agda kernel acceptance is inferred.

**Single authoritative detailed protocol and
phase/deliverable cut:**
`docs/LILA_ORIGINAL_CLAIM_NEURON_STEERING_E8_369_PANTS_ROADMAP_20261111.md`.
It explicitly adds original-source BPC/Leech claim
replication, authentic geometric-on/off training
and LLM neuron steering to P0/P1, with
experimental source provenance and independent
consumer/pants formal acceptance.

**11 Nov physical milestone correction:** the user
executed native GPU 200/200 original-dense AdamW
steps with an actual process restart at 100,
eight independently source-hashed checkpoints,
900.64 / 955.09 target tokens/s and peak
allocation ~526.7 MiB. Final checkpoint digest
`d3c720651c33cda32efda788e0fbd504c14aff5b5e9b62870148de0ad11b5a80`.
Treat earlier source-only open status in this file as
superseded for those exact bounded-training gates.
64 repeated windows and zero resonance penalty
still leave held-out generalisation/geometry OPEN.


## 11 November — full AdamW/Sophontic/LILA efficiency formal-first cut

The consolidated exact proof-obligation and BIDI ledger is
`chboishabba/dashi_agda/Docs/LILAWholeSystemLearningEfficiencySophonticAdamWFormalMaxCut20261111.md`.
It first-class formalises the **actual original LILA
scientific objective**: fewer active neurons/parameters,
faster time-to-heldout-quality, reduced FLOPs and training
wall-time/energy, improved BPC or model/transport bytes,
not only final accuracy. Leech/plain `DASHIg` prelim
already completed and showed mixed timing, so neither
Leech dominance nor failure of the original claim is
licensed from terminal 100% accuracy.

The new **original-source proof cut** comprises
L00–L08 full AdamW/data/RNG/provenance/geometry-state
update plus time-to-quality/resource admission;
S00–S04 Sophontic-style paired semantic/LLM
neuronal steering and stage-aligned intervention;
E00–E01 original root/neural Weyl action;
T00–T02 irreducible trialectic face and typed
pants/history gluing; H00–H01 signed C2/C3
history and future-preserving consumer;
O00–O02 matched AdamW/Dust, true MDL coding
and grokking mechanism; F00–F01 source-to-Agda
kernel acceptance and actual whole-system diagram.
These are **specific outstanding lemmas and
provenance payments**, not declarations proven
by this documentation change. In particular,
optimizer weight decay is NOT literally MDL;
AdamW is NOT unconditionally monotone;
matching current weights is NOT matching future
learning; sign reversal is NOT logical or
ethical negation; geometry is NOT by itself
semantic steering; and finite pants seam
admissibility is NOT full global sheaf gluing.

Prior Agda owner PR #1094: exact semantic paired
intervention and equivariance composition;
#1095: geometric regularizer and Weyl/grokking
same-action boundary; #1265: forward-only Dust
adapter with no universal unbiasedness or
full LILA model claim. Reuse those existing
owners rather than introducing duplicate
learning, steering or topology theories.
