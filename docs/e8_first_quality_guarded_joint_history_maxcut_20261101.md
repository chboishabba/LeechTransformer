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
