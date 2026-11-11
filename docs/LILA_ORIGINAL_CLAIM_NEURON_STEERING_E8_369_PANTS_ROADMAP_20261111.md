# LILA original geometry claim, LLM-guided neuron steering, E8→369/pants and delivery cut

**11 November 2026.** Source/claim audit and corrected research roadmap for
LeechTransformer PR #4 and the existing DASHI Agda PR #1442.

## Correction: the original LILA claim is explicitly geometric superiority

The original Leech-LILA README (Anatolii Kornienko,
DOI 10.5281/zenodo.18784424, AGPL-3.0-or-later) **does explicitly claim**
that a frozen Leech-inspired orthogonal kernel / Leech regularisation
improves transformer efficiency and quality; it advertises TinyStories
compression results, a 20-million-parameter model, 22x compression,
0.129 bits/character and 5–6x benefit over conventional transformers.
Those are **original-author claims**, not something to delete or
retroactively paraphrase as only a consumer-relative network proposal.
They are not independently reproduced by our 384D checkpoint tests.

The authentic saved 12-layer/384D/8-head dense checkpoint has
`lambda_geo=0`, optional resonator unattached, and the core
`generate_leech_kernel(24)` matrix is QR-generated and orthogonal,
applied identically to Q and K. For the ordinary non-rotated dot-product
attention path in exact arithmetic:

```
      (QW)(KW)^T = Q(WW^T)K^T = QK^T
```

Hence the **shared orthogonal Q/K rotation alone** is an
attention-logit gauge and cannot intrinsically improve those
logits relative to the same model weights. This is a concrete
falsifiable *subclaim*, not a dismissal of the original paper's
reported phenomena: separate regularisation, learning dynamics,
random seed, initialisation, resonance bias, dataset processing or
training schedule can have real effects. Investigate each with
actual source-matched arms. Do not call a QR matrix a genuine
Leech-lattice automorphism without membership/action evidence.
The absence of geometry activation in this exact checkpoint
means the 200-step smoke did NOT test the original claimed
regularisation effect.

**Required original-claim replication dossier, first-class P0:**
1. Reconstruct the original author's stated TinyStories and FineWeb-edu
   dataset split, tokenizer, model version, checkpoint, training steps,
   geometry/resonator configuration, metric definition (especially
   bits/character versus next-token CE), sampling and code/commit
   provenance. Do not convert to a new metric without declaring the
   change. Separate reported from independently reproduced results.
2. Controlled **gauge-only** comparison of Q/K attention logits at
   the actual forward and with the shared transform disabled or
   replaced by identity, using original weights and identical input.
   Probe attention and logits for dropout, implementation and FP
   effects. Predict equality modulo roundoff for a shared orthogonal
   transform, not improved expressivity.
3. Direct source-original **geometry-on versus off** training: same
   starting model, token rows, seeds, optimiser, FLOPs budget,
   identical character/token accounting, `lambda_geo=0`
   versus original claimed nonzero value(s); independent
   tests of `use_resonator`, QR/shared-kernel, lattice-grounded
   E8 roots, genuine Leech lattice/coding priors, and matched
   random/learnable orthogonal matrices.
4. Held-out CE, bits-per-character, reasoning/semantic intervention,
   data efficiency, parameter/FLOP/active compute, and confidence
   intervals over multiple seeds. Preregister the primary claim
   and negative controls to avoid post hoc promotion.
5. Accept an actual original-claim advantage **only** when
   the original metric and a genuinely matched conventional
   architecture are reproduced on held-out data.

## LLM-driven neuron steering is a distinct P0 experimental lane

LLM-guided **proposal**, not LLM-certified evidence: a
separately identified proposal model chooses candidate target
concepts, token spans, layers and neuron/direction hypotheses
from a **discovery** split. A fixed independent evaluator scores
the resulting *actual LILA* model on a disjoint assessment
split, not the proposal model's self-reported confidence.

Intervention operators on source-bound native LILA forward
activations (start with dense first, then C2):
- **single unit**: edit one coordinate of an explicitly recorded
  tensor after `blocks[i].ln2` or another precisely defined
  hook boundary; fixed additive `+α`, multiplicative gain
  `×γ` and ablation `0`, with matched norm controls.
- **direction**: source-concept contrastive activation difference
  or learned probe direction `d`, `h ↦ h + α d`;
  compare to random, permuted, equal-norm and shuffled-label
  directions and equally budgeted unguided searches.
- **conditional history**: jointly intervene on `(x,H)`
  only after an explicit history carrier is available; compare
  one-time stimulus, per-token and persistent interventions.
- **geometry-aligned**: compare 8D E8 root/reflection directions,
  24D Leech candidates, punctured-nine C2/C3 Fourier charts,
  and arbitrary equal-norm directions. The point is to
  **test** whether geometric steering generalises or requires
  fewer experiments, not to assign semantic meanings to roots.

For every proposed concept, record:
`proposal_model_id, proposal_prompt_hash, LILA_checkpoint_sha256,
 tokenizer_sha256, source_examples_hash, assessment_examples_hash,
 layer, hook_site, coordinate_or_direction_hash, intervention,
 α, seeds, matched_control_family, target_logit_difference,
 target_behavior_metric, off_target_delta, KL_to_base,
 heldout_CE_delta, runtime, peak_VRAM`.

**Causal acceptance:** interventions must change the targeted
behavior on independently labelled counterfactual prompts,
with nuisance robustness and acceptable off-target damage.
Measure both **necessity** (ablation) and **sufficiency**
(injection/activation patch) and handle prompt/order leakage.
Use paired prompt datasets with predeclared correct outcomes;
LLM-generated labels need human or independent-source validation.
A visually convincing output or one LLM judge score is not proof
of causal neuron meaning. Do not conflate a C2 activation
sign flip with dialectical opposition or ethical stance.

**Deliverables N1–N5:**
N1 exact checkpoint/activation provenance and safe hook inventory;
N2 fixed-proposal LLM candidate JSON and labelled discovery/test split;
N3 same-token intervention runner with no gradient-based
test-set tuning, control directions and dose-response;
N4 trained probe and off-target/necessity/sufficiency report;
N5 independent replication of best candidate on original
checkpoint and step-200 checkpoint, then separate C2 only
if scientifically useful.

## Existing Agda donors: E8 → 369, dialectic/trialectic, pants

These are **read in-repo existing source owners**, not newly
invented primitives and not automatically Golden-kernel-accepted
by this document. The cross-owner files often deliberately omit
`{-# OPTIONS --safe #-}` due to import closures.

| Owner | Exact source-backed reusable result | LILA bridge / obligation |
| --- | --- | --- |
| `DASHI/Reasoning/LeechE8PriorStrataBidiExact.agda` | E8 rank 8, 240 roots, 24=3×8, 384=48×8; distinguishes 8D lattice from eight punctured-plane points | choose genuine root candidate and typed source action; no '8=8' isometry |
| `DASHI/Reasoning/LeechPuncturedNineLearningBridgeExact.agda` | explicit eight nonzero `F3²` sites, antipode, distinct coefficient sign, Fourier chart | test selected neural site/chiral and sign actions separately |
| `DASHI/Reasoning/Leech369TrialecticWaveLearningArchitectureBidiExact.agda` | 384=16×24=128×3, C2/C3, directed channels, independent trialectic face | source-bound layer/activation-to-carrier interpretation |
| `DASHI/Reasoning/TrialecticBoundaryFaceNonfactorabilityExact.agda` | identical three dyadic edges can have distinct independent face; face cannot factor through the boundary | preserve explicit face evidence or residual in steering/composition |
| `DASHI/Reasoning/DialecticInvariantGeometry.agda` | thesis/antithesis predicate field with stance involution and invariant intersection | semantic observations must be typed predicates, never sign(activation) |
| `DASHI/Topology/TernaryCylinderPantsGeometryExact.agda` | finite radix-three refinement and a three-output 1→3 pants correspondence; no smooth TQFT | multi-context branch indexing and matching, not geometric physics |
| `DASHI/Reasoning/TernaryPantsSynthesisS3BridgeExact.agda` | recursive 27 synthesis cells, branch-slot ↔ TriTruth and exact S3 inverse transport | controlled context/order perturbations and branch-history tests |
| `DASHI/Reasoning/TypedHyperfabricPantsGluingBridgeExact.agda` | `HyperfabricPantsGluing` requires realized edge stalks and `InterfaceMatch`, equating proposition/capacity/phase (and other canonical interface fields) | steering/gluing only across source-typed compatible boundary channels |
| `DASHI/Reasoning/WavePantsBundleGluingAdapterExact.agda` | exact zero canonical transport residual, explicit phase-changed nonzero residual; **arbitrary bundle/sheaf gluing still false** | negative-control seam mismatch; never promote an arbitrary local-family weld |
| `DASHI/Reasoning/LeechJointStateConsumerQuotientExact.agda` | selected history is reconstructible under retained inactive state and consumer fibre-constancy | do not elide history/face if a downstream steering consumer observes it |

The architecture does **not** identify E8 root sign with semantic
contradiction, C3 cyclic order with a time evolution law, or
pair-of-pants finite-combinatorial glue with a smooth cobordism.
A typed BIDI seam must carry original model revision, layer,
activation and history coordinate, observation semantics,
selected branch, phase/action, face evidence, source provenance,
and a consumer-preserving admission check. The exact
`TypedHyperfabricPantsGluingBridgeExact` proves interface
admissibility for a constructed section; it explicitly does
NOT build a restructured hyperfabric or erase path memory.

### New bridge priorities (source-written roadmap, proof gates open)

**A1 — original learned-carrier interpretation:** explicit
`LeechGPT` layer tensor → typed 24D/8D chart and
`T³/T²* / T⁹` view, with exact dimensions, residual
coordinates, no silent non-isometric quotient.

**A2 — action-intervention commuting square:** for any chosen
Agda C2/C3/S3 action `g`, compare
`observe(steer(g · z))` and
`observe(g · steer(z))`, with explicit action
transport on target observations. Prove an implication
**conditional** on full source neural intertwining;
measure its error on saved real tensors. If the network
does not intertwine, record that failure as a result.

**A3 — pants/history source seam:** lift a selected
branch/historical state to actual `BranchChannel`,
construct `InterfaceMatch` with real source
revision/phase/face/history fields, perform the
actual gluing/reorganisation in a separate owner,
and prove selected-consumer preservation. A seam-only
certificate is not the completed operation.

**A4 — semantic steering non-descent:** use two
same-dyad/different-face contexts as a negative
control; a neuron selector trained on only pairwise
edges cannot be credited for a face-sensitive
semantic decision without retaining extra evidence.

**A5 — kernel/empirical correspondence:** Agda
2.9 acceptance of new/affected safe leaf modules,
source-to-theorem provenance, and tests with
real checkpoint/tensor hashes. Do NOT infer that
formal finite carrier laws guarantee behavioral
steering quality or LLM causality.

## Phased deliveries with independent gates

| Phase | Delivery | Measurable exit gate | Status |
| --- | --- | --- | --- |
| P0 | Prior DASHIg Leech-vs-plain modular benchmark | original source CSVs, two WDs, seed 0, 100% test acc both; no dominance | **PRELIM RUNS COMPLETED** |\n| P0 | Original Leech-LILA TinyStories/BPC claim and strict same-model gauge/resonance controls | original BPC/compression protocol, nonzero geometry and multi-seed original task | **PARTIAL PRIOR ART; ORIGINAL CLAIM OPEN** |
| P1 | Held-out original/step-200 dense checkpoint evaluation; expanded source-disjoint GPU training | honest test CE/BPC, sample efficiency, checkpoint identity | OPEN |
| P2 | LLM-driven neuron/direction steering with source-disjoint paired interventions | target improvement, random/unguided controls, necessity/sufficiency, off-target KL | NEW / OPEN |
| P3 | E8→369/C2/C3/S3 chart and typed semantic/pants bridge | real source tensor carriers, original-object equalities, mismatch negative controls | AGDA DONORS PRESENT / WELD OPEN |
| P4 | Signed/history C2 architecture rescue or decisive negative result | strict student CE <= matched dense or a distinct predeclared systems Pareto advantage | CURRENT CE NEGATIVE |
| P5 | Consumer-preserving compression and federated branch glue | complete encode/wire/decode/ACK cost and consumer semantics | PARTIAL |
| P6 | Source-bound Golden kernel and release | actual import-closure Agda check, same revision experiment/proof | OPEN |

**Already closed:** Original dense gfx803 bounded training smoke,
200/200 updates, real restart at 100, eight rehashed
checkpoints, 900.64/955.09 target tokens/s and peak
allocated 526.7 MiB. Source `verified_200.json`;
final checkpoint SHA256
`d3c720651c33cda32efda788e0fbd504c14aff5b5e9b62870148de0ad11b5a80`.
64 repeated token windows and `geo_regularizer_weight=0.0`
mean neither independent generalisation nor geometric
benefit was tested in that smoke.

**Acceptance order:** original LILA claim reproduction
and independent geometry-controlled evaluation have
precedence over all new geometry narrative. In parallel
instrument original-model neuron steering on trusted
saved activations and source-disjoint examples. Pull
Agda donor lemmas into those exact interfaces, not
another abstract numerical analogy. Only then
promote cross-context pants/consumer transport,
C2 geometry and federated end-to-end claims.

**Attribution:** Original LILA scientific claims and
neural source remain Kornienko's. Existing DASHI
finite theorems and topology belong to their recorded
authors; AGDA is not evidence of a neural benchmark.
The imported Schaka ROCm/TopK work is separately
owned. No fabricated new GPU run, model-quality gain,
neuron steering result or Agda kernel check is asserted.


## 11 November repository recovery: DASHIg already ran a partial Leech-vs-plain test

**Important correction to this roadmap's earlier 'OPEN' shorthand.**
The separate `chboishabba/DASHIg` repository already
**implemented and executed** a first Leech architectural-prior
comparison against a neutral ordinary transformer on modular
arithmetic/grokking. Do not mark this as a test that nobody
has tried. The original user suggestion to inspect
`FRACDASH`, `dashiQ`, `DASHIg` was correct:
`DASHIg` owns this exact experimental lane.
`FRACDASH` README concerns FRACTRAN/DASHI
translation and shard compression; `dashiQ` README
concerns quantum/MDL carriers. Neither is the source
of these Leech-language-model comparison receipts.

**Source/evidence owners, already present:**
- `DASHIg/phase2_validation/leech_modular_benchmark.py`:
  translated Leech 24D shared QR Q/K and
  configurable resonance, not the full original
  TinyStories LM; `DASHIg/phase2_validation/plain_modular_benchmark.py`:
  neutral `nn.MultiheadAttention` baseline.
- `DASHIg/leech_arch_ablation_prelim/mul_adamw_lambda_0/scan.csv`;
  `DASHIg/plain_baseline_prelim/scan.csv`;
  `DASHIg/derivative_comparison_prelim.csv`;
  `DASHIg/ROADMAP.md`, `GROKKING_TIME_RESCALING_NOTE.md`,
  `COMPACTIFIED_CONTEXT.md`, and
  `LILA_PHASE2_VALIDATION.md`.
- Actual benchmark: modular multiplication modulo 97,
  single seed 0, 30% of 97² pairs used to train,
  AdamW LR=0.001, 192D, 8 heads, 2 layers,
  FF multiplier 2, weight decay 0.22 and 0.30,
  reported model `lambda_geo=0` on Leech arm.
  Both trained to 100% **test accuracy**, not
  original paper TinyStories bits-per-character.
- Exact Leech `lambda_geo=0` saved `t50`:
  5060 / 6220 epochs (WD .22/.30), `t95`:
  5500 / 12500; final heldout test CE 0.010609 /
  0.011186. Plain transformer `t50`:
  4520 / 8320, `t95`: 6900 / 9300;
  test CE 0.005975 / 0.010468.
  At WD=.22, plain t50 earlier and Leech t95
  earlier; at WD=.30, Leech t50 earlier and
  plain t95 earlier. **No consistent Leech
  superiority even on timing.** Only two
  weight-decay points, one seed; both reach
  ceiling test accuracy.
- `DASHIg/derivative_comparison_prelim.csv` records
  mean normalized derivative peak about 1.069
  for Leech vs 1.271 for plain, two runs each.
  The plain onset/logistic fit MSE was slightly
  lower than Leech on this slice. Existing FFT
  checks found no distinctive Leech-only
  frequency signal.
- DASHIg itself explicitly marks its next
  `lambda_geo=1e-3` and full `0/1e-3/1e-2`
  architecture ladder as **not finished**,
  followed by optimizer/task/seed expansion.
  Its code comparison uses different attention
  implementations (manual attention versus
  `nn.MultiheadAttention`); a causally pure
  geometric-prior ablation requires a **same
  architecture/weights** gauge-on/off arm
  alongside these broader architecture controls.

**Status change:** 'Leech versus plain control
implemented and preliminary runs executed' is
**COMPLETE**, while 'faithful replication of
Kornienko's original TinyStories/22×/0.129 BPC
metrics' is still **UNVERIFIED**, and
'geometry-attributable superiority under a
complete multi-seed, matched, original-task
ablation' is still **OPEN**.
Reuse the existing DASHIg source matrices and
saved results rather than rerun them as if absent.
The current 200-step native GPU original-model
smoke is a separate checkpoint/resume result,
not a replacement for DASHIg's actual grokking
controls. Future comparison claims should cite
DASHIg's source-run receipt and preserve
different task/architecture/metric boundaries.

**Immediate shared maximum cut:**
1. Reconcile original paper/TinyStories BPC
   and compression definition with the existing
   `LeechTransformer` train logs; do not conflate
   token CE with BPC or compression rate.
2. Run the **unpaid** nonzero-`lambda_geo`
   representative band with the DASHIg
   control harness (or source-identical GPU
   port), preserving its existing baselines.
3. Make shared-Q/K on/off **within the same
   attention implementation** an independent
   symmetry-gauge negative control and measure
   resonance/non-gauge geometry separately.
4. Assess heldout source-disjoint TinyStories
   original-checkpoint results and geometry
   effects only after preserving tokenizer,
   character counts and same data/compute.

**User-origin and original-owner attribution:**
The first-party DASHIg architectural-prior
studies predate this roadmap; original
Leech-LILA neural claims belong to
Anatolii Kornienko. No new GPU test or
DASHIg run is claimed by this documentation.
