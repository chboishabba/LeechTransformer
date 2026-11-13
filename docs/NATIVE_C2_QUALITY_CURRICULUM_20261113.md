# Quality-first LILA C2 student curriculum — 13 Nov 2026

## Decision / source baseline

The native RX 580 ROCm 10.1 environment and actual model checkpoint already work.
Federation is explicitly **parked**: 24/24 local protocol tests PASS (user run,
Leech PR #4 `affd660`), the local native selected-expert baseline completed
64 real 64-token rows at 29.593574 rows/s with 4/4 sampled output/history
parity checks at 0.0 max error, ~259.5 MB peak allocated / ~274.7 MB
PyTorch reserved, but physical 2+ GPU federation is not needed for the
next learning experiment.

The matched native C2-vs-dense quality negative control is unchanged:
- dense development CE 2.216053
- C2 gated development CE 2.564707 (worse by 0.348654)
- C2 hybrid held-out CE 2.515797
- **strict full C2 student-only held-out CE 11.565138**
- historical matched optimizer rows 45; the old 64x64 native federation
  token capture is a small benchmark source, **not sufficient evidence of
  training quality or a replacement for a real train/dev/heldout dataset**.

## New implementation in LeechTransformer PR #4

`scripts/train_native_student_quality_curriculum.py` is a direct native-GPU
quality-focused trainer, not a ROCm correctness requalification:

1. Strictly loads real source dense teacher and complete signed-MoE student.
   The guarded builder now **always restores each dense `.ffn.` from the
   original teacher checkpoint**, not from any candidate student's FFNs.
   Before fitting, it checks that the all-zero-gate hybrid matches dense
   teacher development CE at tolerance 1e-5.
2. Uses real train IDs and independent heldout IDs. Groups duplicate train
   windows by exact bytes so none can leak between fit and calibration;
   rejects any token window shared with heldout.
3. On `--stages auto` (default), ranks **isolated one-layer student
   substitutions only on fit-derived development rows**, then trains
   cumulative 1,2,4,8,...,all original layers. Or supply a strictly
   cumulative explicit layer schedule. This is *progressive* source-to-student
   replacement, not a claim all initially random students are good.
4. Each focused stage trains real signed C2/MoE and even-channel parameters
   with native GPU end-to-end next-token CE, teacher next-token KL, and
   local teacher-FFN MSE. All non-student shared weights remain frozen.
   Each epoch is evaluated **only on development** and the best-scoring epoch
   is restored before advancing to the next cumulative stage.
5. The independent heldout is **scored only at the end**, never used for
   stage ordering, best-epoch choice, LR or quality admission.
   Reports teacher heldout, focused student heldout and **full teacher-free
   student rollout heldout** separately; retaining source teacher FFNs in
   exported guarded checkpoint is explicit. `full_student_quality_pass`
   is false until all layers are active and strict student CE meets the
   teacher heldout plus tolerance. No bare architecture victory or
   federation speedup implied.
6. Native CUDA-sync progress/VRAM allocator JSONL is saved separately
   for each stage; primary checkpoint and JSON receipt are new files,
   never overwrite earlier training results.

**Source modifications:** `scripts/train_guarded_moe_continuation.py` fixes
the source FFN identity, and `scripts/run_native_schaka101_lila.sh` admits
the new trainer via the already-working Schaka ROCm 10.1 wrapper.
**Regression tests:** `tests/test_native_student_quality_curriculum.py`
checks cumulative stage selection, deterministic duplicate-safe train/dev
partition and heldout disjointness; `tests/test_guarded_original_teacher_identity.py`
checks a deliberately modified student FFN cannot alter the zero-gate
source teacher. One-command test `scripts/check_native_quality_contracts.sh`.
All new code is source-committed but **not yet user-host executed**.

## One substantive native GPU quality experiment

Use the **actual original train and independently heldout token files** from
the matched quality experiment; do not use the 64-row federation smoke
capture as a full training corpus. Replace variables below with the real
existing host paths. A full MoE model checkpoint is required; the
`c2_layer1.pt` adapter alone is NOT admissible.

```bash
cd /path/to/LeechTransformer
git fetch origin
git switch experiment/consumer-preserving-multiscale-codec-20261031
git pull --ff-only

bash scripts/check_native_quality_contracts.sh

export TEACHER=/actual/full_dense_teacher.pt
export STUDENT=/actual/full_moe_student.pt
export TRAIN_IDS=/actual/train_ids.npy
export HELDOUT_IDS=/actual/heldout_ids.npy
export CONFIG=/actual/config.json

bash scripts/run_native_schaka101_lila.sh \
  --module scripts.train_native_student_quality_curriculum -- \
  --teacher "$TEACHER" --student "$STUDENT" --config "$CONFIG" \
  --train-ids "$TRAIN_IDS" --heldout-ids "$HELDOUT_IDS" \
  --stages auto --epochs 3 --batch 1 --max-rows 4096 \
  --student-ce-weight 1 --teacher-kl-weight 1 --local-mse-weight 1 \
  --calibration-fraction .15 --quality-budget .02 --progress-every 5 \
  --output /mnt/truenas/comet-net/experiments/lila_quality_curriculum_20261113/candidate.pt
```

If the source checkpoint already has a complete inferred config, omit
`--config`. `--max-rows` is a bound, not a request to invent additional
training data. This experiment can be expensive on the RX 580, particularly
when all layers are active. Run on an idle GPU with preserved logs; nothing
in the source implies a performance or quality win before the real receipt.

## Exact acceptance and remaining lemmas

| Source/quality gate | What this round pays | Still open |
|---|---|---|
| A01–A06 source identity and native model learning | Original FFN source restore, zero-gate dev CE parity check, native learning on original C2/even student parameters | Host contracts and physical training receipt, wider training corpus, stable teacher/student original checkpoint matching |
| O01–O03 original teacher-to-student observation | Focused CE, next-token KL, FFN MSE; original teacher remains independent | Source parameter/activation trace weld over full sequence and multi-seed quality confirmation |
| R02 / strict teacher-free replacement | Full student-only heldout CE admitted **only after** all layers trained; no hidden dense fallback for rollout | Actually improve 11.565138 against same heldout dense CE; prove/measure standalone stripped teacher checkpoint correspondence and compression |
| PERF01–08 | Per-stage completed train rows/target tokens/s, elapsed timing and allocator peaks | Real host run, same-token benchmark, compute/memory/energy Pareto vs dense |
| SBD04/05/06 diffusion | Preserved as separate optional research lane; no AR checkpoint reinterpretation | Real trained block denoiser, asynchronous refinement theorem and quality comparison |
| N/TNET/GPN federation | Earlier finite proofs, source code and local tests retained; local native baseline measured | Physical remote run paused by user; no new blocker on quality experiments |

**Priority change:** P0 validated existing runtime; **Q1** source-controlled student-only
learning and dev-selected curriculum now ACTIVE; **Q2** original quality improvement
and teacher-free checkpoint export; **Q3** native quality/resource Pareto across
seeds; Q4 trained block-diffusion research if it addresses a measured quality or
compute bottleneck; federation F2–F6 parked until user reopens it.

Quality gain is the only current promotion criterion. A training receipt can
show progress without CE improvement; no claim is made before results.
