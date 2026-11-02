# Learned history-orbit MoE implementation ↔ Agda BIDI (2026-10-31)

## Corrected definition of a mixture

Original `SignedHistoryExpert` is one parametrized function with two
sign-conjugate charts: useful as a baseline, but **not a Mixture of Experts**.
This PR now also contains `models/history_orbit_moe.py`:

- `num_experts >= 2` *independently learned* FFN networks, each with its
  own learned recurrent history transition.
- A shared learned router scores (current activation, expert-specific history)
  for each expert, selects top-k expert identities, and normalizes their
  selected scores with softmax. This is genuine sparse dispatch, not
  dense averaging of all expert evaluations.
- A separate antisymmetric learned orientation score chooses ±1 chart
  *within* each chosen expert. Expert ID and chart sign are orthogonal.
- Only selected expert histories advance; unselected histories retain
  their previous state. Expert outputs are weighted and summed, but
  history vectors are never averaged.
- `forward([B,T,D], [B,E,D])` returns outputs, terminal per-expert history,
  route IDs, signed witnesses and routing probabilities, allowing
  segmented causal replay.
- Each expert uses GELU internally; it is **not assumed odd**. Sign
  conjugation is applied outside its learned function.

This is a first-stage stateful sparse MoE, not yet integrated into the
LeechTransformer checkpoint format or trained for language modelling.

## Agda correspondence

Companion `chboishabba/dashi_agda` PR #1442 adds
`DASHI/Reasoning/LeechLearnedOrbitMixtureBidiExact.agda`:
two distinct `ExpertId` constructors, an `ExpertFamily` indexed by
expert ID, a separate routing selector, signed conjugation within
a selected expert, and a *conditional* commuting-square theorem that
requires route-inversion invariance.

This theorem is **not** a proof that arbitrary learnt PyTorch routing
is inversion invariant: the actual router depends on individual
history states and can choose different expert IDs on inversion.
A future implementation can either enforce the invariant, transport
the change of expert ID as an explicit group witness, or count a
corrective route. The current Agda finite dispatch skeleton does not
model real softmax weights or the batched `top_k` algorithm.

## Why this matters for symmetry compression

An encoder must communicate or deterministically reproduce:
model version, chosen expert IDs, sign witness per selected expert,
the exact activation, and histories or agreed history versions.
The sign group allows *parameter sharing across orientation charts
within one learned expert*, but does not make all E learned experts
the same. The history itself can dominate payload size unless
it is receiver-local and synchronized; do not assume it is free.

Measure per-stage, per-axis traffic for hidden [B,T,384],
Q/K [B,8,T,2,24], attention [B,8,T,T], FFN intermediate
[B,T,1536] and K/V [B,8,T,48]. A 24D chart is not automatically a
faithful Leech orbit. Benchmark lossless payload ratio against
the prior 1.0496x 444-frame real capture, with full transport costs.

## Test and source boundary

`tests/test_history_orbit_moe.py` checks independent parameter
storage, shapes, top-k weights, selected-only history updates, full
vs segmented execution, and training gradients into the router,
orientation and selected experts. The tests were added through the
GitHub connector; there is no claim that they have executed in the
current runtime. A local container has PyTorch installed but no
network route to the GitHub raw-file endpoint, so no exact checked-out
branch was available for execution from that runtime.

Do not assign geometric orientation to 'good' or 'evil': Anatolii
Kornienko's March 2026 observation was an exploratory qualitative
comparison across model training phases and did not measure an
ethical axis. See `signed_orientation_learning_and_safety_boundary.md`.

## Remaining high-value cuts

1. Run the new tests and fix any real errors before merging.
2. Expose a configuration-gated MoE replacement in real LeechBlock,
   including sequence/step state and KV-cache interactions, without
   breaking original dense checkpoint loading.
3. Specify a real signed 24D Leech action and verify the complete
   consumer rather than relying on a placeholder QR kernel.
4. Train ordinary-history top-k vs signed-history top-k and dense
   baselines on matched active FLOPs, model memory, prompts and seeds.
5. Add actual decoder receipts, exact float-byte roundtrip and
   network-cut latency/throughput; never equate lossless state
   representation with observed compression gain.

## New finite weighted MoE owner

The companion DASHI PR #1442 adds
`DASHI/Reasoning/LeechWeightedHistoryMoEBidiExact.agda`:
two independently indexed expert functions and recurrent
transitions; history-indexed per-expert scoring; deterministic
top-one/top-two selection; natural-number weighted selected
aggregation; unselected history preservation; exact bidirectional
route/output/history receipt; and finite memory-only segment replay.
See `Docs/LeechLearnedMoEImplementationParity20261031.md`.

The finite formalism is now closer to a genuine mixture, but it
**does not yet match Python's generic E/k, float softmax or autograd**
and has not been kernel-checked. An Agda source function selected by
expert ID is an abstraction of learned weights, not a theorem that
optimization produces specialization. Do not call formal parity
complete until those interfaces are discharged. The original
`models/history_orbit_moe.py` implementation remains a standalone
prototype, not yet wired into the pretrained model.


## LeechGPT integration completed in source (2026-10-31)

`models/model.py` now offers default-off config keys
`moe_experts=0`, `moe_top_k=2`, `moe_expert_width=None`;
`config/config.py` is synchronized. In dense mode,
`LeechGPT.forward` retains its historical four-item return and
dense FFN state-dict topology. With `moe_experts>=2`,
`LeechBlock` uses `HistoryOrbitMoE` in place of its FFN, with
the original LeechAttention, resonance and residual pathways
left intact. The unused dense FFN is not stored on MoE models.

For MoE inference `past_expert_histories` accepts an explicit
per-layer `[B,E,D]` state, while `return_expert_histories=True`
opt-in exposes a fifth return value. Cached continuation rejects
existing KV cache with missing expert state rather than silently
resetting historical expert memory. In train mode, the learned
sign router uses its biased straight-through estimator.
`tests/test_leech_moe_integration.py` covers dense strict load,
MoE gradient flow, full-sequence versus token-cached equivalence
and fail-closed history loss. Tests have been COMMITTED, not
executed in this runtime.

**Outstanding formal correspondence:** The companion DASHI
`LeechWeightedHistoryMoEBidiExact.agda` proves finite selected
state replay and route/output/history receipt roundtrip, but
does not simulate this PyTorch per-layer stateful KV/cache
implementation, real-valued top-k or straight-through gradients.
Companion `IEEEOrbitSignWitnessBidiExact.agda` establishes a
symbolic sign-bit and witness roundtrip, NOT the real codec's
whole-block lexical canonicalizer, packed mask and framed DEFLATE.
Please treat these as explicit proof gaps, not completed formal
bisimulation.

**Execution blocker:** The model runtime has PyTorch 2.10 CPU,
but a GitHub checkout in that runtime failed at DNS resolution,
so it could not run exact committed branch files. This statement
is not evidence of an implementation defect or passed tests.
