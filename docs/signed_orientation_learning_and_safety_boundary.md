# History-conditioned signed-conjugate routing: hypothesis and tests

Source: Anatolii Kornienko, Leech-LILA README and models/model.py (AGPL-3.0-or-later).
DASHI: finite antipodal 5/9 and 14/27 carriers and time-indexed consumer fibres,
as already attributed in `docs/orbit_equivariant_moe_symmetry_compression_20261031.md`.

## Research hypothesis (not an ethical polarity claim)

A learned router may identify two *related computational orientations* under an
involution, such that one shared expert is useful for both. This does NOT make
benevolent and harmful content numerical negatives, nor predict safety from a
resonance score. Safety requires independently specified evaluation, datasets,
robustness to adversarial prompts, and fail-closed policy controls.

The upstream Leech-LILA reference README describes an orthogonal 24D placeholder
kernel, geometric loss and dream-state resonance diagnostics. Its documented
'DREAMING/AWAKE/ABSOLUTE GENESIS' states are a geometric score/status threshold,
not a measured moral, truth or harmlessness classifier. No such mapping was found
in the audited reference README or `models/model.py`.

## Executable first cut

`models/signed_history_expert.py` defines a history-conditioned learned
sign logit and a shared conjugate expert. For s in {+1,-1} and state H,

- canon = (s*x, s*H)
- y_c = F(canon) with GELU internal to F (not assumed odd)
- H'_c = U(canon,y_c)
- output = (s*y_c,s*H'_c)

This gives an algebraic relation between two *forced orientation charts*:
E_{-}(-x,-H)=-E_{+}(x,H) including state output.
The implementation now antisymmetrizes the learned router logit as\n`r(x,H)-r(-x,-H)`; except at exact zero logit, learned orientation switches\nunder joint inversion. At the zero tie the positive convention cannot obey\nstrict `s(-x,-H)=-s(x,H)` because both inputs coincide under inversion;\nexplicitly retain that fixed-locus exception. Autograd of the hard selected\nroute remains nondifferentiable and requires separately trained routing loss.

Crucially, `forced_sign` permits exact tests of paired expert and history
transition without confusing hard routing with differentiated training.
Autograd propagates gradients through the input/output sign conjugation into
shared expert and history-update parameters. A hard sign threshold has zero/
undefined router gradient: `signed_training_objective` does not claim to
train it. Router loss, straight-through estimators or smooth gating remain
future experiments, and moral alignment is **not** a routing objective inferred
from lattice geometry.

## BIDI / compression consequence

Store/communicate the orientation witness `s` and its history version.
For inference routing, the paired expert can share physical parameters and
history in a canonical chart, potentially reducing expert model duplication.
For codec use, a single sign bit alone does NOT compress a real activation:
encode canonical input plus a signed/corrective residual and count all bytes,
including reference synchronization. Unless the actual consumer is proven
equivariant, decode the exact activation before consuming it.

For sign on binary32/16: bitwise sign-bit toggling is an exact involution,
including NaN payloads and signed zero. Floating arithmetic using `-x`
often preserves NaN sign/payload on particular hardware but should not be
claimed universally bit-exact without checking. The current prototype is a
computational MoE experiment rather than a production lossless wire codec.

## Verification

Committed `tests/test_signed_history_expert.py` exercises forced-sign
equivariance, history-dependent route selection and nonzero shared-expert
gradients under negative orientation. No actual Leech-LILA checkpoint
fine-tuning, natural language harm classification, or performance comparison
is claimed.

Next max-cut is to (i) train a documented, independently evaluated router,
(ii) add and measure learned canonical historical expert state versus KV-only,
(iii) check signed action on true Q/K/24D Leech geometry and actual lattice
stabilizers if imported, and (iv) jointly benchmark bytes/latency and held-out
model quality under matched active compute.


## Implemented next cut: training and time unrolling

The signed history module now supports `train_router=True`, which retains
a hard ±1 *forward* action but supplies a `tanh(score / temperature)`
straight-through *backward surrogate*. This produces trainable router
gradients but is **biased**, may be unstable at boundaries, and does
not make route selection a mathematically differentiable discrete map.
Forced signs intentionally bypass router training.

`SignedHistorySequence` explicitly unrolls the recurrent historical
expert over the token dimension, returns the terminal history and
supports exact continuation into later invocations. This is more than
feeding a static history placeholder to a single expert. It is a
separate experimental module, NOT wired into Kornienko's original
LeechGPT block, preserving pretrained inference compatibility.

The repository regression checks: history-inverted conjugation,
learned route antisymmetry, nonzero straight-through router gradients,
negative-sign canonical-expert gradients, and equivalence of full
versus split sequence unrolling. These tests have been committed but
**not run against this branch in the available runtime**. No held-out
model training or quality evaluation has occurred.

The remaining high-alpha interface is to incorporate the sequence
cell as a configurable FFN alternative (not a transparent replacement:
it changes recurrent state and output semantics), checkpoint and
synchronize that state with KV cache, then benchmark matched capacity
and active FLOPs with a production-grade wire decoder and complete
routing metadata. No evidence establishes beneficial/harmful vectors
as an antipodal pair; evaluate behavioural interpretations separately.
