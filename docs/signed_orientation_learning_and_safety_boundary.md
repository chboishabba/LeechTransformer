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
For a learned router to be globally equivariant, the router itself would need
s(-x,-H)=-s(x,H), which an ordinary biased affine hard-sign router does not
guarantee. An orientation-aware router can instead be trained or designed with
an explicit antisymmetric score `r(x,H)-r(-x,-H)`, with the zero/tie case
specified; implement that as a separate experiment if empirical tests justify it.

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
