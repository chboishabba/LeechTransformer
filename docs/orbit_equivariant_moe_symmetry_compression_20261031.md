# Orbit-equivariant MoE and exact symmetry compression: joint design cut

Date: 2026-10-31. Source attribution: upstream Leech-LILA model
(Anatolii Kornienko); DASHI-authored finite orbit/involution formalism.
This is a design/specification receipt, not measured quality or a
kernel-verified Leech automorphism implementation.

## Verified mathematical owners

- `DASHI/Foundations/BalancedTernaryAntipodalOrbitExact.agda`
  defines the strict antipodal action on T^3. It has 27 points, one
  fixed zero, 13 nontrivial two-element antipodal orbits, hence
  14 quotient classes (not 13 total classes).
- `DASHI/Biology/TriadicKernelLiftQuotientExact.agda`
  gives T^2 (nine raw states), five quotient classes under
  simultaneous negation, including the zero fixed class.
- `DASHI/Moonshine/Monster369SignedPairNineOrbitExact.agda`
  constructs signed pairs and descends quarter-turn and reflection
  actions to the nine-state quotient. These do not by themselves
  identify a Monster representation acting on neural weights.
- `DASHI/Foundations/UBP/LeechValidMoveSet.agda`
  requires explicit membership/closure receipts for lattice-valid
  moves. A finite valid move is not automatically a full automorphism
  of the Leech lattice or the trained network.
- `DASHI/Biology/ExceptionalLatticeGrokkingProtocolExact.agda`
  records Leech dimension 24 and 196560 minimal vectors as benchmark
  data; the inspected file does not implement the whole lattice.

## Tensor-object matrix (do not flatten geometry away)

- hidden token [B,384] = [B,16,24] as a feature chart
- sequence hidden [B,T,384] = [B,T,16,24]
- Q and K before/after QR kernel: [B,8,T,2,24]
- attention score [B,8,T,T] = token-token interactions
- FFN expanded [B,T,1536] = [B,T,64,24] as a possible chart
- K/V cache [B,8,T,48] = [B,8,T,2,24] as storage layout

The [B,16,24] FFN input is not sixteen independent Leech lattice
points. Different axes have different physical and mathematical roles.

## Architecture alternatives to benchmark against the original model

(A) Dense original LeechBlock FFN, with existing shared attention
and optional resonance bias, as the quality/compute reference.

(B) Conventional sparse MoE FFN with E/k/d_ff matched active FLOPs;
preserve existing Q/K transform and geometric training objective.

(C) Signed-orbit MoE on a *declared* finite carrier:
a canonical expert per orbit, with explicit group-action witness.
For the two-element group, tie parameters and enforce
 E_minus(x) = -E_plus(-x).
If E_plus is a standard GELU network, this definition nonetheless
induces a mathematically equivariant paired family; it does NOT assert
the base GELU itself is odd. Sign actions on inputs and outputs are
exact real operations, while finite-precision exactness must be
verified for a specified dtype and operation ordering.

(D) Hierarchical 5/9 and 14/27 routers:
coarse sign-pair orbit first, finer sign-triple orbit second;
record the orientation witness and fixed-zero case. These are
candidate routing labels, NOT a discovered mapping of trained
384D model activations into 5 or 14 experts.

(E) Geometry-conditioned shared experts:
route on signed/orbit summaries from 24D blocks, but retain full
384D expert inputs unless an independently certified sufficient
statistic for the downstream consumer has been proved.

All candidates must be compared at matched active FLOPs,
matched total parameters, and under the actual network layout.

## Coupled compression contract

Given exact data x and finite action g, choose a deterministic
canonicalizer c and orientation witness g such that x=g(c(x)).
Then transmit canonical payload + action ID + residual correction.
The side information must identify the orbit class, fixed-zero
case, operation, data dtype, axis layout, time/version dependencies,
and any approximation defect. A quotient alone is lossy.
For neural floats, do not quantize arbitrary float32 bytes to
balanced trits and call it exact; use the trit stream as a symbol
model while preserving an exact correction/bypass route.

Codec consumption must preserve the original tensor bytewise
BEFORE the next computation. A stronger computational shortcut
requires an equivariance theorem for the entire consuming
operator (including normalization, nonlinearities, learned
weights, finite-precision effects), rather than only its
geometric kernel.

## Experiment discrimination

Measure model-specific orbit canonicalization on actual Q/K,
FFN inputs/outputs and time-indexed KV state. Compare separate
axis-aligned codecs, orbit-action metadata + exact correction,
and generic byte shuffle; retain the signed details at every
scale. Track both per-frame compression AND per-expert
WAN round-trip, batching, processing time, placement, quality,
balance, dead experts and effective tokens/s.

No codec ratio or training advantage for these candidate orbit
experts is yet measured, and no global Leech/Co0 network
equivariance is claimed.
