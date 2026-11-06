# Signed shell max-cut for real Leech activation transport (2026-10-31)

## Source ownership / import boundary

Implementation: `scripts/benchmark_multiscale_fibre_codec.py` and
`tests/test_activation_signed_shell.py`. Leech-LILA original model credit:
Anatolii Kornienko (upstream author, as acknowledged in model source).
Relevant predecessor inspirations, **not copied proofs**:
`dashitest/compression/video_bench.py`,
`DASHI/Codec/TriadicExplodedTransformCompletion.agda`,
`DASHI/Combinatorics/TriadicVideoCodecObservationQuotient.agda`,
`DASHI/Reasoning/HistoryTimeIndexedConsumerFibreExact.agda`,
and dashiRTX signed transport. The finite periodic PDE Littlewood–Paley
owner is an analogy about keeping signed shell contributions, not an
activation compression theorem. Prime correlations are out of scope.

## Exact implementation

For each consecutive 8-channel subgroup in a 24-channel block (three
subgroups), three reversible integer-lifting stages apply to each original
storage byte lane independently. At one stage the bijection is

```
(a, b) -> (a, (b-a) mod 256)
(a, d) -> (a, (a+d) mod 256).
```

The signed residual is represented as a two's-complement byte (the
modulo-256 detail *retains sign information*; it is not absolute-valued,
squared, or discarded). Three rounds are 8->4->2->1. Inverse applies
2->4->8. All coefficients are retained. Thus no cross-shell term is
lost, and each input IEEE floating-point bit pattern including NaN payload
or signed zero is reconstructed unchanged. However, this is **not**
a proof of GELU/softmax equivariance, nor a finite-precision approximation
to the Leech lattice automorphism group. Byte-lane lifting differs from
integer-value lifting on 32-bit floats; its performance is empirical.

The chart selector compares per-frame raw bytes, byte-plane shuffle,
one-stage 24-coordinate lifting, three-stage signed shell lifting,
their byte-shuffled outputs, and causal delta/XOR variants. Decoder
round-trip checks apply to every candidate. `mode` is costed by an
optimistic single-byte identifier. Negotiation, acknowledgment, frame
length, checkpointing, stream version, network packet latency, encoder
CPU load outside the zlib call, and state synchronization are not
completely charged. A production stream cannot assume that a prior
frame on the sender is already accepted at the receiver.

## Verification and empirical status

A local isolated NumPy check of three-stage signed shell mapping and
its inverse ran 600 randomly generated frame roundtrips (2- and 4-byte
element widths; 24, 48 and 384 channel cases; 100 draws each), PASS.
The exact code is also guarded by the repository regression module,
which was committed but has **not been run as a full repository test**
in this connector-only pass. No real captured activation data was
accessible to the execution runtime here.

Real Leech-LILA evidence from the previous 12-frame, 37-stream capture:
681,984 bytes raw; former adaptive prototype 649,773 payload bytes,
ratio 1.0496x. This is baseline only. Three-stage signed shell
performance on actual captured Leech-LILA activations remains UNKNOWN.
Do not treat 1.0496x as the ratio of the new transform.

## Bidirectional max-cut

Producer obligation: encoder must emit a finite chart identifier, each
signed detail, and the exact preserved coarse coefficients. Every codec
mode must admit deterministic reverse mapping.

Consumer obligation: the decoder reconstructs exactly the original
activation bits before GELU, softmax, addition or downstream use.
For a nonlinear consumer F, preserving output after decompression is
trivial from exact input equality; predicting F from shell energies
is *not* allowed without a full error/correction and arithmetic proof.

Decision obligation: to improve *inference throughput* rather than
only payload bytes, measured savings must exceed compression cost and
context synchronization overhead on the actual critical network cut.

RPD / steelman next: test the real Leech checkpoint on longer streams;
add a genuinely admitted model symmetry group if provable from the
actual kernel and rounding conventions; test triadic support/sign
residual streams with exact entropy decoder roundtrips. Avoid importing
unrelated analytic claims as theorem authority.
