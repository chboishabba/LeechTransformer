# Sibling triadic reuse and DeepSeek/Kimi steelman BIDI (2026-10-31)

## Actual in-repo source audit

Directly inspected from `chboishabba/dashitest`:
- `compression/video_bench.py`: temporal signed-residual/video benchmark owner (see sibling evidence inventory).
- `video_bench_reuse.py`: actually contains `_canonical_block_signature`, choosing the lexicographically smaller byte signature between a balanced-trit block and its simultaneous sign inversion. This is an executable finite antipodal *canonicalizer*, potentially useful for exact orbit reuse. A canonical signature alone is NOT sufficient to decode the original block unless action orientation and any correction are retained.
- `scripts/gen_codec_E_seq.py`: computes `mean(abs(planes), axis=(2,3))` and normalizes by mean band activity. Output `[T,B]` is a **nonnegative magnitude-only diagnostic**, not a lossless signed stream or reconstructible video codec. Keep the signs/planes in the codec itself.
- `DASHI/Combinatorics/TriadicVideoCodecObservationQuotient.agda`: explicitly requires deterministic sign/magnitude quotient, reuse equality/correction, side information and reconstruction.
- `DASHI/Physics/Closure/LightTransportFibreClosure.agda`: abstract closure surface with non-promotion gates. No 600-frame/0.032-bpp numerical receipt was found in the inspected file; do not attribute that benchmark to it without tracing the producing source.

The video codec acts on frame/block spatiotemporal arrays. Leech-LILA compression acts on typed activation objects, including Q/K [B,8,T,2,24] and per-expert hidden histories [B,E,384]. A quotient action must be explicitly translated between *these* carriers; raw video rates are not activation rates. Never transfer a lossy rate/distortion number to a lossless tensor transport claim.

## External MoE bidi steelman (architecture, not scale copy)

DeepSeek-V3 technical report, arXiv:2412.19437:
https://arxiv.org/abs/2412.19437
- One shared expert and 256 routed experts, 8 active routed experts per token.
- Auxiliary-loss-free load-balancing approach and node-limited routing.
- Multi-head latent attention and compressed KV representation.
For our experiment: shared expert as a common (possibly even/parity)
component; independent routed families for learnt specialized effects;
balanced top-k use and placement-constrained WAN routing; measure
changes to expert quality, not only entropy.

Kimi K3 primary release:
https://www.kimi.com/en/blog/kimi-k3
- 896 experts, 16 selected per token, Stable LatentMoE, Kimi Delta
Attention and Attention Residuals; long historical context.
- Different from Kimi K2's 384 experts/8 selected; do not conflate versions.
For our experiment: latent history/activation routing, model-depth
history retention, state replay and expert-load stability, with a
matched quality/parameter/active-FLOP sweep. 896/16 is NOT an
empirical optimum for a 384-dimensional Leech-LILA model.

## BIDI target and proof/experiment map

A. **Producer -> consumer**: source tensor, tensor-axis schema,
    model+expert history version, selected IDs, orientation witnesses,
    signed residuals and deterministic exact reconstruction.
B. **Consumer -> producer**: prove which transformation or quotient
    commutes with the actual downstream operator. If not, restore
    the exact tensor bytes before consumption.
C. **Learned MoE**: independent learned expert parameters, shared
    component, learned top-k, normalized weights, load balancing,
    no dropped tokens, explicit selected-only memory transitions,
    and reproducible sequence continuation.
D. **Symmetry**: test the existing dashitest Z2 canonical signature as
    a codec baseline on separately typed 24D activation *storage*;
    retain the full orientation action and all correction bits.
    Do not assume arbitrary float tensor inversion is a learned
    Leech automorphism or that "evil" aligns with sign.
E. **Sizing and network**: measure active FLOPs, stored params,
    real expert utilization, decoder availability, traffic per
    expert/packet, placement, WAN latency and held-out quality.
    Use DeepSeek/Kimi only as architecture and evaluation targets,
    not a mandate to copy a trillion-parameter configuration.

## Formal status

Companion DASHI PR #1442 includes finite route/state receipts and
sign-conjugate algebra, but NOT a proof of the actual batched PyTorch
softmax, generic E/k or gradient updates; this remains explicitly OPEN.
The highest-alpha coding next step is a lossless *bitwise* sign-orbit
canonicalization with side-information accounting, tested on real
Leech captures and compared against byte shuffle.
