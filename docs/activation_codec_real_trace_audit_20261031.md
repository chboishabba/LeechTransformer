# Real Leech-LILA activation codec: first trace and next max-cut

Date: 2026-10-31. Empirical source: the locally executed 12-token
Leech-LILA CPU capture described by the experiment operator. These numbers
were supplied from that execution; they were not independently re-executed
by this GitHub connector pass.

## Per-frame lossless comparison

The real trace consists of 444 float32 frames across 37 module streams,
681,984 raw bytes. Per-frame compression resets the DEFLATE stream.

| Candidate | Ratio |
|---|---:|
| Byte shuffle + DEFLATE | 1.0461x |
| XOR then byte shuffle + DEFLATE | 1.0426x |
| Raw DEFLATE | 1.0399x |
| XOR + DEFLATE | 1.0368x |
| Byte delta + DEFLATE | 1.0349x |
| Raw | 1.0000x |
| Bit planes + DEFLATE | 0.9973x |

Current adaptive chart selector, after the raw-mode decoder fix
(commit 98923d3): 681,984 raw bytes -> 649,773 encoded **payload** bytes,
1.0496x, exact decoder inverse checks passed according to local execution.

Selected modes:
- Byte shuffle: 335 frames; aggregate mode-specific 1.0463x
- Byte delta: 49 frames; 1.0641x
- XOR: 49 frames; 1.0593x
- 24-channel byte permutation: 11 frames; 1.0424x
- 24-channel modular lifting: 0 frames.

These mode-specific ratios are conditioned on the selector choosing them;
they are NOT independent full-dataset comparisons, and selection overhead,
framing, context-state signalling and WAN critical path have not been charged.
The adaptive 1.0496x should not be interpreted as a statistically
validated held-out win: the candidate selection tested each frame before
choosing, and only 12 steps of a single trace were observed.

## Exact source ownership

Upstream Leech-LILA model architecture and original historical research:
Anatolii Kornienko, DOI links in this repository README. The model's
`generate_leech_kernel` uses QR on an explicit 24x24 input matrix; a
24-channel reversal or modular pair lifting is NOT established as an action
of the Leech lattice automorphism group.

Original triadic-video codec algorithms/engineering provenance:
`chboishabba/dashitest/compression/video_bench.py` and
`compression/triadic_pipeline.md`; this repository's prototype is an
independent implementation, not a copied implementation.
DASHI's existing proof owners include
`DASHI/Codec/TriadicExplodedTransformCompletion.agda` and
`DASHI/Combinatorics/TriadicVideoCodecObservationQuotient.agda`.
Do not upgrade video-compression diagnostics or lossless finite
representations into a theorem about neural activation entropy.

## Next high-alpha discriminator

1. Collect longer, multiple-prompt float32 traces and split results by
   attention, FFN and residual-block module. Retain exact bytes and
   causal dependencies.
2. Compare an admitted symmetry group of the *actual computation*, not
   arbitrary coordinate permutations. Prove exact invertibility and include
   group-element/chart signalling costs. The GELU and softmax computations
   must be respected; symmetry on 24-dimensional query/key coordinates
   does not imply global neural-network equivariance.
3. Test invertible integer lifting families, learned causal context charts,
   cross-channel prediction and triadic mask/sign entropy models, against
   byte shuffle at identical framing and decoder availability.
4. Model total codec cost per network cut (compression and decompression time,
   CPU/GPU load, bytes, network latency, queueing and federation throughput).
5. Link *verified* finite source/decoder round trips to DASHI's time-indexed
   fibre/hyperfabric and existing codec proof owners. Keep empirical
   compression claims outside kernel theorem claims.

## Repo topology

`chboishabba/DASHIg/.gitmodules` currently refers to
`SPUTNIKAI/LeechTransformer` which does not resolve through current
GitHub API lookup. `meta-introspector/LeechTransformer` is public,
but GitHub existence alone does not establish it as a rename of
SPUTNIKAI or as the exact submodule commit ancestry. Do not blindly update
the submodule remote/pointer without a commit-identity and access check.
