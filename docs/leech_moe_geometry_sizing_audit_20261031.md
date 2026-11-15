# Leech-LILA MoE geometry and sizing audit (2026-10-31)

## Actual model mechanism (read from models/model.py)

Default d_model=384, n_heads=8, head_dim=48=2*24, n_layers=12,
FFN intermediate=4*d_model=1536. Q/K projected by same fixed QR-derived
24x24 orthogonal kernel, applied blockwise to 16 head subblocks per token.
V is not transformed. The registered block-diagonal W_leech is allocated,
but the Q/K forward actually uses its first 24x24 block and repeats it.
This is **not** an explicit Leech-lattice construction or an implementation
of Co_0 symmetry. Do not confuse the implemented QR kernel with an
automorphism of the Leech lattice.

For exact real arithmetic, Q'=QW and K'=KW with orthogonal W imply
Q'K'^T=QWW^TK^T=QK^T. This is a gauge-coordinate cancellation
of the fixed kernel in attention scores, subject to floating-point
rounding and any nonuniform implementation effects. Geometry affects
training through LeechResonanceLoss on hidden vectors, if invoked
in the training objective. An optional LeechResonanceBiasing layer
produces a learned logits correction using the 24D basis.
These are separate from Q/K cancellation.

## MoE placement preserving these mechanisms

First replace only LeechBlock.ffn with sparse experts. Preserve:
1) the entire shared attention layer (including its kernel),
2) layer norm and residual connections,
3) the geometric loss, whose target/gradient can condition expert routing,
4) optional final geometric resonator.

Expert e: E_e(x)=W_down,e GELU(W_up,e x). To preserve activation
FLOPs relative to dense 4d FFN while using top-k experts, choose each
expert intermediate d_ff,e ~= (4*d)/k, subject to kernel packing and
memory alignment. Router may use the raw hidden state h or a
24-coordinate geometric summary. A geometric router can incorporate
sign-aware orbit/shell features but is NOT automatically equivariant.

Per-layer linear-weight arithmetic ignoring bias/norm/router:
shared attention = 4*d^2; baseline FFN = 8*d^2.
MoE stored expert weights = E*2*d*d_ff,e = (8E/k)*d^2,
active expert weights = k*2*d*d_ff,e = 8*d^2.
Router weights E*d. Overall baseline 12*d^2 per layer, versus
(4+8E/k)*d^2 + E*d stored per MoE layer, and
about 12*d^2 + routing/gather active. At d=384, L=12,
one dense FFN stores 1,179,648 linear weights. Embeddings
and any tied head are separately accounted for; checkpoint size
and optimizer states are NOT model parameter count.

Example configurations preserving approximately dense active FFN:
- E=4, k=1, d_ff,e=1536: 4x FFN storage.
- E=8, k=2, d_ff,e=768: 4x FFN storage.
- E=16, k=2, d_ff,e=768: 8x FFN storage.
Each requires a capacity/loss/latency sweep; none is an optimum.

## The geometric object being routed

For a last-token hidden vector [B,d] with d=384, reshape into
[B,16,24] *as a feature chart*, not necessarily 16 independent
lattice vectors. Q/K internally are [B,H,T,2,24]. The expansion
after the first FFN linear is [B,T,1536], and the KV cache
has [B,H,T,48] per K/V. A sparse expert's input remains [B,T,384]
unless its architecture is explicitly changed; a 24-dimensional
summary is an optional ROUTER FEATURE, not a lossless replacement
for the original expert input. Different objects have different
communication costs and locality.

## Geometry-aware router candidates and their tests

Baseline: standard learned softmax-top-k router on h.
Geometry A: learned router on flattened 24D block coordinates.
Geometry B: sign-aware summary per block (norm, projection,
resonance angles, signed coefficients), with learned router.
Geometry C: tied or grouped expert routing across 24D block orbits
ONLY after an admitted finite group action is explicitly defined
and proven compatible with the trained computation.

Compare held-out perplexity, training stability, expert utilization,
max load/average load, dead experts, throughput, active FLOPs,
total parameter bytes, inter-worker bytes/token, and fraction
of same-machine expert calls. Include router and network scheduling
costs. Run dense baseline and a matched-active-FLOP sparse family,
then a matched-total-parameter family. Do NOT claim that 24D
geometry determines E or k.

## Distributed inference sizing objective

Optimize measured token throughput/latency subject to a quality
constraint, GPU VRAM/RAM, model availability and queueing limits.
For each layer: send only representations needed by the chosen expert.
A remote expert call has input of d * bytes_per_element per token
and output of d * bytes_per_element, plus routing/packet headers.
At d=384 and FP32 this is 1536 bytes each way per expert call,
before compression and batching. The same tensor's *tensor-axis
structure* (tokens, layer, experts, 24D blocks) must be preserved
as metadata without replicating the raw payload.

The earlier 1.0496x lossless payload ratio was measured on 444
module-output frames (12 decoding steps), not expert input traffic;
do not extrapolate it directly to MoE WAN transfers.

## Next falsifiable cuts

1. Experimentally remove or toggle Q/K kernel and compare logits
   in FP32: exact-math equality is predicted, finite precision
   deviations should be quantified. Repeat with geometric training
   loss enabled/disabled during training, separately.
2. Preserve original dense checkpoint and add an MoE FFN behind
   a configuration flag. Transfer baseline FFN into a shared
   residual branch or initialize each expert from the dense FFN;
   evaluate cold-start specialization without pretending identical
   outputs after uncalibrated routing.
3. Build comparable E/k/d_ff sweeps under active FLOPs and memory
   budgets; report detailed routing traces, per-module compression,
   and full communication critical paths.
4. Only then select admitted 24D symmetry transformations for
   routing or codec and formalize their exact consumer relation
   in DASHI Agda. No protected ESD files modified.
