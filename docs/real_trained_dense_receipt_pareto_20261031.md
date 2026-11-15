# First real trained-checkpoint receipt — dense baseline (2026-10-31)

Owner: LeechTransformer PR #4, verifier fixed in user commit e5ab5f4.
User-supplied receipt: `leech_dense_verifier_receipt.json`.
Configuration: LeechGPT vocab 2048, D384, 12 layers, 8 heads,
block_size 512, dense MoE_experts=0; checkpoint best_model.pt.

## Verified user-executed claims

Targeted tests: 11/11 passed (user report).
Strict real trained-checkpoint load: passed.
Full vs incremental KV cache logits max |delta| 9.5367431640625e-06
(RMSE 1.5918701592454454e-06).
Activations exactly restored: source 24576 bytes; raw packet
24784 bytes; adaptive packet 23004 bytes. Identical SHA-256:
b9f31d10031dfde6e313b3fb0fbead7b032fc3c7ce379ca0c40ed71b66fe7c63.

## Codec Pareto: *raw wins on 1/10 Gbit* with measured codec overhead

Measured per-run elapsed milliseconds:
raw encode .139397 / decode .087842;
adaptive encode 1.472265 / decode .149940.
Saved 1780 wire bytes per direction, 1.07738x ratio;
extra encode+decode cost ~1.394966 ms.

The existing verifier's model assumes an equal-sized reply and 5 ms RTT:
  1 Gbit/s: raw 5.623783 ms; adaptive 6.990269 ms.
 10 Gbit/s: raw 5.266893 ms; adaptive 6.659011 ms.
Equal sized response assumption is NOT measured, and one-run codec
timings are not stable p95 WAN measurements.
Under this model, break-even bandwidth = 16*1780 bytes*1000/
1.394966 ms = ~20.42 Mbit/s; below that compression can start to win,
above that raw is faster if all other terms equal.

## Exact inference scope and next cut

The checkpoint is DENSE; `geometry_diagnostics=[]` and
`final_history_parity=[]` are therefore expected, not missing
successful MoE geometry certification. No trained signed-history MoE
equivariance, Sophontic paired semantic action, safety interpretation,
federated expert placement, network RTT measurement or authenticated
remote worker execution is verified by this receipt.

Leverage this as dense Pareto control; train/load an opt-in history MoE
using the original model's actual data, then require cache/history
parity and actual learned route/permutation/correction checks.
For networking, test no-round-trip/local computation and batched remote
execution before investing in marginal 1.077x byte compression.
Retain DASHI 369/5-in-9/14-in-27, non-antipode Sophontic,
surface+residual consumer quotient and prior source ownership.
