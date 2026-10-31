# Run the trained Leech-LILA / federated expert verification locally

Branch: `experiment/consumer-preserving-multiscale-codec-20261031`, PR #4.
This harness uses original saved weights and **never silently substitutes synthetic weights**.

```bash
git fetch origin experiment/consumer-preserving-multiscale-codec-20261031
git switch experiment/consumer-preserving-multiscale-codec-20261031
python -m pip install torch numpy
cat > my_trained_config.json <<'JSON'
{"n_heads":8,"moe_top_k":2}
JSON
python -m scripts.verify_trained_federated_moe \
  --checkpoint /absolute/path/to/your/trained-checkpoint.pt \
  --config my_trained_config.json \
  --device cpu \
  --tokens 8 --batch 2 \
  --output my_trained_moe_receipt.json
```

Replace `n_heads` and `moe_top_k` with the **actual training configuration**, not the sample values above. The module refuses to guess them if the checkpoint does not store them. Use `--device cuda` only if the installed PyTorch backend supports it; start CPU for deterministic parity diagnosis.

Supported input: a plain PyTorch tensor `state_dict`, or an object with `model_state_dict`, `state_dict`, `model`, or `model_state` mapping to a state dict. `torch.load(weights_only=True)` is used for security. Other model/checkpoint classes require a separately attributed loader; no unsafe generic pickle fallback is provided. Config JSON can override an embedded plain-dict config. Strict state-dict matching and inferred embedding dimension/layers/MoE count reject incompatible checkpoints.

**The suite checks:**
- identical saved trained parameters loaded under strict structural compatibility;
- full token forward vs sequential KV+expert-history continuation (logits and every final expert history);
- actual last-layer hidden activation bytes: raw vs adaptive codec, decode exactness and SHA-256;
- total complete packet costs at 1 and 10 Gbit/s, with RTT and locally measured encode/decode time;
- diagnostic router IDs, weights and output parity under joint sign inversion, **without** claiming a learned expert sign means semantic opposition, unethical output or a Leech group automorphism.

**Scope and current blockers:** The suite has not been executed on the user's private trained checkpoint. It does not send requests over a remote socket, obtain true p95 network latency, simulate acknowledged temporal-delta state, train or fit a Sophontic semantic action, or prove full action equivariance. A dense checkpoint cannot certify learned expert specialization. A model checkpoint with different architecture must be adapted by source-specific code, not silently reshaped.

**Source attribution:** Leech-LILA original architect Anatolii Kornienko; historical/sophontic and 369/hypervoxel donors retain separate primary/PR ownership. DASHI PR #1442 contains conditional Agda finite and general transport theorems but they are not a kernel check of this Python program.

## The decisive next evidence

Return `my_trained_moe_receipt.json` (or paste output). It will expose the *actual* failure/success discriminators needed to construct concrete model-dependent certificates, especially router sign permutation failure and any exact history replay divergence. For true WAN results add a separately authenticated remote worker and held-out model-quality tasks; don't label ideal 1/10 Gbit/s serialization as a measured benchmark.
