# X369 authenticated original source: marked-only consumer non-descent witness

**Evidence owner:** [original native gfx803 receipt](x369_original_native_minimal_consumer_20261128.json) at LeechTransformer PR #5 commit ff525f0. Code that checks this receipt without another GPU run: `scripts/x369_original_receipt_mark_collision_weld.py`; regression `tests/test_x369_original_receipt_mark_collision_weld.py`. Original Leech-LILA by Anatolii Kornienko, DASHI source bridge separately attributed.

**Scope:** authenticated source HEAD `862818f58284522c48ccb095b3afaa3021571fdd`, translated candidate checkpoint SHA-256 `fa473dcaab3afd4ce87d6d175539b2d05b09ea7879b1589d29eebc263a241a44`, model SHA-256 `54bc2a445dd9c520b2f3b3e34544cb632fce42871c69c0d4c4a93dfe89e91775`, original eval source `SourceGeometryOrderedLearner` on RX580 gfx803. Admission passes, but `training_provenance_authenticated=false`, `release_admitted=false`, and no entire float-history source universe was certified.

## Two actual reachable original histories

Original seed indices 0 and 1 each receive the SAME first original control `[8,0]`. Their first learned recurrent successor is an actual 384-byte `E=4 × 2 × W=12 × float32` source history, not the canonical 189-signed-seed input form. The independently verified exact canonical decoder cannot decode either as an original mean+residual seed. Parent original source-mark audit separately confirms **756/756** marked-seed transitions leave the canonical original-seed image.

- Original history A (after one step), SHA-256: `5352748e1fc85c0eb6e0c77c270f7b6444a24a878cece8b65a4331d743b15c92`.
- Original history B (after one step), SHA-256: `df6fcb88f798bd9e4610c2287827ab0caee04676df4406aa2f76db4406aa2f76d???`.

Both map to **the same** candidate coarse value `('unmarked_original_recurrent_history',)`. Then the SAME second original control `[4,1]` is applied. The source-owned typed byte observations in the original word-residual receipt disagree on `logits_0`, `logits_1`, `logits_2`, `output`, `ids`, `weights`. They agree on `local_signs` in this particular pair. All seven fields are original model outputs, not task labels; no artificial source step was inserted.

Thus there exist original reachable h≠h' with K(h)=K(h') but C_[4,1](h)≠C_[4,1](h'). The factorization `C_[4,1] = c_[4,1] ∘ K` is impossible on this admitted reachable domain. The existing DASHI consumer-fibre repair theorem implies **every** sufficient repair of this coarse K must distinguish this pair. This negative requires NO finite global successor closure.

Important: This proves a coarse-source observer collision, NOT that a learned recurrent model is non-equivariant under the original group action. Likewise it does not refute full native history bytes, the actual constructed semantic six, semantic+capacity, original source Stage13 or the genuine five-field pants seam. It establishes no unique/minimum repaired code for the unrestricted source.

## Exact offline validation / follow-up

Run:
```bash
python -m unittest -v tests.test_x369_original_receipt_mark_collision_weld
python -m scripts.x369_original_receipt_mark_collision_weld \
  --receipt docs/receipts/x369_original_native_minimal_consumer_20261128.json \
  --output runs/x369_original_mark_only_collision_offline.json
```

This does NOT use the GPU. It checks original stage/checkpoint identity flags, raw initial seed canonical membership, both actual first-step state hashes and exact byte layout, nonmembership of two post-step source histories, original second control and distinct serialized typed original consumer observations. It produces a scoped receipt; it is **receipt-based**, not an independently reexecuted GPU source step.

The new `scripts/x369_original_source_witness_first_probe.py` can later replay such a witness against the native checkpoint and extend the search to finer charts, sampling same-fibre pairs first. Avoid repeating the entire original 1000-state BFS merely to obtain a first separating coarse witness.

## Corrections to prior summary

- The 384-byte original source history size is fixed by 4 experts, 2 modes, 12 width and float32. Counts of 284–287 nonzero XOR **bytes** are out of 384, not out of 288.
- `state_budget` after 819 checked edges and 1000 discovered states does **not** prove the full neural float source is nonfinite; it only proves that the specified exploration budget did not close this graph.
- A byte mismatch between learned step and independently specified signed-task seed is **not by itself** a failure of group equivariance: equivariance requires separately comparing transformed source inputs and outputs under the declared group action.
