# Native GPU selected-expert federation: integrated two-host handoff (13 Nov 2026)

## What this tranche implements

Use the **already provided** Schaka ROCm 10.1 / patched PyTorch environment. Do not rebuild ROCm, rerun generic OpInfo qualification, substitute CPU GEMMs, or create a new container. The owner of the original Leech-LILA design remains Anatolii Kornienko.

The prior RX 580 PCIe-only run completed with **zero selected-output/history error** on 16 real rows × 3 repeats, but its pipeline averaged **168.5 versus 188.1 microbatches/s** sequential (about 10.4% less throughput). Stop treating one-card CUDA stream overlap as a proven federation speedup.

This tranche adds `scripts/native_federated_selected_worker.py` with **real TCP frames**; source-selected IDs, exact JQ1 packed activation/selected histories, model/history revisions, HMAC-SHA256, replay-checked session counters, a bounded outstanding-request window, and true native GPU selected-expert execution on the worker. The coordinator retains source router weights and all inactive histories, reduces unweighted worker expert outputs, verifies the selected histories, and reopens only selected histories. It records request/reply bytes, per-request RTT, worker compute wall time, throughput and precise checkpoint hashes.

The worker loads a full student checkpoint into CPU RAM but moves **only the chosen MoE layer** to worker GPU, not the transformer attention/embedding stacks. It still carries **all E experts of that one layer**; sparse expert weight residency, actual multiple miners and model tensor partition require later implementation.

Crucial current limits: **one worker per service process**, one TCP connection, independent first-position rows each starting with zero history, sequential worker compute, one shared authentication key per session, in-memory commit counter, no durable ledger, no ciphertext confidentiality, no diffusion model, no full autoregressive temporal history, and no multi-worker makespan result. The bounded window permits **network request pipelining**, not parallel selected expert computation across workers. Do not describe it as a deployed federation or release.

## Preconditions (source and worker on separate physical GPU machines if available)

Both sides need exactly the same full student checkpoint (SHA-256 equality), compatible original Leech code, native Schaka 10.1 interpreter, and a securely provisioned identical random 32-byte-or-longer shared HMAC key in a file with permissions `0600`. The key MUST NOT be sent in plaintext over an insecure channel, copied into commits or receipts, or echoed into terminal logs.

Use a **fresh high-entropy session ID per run**. Both sides must agree before the run. HMAC authenticates packets but does not encrypt. The default service binds loopback. The supported inter-host method is an independently authenticated **SSH port-forward**; do not expose the plaintext service to the public Internet. `--allow-remote` relaxes only the bind guard and does not establish encryption.

No physical speedup estimate transfers from 16-row local PCIe-only run to LAN/WAN. In particular, a single RX 580 holding two model processes can OOM and must not be treated as independent workers.

## Source-level contract check (once per updated branch)

```bash
cd /path/to/LeechTransformer
git fetch origin
git switch experiment/consumer-preserving-multiscale-codec-20261031
git pull --ff-only
/nix/store/vq1lgpc49im7qvvxgxabykyfs47l2ykb-gfx803-schaka101-torch-candidate-overlay/bin/gfx803-schaka101-python \
  -m unittest discover -s tests -p 'test_native_federated_selected_worker.py' -v
```

These tests exercise the existing original `HistoryOrbitMoE`, JQ1, TCP framing and HMAC with CPU tensors as a **contract oracle**, not a new physical GPU qualification.

## One native two-host measurement, no piecemeal retest campaign

Define the following *real paths* from your existing GPU training and paired-quality experiment:

- `STUDENT`: full trained MoE checkpoint (not the `c2_layer1.pt` adapter)
- `TEACHER`: original dense checkpoint
- `TOKENS`: original real `.npy` token rows
- `CONFIG`: original teacher configuration (only if needed)
- `KEY`: the same securely exchanged 0600 key file on both hosts
- `SESSION`: shared fresh random hex ID, for example from `openssl rand -hex 16`

**On worker host** (new native ROCm GPU process, separate terminal):

```bash
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.native_federated_selected_worker -- \
  --mode serve --host 127.0.0.1 --port 48123 \
  --student "$STUDENT" --key-file "$KEY" --session "$SESSION" \
  --layer 1 --max-requests 256
```

**On coordinator host**, first establish an SSH tunnel to the worker host (separate terminal):

```bash
ssh -N -L 48123:127.0.0.1:48123 USER@WORKER_HOST
```

**On coordinator host**, run one bounded end-to-end batch, with a fresh output path (the same student and agreed session ID; model/hash binding is enforced):

```bash
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.native_federated_selected_worker -- \
  --mode coordinator --host 127.0.0.1 --port 48123 \
  --teacher "$TEACHER" --student "$STUDENT" \
  --config "$CONFIG" --token-ids "$TOKENS" \
  --key-file "$KEY" --session "$SESSION" --layer 1 \
  --max-rows 256 --window 8 --progress-every 16 \
  --output /mnt/truenas/comet-net/experiments/lila_native_federated_selected_20261113/receipt.json
```

Do not pass `--config` unless an actual file is available/required. This **single integrated run** covers authenticated transport, model parity, selected-state rejoin and request-window throughput. The worker stops at `--max-requests` or connection end; it cannot serve a second run without restart and a fresh session. Capture each process's terminal log separately, preserving nonzero exits.

## Acceptance and exact remaining frontier

| Gate | Current state | Required for closure |
|---|---|---|
| SBD01/TNET08 | Native GPU pipeline and TCP/window code written; local 16-row PCIe-only pipeline slower | A multi-host measured overlap/saturation sweep using **real** link latency and actual source compute |
| SBD02/TNET02 | Bounded outstanding-request window; worker only one MoE layer resident on GPU | Independent peer capacities, actual VRAM reservation/caches, failover/backpressure under load |
| SBD03/N07/N12 | HMAC monotonic request/reply sessions and exact independent selected-history reopen | Durable ACK/commit and real recurrent signed-history state across generations, atomic retry |
| SBD04/TNET04/TNET13 | Actual selected expert IDs and unweighted output on worker, source-owned weights; original JQ1 | Full temporal state/BIDI, original source model shard tensor identities, actual diffusion denoiser |
| SBD05 | Not begun | Chosen refinement contraction or counterexample; actual asynchronous update semantics |
| SBD06/PERF07–08 | Native GPU experiment planned | Matched AR/diffusion quality, p50/p95 latency, bytes/tokens/Joule, multiple peers, enough independent work |
| TNET05/TNET16 | Original layer module loaded on worker native GPU | Entire program graph/tensor manifest and source-to-Agda correspondence |
| Model quality | Matched dev dense 2.216053, C2 gated 2.564707, heldout strict C2 11.565138 | Student-only learning remains behind dense; no teacher-free promotion |

This tranche is **source-written and requires local contract tests and GPU host receipts**. It is not an assistant-performed GPU deployment. Its 256-row windowed measurement is much larger than the prior 16-row PCIe-only diagnostic, but even that does not establish a general WAN or diffusion speedup without subsequent independent physical peers.
