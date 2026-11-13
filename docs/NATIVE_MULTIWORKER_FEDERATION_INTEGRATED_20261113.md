# Multi-worker selected-expert federation — integrated native-gfx803 tranche (13 Nov 2026)

## Decision from measured local negative control

User's completed native RX 580 test: **16 real rows, eight microbatches, three repeats, exact output/history parity 0 error**. Same-card PCIe-staging streaming pipeline averaged **168.5 microbatches/s**, versus **188.1 microbatches/s sequential**: pipeline is **10.42% slower** in that workload. This rules out a claimed improvement from that local scheduling change. It does NOT rule out inter-host or block-parallel overlap.

No more piecemeal one-card queue sweeps on that path. The new integrated cut runs actual source-selected expert workloads on **two separate GPU worker owners**, with bounded in-flight rows, JQ1 exact codec, authenticated versioned requests, independent per-worker sessions, native GPU modules and source-side weighted reduction.

## Source and verification ownership

- Original Leech-LILA architecture: Anatolii Kornienko; do not attribute kernel or student architecture to the federation adapter.
- ROCm: reuse the existing working Schaka 10.1 Nix wrapper, not Docker or a requalification campaign.
- Agda finite streaming/revision module accepted at `ba6af7cd2` with user-reported **12/12 scheduler tests and Agda 2.9 kernel check**. General model correspondence remains open.
- Implemented on LeechTransformer PR #4 branch `experiment/consumer-preserving-multiscale-codec-20261031`: `HistoryOrbitMoE.execute_selected_subset`, `native_federated_selected_worker.py` and `native_multiworker_coordinator.py`. Source owner alone computes top-k routes, signed histories and weights, while workers compute **unweighted** output and selected updated history only. The source never discards inactive histories.
- Worker GPU residency: full checkpoint is read into CPU memory; worker GPU holds only orientation scorer and locally **owned** expert FFNs/transitions, not the whole LeechGPT model. This is *expert-level* placement, not model neuron/weight shard execution yet.
- Protocol: JQ1 exact selected activation/history packet; HMAC-SHA256 session envelopes with source model SHA, per-worker monotonically increasing sequence, history-revision label, reply ID/shape validation, source weighted reduction, original output/history parity, inactive-history preservation; bounded source row window. Source advances session state only after selected outputs/history and inactive-history checks. Worker-side commit is in memory, not durable.
- Peer telemetry: per-worker request/response bytes, completed requests, self-reported HIP/GPU identity, p50/p95/p99 row latency, completed rows/s, original source/model/tokens SHA and max parity errors.

**Security boundary:** HMAC authenticates but does not encrypt; create separate authenticated SSH tunnels. Different peers get unique high-entropy session IDs and per-peer private key files (0600) securely supplied on both ends. An arbitrary random session must not be reused with the same key after a worker restart. Worker-reported GPU metadata is signed by that worker but is *not independent hardware attestation*.

**Empirical boundary:** requires separate GPU hosts to claim parallel multi-worker compute. A two-process loopback test sharing one RX 580 is protocol-only, can OOM and is not evidence of parallel hardware. This branch contains no diffusion denoiser, recurrent signed temporal state across rows, asynchronous convergence proof, durable exactly-once retry, peer failure recovery or full distributed training.

## One integrated physical run, not a ladder of tiny smokes

Use the existing **full trained MoE student** from the matched-quality experiment and the exact dense teacher and token-array files. A single layer-adapter file such as `c2_layer1.pt` is **not** a complete student checkpoint. Use actual paths from your existing TrueNAS run.

**Contract tests (one command on source checkout):**

```bash
cd /path/to/LeechTransformer
git fetch origin
git switch experiment/consumer-preserving-multiscale-codec-20261031
git pull --ff-only
bash scripts/check_native_federation_contracts.sh
```

Tests use small CPU tensors as a protocol oracle through the existing Torch interpreter, but do not launch another ROCm diagnostic campaign.

**Worker A**, on a separate native GPU host with a copy of the full student and owner key:

```bash
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.native_federated_selected_worker -- \
  --mode serve --host 127.0.0.1 --port 48123 \
  --student "$STUDENT" --owned-experts 0,2 \
  --key-file "$WORKER_A_KEY" --session "$WORKER_A_SESSION" \
  --layer 1 --max-requests 256
```

**Worker B**, on another physical GPU host with exactly the same student SHA and own key:

```bash
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.native_federated_selected_worker -- \
  --mode serve --host 127.0.0.1 --port 48123 \
  --student "$STUDENT" --owned-experts 1,3 \
  --key-file "$WORKER_B_KEY" --session "$WORKER_B_SESSION" \
  --layer 1 --max-requests 256
```

Use SSH tunnels from source to the two hosts (separate long-lived terminal sessions). Both workers bind their *own* loopback port 48123; source connects to two **different local** ports:

```bash
ssh -N -L 48123:127.0.0.1:48123 USER_A@WORKER_A_HOST
ssh -N -L 48124:127.0.0.1:48123 USER_B@WORKER_B_HOST
```

Create a source-only manifest `/path/to/real_worker_manifest.json` with private **key paths only**, never actual key bytes. Replace every placeholder below with the existing key/session values; sessions should be generated independently from a secure random source:

```json
{
  "workers": [
    {
      "name": "miner-a-worker-0",
      "host": "127.0.0.1",
      "port": 48123,
      "session": "FRESH_RANDOM_A",
      "key_file": "/source/private/key-a",
      "experts": [0, 2]
    },
    {
      "name": "miner-b-worker-0",
      "host": "127.0.0.1",
      "port": 48124,
      "session": "FRESH_RANDOM_B",
      "key_file": "/source/private/key-b",
      "experts": [1, 3]
    }
  ]
}
```

The manifest is validated against the **actual** `student.cfg.moe_experts`: coverage of *every* expert is required, there can be no duplicate owners, and at least two physical worker endpoints must be described. The example partitions apply to the historical four-expert C2 student only; use a model-derived partition if the actual count differs.

**Coordinator**, on its own native Schaka GPU, with no extra ROCm qualification:

```bash
bash scripts/run_native_schaka101_lila.sh \
  --module scripts.native_multiworker_coordinator -- \
  --workers /path/to/real_worker_manifest.json \
  --teacher "$TEACHER" --student "$STUDENT" \
  --config "$CONFIG" --token-ids "$TOKENS" \
  --layer 1 --max-rows 256 --window 8 --progress-every 16 \
  --output /mnt/truenas/comet-net/experiments/lila_native_multiworker_20261113/receipt.json
```

Omit `--config` when the checkpoint already supplies it and there is no real config file. Use the original exact source/token windows and a fresh output directory. Preserve separate coordinator, A, B and SSH tunnel logs.

**Interpretation:** A passing receipt establishes limited selected-expert, native GPU, two-worker source parity and authenticated transport for **independent zero-history first-position rows**. It can report physical latency and throughput for that deployment, but does not establish teacher-free quality, persistent state replay across tokens, global network scheduling optimality, or diffusion superiority. The baseline to compare against is a matched local **same work** selected-expert run, not 188.1 microbatches/s with a different workload.

## Exact next max-cut after a measured multiworker receipt

1. **Recurrent history**: source model's actual full temporal per-token history; ordered revisions/ACKed replication, cancellation, retry after process failure, durable source/worker commit and source↔distributed BIDI.
2. **Original expert tensor manifests**: host/GPU/SSD/remote capacity admission, real expert/subexpert neuron shards, source identity and actual sparse storage (TNET02/05/13/16).
3. **Parallel throughput economics**: same model workload vs local resident baseline; vary workers, window, bandwidth, per-peer queue, microbatch sizes; collect true physical compute, transfer, load, queue and p95 latency. One 256-row measurement is the first *admission* check, not a universal speedup claim.
4. **Separate block diffusion branch**: actual trained denoiser/corruption/mask schedule, native training objective and comparable output/task quality; only then proof obligations SBD04/05/06.
5. **Model quality**: restore / improve strict student-only heldout CE (previous 11.565138) against dense development control 2.216053; never present the gated hybrid 2.515797 as teacher-free compression.
