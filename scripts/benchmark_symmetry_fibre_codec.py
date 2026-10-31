"""Lossless activation-frame codec tournament: causal, symmetry-safe charts.

Source model: SPUTNIKAI/LeechTransformer (Anatolii Kornienko).
DASHI/dashitest inspired chart selection; not a claim of Leech automorphism.
"""
import argparse
import csv
import time
import zlib
from pathlib import Path

import numpy as np

# Frame-local modes, all byte-exact and independently decodable.
def transform(data: bytes, itemsize: int, mode: str, prior: bytes | None = None) -> bytes:
    if mode == "raw":
        return data
    if mode in ("shuffle", "block24", "haar24"):
        if len(data) % itemsize:
            raise ValueError("non-integral elements")
        a = np.frombuffer(data, np.uint8).reshape(-1, itemsize)
        if mode == "shuffle":
            return a.T.copy().tobytes()
        if mode == "haar24":
            if len(a) % 24:
                raise ValueError("non-integral 24-channel block")
            # Byte-plane integer lifting, 12 pairs per 24-channel block.
            # a,b -> a,(b-a) modulo 256; inverse reconstructs b.
            blocks = a.reshape(-1, 24, itemsize).copy()
            blocks[:, 1::2, :] = (blocks[:, 1::2, :].astype(np.int16) - blocks[:, 0::2, :].astype(np.int16)).astype(np.uint8)
            return blocks.tobytes()
        # Reversible 24-channel block coordinate *permutation*, not an
        # established automorphism of the physical Leech lattice.
        if len(a) % 24:
            raise ValueError("non-integral 24-channel block")
        return a.reshape(-1, 24, itemsize)[:, ::-1, :].copy().tobytes()
    if mode == "xor":
        if prior is None or len(prior) != len(data):
            raise ValueError("missing causal reference")
        return bytes(a ^ b for a, b in zip(data, prior))
    if mode == "delta":
        if prior is None or len(prior) != len(data):
            raise ValueError("missing causal reference")
        return bytes((a - b) & 255 for a, b in zip(data, prior))
    raise ValueError(mode)


def inverse(data: bytes, itemsize: int, mode: str, prior: bytes | None = None) -> bytes:
    if mode in ("raw", "shuffle", "block24", "haar24"):
        if mode == "raw":
            return data
        a = np.frombuffer(data, np.uint8)
        if mode == "shuffle":
            return a.reshape(itemsize, -1).T.copy().tobytes()
        if mode == "haar24":
            if len(a) % (24 * itemsize):
                raise ValueError("invalid 24-channel block")
            blocks = a.reshape(-1, 24, itemsize).copy()
            blocks[:, 1::2, :] = (blocks[:, 1::2, :].astype(np.int16) + blocks[:, 0::2, :].astype(np.int16)).astype(np.uint8)
            return blocks.tobytes()
        if len(a) % (24 * itemsize):
            raise ValueError("invalid 24-channel block")
        return a.reshape(-1, 24, itemsize)[:, ::-1, :].copy().tobytes()
    if prior is None or len(prior) != len(data):
        raise ValueError("missing causal reference")
    if mode == "xor":
        return bytes(a ^ b for a, b in zip(data, prior))
    if mode == "delta":
        return bytes((a + b) & 255 for a, b in zip(data, prior))
    raise ValueError(mode)


def choose(frame: bytes, itemsize: int, prior: bytes | None):
    modes = ["raw", "shuffle"]
    if len(frame) % (24 * itemsize) == 0:
        modes.extend(["block24", "haar24"])
    if prior is not None:
        modes.extend(["xor", "delta"])
    best = (len(frame), "raw", frame)
    for mode in modes:
        t = transform(frame, itemsize, mode, prior)
        encoded = zlib.compress(t, level=6)
        if len(encoded) < best[0]:
            best = len(encoded), mode, encoded
    length, mode, payload = best
    reconstructed = inverse(zlib.decompress(payload) if mode != "raw" else payload,
                            itemsize, mode, prior)
    assert reconstructed == frame, f"failed lossless reconstruction: {mode}"
    return length, mode


def benchmark(path, out, limit=None):
    with np.load(path, allow_pickle=False) as archive:
        rows = []
        for stream in archive.files:
            if stream == "metadata_json":
                continue
            a = archive[stream]
            if a.dtype.kind != "f" or a.dtype.itemsize not in (2, 4):
                continue
            prior = None
            for idx, sample in enumerate(a[:limit] if limit else a):
                b = np.ascontiguousarray(sample).tobytes()
                start = time.perf_counter_ns()
                encoded, mode = choose(b, a.dtype.itemsize, prior)
                elapsed = time.perf_counter_ns() - start
                rows.append(dict(stream=stream, frame=idx, bytes=len(b),
                                 encoded_bytes=encoded, mode=mode,
                                 ratio=len(b)/encoded, select_ns=elapsed))
                prior = b
    if not rows:
        raise RuntimeError("no floating-point frames found")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    total = sum(int(r["bytes"]) for r in rows)
    packed = sum(int(r["encoded_bytes"]) for r in rows)
    print(f"{len(rows)} frames, raw={total}, packed={packed}, ratio={total/packed:.4f}x")
    print("NOTE: payload-only results: add real packet framing, chart signalling, ACKs and transport latency.")


def self_test():
    rng = np.random.default_rng(20261031)
    for itemsize in (2, 4):
        for width in (24, 48, 384):
            old = None
            for kind in (0, 1, 2):
                frame = (rng.integers(0, 256, width * itemsize, dtype=np.uint8)
                         if kind == 0 else np.zeros(width * itemsize, np.uint8)
                         if kind == 1 else np.arange(width * itemsize, dtype=np.uint8)).tobytes()
                for mode in ["raw", "shuffle", "block24", "haar24"] + (["xor", "delta"] if old else []):
                    assert inverse(transform(frame, itemsize, mode, old), itemsize, mode, old) == frame
                choose(frame, itemsize, old)
                old = frame
    print("PASS: exact inverse charts and selector, seeded tests")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", nargs="?")
    ap.add_argument("--out", default="activation_chart_comparison.csv")
    ap.add_argument("--frames", type=int, default=None)
    args = ap.parse_args()
    self_test()
    if args.capture:
        benchmark(args.capture, args.out, args.frames)
