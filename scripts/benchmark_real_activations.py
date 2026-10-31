#!/usr/bin/env python
"""Lossless raw, byte-shuffled DEFLATE, and XOR-residual DEFLATE benchmark."""
import argparse, csv, zlib
import numpy as np

def shuffled_bytes(a):
    b = np.ascontiguousarray(a).view(np.uint8).reshape(-1, a.dtype.itemsize)
    return b.T.reshape(-1).tobytes()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    source = np.load(a.input, allow_pickle=False)
    rows = []
    for name in source.files:
        if name == "metadata_json": continue
        values = source[name]
        if values.dtype.kind not in "fiu" or values.ndim < 1: continue
        values = values[:a.frames] if a.frames else values
        raw = values.tobytes()
        shuffle = shuffled_bytes(values)
        xor = b"" if len(values) < 2 else np.bitwise_xor(values[1:].view(np.uint8), values[:-1].view(np.uint8)).tobytes()
        candidates = {"raw": len(raw), "deflate_raw": len(zlib.compress(raw)),
                      "deflate_byte_shuffle": len(zlib.compress(shuffle)),
                      "deflate_xor_residual": len(raw[:values[0].nbytes]) + len(zlib.compress(xor))}
        for method, encoded in candidates.items():
            rows.append({"stream": name, "frames": len(values), "dtype": str(values.dtype),
                         "raw_bytes": len(raw), "method": method, "encoded_bytes": encoded,
                         "ratio": len(raw) / encoded if encoded else float("inf")})
    with open(a.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")

if __name__ == "__main__": main()
