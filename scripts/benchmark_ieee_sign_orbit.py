"""Compare sign-orbit, byte shuffle and zlib on identical exact tensor bytes.

Example:
python scripts/benchmark_ieee_sign_orbit.py --n-blocks 256 --dtype-size 4 --seed 7
For real captures use --raw PATH (little-endian IEEE binary16/binary32).
All costs include codec headers and orientation witnesses. No checkpoint
or hardware inference performance is implied by synthetic inputs.
"""
import argparse
import json
import random
import zlib
from pathlib import Path
from scripts.ieee_sign_orbit_codec import encode_orbit, decode_orbit


def shuffle_bytes(raw: bytes, itemsize: int) -> bytes:
    if len(raw) % itemsize:
        raise ValueError("unaligned source")
    return b"".join(raw[offset::itemsize] for offset in range(itemsize))


def benchmark(raw: bytes, itemsize: int, width: int = 24):
    if not raw:
        raise ValueError("empty source")
    orbit = encode_orbit(raw, itemsize, width)
    if decode_orbit(orbit) != raw:
        raise AssertionError("bit-exact sign orbit failed")
    base = zlib.compress(raw, 6)
    shuffled = zlib.compress(shuffle_bytes(raw, itemsize), 6)
    return {"source_bytes":len(raw), "zlib_bytes":len(base),
            "shuffle_zlib_bytes":len(shuffled),
            "sign_orbit_total_bytes":len(orbit),
            "ratio_sign_to_shuffle":len(shuffled)/len(orbit),
            "sign_orbit_better_than_shuffle":len(orbit)<len(shuffled),
            "blocks":len(raw)//(itemsize*width), "itemsize":itemsize,
            "width":width, "bit_exact":True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--raw", type=Path)
    p.add_argument("--n-blocks", type=int, default=256)
    p.add_argument("--dtype-size", type=int, choices=(2,4), default=4)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--seed", type=int, default=7)
    a=p.parse_args()
    raw=(a.raw.read_bytes() if a.raw else
         random.Random(a.seed).randbytes(a.n_blocks*a.dtype_size*a.width))
    print(json.dumps(benchmark(raw,a.dtype_size,a.width),sort_keys=True,indent=2))


if __name__=="__main__":
    main()
