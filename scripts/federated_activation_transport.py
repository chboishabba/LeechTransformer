"""Lossless, versioned federated activation transport with measured codec selection.

Compress BEFORE authenticated encryption. No encryption is implemented here.
Never mix secrets in a compression context: compression ratio can leak
information through sizes. Policy controls whether compression is allowed.

Only self-contained packets are emitted: no unacknowledged remote history
assumptions, and exact payload hashes are checked on reception.
"""
from __future__ import annotations
import hashlib
import json
import struct
import zlib
from scripts.ieee_sign_orbit_codec import encode_orbit, decode_orbit

MAGIC = b"FAI1"


def pack_activation(raw: bytes, *, dtype_size: int, block_width: int,
                    model_revision: str, history_revision: str,
                    permit_compression: bool = True) -> bytes:
    if dtype_size not in (2, 4) or not 1 <= block_width <= 65535:
        raise ValueError("unsupported scalar/block layout")
    if len(raw) % (dtype_size * block_width):
        raise ValueError("not aligned to block layout")
    if not model_revision or not history_revision:
        raise ValueError("explicit model and history revisions are required")
    options = [("raw", raw)]
    if permit_compression:
        options.extend([("zlib", zlib.compress(raw, 6)),
                        ("orbit", encode_orbit(raw, dtype_size, block_width))])
    meta = dict(model=model_revision, history=history_revision,
                itemsize=dtype_size, block_width=block_width,
                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                codec="", version=1)
    # Measure total framed bytes including metadata, not payload only.
    packets = []
    for name, body in options:
        meta["codec"] = name
        header = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode()
        packets.append((len(header)+len(body)+8, MAGIC + struct.pack("<I",len(header))+header+body))
    return min(packets, key=lambda p:p[0])[1]


def unpack_activation(packet: bytes, *, expected_model: str, expected_history: str,
                      max_uncompressed: int = 64*1024*1024) -> bytes:
    if len(packet) < 8 or packet[:4] != MAGIC:
        raise ValueError("invalid packet framing")
    hlen = struct.unpack("<I",packet[4:8])[0]
    if hlen > 8192 or len(packet) < 8+hlen:
        raise ValueError("invalid header length")
    meta=json.loads(packet[8:8+hlen])
    if meta.get("version") != 1 or meta.get("model") != expected_model or meta.get("history") != expected_history:
        raise ValueError("incompatible model/history reference")
    n=meta.get("bytes")
    if not isinstance(n,int) or n<0 or n>max_uncompressed:
        raise ValueError("invalid uncompressed length")
    raw_payload=packet[8+hlen:]
    codec=meta.get("codec")
    if codec=="raw":
        raw=raw_payload
    elif codec=="zlib":
        obj=zlib.decompressobj()
        raw=obj.decompress(raw_payload,n+1)
        if not obj.eof or obj.unused_data or obj.unconsumed_tail:
            raise ValueError("invalid compressed stream")
    elif codec=="orbit":
        if meta.get("itemsize") not in (2,4) or not isinstance(meta.get("block_width"),int) or not 1 <= meta["block_width"] <= 65535:
            raise ValueError("invalid orbit header")
        if n != len(raw_payload) and n > max_uncompressed:
            raise ValueError("invalid orbit source length")
        # Inner orbit decoder checks its own layout and bounded decompression.
        raw=decode_orbit(raw_payload)
    else:
        raise ValueError("unknown codec")
    if len(raw)!=n or hashlib.sha256(raw).hexdigest()!=meta.get("sha256"):
        raise ValueError("content digest or length mismatch")
    return raw


def estimate_transfer_time_s(packet_bytes:int, throughput_bytes_per_s:float,
                             round_trip_s:float=0.0, request_count:int=1)->float:
    if throughput_bytes_per_s <= 0 or request_count < 1 or round_trip_s < 0:
        raise ValueError("invalid link model")
    return packet_bytes/throughput_bytes_per_s + request_count*round_trip_s
