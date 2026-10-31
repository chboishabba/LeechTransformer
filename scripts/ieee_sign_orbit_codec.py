"""Exact IEEE sign-orbit candidate codec for 24D activation blocks.

The Z2 action toggles each element's IEEE sign bit (including NaNs and -0)
without numeric negation. This is NOT a Leech lattice automorphism.
All selected orbit orientations are charged as metadata in the packet.
"""
from __future__ import annotations
import zlib


def flip_ieee_signs(block: bytes, itemsize: int) -> bytes:
    if itemsize not in (2, 4) or len(block) % itemsize:
        raise ValueError("require aligned binary16/binary32 storage")
    data = bytearray(block)
    # Little-endian float storage on the reference capture.
    for pos in range(itemsize - 1, len(data), itemsize):
        data[pos] ^= 0x80
    return bytes(data)


def encode_orbit(frame: bytes, itemsize: int = 4, block_width: int = 24) -> bytes:
    if itemsize not in (2, 4) or not 1 <= block_width <= 65535 or len(frame) % (itemsize * block_width):
        raise ValueError("frame must have integral geometric blocks")
    chunk = block_width * itemsize
    blocks = [frame[i:i+chunk] for i in range(0,len(frame),chunk)]
    bits, canonical = [], []
    for b in blocks:
        flipped = flip_ieee_signs(b,itemsize)
        orient = int(flipped < b)
        bits.append(orient)
        canonical.append(flipped if orient else b)
    # Raw orientation witness packing. zlib compresses canonical bytes,
    # but header and witness costs are never hidden.
    mask = bytearray((len(bits)+7)//8)
    for i, orient in enumerate(bits):
        mask[i//8] |= orient << (i%8)
    nblocks = len(bits)
    if nblocks > 0xffffffff:
        raise ValueError("too many blocks")
    payload = zlib.compress(b"".join(canonical),level=6)
    return b"SO1" + bytes((itemsize,)) + block_width.to_bytes(2,"little") + nblocks.to_bytes(4,"little") + bytes(mask) + payload


def decode_orbit(packet: bytes) -> bytes:
    if len(packet) < 10 or packet[:3] != b"SO1":
        raise ValueError("invalid sign-orbit format")
    size = packet[3]
    width = int.from_bytes(packet[4:6],"little")
    count = int.from_bytes(packet[6:10],"little")
    if size not in (2,4) or width < 1:
        raise ValueError("invalid orbit layout")
    mask_bytes = (count+7)//8
    if len(packet) < 10+mask_bytes:
        raise ValueError("incomplete orientation witness")
    mask=packet[10:10+mask_bytes]
    decompressor = zlib.decompressobj()\n    # Prevent decompression bombs: bound the exact expected original size.\n    expected = count * size * width\n    raw = decompressor.decompress(packet[10+mask_bytes:], expected + 1)\n    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:\n        raise ValueError("invalid compressed payload framing")
    chunk=size*width
    if len(raw) != count*chunk:
        raise ValueError("decoded block count mismatch")
    out=bytearray()
    for i in range(count):
        block=raw[i*chunk:(i+1)*chunk]
        if mask[i//8] & (1<<(i%8)):
            block=flip_ieee_signs(block,size)
        out.extend(block)
    return bytes(out)
