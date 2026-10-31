"""Exact multiscale activation fibre codec comparison.

Independent LeechTransformer research adapter. Signed-off source inspirations:
dashitest triadic video (temporal charts), DASHI history-indexed consumer
fibres, finite Fourier shell decomposition (analogy only), and dashiRTX
transport decomposition. No prime-correlation or PDE theorem is imported.

Each candidate reconstructs ORIGINAL float storage bytes bit-for-bit.
No lossy Fourier/DCT or unproven symmetry-group quotient is used.
"""
import argparse
import csv
import time
import zlib
from collections import Counter
import numpy as np

def split_bytes(x, size):
    return np.frombuffer(x, dtype=np.uint8).reshape(-1, size).copy()

def merge_planes(a):
    return a.T.copy().tobytes()

def inverse_planes(data, size):
    return np.frombuffer(data, dtype=np.uint8).reshape(size, -1).T.copy().tobytes()

def lift24(data, size):
    a = split_bytes(data, size)
    if len(a) % 24:
        raise ValueError("length must be divisible by 24")
    b = a.reshape(-1, 24, size).copy()
    # Integer lifting: low + high -> (low, high-low); exact modulo 256.
    b[:, 1::2] = (b[:, 1::2].astype(np.int16) - b[:, 0::2].astype(np.int16)).astype(np.uint8)
    return b.tobytes()

def unlift24(data, size):
    a = split_bytes(data, size)
    if len(a) % 24:
        raise ValueError("length must be divisible by 24")
    b = a.reshape(-1, 24, size).copy()
    b[:, 1::2] = (b[:, 1::2].astype(np.int16) + b[:, 0::2].astype(np.int16)).astype(np.uint8)
    return b.tobytes()

def entropy(data):
    if not data: return 0.
    counts = np.bincount(np.frombuffer(data, np.uint8), minlength=256)
    p = counts[counts>0]/len(data)
    return float(-(p*np.log2(p)).sum())

def candidates(frame, size, prior):
    yield "raw", frame, lambda z: z
    planes = split_bytes(frame, size)
    yield "byte_planes", merge_planes(planes), lambda z: inverse_planes(z, size)
    if len(planes)%24 == 0:
        t = lift24(frame,size)
        yield "lift24", t, lambda z: unlift24(z,size)
        yield "lift24_planes", merge_planes(split_bytes(t,size)), lambda z: unlift24(inverse_planes(z,size),size)
    if prior is not None and len(prior)==len(frame):
        a=np.frombuffer(frame,np.uint8); b=np.frombuffer(prior,np.uint8)
        xor=np.bitwise_xor(a,b).tobytes()
        delta=(a.astype(np.int16)-b.astype(np.int16)).astype(np.uint8).tobytes()
        yield "xor",xor,lambda z: np.bitwise_xor(np.frombuffer(z,np.uint8),b).tobytes()
        yield "xor_planes",merge_planes(split_bytes(xor,size)),lambda z: np.bitwise_xor(np.frombuffer(inverse_planes(z,size),np.uint8),b).tobytes()
        yield "delta",delta,lambda z: (np.frombuffer(z,np.uint8).astype(np.int16)+b.astype(np.int16)).astype(np.uint8).tobytes()
        yield "delta_planes",merge_planes(split_bytes(delta,size)),lambda z: (np.frombuffer(inverse_planes(z,size),np.uint8).astype(np.int16)+b.astype(np.int16)).astype(np.uint8).tobytes()

def compete(frame,size,prior):
    entries=[]
    for mode, transformed, inverse in candidates(frame,size,prior):
        t0=time.perf_counter_ns()
        if mode=="raw": wire=frame
        else: wire=zlib.compress(transformed,6)
        enc_ns=time.perf_counter_ns()-t0
        t1=time.perf_counter_ns()
        decoded=inverse(wire if mode=="raw" else zlib.decompress(wire))
        dec_ns=time.perf_counter_ns()-t1
        if decoded != frame: raise AssertionError(f"not lossless: {mode}")
        entries.append((mode,len(wire),entropy(transformed),enc_ns,dec_ns))
    # Include one-byte chart ID; every candidate pays the same optimistic
    # one-byte signalling lower bound, not a production wire protocol.
    winner=min(entries,key=lambda item:item[1])
    return entries,winner

def run(input_path, output_path, limit):
    rows=[]
    with np.load(input_path,allow_pickle=False) as data:
        for stream in data.files:
            if stream=="metadata_json": continue
            samples=data[stream]
            if samples.dtype.kind!="f" or samples.dtype.itemsize not in (2,4): continue
            prior=None
            for frame_idx,sample in enumerate(samples[:limit] if limit else samples):
                frame=np.ascontiguousarray(sample).tobytes()
                entries,winner=compete(frame,samples.dtype.itemsize,prior)
                for mode,n,e,enc_ns,dec_ns in entries:
                    rows.append(dict(stream=stream,frame=frame_idx,mode=mode,
                       raw_bytes=len(frame),coded_bytes=n,optimistic_signalled_bytes=n+1,
                       ratio=len(frame)/(n+1),byte_entropy=e,
                       encode_ns=enc_ns,decode_ns=dec_ns,selected=(mode==winner[0])))
                prior=frame  # exact prior output must have been ACKed by decoder
    if not rows: raise RuntimeError("no floating-point frames in capture")
    with open(output_path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    picked=[r for r in rows if r["selected"]]
    raw=sum(r["raw_bytes"] for r in picked)
    coded=sum(r["optimistic_signalled_bytes"] for r in picked)
    print(f"real frames={len(picked)} raw={raw} adaptive={coded} ratio={raw/coded:.5f}x")
    print("selected",dict(Counter(r["mode"] for r in picked)))
    print("Per-method aggregate (including 1-byte optimistic chart id):")
    for mode in sorted(set(r["mode"] for r in rows)):
        sub=[r for r in rows if r["mode"]==mode]
        print(mode,len(sub),sum(r["raw_bytes"] for r in sub)/sum(r["optimistic_signalled_bytes"] for r in sub))
    print("NOTE: mode selection runtime and negotiated schema, ACK, reset and transport metadata not charged.")

def self_test():
    rng=np.random.default_rng(17131)
    for size in (2,4):
        for width in (24,48,384,3584):
            prior=None
            for frame in (rng.bytes(width*size),bytes(width*size),rng.bytes(width*size)):
                compete(frame,size,prior)
                prior=frame
    print("PASS reversible multiscale/causal-chart codec candidates")

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("capture",nargs="?")
    a.add_argument("--out",default="fibre_multiscale_results.csv")
    a.add_argument("--frames",type=int,default=0)
    args=a.parse_args()
    self_test()
    if args.capture: run(args.capture,args.out,args.frames)
