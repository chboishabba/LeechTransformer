"""Lossless geometric section of exact semantic-six + cap-profile fibres.

Original Leech-LILA: Anatolii Kornienko. DASHI exact coordinate extension.
Reuse exact_repair, cap_profile, and typed rational serialization.
The residual stores free linear coordinates, one sphere sign per branch,
and original negative-zero positions. No bit-rate or minimality claim.
"""
import argparse
from fractions import Fraction as F
from math import isfinite, isqrt
import json
import struct
from pathlib import Path

from scripts.x369_capped_semantic_geometry import exact_repair, cap_profile
from scripts.x369_future_realization import pack, unpack
from scripts.x369_original_receipt_multichart_weld import (
    RECEIPT_SHA256, digest, require, extract_edges)


def layout(config):
    e,w,dt=config['experts'],config['width'],config['dtype']
    require(type(e) is int and e>=2 and type(w) is int and w>=10
            and dt in ('float32','float64'), 'unsupported paired source layout')
    return e,w,('f' if dt=='float32' else 'd')


def decode_values(h, config):
    e,w,s=layout(config)
    fmt='<'+str(e*2*w)+s
    require(type(h) is bytes and len(h)==struct.calcsize(fmt),
            'original history storage size')
    values=struct.unpack(fmt,h)
    require(all(isfinite(x) for x in values),'nonfinite source history')
    v=tuple(tuple(tuple(F(x) for x in values[(2*i+j)*w:(2*i+j+1)*w])
                  for j in range(2)) for i in range(e))
    size=struct.calcsize(s)
    negative_zero=struct.pack('<'+s,-0.0)
    zeros=tuple(i for i in range(len(values))
                if h[i*size:(i+1)*size]==negative_zero)
    return v,zeros


def free_slots(i,w):
    # Expert zero uses the original six-row branch decoder. Pivot columns
    # j+3, j+6 are solved from v, and the last (unused) coordinate by norm.
    return tuple(j for j in range(w-1) if i!=0 or not 3<=j<9)


def encode(h,config):
    e,w,_=layout(config)
    v,zeros=decode_values(h,config)
    repair=exact_repair(v,(0,)*6,range(e),1)
    z=repair['defect'];r=repair['candidate'];profile=cap_profile(v)
    free=tuple(tuple(tuple(r[i][b][j] for j in free_slots(i,w))
                     for b in range(2)) for i in range(e))
    signs=tuple(tuple(int(r[i][b][-1]<0) for b in range(2)) for i in range(e))
    return dict(schema=1,config={k:config[k] for k in ('experts','width','dtype')},
                chart=pack((z,profile)),free=pack(free),sphere_signs=pack(signs),
                negative_zero_positions=pack(zeros),original_sha256=digest(h))


def rational(x):
    require(type(x) in (int,F),'exact rational coordinate required')
    return F(x)


def rational_sqrt(q):
    q=rational(q)
    require(q>=0,'negative sphere radius')
    n,d=isqrt(q.numerator),isqrt(q.denominator)
    require(n*n==q.numerator and d*d==q.denominator,
            'sphere radius not rational square in declared source')
    return F(n,d)


def reopen(packet):
    require(packet['schema']==1,'unknown section schema')
    config=packet['config'];e,w,s=layout(config)
    z,profile=unpack(packet['chart']);v,norms=profile
    free=unpack(packet['free']);signs=unpack(packet['sphere_signs'])
    zeros=unpack(packet['negative_zero_positions'])
    require(len(z)==len(v)==6 and len(norms)==len(free)==len(signs)==e,
            'invalid exact chart shape')
    require(tuple(sorted(set(zeros)))==zeros and
            all(type(i) is int and 0<=i<e*2*w for i in zeros),
            'invalid signed-zero positions')
    z=tuple(map(rational,z));v=tuple(map(rational,v))
    residual=[]
    for i in range(e):
        require(len(norms[i])==len(free[i])==len(signs[i])==2,
                'invalid paired residual shape')
        branches=[]
        for b in range(2):
            slots=free_slots(i,w)
            require(len(free[i][b])==len(slots) and
                    type(signs[i][b]) is int and signs[i][b] in (0,1),
                    'invalid sphere residual')
            q=[F(0)]*w
            for j,x in zip(slots,free[i][b]):
                q[j]=rational(x)
            if i==0:
                vb=tuple((-1)**b*x for x in v)
                for j in range(3):
                    q[j+3]=2*vb[j]-q[j]
                    q[j+6]=2*(-1)**b*vb[j+3]-q[j]
            q[-1]=(-1)**signs[i][b]*rational_sqrt(
                rational(norms[i][b])-sum(x*x for x in q[:-1]))
            require(q[-1]!=0 or signs[i][b]==0,'noncanonical zero sphere sign')
            branches.append(tuple(q))
        residual.append(tuple(branches))
    residual=tuple(residual)
    zero=exact_repair(residual,(0,)*6,range(e),1)
    require(zero['defect']==(0,)*6,'retained residual not in semantic kernel')
    # Existing orthogonal section r + Q^T z.
    history=exact_repair(residual,z,range(e),1)['candidate']
    require(exact_repair(history,(0,)*6,range(e),1)['defect']==z and
            cap_profile(history)==profile,'reopened history chart mismatch')
    flat=[x for branches in history for branch in branches for x in branch]
    zero_set=set(zeros)
    chunks=[]
    for i,x in enumerate(flat):
        require(i not in zero_set or x==0,'signed-zero bit on nonzero scalar')
        try:
            binary=struct.pack('<'+s,-0.0 if i in zero_set else float(x))
        except (OverflowError,struct.error) as exc:
            raise ValueError('reopened scalar outside source dtype') from exc
        y=struct.unpack('<'+s,binary)[0]
        require(isfinite(y) and F(y)==x,'reopened rational not source dtype exact')
        chunks.append(binary)
    h=b''.join(chunks)
    require(digest(h)==packet['original_sha256'],'original history SHA mismatch')
    require(encode(h,config)==packet,'noncanonical geometric section packet')
    return h


def certificate(raw):
    require(digest(raw)==RECEIPT_SHA256,'original pinned receipt SHA mismatch')
    receipt=json.loads(raw)
    _,states,_=extract_edges(receipt)
    config=receipt['config'];e,w,_=layout(config)
    rows=[]
    for key,h in sorted(states.items()):
        p=encode(h,config)
        require(reopen(json.loads(json.dumps(p)))==h,'receipt history BIDI failed')
        rows.append(dict(history_sha256=key,packet=p))
    return dict(status='exact_semantic_cap_lossless_section',
        receipt_file_sha256=digest(raw),histories=len(rows),
        original_float_scalars=e*2*w,
        free_rational_residual_coordinates=2*(w-7)+2*(e-1)*(w-1),
        sphere_sign_bits=2*e,original_negative_zero_positions_retained=True,
        exact_source_byte_roundtrip=True,packets=rows,
        original_neural_source_reexecuted=False,
        bit_rate_reduction_certified=False,consumer_minimality_certified=False,
        source_graph_closure_certified=False,agda_kernel_checked=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    result=json.dumps(certificate(a.receipt.read_bytes()),sort_keys=True,indent=2)+'\n'
    if a.output:a.output.write_text(result)
    print(result,end='')


if __name__=='__main__':
    main()
