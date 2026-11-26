"""Matched terminal-consumer storage/latency audit; native gfx803 only.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
from collections import Counter
from statistics import median
import time


def packet_account(packet):
    """Tensor logical bytes and deduplicated backing storage separately.
    Python object/allocator overhead is NOT represented by payload bytes.
    The wire metadata count is a declared hypothetical fixed descriptor,
    not a measured serialization format or Python object footprint.
    """
    logical=0;storage={};modes=Counter()
    for row in packet.rows:
        modes[row.mode]+=1
        for v in (row.values,row.signs):
            if v is None:continue
            logical+=v.numel()*v.element_size()
            s=v.untyped_storage()
            if s.nbytes():storage[(str(v.device),s.data_ptr())]=s.nbytes()
    # Four uint64 dimensions, dtype enum, per-row mode and 12-bit subgroup.
    metadata=33+3*len(packet.rows)
    return dict(logical_tensor_bytes=logical,unique_tensor_storage_bytes=sum(storage.values()),descriptor_bytes=metadata,logical_with_descriptor=logical+metadata,modes=dict(modes),python_overhead_measured=False)


def dense_terminal(model,seed,controls,tags):
    """Same terminal consumer as rollout_packed, no retained trace list."""
    from scripts.x369_source_geometry import signed_seed
    from models.geometric_reflection_fibre import GROUP
    h=signed_seed(seed,model.experts,model.width)
    for t in range(controls.shape[1]):
        tag=tags[:,t]
        out=model.history.step_controlled(controls[:,t],h,lambda c:model.control_embed(tag*len(GROUP)+c),train_sign=False)
        h=out.history
        if t+1==controls.shape[1]:return model.consume(out),h
        del out
    raise ValueError('empty word')


def resource_audit(model,rows,*,repeats=5,seconds=90):
    import torch
    from scripts.x369_indexed_residual_closure import native_source
    from scripts.train_x369_ordered_action import encode_inputs
    from scripts.x369_source_geometry import literal_controls
    from models.geometric_stabilizer_codec import rollout_packed,decode_history,same_bits
    native_source(model)
    if not rows or repeats<2 or not 0<seconds<=90:raise ValueError('invalid resource scope')
    p=next(model.parameters());dev=p.device;dt=p.dtype;deadline=time.monotonic()+seconds
    def measure(fn):
        torch.cuda.synchronize(dev);baseline=torch.cuda.memory_allocated(dev);torch.cuda.reset_peak_memory_stats(dev)
        start=time.perf_counter();value=fn();torch.cuda.synchronize(dev);elapsed=(time.perf_counter()-start)*1000
        return value,dict(milliseconds=elapsed,incremental_peak_bytes=max(0,torch.cuda.max_memory_allocated(dev)-baseline),baseline_bytes=baseline)
    results=[]
    with torch.no_grad():
        for r in rows:
            seed,ops=encode_inputs(r,'signed');s=torch.tensor([seed],device=dev,dtype=dt);c,t=literal_controls(torch.tensor([ops],device=dev,dtype=dt))
            a=lambda:dense_terminal(model,s,c,t);b=lambda:rollout_packed(model,s,c,t)
            if time.monotonic()>=deadline:break
            # Exact parity and warm-up outside measurement. No trace list.
            y,h=a();q,packet=b();raw=decode_history(packet)
            if not same_bits(h,raw) or not all(same_bits(v,w) for v,w in zip(y,q)):raise RuntimeError('terminal consumer mismatch')
            storage=packet_account(packet);storage['dense_tensor_bytes']=h.numel()*h.element_size()
            del y,h,q,packet,raw
            stats={'dense':[],'packed':[]}
            for k in range(repeats):
                if time.monotonic()>=deadline:break
                order=(('dense',a),('packed',b)) if k%2==0 else (('packed',b),('dense',a))
                for name,fn in order:
                    value,v=measure(fn);del value;stats[name].append(v)
            if len(stats['dense'])!=repeats:break
            results.append(dict(word=r['word'],guards=r['guards'],mean=r['mean'],residual=r['residual'],storage=storage,bitwise_terminal=True,arms={name:dict(samples=vals,median_ms=median(v['milliseconds'] for v in vals),median_incremental_peak_bytes=median(v['incremental_peak_bytes'] for v in vals)) for name,vals in stats.items()}))
    return dict(status='complete' if len(results)==len(rows) else 'time_budget',requested_rows=len(rows),completed_rows=len(results),rows=results,scope='single-row matched terminal logits and reversible final history; encoding/decoding inside timing; parity checked outside timing',training_steps=0,release_admitted=False,python_metadata_measured=False)


def certificate():
    import torch
    from types import SimpleNamespace as S
    x=torch.arange(16,dtype=torch.float32);y=x[:4];z=x[4:8];b=torch.tensor([3],dtype=torch.uint8)
    p=S(rows=(S(mode='real',values=y,signs=None),S(mode='raw',values=z,signs=None),S(mode='trit',values=b,signs=b)))
    r=packet_account(p)
    assert r['logical_tensor_bytes']==34 and r['unique_tensor_storage_bytes']==65 and r['descriptor_bytes']==42 and r['logical_with_descriptor']==76
    assert r['modes']=={'real':1,'raw':1,'trit':1} and not r['python_overhead_measured']
    print('PASS logical bytes, shared storage dedup, descriptor boundary, modes; no GPU timing')


if __name__=='__main__':certificate()
