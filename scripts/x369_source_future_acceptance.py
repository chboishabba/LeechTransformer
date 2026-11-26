"""Source-bound future closure, no training or checkpoint migration.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
import time
from collections import deque
from fractions import Fraction as F
from scripts.check_x369_nonlinear_closure import refine,witness


def future_acceptance(seeds,controls,step,chart,*,budget,seconds=90):
    """Explore actual states; use existing refinement only on a closed graph.

    step returns (named hashable consumer observations, successor).
    Budget/time exhaustion never certifies closure. Initial chart is a
    declared observation, not an independently verified semantic target.
    """
    seeds=tuple(dict.fromkeys(seeds));controls=tuple(controls)
    if not seeds or not controls or len(set(controls))!=len(controls) or budget<1 or not 0<seconds<=90:
        raise ValueError('invalid future audit scope')
    end=time.monotonic()+seconds;parents={};q=deque();table={}
    def result(status,**kw):
        return dict(status=status,states=len(parents),edges=len(table),universal_closure_certified=False,**kw)
    for h in seeds:
        if len(parents)>=budget:return result('state_budget')
        parents[h]=(None,None);q.append(h)
    while q:
        h=q.popleft()
        for u in controls:
            if time.monotonic()>=end:return result('time_budget')
            obs,n=step(h,u)
            if n not in parents:
                if len(parents)>=budget:return result('state_budget')
                parents[n]=(h,u);q.append(n)
            table[h,u]=(obs,n)
    # Refinement itself is bounded by the finite carrier size, not the
    # discovery wall-clock deadline. No interruption inside a source step.
    codes={h:chart(h) for h in parents}
    ps,_=refine(tuple(parents),controls,codes.__getitem__,lambda h,u:table[h,u])
    final=ps[-1];groups={};pair=None
    for h in parents:
        a=groups.setdefault(codes[h],h)
        if final[a]!=final[h] and pair is None:pair=(a,h)
    def path(h):
        word=[]
        while parents[h][0] is not None:
            h,u=parents[h];word.append(u)
        return dict(seed=seeds.index(h),word=list(reversed(word)))
    separator=None
    if pair:
        a,b=pair;word=witness(a,b,controls,codes.__getitem__,table)
        assert word is not None
        left,right=a,b;fields=[]
        for u in word:
            A,left=table[left,u];B,right=table[right,u]
            if A!=B:fields.append('consumer')
            if codes[left]!=codes[right]:fields.append('successor_chart')
        separator=dict(left=path(a),right=path(b),word=list(word),fields=sorted(set(fields)))
    assert len(table)==len(parents)*len(controls)
    return result('closed',partition_counts=[len(set(p.values())) for p in ps],candidate_descends=pair is None,
                  shortest_pair_separator=separator,closed_seed_control_future_certified=True,
                  scope='declared deterministic state, seed, control and observation contract')


def native_future(model,controls,*,budget,seconds=90,consumer='raw'):
    """Actual new-architecture source; no reimplemented router or readout.
    Fixed single-row native execution. Candidate is rational-value z/profile;
    original state keys and consumer observations remain byte-exact.
    """
    import torch
    from itertools import product
    from scripts.x369_indexed_residual_closure import native_source
    from scripts.x369_capped_semantic_geometry import exact_repair,cap_profile
    from scripts.audit_x369_joint_action import MEANS
    from scripts.audit_x369_ordered_signed_task import TRIADS
    from models.geometric_reflection_fibre import GROUP
    if consumer not in ('raw','typed'):raise ValueError('unknown consumer')
    if bool(model.typed_consumer)!=(consumer=='typed'):raise ValueError('consumer/source mismatch')
    seed,_,_=native_source(model);p=next(model.parameters());dev=p.device;dt=p.dtype
    dims=(1,model.experts,2,model.width)
    def reopen(h):return torch.frombuffer(bytearray(h),dtype=dt).reshape(dims).to(dev)
    def key(t):
        if t.is_floating_point() and not torch.isfinite(t).all():raise ValueError('nonfinite source observation')
        a=t.detach().cpu().contiguous();return str(a.dtype),tuple(a.shape),a.reshape(-1).view(torch.uint8).numpy().tobytes()
    def once(h,u):
        g,tag=u
        if not 0<=g<len(GROUP) or not 0<=tag<4:raise ValueError('invalid control')
        with torch.no_grad():
            out=model.history.step_controlled(torch.tensor([g],device=dev),reopen(h),lambda c:model.control_embed(tag*len(GROUP)+c),train_sign=False)
            obs=tuple((f'logits_{i}',key(v)) for i,v in enumerate(model.consume(out)))
            if consumer=='raw':obs+=tuple((n,key(getattr(out,n))) for n in ('output','ids','weights','local_signs'))
            return obs,key(out.history)[2]
    def step(h,u):
        a=once(h,u);b=once(h,u)
        if a!=b:raise RuntimeError('nonrepeatable native source edge or consumer')
        return a
    def chart(h):
        v=tuple(tuple(tuple(F(float(x)) for x in b) for b in e) for e in reopen(h)[0].cpu().tolist())
        return exact_repair(v,(0,)*6,range(model.experts),1)['defect'],cap_profile(v)
    out=future_acceptance((seed(m,d) for m,d in product(MEANS,TRIADS)),controls,step,chart,budget=budget,seconds=seconds)
    out.update(consumer=consumer,byte_state=True,byte_observation=True,chart='exact rational semantic and cap profile',training_steps=0)
    return out


def certificate():
    # Actual finite refinement owner; exact delayed separator and closed positive.
    states=(0,1,2,3);chart=lambda x:0
    def step(h,u):return (h==3,(h+2)%4)
    a=future_acceptance(states,('tick',),step,chart,budget=4)
    assert a['partition_counts']==[1,2,3] and not a['candidate_descends']
    assert a['shortest_pair_separator']['word']==['tick','tick']
    b=future_acceptance((0,),('tick',),lambda h,u:(0,1-h),chart,budget=2)
    assert b['status']=='closed' and b['candidate_descends'] and b['partition_counts']==[1]
    c=future_acceptance((0,),('tick',),lambda h,u:(0,h+1),chart,budget=3)
    assert c['status']=='state_budget' and 'candidate_descends' not in c and not c['universal_closure_certified']
    return dict(delayed_separator=a,positive=b,budget=c)


if __name__=='__main__':print(certificate())
