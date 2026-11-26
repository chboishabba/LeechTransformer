"""Controlled-source factorial interventions and checkpoint-indexed charts.
Original Leech-LILA: Anatolii Kornienko; DASHI extension. No training.
"""
from itertools import product
import time


def matched_null(old,edited,V,direction):
    d=direction-(direction@V)@V.T;n=d.norm();m=(edited-old).norm()
    if not bool(n>0) or not bool(m>0):return None
    return old+d*(m/n)


def phase_chart(rows,f):
    import torch
    from scripts.audit_x369_joint_action import state
    tr=torch.tensor([r['split']=='train' for r in rows]);ev=~tr
    Y=torch.tensor([sum(state(r),()) for r in rows],dtype=torch.float64)
    V,q,A=(f[k] for k in ('V','q','A'))
    return dict(U=(torch.linalg.pinv(Y[tr])@q[tr]).tolist(),V=V.tolist(),A={g:a.tolist() for g,a in A.items()},decoder=(torch.linalg.pinv(q[tr])@Y[tr]).tolist(),composition_q=q[ev].tolist(),composition_keys=[[r['word'],list(r['guards']),list(r['mean']),list(r['residual'])] for r in rows if r['split']=='composition'])


def source_factorial(model,rows,*,contexts=1,pairs=2,horizon=1,seconds=90):
    import torch
    from hashlib import sha256
    from scripts.audit_x369_joint_action import select_contexts,fit_joint,task_digest,state
    from scripts.x369_common_hom import capture_natural_rows
    from scripts.x369_indexed_residual_closure import native_source
    from scripts.x369_source_future_acceptance import native_future_contract
    from scripts.x369_source_acceptance import guard_controls
    from scripts.audit_x369_consumer_closure import factorial_states,factorial_effect
    from scripts.x369_simultaneous_weld import literal_matrix,apply_matrix
    from scripts.audit_x369_ordered_signed_task import gram
    from scripts.train_x369_ordered_action import LABELS
    if min(contexts,pairs)<1 or not 1<=horizon<=3 or not 0<seconds<=90:raise ValueError('invalid factorial scope')
    end=time.monotonic()+seconds;rows=select_contexts(rows,contexts);seed,step,_=native_source(model)
    def controls(r,word):return tuple(dict(zip('abjn',guard_controls(r['guards'])))[c] for c in word)
    hs,edges=capture_natural_rows(rows,seed,step,lambda r:controls(r,r['word']),seconds)
    report=dict(status='capture_time_budget',rows=len(rows),task_sha256=task_digest(rows),unique_capture_edges=edges,training_steps=0,external_consumer_certified=False,release_admitted=False)
    if hs is None:return report
    dt=next(model.parameters()).dtype;Z=torch.stack([torch.frombuffer(bytearray(h),dtype=dt) for h in hs]).double();f=fit_joint(rows,Z)
    report.update(history_sha256=sha256(b''.join(hs)).hexdigest(),chart_admitted=f['admitted'],fit_rank=f['rank'])
    if not f['admitted']:report['status']='chart_not_admitted';return report
    V,q,A,ix=(f[k] for k in ('V','q','A','targets'));report['phase']=dict(chart=phase_chart(rows,f));report['status']='intervention_time_budget';report['comparisons']=[]
    if time.monotonic()>=end:return report
    _,observe,_=native_future_contract(model,'raw')
    def key(h):return h.to(dt).contiguous().view(torch.uint8).numpy().tobytes()
    def logits(obs):
        obs=dict(obs);return tuple(torch.frombuffer(bytearray(obs[f'logits_{j}'][2]),dtype=dt).double() for j in range(3))
    def labels(y):return y[:3],gram(y[3:]),y[3:]
    def evaluate(vals,y):
        out=[]
        for v,L,t in zip(vals,LABELS,labels(y)):
            i=L.index(t);top=(v==v.max()).nonzero().flatten().tolist();out.append(dict(unique_correct=top==[i],target_in_max=i in top,margin=float(v[i]-torch.cat((v[:i],v[i+1:])).max())))
        return out
    for g,t in ix.items():
        elig=[i for i,r in enumerate(rows) if r['split']=='composition' and i!=int(t[i])]
        chosen=elig if len(elig)<=pairs else [elig[k*len(elig)//pairs] for k in range(pairs)]
        for i in chosen:
            j=int(t[i]);arms=[a[0] for a in factorial_states(Z[i:i+1],Z[j:j+1],V,q[i:i+1],q[j:j+1],A[g])];names=['fitted_old','oracle_chart_old','fitted_natural_residual','natural_full','unedited'];arms.append(Z[i])
            null=matched_null(arms[0],arms[1],V,Z[j]-Z[i])
            if null is not None:names.append('chart_null_matched');arms.append(null)
            keys=[key(a) for a in arms];keys[3]=hs[j];keys[4]=hs[i] # original natural bytes, including signed zero
            cast=[torch.frombuffer(bytearray(h),dtype=dt).double() for h in keys];edit=dict(chart_displacement=float((cast[1]-cast[0]).norm()),null_available=null is not None)
            if null is not None:edit.update(null_displacement=float((cast[-1]-cast[0]).norm()),null_chart_leakage=float(((cast[-1]-cast[0])@V).norm()))
            cap=model.history.source.history_cap;edit['over_cap_input_branches']=[int((v.reshape(model.experts,2,model.width).norm(dim=-1)>cap).sum()) if cap is not None else 0 for v in cast]
            for depth in range(1,horizon+1):
                for word in product('abjn',repeat=depth):
                    if time.monotonic()>=end:return report
                    us=controls(rows[i],word);y=sum(state(rows[j]),());old=sum(state(rows[i]),());out=[];final=[]
                    for u in us:
                        M=literal_matrix(u);y=apply_matrix(M,y);old=apply_matrix(M,old)
                    for h in keys:
                        for u in us:obs,h=observe(h,u)
                        out.append(logits(obs));final.append(h)
                    natural=out[3];effects=[]
                    for k in range(3):
                        c,r,m=factorial_effect([v[k] for v in out[:4]]);effects.append(dict(chart_l2=float(c.norm()),residual_l2=float(r.norm()),mixed_l2=float(m.norm())))
                    report['comparisons'].append(dict(generator=g,source_row=i,target_row=j,guards=rows[i]['guards'],prefix=rows[i]['word'],continuation=''.join(word),edit=edit,unchanged_heads=[k for k,(a,b) in enumerate(zip(labels(y),labels(old))) if a==b],effects=effects,arms={name:dict(target=evaluate(v,y),original=evaluate(v,old),logit_l2_to_natural=[float((a-b).norm()) for a,b in zip(v,natural)],history_sha256=sha256(h).hexdigest()) for name,v,h in zip(names,out,final)}))
    report['status']='complete';return report


def compare_frozen_phases(previous,current):
    """Reuse existing cross_time on compatible admitted source receipts.
    Input order is not authenticated optimizer chronology.
    """
    from scripts.x369_phase_geometry import cross_time
    import json
    canonical=lambda v:json.dumps(v,sort_keys=True,allow_nan=False)
    if previous.get('kind')!='new_source_acceptance' or current.get('kind')!='new_source_acceptance' or not previous.get('complete'):
        raise ValueError('expected original completed previous source receipt')
    if any(canonical(previous[k])!=canonical(current[k]) for k in ('config','runtime','task_sha256')) or previous['source']['files']!=current['source']['files']:
        raise ValueError('incompatible source, task, runtime or configuration')
    a=previous['stages'].get('factorial',{});b=current['stages'].get('factorial',{})
    if 'phase' not in a or 'phase' not in b:return dict(admitted=False,reason='unidentified chart',training_clock_authenticated=False)
    if a['task_sha256']!=b['task_sha256']:raise ValueError('different chart probe sample')
    return dict(comparison=cross_time(a['phase'],b['phase']),previous_checkpoint=previous['checkpoint_sha256'],current_checkpoint=current['checkpoint_sha256'],training_clock_authenticated=False,optimizer_correspondence_certified=False)


def certificate():
    import torch
    V=torch.eye(4,dtype=torch.float64)[:,:2];old=torch.tensor([1.,2.,3.,4.],dtype=V.dtype);chart=old+torch.tensor([3.,4.,0.,0.],dtype=V.dtype);d=torch.tensor([2.,5.,3.,4.],dtype=V.dtype);v=matched_null(old,chart,V,d)
    assert torch.equal((v-old)@V,torch.zeros(2,dtype=V.dtype)) and (v-old).norm()==(chart-old).norm()==5
    assert matched_null(old,chart,V,d*torch.tensor([1,1,0,0])) is None and matched_null(old,old,V,d) is None
    print('PASS matched chart-null distance and zero-direction/zero-edit refusal; no native intervention')


if __name__=='__main__':certificate()
