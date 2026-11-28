"""Actual source-aligned Leech action on a registered native learner.

Read-only diagnostic outside all registered training source owner hashes.
Two distinct squares:
  retained rechart -> reopen -> original step (bit exact);
  actual raw action on original frozen operator (empirical defect).
No lattice membership claim for real neural histories, no semantic
opposition inference, no parameter changes or heldout selection.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts.run_x369_mod3_checkpointed import identity,make_arm,sha
from scripts.run_x369_mod3_maxcut_gpu import state_sha
from scripts.x369_r4_execution_device import resolve_device,runtime,tensor_bytes
from scripts.audit_x369_typed_mod3_prediction_consumer import TASK_PAIRS,typed_mod3_prediction_consumer


def require(ok,msg):
    if not ok:raise ValueError(msg)


def hash_bytes(x):return hashlib.sha256(tensor_bytes(x)).hexdigest()


def matrix(raw,device):
    require(len(raw)==24 and all(len(r)==24 for r in raw),'24-dimensional source')
    q=[[F(v) for v in r] for r in raw]
    require(all(sum(q[i][k]*q[j][k] for k in range(24))==int(i==j)
                for i in range(24) for j in range(24)),'exact real orthogonality')
    require(all(v.denominator&(v.denominator-1)==0 for r in q for v in r),'dyadic source')
    return torch.tensor([[float(v) for v in r] for r in q],device=device,dtype=torch.float64)


def transform(x,R):
    require(x.shape[-1]%24==0 and x.dtype in (torch.float32,torch.float64),'Leech block alignment')
    y=(x.reshape(*x.shape[:-1],-1,24).double()@R.T).reshape(x.shape).to(x.dtype)
    require(bool(torch.isfinite(y).all()),'nonfinite rechart')
    return y.contiguous()


def rawbytes(x):return x.contiguous().reshape(-1).view(torch.uint8)


def retained(x,R):
    y=transform(x,R)
    back=transform(y,R.T)
    correction=torch.bitwise_xor(rawbytes(x),rawbytes(back))
    return y,correction


def reopen(packet,R):
    y,c=packet
    back=transform(y,R.T)
    require(c.shape==rawbytes(back).shape and c.dtype==torch.uint8,'exact residual scope')
    return torch.bitwise_xor(rawbytes(back),c).view(y.dtype).reshape(y.shape)


def exact_rechart(x,R):
    p=retained(x,R);y=reopen(p,R)
    require(torch.equal(rawbytes(y),rawbytes(x)),'bitwise retained reopening')
    return y,dict(source_sha256=hash_bytes(x),chart_sha256=hash_bytes(p[0]),
        residual_nonzero_bytes=int((p[1]!=0).sum()),source_bytes=p[1].numel(),
        residual_retained=True,compression_claimed=False)


def execute(m,selector,x,h):
    if selector is None:return m.history.step(x,h,train_sign=False)
    ids,w=selector(m.history,x,h)
    idx=ids.unsqueeze(-1).expand(-1,-1,m.history.d_model)
    y,u,signs=m.history.execute_selected(x,ids,h.gather(1,idx),w,train_sign=False)
    return y,h.scatter(1,idx,u),ids,signs,w


def difference(a,b):
    require(a.shape==b.shape,'consumer shape changed')
    return dict(bitwise_equal=torch.equal(rawbytes(a),rawbytes(b)),
                max_abs=float((a.double()-b.double()).abs().max()))


def probe(m,selector,R,pairs):
    observed=[]
    def observer(t,x,y,h0,h,ids,signs,w):
        # No mutation of actual source state. Replay original native step
        # via exact retained coordinate reopening.
        xr,rx=exact_rechart(x,R);hr,rh=exact_rechart(h0,R)
        base=(y,h,ids,signs,w)
        same=execute(m,selector,xr,hr)
        require(all(torch.equal(rawbytes(a),rawbytes(b)) for a,b in zip(base,same)),
                'retained source step changed')
        yr,ry=exact_rechart(y,R);nr,rn=exact_rechart(h,R)
        moved=execute(m,selector,transform(x,R),transform(h0,R))
        row=dict(token_step=t,retained_reopen=dict(input=rx,previous_history=rh,
            output=ry,successor_history=rn),actual_original_step_bitwise_equal=True,
            raw_geometry_action=dict(output=difference(moved[0],transform(y,R)),
            successor=difference(moved[1],transform(h,R)),
            selected_ids_equal=torch.equal(ids,moved[2]),
            local_signs_equal=torch.equal(signs,moved[3]),
            selected_weights=difference(w,moved[4])))
        if t==1:
            original=m.readout(y+x)
            preserved=m.readout(yr+xr)
            require(torch.equal(rawbytes(original),rawbytes(preserved)),'original readout rechart')
            raw=m.readout(moved[0]+transform(x,R))
            row['readout']=dict(original_retained_bitwise_equal=True,
                raw_intervention=difference(original,raw),
                original_predictions=original.argmax(-1).tolist(),
                intervention_predictions=raw.argmax(-1).tolist())
        observed.append(row)
    with torch.no_grad():
        logits=m(pairs,selector,observer=observer)
    return dict(original_task=typed_mod3_prediction_consumer(logits),steps=observed,
        all_original_step_squares_exact=True,
        raw_action_equivariance_observed=all(r['raw_geometry_action']['output']['bitwise_equal']
            and r['raw_geometry_action']['successor']['bitwise_equal']
            and r['raw_geometry_action']['selected_ids_equal']
            and r['raw_geometry_action']['local_signs_equal']
            and r['raw_geometry_action']['selected_weights']['bitwise_equal'] for r in observed),
        original_neural_lattice_membership_claimed=False,
        semantic_causal_intervention_certified=False)


def run(a):
    dev=resolve_device(a.device)
    require(sha(a.checkpoint)==a.checkpoint_sha256,'registered checkpoint SHA')
    p=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    c=p['contract']
    spec=SimpleNamespace(arm=c['arm'],seed=c['seed'],width=c['width'],experts=c['experts'],
        dtype=c['dtype'],steps=c['horizon'],report_every=c['report_every'],
        lr=c['lr'],weight_decay=c['weight_decay'])
    require(identity(spec,dev)==c,'active source contract changed')
    m,selector,is_learner=make_arm(spec,dev)
    require(is_learner,'actual recurrent learner required')
    m.load_state_dict(p['model'],strict=True);m.eval()
    require(state_sha(m)==p['model_sha256'],'registered actual model identity')
    raw=a.lift.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==a.lift_sha256,'pinned actual Conway source')
    lift=json.loads(raw)
    require(lift['status']=='actual_Gx0_real_Leech_lift' and lift['forward']['quotient_basis_checks']==24,
            'actual source action receipt required')
    R=matrix(lift['forward']['real_matrix'],dev)
    result=probe(m,selector,R,torch.tensor(TASK_PAIRS,device=dev,dtype=torch.long))
    require(state_sha(m)==p['model_sha256'],'read-only audit mutated model')
    return dict(status='actual_registered_Leech_action_probe',checkpoint_sha256=a.checkpoint_sha256,
        model_sha256=p['model_sha256'],completed_steps=p['completed_steps'],arm=c['arm'],seed=c['seed'],
        runtime=runtime(dev),lift_sha256=a.lift_sha256,literal_source_word=lift['word'],result=result,
        registered_training_modified=False,heldout_selection=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--lift',type=Path,required=True);p.add_argument('--lift-sha256',required=True)
    p.add_argument('--device',choices=('gfx803','cpu'),default='gfx803')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=run(a);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(r,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(r['status'],r['result']['raw_action_equivariance_observed'])

if __name__=='__main__':main()
