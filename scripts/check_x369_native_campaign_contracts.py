"""CPU contracts for integrated native campaign; no trained-model claim."""
from itertools import product
from pathlib import Path
import json,tempfile
import torch
from scripts.audit_x369_ordered_signed_task import closed_action
from scripts.audit_x369_joint_action import joint_records,joint_generator_control,task_digest,select_contexts,state,fit_joint
from scripts.audit_x369_reachable_bins import bounded_witnesses
from scripts.audit_x369_precision_guards import ordered_gap,measured_order_bound
from scripts.run_x369_ordered_campaign import arms,arm_name,checkpoint_receipt,replay_valid,sha


def main():
    torch.set_num_threads(1);torch.manual_seed(41)
    rows=list(joint_records());assert len(rows)==81648
    assert all((r['mean_label'],r['signed_label'],r['gram_label'])==closed_action(r['word'],r['guards'],r['mean'],r['residual']) for r in rows)
    control=joint_generator_control(rows);assert control['generators']==10 and control['correct']=={'train':34020,'composition':47628} and not control['mean_twist_given']
    subset=select_contexts(rows,12);Y=torch.tensor([sum(state(r),()) for r in subset],dtype=torch.float64);Z=Y@torch.randn(6,12,dtype=torch.float64)+torch.randn(12,dtype=torch.float64)
    fit=fit_joint(subset,Z);assert fit['rank']==6 and max(fit['relations'].values())<1e-12 and fit['composition']['signed_rmse']<1e-12
    assert not fit_joint(subset,Z*0)['admitted']
    print('joint',len(rows),task_digest(rows),control['correct'],max(fit['relations'].values()))
    h=torch.tensor([[0.,1.],[0.,-1.],[1.,0.],[1.,0.]])
    def obs(w):
        x=h.clone()
        for u in w:
            y=x[:,0].clone()
            if u=='swap':x=x.flip(-1)
        return {'y':y}
    r=bounded_witnesses([0,0,1,1],('hold','swap'),obs,2)
    assert r['first_separation_depth']=={'1':0,'2':1} and r['witnesses'][0]['word']==['swap','hold'] and r['witnesses'][0]['max_abs_delta']=={'y':2.} and r['separating_words_replayed']==1
    r=bounded_witnesses([0]*189,('a',),lambda w:{'y':torch.arange(189)},1);assert r['separated_pairs']==17766
    counter=[0]
    def bad(w):
        counter[0]+=1;return {'y':torch.tensor([0.,float(counter[0])])}
    try:bounded_witnesses([0,0],('a',),bad,1)
    except RuntimeError:pass
    else:raise AssertionError('nonrepeatable witness admitted')
    b=torch.tensor(list(product(range(-1,2),repeat=4)),dtype=torch.float64)
    ids=b.topk(2,dim=1).indices;deltas=torch.tensor(list(product((-.1,0.,.1),repeat=4)),dtype=b.dtype)
    ref=b.repeat_interleave(len(deltas),0);a=ref+deltas.repeat(len(b),1);ii=ids.repeat_interleave(len(deltas),0);gap,eps,paid=measured_order_bound(a,ref,ii)
    assert paid.any() and (~paid).any() and torch.equal(a.topk(2,dim=1).indices[paid],ii[paid]);assert (ordered_gap(b,ids)==0).any()
    plan=arms([41,42,43],['c3','s3','full','entry'],['single','paired']);assert len(plan)==len({arm_name(a) for a in plan})==96
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);arm=plan[0];name=arm_name(arm);settings=dict(width=12,steps=2000,batch_size=128)
        assert checkpoint_receipt(root,name,arm,settings) is None
        target=root/(name+'.pt');target.write_bytes(b'contract only')
        try:checkpoint_receipt(root,name,arm,settings)
        except RuntimeError:pass
        else:raise AssertionError('unreceipted checkpoint admitted')
        m=dict(**arm,**settings,joint_mean=True,fixed_horizon_completed=True,public_task_only=True)
        cp=dict(kind='final_checkpoint',path=str(target),sha256=sha(target),metadata=m);(root/'train.0.log').write_text(json.dumps(cp)+'\n');assert checkpoint_receipt(root,name,arm,settings)==cp
        receipt=root/'replay.json';scope={'horizon':2};assert not replay_valid(receipt,cp,scope)
        receipt.write_text(json.dumps(dict(kind='ordered_checkpoint_replay',sha256=cp['sha256'],metadata=m,trace_probe_equal=True,scope=scope)))
        try:replay_valid(receipt,cp,scope)
        except RuntimeError:pass
        else:raise AssertionError('incomplete replay admitted')
    print('PASS 17766 collision pairs; repeatable witnesses; 6561 observed margin cases; 96-arm plan and fail-closed receipt controls')
if __name__=='__main__':main()
