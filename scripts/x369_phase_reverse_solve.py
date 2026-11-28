"""Offline phase receipts and reverse-solved geometry. No model training or release claim."""
import argparse, hashlib, json
from fractions import Fraction as F
from pathlib import Path
import torch


def matrices():
    r=torch.tensor([[0.,1,0],[0,0,1],[1,0,0]],dtype=torch.float64)
    j=torch.tensor([[1.,0,0],[0,0,1],[0,1,0]],dtype=torch.float64)
    return {'R':torch.block_diag(r,r),'J':torch.block_diag(j,-j),'N':-torch.eye(6,dtype=r.dtype)}


def rms(x):return float(x.square().mean().sqrt())


def gauge_audit(U,A,q,Y,targets,rtol=1e-10):
    """U independently fitted on train, q,Y,targets from composition only.
    Row gauge. Numerical diagnostics, no claim of exact fitted action.
    """
    if U.shape!=(6,6) or q.shape!=Y.shape or q.shape[1]!=6 or any(not torch.isfinite(x).all() for x in [U,q,Y,*A.values()]):raise ValueError('invalid finite chart')
    if int(torch.linalg.matrix_rank(U,rtol=rtol))!=6:return {'admitted':False,'reason':'anchor_rank'}
    inv=torch.linalg.inv(U);z=q@inv;e=z-Y;rho=matrices();report={}
    for g,R in rho.items():
        j=targets[g]
        if not torch.equal(Y[j],Y@R):raise ValueError('natural target disagrees with task action')
        G=U@A[g]@inv;F_g=G-R;corrected=inv@R@U
        raw=z@G-z[j];closure=e@R-e[j];defect=z@F_g
        # Exact identity over reals: raw = action defect + natural-chart defect.
        report[g]=dict(task_gauge_operator_defect=rms(F_g),raw_natural_rmse=rms(raw),action_defect_rmse=rms(defect),corrected_natural_rmse=rms(closure),decomposition_error=rms(raw-defect-closure),correction_latent_norm=float((A[g]-corrected).norm()),anchored_operator=G.tolist())
    return dict(admitted=True,condition=float(torch.linalg.cond(U)),anchor_rmse=rms(e),anchor_max=float(e.abs().max()),generators=report,exact_learned_action_certified=False)


def affine_cell_transport(A,b,c,delta):
    """Exact rational descent on one complete ties-to-even input fibre.
    Column convention. Endpoints attained iff all nonzero coefficients
    use even input bins. Returns None iff output code varies on the fibre.
    No claim about floating execution or non-affine guards.
    """
    delta=F(delta);c=list(c)
    if delta<=0 or not all(isinstance(v,int) for v in c) or len(A)!=len(b) or any(len(row)!=len(c) for row in A):raise ValueError('invalid affine code')
    result=[];bounds=[];witnesses=[]
    for i,(row,offset) in enumerate(zip(A,b)):
        a=list(map(F,row));center=sum((w*v for w,v in zip(a,c)),F(offset)/delta);rad=sum(map(abs,a))/2
        lo,hi=center-rad,center+rad;k=round(center);left,right=F(k)-F(1,2),F(k)+F(1,2)
        attained=all(v%2==0 for w,v in zip(a,c) if w)
        ok=(lo>=left and hi<=right) if k%2==0 else ((lo>left or lo==left and not attained) and (hi<right or hi==right and not attained))
        result.append(k if ok else None);bounds.append(dict(lo=str(lo),hi=str(hi),endpoints_attained=attained))
        if not ok:
            low=lo<left or (lo==left and k%2==1 and attained)
            threshold=left if low else right;direction=-1 if low else 1
            edge=lo if low else hi
            t=F(1) if edge==threshold else (rad+abs(center-threshold))/(2*rad)
            point=[delta*(v+direction*t*F((w>0)-(w<0),2)) for v,w in zip(c,a)]
            value=sum((w*z for w,z in zip(a,point)),F(offset))/delta
            assert all(round(z/delta)==v for z,v in zip(point,c)) and round(value)!=k
            witnesses.append(dict(coordinate=i,center=[str(delta*v) for v in c],other=[str(z) for z in point],center_code=k,other_code=round(value)))
    return dict(code=result if all(v is not None for v in result) else None,coordinate_certificates=result,bounds=bounds,witnesses=witnesses)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read_json(p):
    return json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON '+x)))


def arm_name(a):return f"ordered_{a['mode']}_paired{int(a['paired'])}_seed{a['seed']}_null{int(a['null'])}_ternary{int(a['ternary'])}"


def chart_metrics(chart):
    from scripts.audit_x369_ordered_signed_task import endpoint
    from scripts.audit_x369_joint_action import acted
    keys=chart['composition_keys'];ys=[];lookup={}
    for i,(w,g,m,d) in enumerate(keys):
        state=endpoint(w,g,m,d)[:2];ys.append(sum(state,()));k=(w,tuple(g),state)
        if k in lookup:raise ValueError('duplicate natural state')
        lookup[k]=i
    target={g:torch.tensor([lookup[w,tuple(guards),acted(g,(tuple(y[:3]),tuple(y[3:])))] for (w,guards,_,_),y in zip(keys,ys)]) for g in 'RJN'}
    mat=lambda x:torch.tensor(x,dtype=torch.float64)
    return gauge_audit(mat(chart['U']),{g:mat(chart['A'][g]) for g in 'RJN'},mat(chart['composition_q']),mat(ys),target)


def summarize(a):
    p=a['phase'];out=dict(step=a['step'],rank=p['rank'],chart_admitted=p['admitted'],train=a['train'],composition=a['composition'],cross_time=a.get('cross_time'))
    o=p.get('orbit',{});out['phase_energy_fraction']=o['phase_energy']/o['total_energy'] if o.get('total_energy',0)>0 else None
    out['anchored']=p.get('anchored');out['common_hom']=p.get('common_hom');out['cube']=p.get('cube');out['intervention']=p.get('intervention')
    if 'chart' in p:out['task_gauge']=chart_metrics(p['chart'])
    return out


def read_campaign(root,repo):
    """Read existing trajectory without modifying original manifest or receipts.
    Validate hashes of snapshot bytes (not deserialize), manifest source bytes,
    and cross-file IDs. Audit JSON is not cryptographically authenticated by
    the original runner; recompute chart diagnostics only from stored arrays.
    """
    mp=root/'manifest.json';m=read_json(mp);mh=digest(mp)
    if m.get('kind')!='x369_phase_trajectory' or m.get('release_admitted') is not False:raise ValueError('foreign manifest')
    for name,h in m['sources'].items():
        if digest(repo/name)!=h:raise ValueError('source mismatch: '+name)
    cfg=m['settings'];steps=cfg['schedule'];arms=[]
    for arm in m['arms']:
        d=root/arm_name(arm);identity=dict(arm=arm,settings=cfg,manifest_sha256=mh);records=[]
        for t in steps:
            sp=d/f'step_{t:08d}.pt';rp=d/f'step_{t:08d}.json';ap=d/f'audit_{t:08d}.json'
            if not any(p.exists() for p in (sp,rp,ap)):continue
            if not sp.exists() or not rp.exists():raise ValueError('orphan snapshot '+str(sp))
            r=read_json(rp)
            if r['identity']!=identity or r['step']!=t or digest(sp)!=r['sha256']:raise ValueError('snapshot identity mismatch')
            if not ap.exists():continue
            a=read_json(ap)
            if a['identity']!=identity or a['step']!=t or a['snapshot_sha256']!=r['sha256'] or a.get('release_admitted') is not False:raise ValueError('audit identity mismatch')
            records.append(dict(snapshot_sha256=r['sha256'],audit_sha256=digest(ap),**summarize(a)))
        arms.append(dict(arm=arm,expected=len(steps),audited=len(records),records=records))
    return dict(manifest_sha256=mh,arms=arms,planned=len(arms),complete=sum(a['audited']==a['expected'] for a in arms),audits_reexecuted=False,phase_sequence_certified=False,release_admitted=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    result=read_campaign(a.input,a.source_root)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(planned=result['planned'],complete=result['complete'],output=str(a.output))))
if __name__=='__main__':main()
