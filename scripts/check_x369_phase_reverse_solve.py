from itertools import product
from fractions import Fraction as F
import torch
import sympy as s
from x369_phase_reverse_solve import matrices,gauge_audit,affine_cell_transport,read_campaign,digest,arm_name
import tempfile,json
from pathlib import Path

def main():
    torch.set_num_threads(1);torch.manual_seed(41);dt=torch.float64
    Y=torch.tensor(list(product((-1,0,1),repeat=6)),dtype=dt);lookup={tuple(y.tolist()):i for i,y in enumerate(Y)};rho=matrices();ix={g:torch.tensor([lookup[tuple(y.tolist())] for y in Y@a]) for g,a in rho.items()}
    U=torch.eye(6,dtype=dt)+.2*torch.randn(6,6,dtype=dt);A={g:torch.linalg.solve(U,a@U) for g,a in rho.items()}
    q=Y@U;good=gauge_audit(U,A,q,Y,ix);assert max(v['raw_natural_rmse'] for v in good['generators'].values())<1e-12
    bad=dict(A,J=torch.eye(6,dtype=dt));r=gauge_audit(U,bad,q,Y,ix);assert r['generators']['J']['raw_natural_rmse']>.1 and r['generators']['J']['corrected_natural_rmse']<1e-12
    q=q+.1*torch.randn(q.shape,dtype=dt);r=gauge_audit(U,bad,q,Y,ix);assert max(v['decomposition_error'] for v in r['generators'].values())<1e-12 and r['generators']['J']['corrected_natural_rmse']>.01
    assert not gauge_audit(torch.zeros_like(U),A,q,Y,ix)['admitted']
    # General identity with unconstrained symbolic matrices (2D witness).
    mat=lambda prefix:s.Matrix(2,2,s.symbols(prefix+'0:4'));Z,G,R,E,H=map(mat,'zgreh');Y0=Z-E
    assert s.expand(Z*G-(Y0*R+H)-Z*(G-R)-(E*R-H))==s.zeros(2)
    # All intervals certified by exact cell test must have same rounded image
    # on a rational grid, including ties; odd cells exclude their boundary.
    n=paid=0
    for a,b,c,d in product((F(-1),F(-1,2),F(0),F(1,2),F(1)),repeat=4):
        A=[[a,b]];offset=[d];code=[int(c*2),1];cert=affine_cell_transport(A,offset,code,1);n+=1
        if cert['code'] is None:
            assert cert['witnesses']
        else:
            paid+=1
            for r0,r1 in product((F(-1,2),F(-1,4),F(0),F(1,4),F(1,2)),repeat=2):
                if round(code[0]+r0)!=code[0] or round(code[1]+r1)!=code[1]:continue
                assert round(a*(code[0]+r0)+b*(code[1]+r1)+d)==cert['code'][0]
    assert affine_cell_transport([[1,1]],[0],[0,0],1)['code'] is None
    assert affine_cell_transport([[1]],[0],[0],1)['code']==[0]
    assert affine_cell_transport([[1]],[0],[1],1)['code'] ==[1]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);source=root/'source';source.write_text('fixture');arm=dict(seed=41,mode='c3',paired=False,null=False,ternary=False);cfg={'schedule':[0]};m=dict(kind='x369_phase_trajectory',release_admitted=False,sources={'source':digest(source)},settings=cfg,arms=[arm]);mp=root/'manifest.json';mp.write_text(json.dumps(m));d=root/arm_name(arm);d.mkdir();p=d/'step_00000000.pt';p.write_bytes(b'fixture, not actual checkpoint');identity=dict(arm=arm,settings=cfg,manifest_sha256=digest(mp));r=dict(identity=identity,step=0,sha256=digest(p));(d/'step_00000000.json').write_text(json.dumps(r));a=dict(identity=identity,step=0,snapshot_sha256=r['sha256'],release_admitted=False,phase=dict(rank=0,admitted=False),train={},composition={});(d/'audit_00000000.json').write_text(json.dumps(a));assert read_campaign(root,root)['complete']==1
        p.write_bytes(b'changed')
        try:read_campaign(root,root)
        except ValueError:pass
        else:raise AssertionError('changed snapshot admitted')
    print('PASS 729 natural states, wrong J correction, nonzero outside-anchor defect, exact symbolic decomposition,',n,'interval cases,',paid,'certified')
if __name__=='__main__':main()
