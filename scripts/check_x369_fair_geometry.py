import torch
from torch import nn
from models.geometric_fair_residual import block_frame,fair_arm,FairResidualLinear
from models.geometric_paired_history import PairedHistoryMoE
from scripts.check_lila_paired_history import Fixture
from scripts.audit_x369_hidden_chart import fit_chart,chart_metrics,transport_delta
from scripts.audit_x369_ordered_signed_task import records


def main():
    torch.manual_seed(41);torch.set_num_threads(1)
    source=Fixture();source.temperature=1.;x=torch.randn(5,3,dtype=torch.float64);h=torch.randn(5,4,2,3,dtype=torch.float64)
    baseline=PairedHistoryMoE(source).step(x,h);checks=0
    for mode,rank in [('c3',3),('s3',5),('full',9),('entry',9)]:
        G=block_frame(mode).reshape(rank,9);N=block_frame(mode,True).reshape(rank,9)
        for B in (G,N):assert torch.allclose(B@B.T,torch.eye(rank,dtype=B.dtype),atol=1e-14)
        if rank<9:assert (G.T@G-N.T@N).norm()>0.1
        else:assert torch.allclose(G.T@G,N.T@N,atol=1e-14)
        arms=[]
        for null in (False,True):
            for ternary in (False,True):
                m=PairedHistoryMoE(fair_arm(source,mode,null,ternary));arms.append(m)
                a=m.step(x,h,train_sign=True)
                assert torch.equal(a.branches,baseline.branches) and torch.equal(a.history,baseline.history) and torch.equal(a.ids,baseline.ids) and torch.equal(a.signs,baseline.signs) and torch.equal(a.weights,baseline.weights)
                (a.branches.square().sum()+a.history.square().sum()).backward()
                assert m.source.orientation.weight.grad is not None and m.source.orientation.weight.grad.abs().sum()>0
                for l in m.modules():
                    if isinstance(l,FairResidualLinear) and l.coefficients.grad is not None:assert torch.isfinite(l.coefficients.grad).all()
                checks+=1
        assert len(set(sum(p.numel() for p in m.parameters()) for m in arms))==1
        for i in (0,2):
            for a,b in zip(arms[i].parameters(),arms[i+1].parameters()):
                assert (a.grad is None)==(b.grad is None)
                if a.grad is not None:assert torch.equal(a.grad,b.grad)
    rows=list(records());Y=torch.tensor([r['mean_label']+r['signed_label'] for r in rows],dtype=torch.float64)
    mask=torch.tensor([r['split']=='train' for r in rows]);U=torch.randn(6,12,dtype=torch.float64);Z=Y@U+torch.randn(12,dtype=torch.float64)
    c=fit_chart(Y[mask],Z[mask]);a=chart_metrics(c,Y[~mask],Z[~mask]);assert a['full_rank'] and a['task_rmse']<1e-12 and a['relative_history_error']<1e-12
    other=Y[~mask].clone();other[:,3:]*=-1;assert torch.allclose(Z[~mask]+transport_delta(c,Y[~mask],other),other@U+(Z-Y@U)[0],atol=1e-12)
    collapsed=Z.clone()*0;c0=fit_chart(Y[mask],collapsed[mask]);assert not c0['full_rank']
    print('PASS',torch.__version__,checks,'common-function paired fixture arms; real/STE initial gradients; 6804 heldout chart rows; collapse rejected')
if __name__=='__main__':main()
