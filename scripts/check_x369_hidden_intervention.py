"""Exact synthetic chart/consumer and untrained source-interface negative control."""
import torch
from scripts.run_x369_fair_ordered_gpu import FairLearner,audit_hidden
from scripts.check_lila_paired_history import Fixture
from scripts.audit_x369_ordered_signed_task import records
from scripts.train_x369_ordered_action import LABELS,gram

class Toy:
    experts=1;width=6;paired=False
    def advance(self,op,h):
        z=h.reshape(-1,2,3);kind=op[:,:4].argmax(-1);v=op[:,4:].argmax(-1);new=[]
        for a,k,t in zip(z,kind.tolist(),v.tolist()):
            if k<2:a=a.roll(t,-1)
            elif k==2 and t:a=a[:,[0,2,1]]*a.new_tensor([[1],[-1]])
            elif k==3 and t:a=-a
            new.append(a)
        h=torch.stack(new).reshape(-1,1,6);ids=torch.zeros(len(h),1,dtype=torch.long);w=h.new_ones(len(h),1)
        return h[:,0],h,ids,w,w
    def rollout(self,s,o):
        h=s[:,:6].reshape(-1,1,6)
        for t in range(o.shape[1]):y,h,*_=self.advance(o[:,t],h)
        return y,h
    def consume(self,y,h):
        targets=[tuple(v[:3]) for v in y.tolist()], [gram(tuple(v[3:])) for v in y.tolist()], [tuple(v[3:]) for v in y.tolist()]
        return tuple(-((torch.tensor(t,dtype=y.dtype)[:,None]-torch.tensor(L,dtype=y.dtype)[None])**2).sum(-1) for t,L in zip(targets,LABELS))

if __name__=='__main__':
    torch.set_num_threads(1);torch.manual_seed(41);rows=list(records())
    r=audit_hidden(Toy(),rows,'cpu',torch.float64);print('synthetic',r)
    assert r['chart_rank']==6 and r['intervention_pairs']==2808 and r['probability_l1_sum_per_row'][1]<1e-12
    f=Fixture();f.temperature=1.;m=FairLearner(3,4,True,'dense',f).double().eval()
    r=audit_hidden(m,rows,'cpu',torch.float64);print('untrained fixture',r)
    assert r['chart_rank']==5 and r['intervention']=='rank_deficient'
