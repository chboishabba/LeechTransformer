from itertools import product
import math
import sympy as s
import torch
from torch import nn
from models.geometric_paired_history import PairedHistoryMoE
from models.geometric_residual_observability import triad_probes,squared_probe_packet,gram_from_squared_packet,gelu_curvature_bound,propagate_envelope

class Fixture(nn.Module):
    """Source-legal parameter layout and strict constant routing; not original import."""
    def __init__(self):
        super().__init__();self.d_model=6;self.num_experts=4;self.top_k=2;self.route_geometry='c2_c3_residual'
        self.orientation=nn.Linear(12,1,bias=False,dtype=torch.float64)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(12,6,dtype=torch.float64),nn.GELU(),nn.Linear(6,6,dtype=torch.float64)) for _ in range(4)])
        self.transitions=nn.ModuleList([nn.Linear(18,6,dtype=torch.float64) for _ in range(4)])
        with torch.no_grad():
            self.orientation.weight.zero_();self.orientation.weight[0,3:6]=1
            for e,(a,_,b) in enumerate(self.experts):
                a.weight.zero_();a.bias.zero_();b.weight.copy_(torch.eye(6,dtype=torch.float64));b.bias.zero_()
                if e==0:
                    p=triad_probes(dtype=torch.float64);a.weight[:,:3]=-p;a.weight[:,6:9]=p
                t=self.transitions[e];t.weight.zero_();t.bias.zero_();t.weight[:,:6]=torch.eye(6,dtype=torch.float64)
    def select_source_experts(self,x,h):
        logits=x.new_tensor([2.,1.,0.,-1.]).expand(x.shape[0],-1)
        val,ids=logits.topk(2,dim=-1);return ids,val.softmax(-1)
    def cap_history(self,h):return h

def original_factory():
    from models.history_orbit_moe import HistoryOrbitMoE
    f=Fixture();m=HistoryOrbitMoE(d_model=6,num_experts=4,top_k=2,expert_width=6,route_geometry='c2_c3_residual').double().eval()
    with torch.no_grad():
        m.router.weight.zero_();m.router.bias.zero_();m.c3_residual_weight.zero_();m.expert_bias.copy_(torch.tensor([2.,1.,0.,-1.],dtype=torch.float64))
        m.orientation.weight.copy_(f.orientation.weight);m.experts.load_state_dict(f.experts.state_dict());m.transitions.load_state_dict(f.transitions.state_dict())
    return m

def main(factory=Fixture):
    torch.manual_seed(42);torch.set_num_threads(1)
    dtype=torch.float64
    # Exact polynomial six-probe inverse; signed ternary enumeration.
    d=s.Matrix(s.symbols('d:3'));p=s.Matrix([[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0]])
    a,b,c,bc,ca,ab=[v**2 for v in p*d];xy=(ab-a-b)/2;yz=(bc-b-c)/2;zx=(ca-c-a)/2
    assert s.expand(s.Matrix([[a,xy,zx],[xy,b,yz],[zx,yz,c]])-d*d.T)==s.zeros(3)
    vals=torch.tensor(list(product((-1,0,1),repeat=3)),dtype=dtype)
    packets=squared_probe_packet(vals); grams=gram_from_squared_packet(packets)
    assert torch.equal(grams,vals.unsqueeze(-1)*vals.unsqueeze(-2))
    assert len(set(map(tuple,packets.tolist())))==14
    for i,j in product(range(len(vals)),repeat=2):
        assert torch.equal(packets[i],packets[j])==(torch.equal(vals[i],vals[j]) or torch.equal(vals[i],-vals[j]))
    # Delayed observability: coordinate-square observers plus symmetric shear
    # reveal the hidden cross product on the next labelled mode.
    A=s.Matrix([[1,1],[0,1]]);q=s.Matrix([[1,0,0],[0,0,1]])
    # Symmetric coordinates (D1², D1*D2, D2²).
    L=s.Matrix([[1,2,1],[0,1,1],[0,0,1]])
    assert s.Matrix.vstack(q,q*L).rank()==3
    assert s.Matrix([[(A*s.Matrix([1,1]))[0]**2]])!=s.Matrix([[(A*s.Matrix([1,-1]))[0]**2]])
    # Four experts, original-interface paired recurrence. Two control triads:
    # x first triad centers known M, x second triad fixes positive sign.
    model=PairedHistoryMoE(factory())
    cases=0
    for m in (torch.zeros(3,dtype=dtype),torch.tensor([.25,-.5,1.],dtype=dtype),torch.tensor([-1.,2.,-.25],dtype=dtype)):
        x=torch.cat((m,torch.ones(3,dtype=dtype))).expand(len(vals),-1).clone()
        hist=torch.zeros(len(vals),4,2,6,dtype=dtype);hist[:,0,0,:3]=m+vals;hist[:,0,1,:3]=m-vals
        a=model.step(x,hist);b=model.step(x,hist.flip(-2))
        proj=vals@triad_probes(dtype=dtype).T;alpha=a.weights[:,0:1]
        assert torch.equal(a.ids,torch.tensor([[0,1]]).expand(len(vals),-1))
        assert torch.equal(a.signs,torch.ones_like(a.signs));assert torch.equal(a.history,hist)
        assert torch.allclose(a.mean,alpha*proj*torch.erf(proj/math.sqrt(2))/2,atol=1e-14,rtol=1e-14)
        assert torch.allclose(a.residual,alpha*proj/2,atol=1e-14,rtol=1e-14)
        assert torch.equal(a.mean,b.mean) and torch.equal(a.residual,-b.residual)
        cases+=len(vals)
    # Curvature certificate: all pairs of ternary M,D in a triad and 1000 random rows.
    m=vals.repeat_interleave(27,0);d=vals.repeat((27,1));m=torch.cat((m,torch.randn(1000,3,dtype=dtype)*3));d=torch.cat((d,torch.randn(1000,3,dtype=dtype)))
    g=torch.nn.functional.gelu;defect=((g(m+d)+g(m-d))/2-g(m)).norm(dim=-1)
    bound=gelu_curvature_bound(d);assert torch.all(defect<=bound+1e-13)
    # Cap kink is first order, exact rational identity.
    for eps in (s.Rational(1,2),s.Rational(1,8),s.Rational(1,1024)):
        assert (1+(1-eps))/2-1==-eps/2
    assert propagate_envelope([2.,3.],[.125,.25])==[0.,.125,.625]
    print('PASS',s.__version__,torch.__version__,'729 exact squared-packet collision pairs; 14 antipodal classes; 81 centered paired rows; curvature cases',len(m),'delayed quadratic rank 2->3; cap and envelope')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--original',action='store_true');a=p.parse_args();print('source mode:', 'original' if a.original else 'fixture');main(original_factory if a.original else Fixture)
