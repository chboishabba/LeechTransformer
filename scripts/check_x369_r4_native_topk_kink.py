#!/usr/bin/env python3
"""Source-NATIVE R4 topk/ResponseKink architecture fixture on CPU.

Runs the actual HistoryOrbitMoE model and its original selected-execution
consumer, not a duplicate scorer or substitute GELU kernel. The weights
are explicit synthetic source-legal parameters, not a trained checkpoint.
No GPU, Agda, CI or physical performance inference.
"""
import math
import torch
from models.history_orbit_moe import HistoryOrbitMoE


def test_original_source_topk_replacement_kink():
    m=HistoryOrbitMoE(d_model=6,num_experts=4,top_k=2,
                      route_geometry="c2_c3_residual",
                      router_selection="topk").double().eval()
    slopes=[0,4,9,10]
    biases=[8,3,-11,-13]
    with torch.no_grad():
        m.router.weight.zero_()
        m.router.bias.zero_()
        m.c3_residual_weight.zero_()
        # Original R4 residual features consist of unary 6 +
        # directed 6; channel 0 selects the first self inner product.
        m.c3_residual_weight[0,6]=1
        m.expert_bias.copy_(torch.tensor(biases,dtype=torch.double))
        m.orientation.weight.zero_()
        m.orientation.weight[0,6]=1
        m.orientation.bias.zero_()
        for e in range(4):
            m.experts[e][0].weight.zero_()
            m.experts[e][0].bias.zero_()
            m.experts[e][2].weight.zero_()
            m.experts[e][2].bias.zero_()
            m.experts[e][2].bias[0]=1 if e==1 else 0
            m.transitions[e].weight.zero_()
            m.transitions[e].bias.zero_()
    h=torch.zeros(1,4,6,dtype=torch.double)
    for e,slope in enumerate(slopes):
        h[0,e,0]=1
        h[0,e,2]=slope
    expected_ids=[[0,1],[0,1],[1,0],[3,2]]
    expected_logits=[[8,3,-11,-13],[8,7,-2,-3],
                     [8,11,7,7],[8,15,16,17]]
    selected_obs=[]
    forced_obs=[]
    nat=[]
    fixed=None
    with torch.no_grad():
        for t in range(4):
            x=torch.zeros(1,6,dtype=torch.double)
            x[0,0]=t
            ids,weights,logits=m.select_source_experts(x,h,return_logits=True)
            assert ids[0].tolist()==expected_ids[t]
            assert logits[0].tolist()==expected_logits[t]
            assert (weights>0).all()
            y,hnew,ids2,signs,w2=m.step(x,h)
            assert torch.equal(ids,ids2)
            assert torch.equal(weights,w2)
            assert torch.equal(signs,torch.ones_like(signs))
            selected_h=h.gather(1,ids.unsqueeze(-1).expand(-1,-1,6))
            replay,updated,rsigns=m.execute_selected(x,ids,selected_h,weights)
            assert torch.allclose(replay,y,atol=1e-12,rtol=1e-12)
            assert torch.equal(signs,rsigns)
            for slot,e in enumerate(ids[0]):
                assert torch.allclose(updated[0,slot],hnew[0,int(e)],
                                      atol=1e-12,rtol=1e-12)
            if fixed is None:fixed=ids.clone()
            fw=torch.softmax(logits.gather(1,fixed),dim=-1)
            fh=h.gather(1,fixed.unsqueeze(-1).expand(-1,-1,6))
            fy,_,_=m.execute_selected(x,fixed,fh,fw)
            v=float(y[0,0])
            selected_obs.append(v)
            forced_obs.append(float(fy[0,0]))
            nat.append(math.floor(10*v))
    assert nat==[0,2,9,0]
    assert selected_obs[0]<selected_obs[1]<selected_obs[2]
    assert selected_obs[3]<selected_obs[2]
    assert all(a<b for a,b in zip(forced_obs,forced_obs[1:]))
    # The original forced selection removes the overload downturn.
    assert forced_obs[-1]>forced_obs[-2]
    print("X369_R4_NATIVE_SOURCE_KINK_PASS",
          "Nat",nat,"native",selected_obs,"fixed",forced_obs)


if __name__=="__main__":
    test_original_source_topk_replacement_kink()
