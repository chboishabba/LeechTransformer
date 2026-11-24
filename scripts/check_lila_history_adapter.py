"""Actual original HistoryOrbitMoE adapter acceptance; not executed in authoring environment."""
from itertools import product
import torch
from torch import nn
from models.geometric_cross_triads import CrossTriadBitLinear
from models.geometric_history_adapter import adapt_history_experts
from models.history_orbit_moe import HistoryOrbitMoE

def compare(a,b):
    assert len(a)==len(b)
    for i,(x,y) in enumerate(zip(a,b)):
        if not x.is_floating_point():
            assert torch.equal(x,y),i
        else:
            assert torch.allclose(x,y,atol=1e-11,rtol=1e-11),i

def main():
    torch.manual_seed(42)
    count=0
    for mode,rank,cap,geometry in product(('c3','s3','full'),(None,3),(None,1.0),('c2_c3_residual','raw')):
        d=6
        src=HistoryOrbitMoE(d_model=d,num_experts=4,top_k=2,expert_width=12,route_geometry=geometry,transition_rank=rank).double().eval()
        src.history_cap=cap
        witnesses={}
        with torch.no_grad():
            for name,layer in src.named_modules():
                if not isinstance(layer,nn.Linear) or name.split('.')[0] not in ('experts','transitions'):
                    continue
                tmp=CrossTriadBitLinear(layer.in_features,layer.out_features,mode=mode,scale=.125,bias=layer.bias is not None,dtype=torch.float64)
                tmp.coefficients.copy_(torch.randint(-1,2,tmp.coefficients.shape).double())
                layer.weight.copy_(tmp.dense_weight())
                witnesses[name]=tmp.coefficients.detach().clone()
            # Fixed strict biases and nonzero learned direction for this acceptance;
            # actual source router still evaluated, not replaced.
            src.expert_bias.copy_(torch.tensor([12.,4.,-4.,-12.],dtype=torch.float64))
        dst=adapt_history_experts(src,witnesses,mode=mode,scale=.125)
        x=torch.randn(3,4,d,dtype=torch.float64)/8
        h=torch.randn(3,4,d,dtype=torch.float64)/8
        compare(src(x,h),dst(x,h))
        a=src.step(x[:,0],h); b=dst.step(x[:,0],h)
        compare(a,b)
        assert torch.equal(a[2],b[2]) and torch.equal(a[3],b[3])
        ids,w=src.select_source_experts(x[:,0],h)
        selected=h.gather(1,ids[...,None].expand(-1,-1,d))
        compare(src.execute_selected(x[:,0],ids,selected,w),dst.execute_selected(x[:,0],ids,selected,w))
        compare(src.execute_selected(x[:,0],ids,selected),dst.execute_selected(x[:,0],ids,selected))
        try:
            adapt_history_experts(src,{k:v for k,v in witnesses.items() if k!=next(iter(witnesses))},mode=mode,scale=.125)
        except ValueError: pass
        else: raise AssertionError('incomplete witness accepted')
        with torch.no_grad():
            layer=src.get_submodule(next(iter(witnesses)))
            layer.weight[0,0]+=.01
        try: adapt_history_experts(src,witnesses,mode=mode,scale=.125)
        except ValueError: pass
        else: raise AssertionError('mismatched source accepted')
        count+=1
    print('PASS actual source adapter combinations',count)

if __name__=='__main__': main()
