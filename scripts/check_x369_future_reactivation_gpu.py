"""GPU-only native future-reactivation non-descent witness.
Two full histories -> identical present selected-only packet -> different future route.
Uses original HistoryOrbitMoE parameters and transition (synthetic fixture).
Not a general hysteresis or fitted-model theorem.
"""
import argparse,json,torch
from models.history_orbit_moe import HistoryOrbitMoE

def run(device,dtype):
    m=HistoryOrbitMoE(d_model=6,num_experts=4,top_k=2,route_geometry="c2_c3_residual",router_selection="topk").to(device,dtype).eval()
    with torch.no_grad():
        for p in m.parameters():p.zero_()
        m.c3_residual_weight[0,6]=1
        m.expert_bias.copy_(torch.tensor([8,3,-11,-16],dtype=dtype,device=device))
        m.orientation.weight[0,6]=1
        m.experts[1][2].bias[0]=1
    a=torch.zeros(1,4,6,device=device,dtype=dtype)
    a[0,:,0]=1
    a[0,:,2]=torch.tensor([0,4,9,11],device=device,dtype=dtype)
    b=a.clone(); b[0,2,2]=0 # inactive e2; selected packet must be unchanged
    x0=torch.zeros(1,6,device=device,dtype=dtype)
    x2=x0.clone(); x2[0,0]=2
    with torch.no_grad():
        ids_a,weights_a=m.select_source_experts(x0,a)
        ids_b,weights_b=m.select_source_experts(x0,b)
        assert ids_a.tolist()==ids_b.tolist()==[[0,1]]
        assert torch.equal(weights_a,weights_b)
        idx=ids_a.unsqueeze(-1).expand(-1,-1,6)
        assert torch.equal(a.gather(1,idx),b.gather(1,idx))
        first_a=m.step(x0,a);first_b=m.step(x0,b)
        assert torch.equal(first_a[2],first_b[2])
        assert torch.equal(first_a[4],first_b[4])
        # Current selected histories, signs, outputs coincide; inactive state differs.
        assert torch.equal(first_a[0],first_b[0])
        assert torch.equal(first_a[3],first_b[3])
        assert torch.equal(first_a[1].gather(1,idx),first_b[1].gather(1,idx))
        second_a=m.step(x2,first_a[1]);second_b=m.step(x2,first_b[1])
        assert second_a[2].tolist()==[[0,2]],second_a[2].tolist()
        assert second_b[2].tolist()==[[3,0]] or second_b[2].tolist()==[[0,3]],second_b[2].tolist()
        assert not torch.equal(second_a[2],second_b[2])
    return dict(kind="original_source_future_reactivation_non_descent",
                current_ids=ids_a.tolist(),present_packet_equal=True,
                future_a_ids=second_a[2].tolist(),future_b_ids=second_b[2].tolist(),
                stable_fibres_refuted_for_selected_only=True,
                general_future_sufficiency_certified=False,
                fitted_history_claim=False,source_original=True)
def main():
    p=argparse.ArgumentParser();p.add_argument("--require-gfx803",action="store_true");p.add_argument("--dtype",choices=("float32","float64"),default="float64")
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("GPU required, no CPU fallback")
    dev=torch.device("cuda",torch.cuda.current_device());prop=torch.cuda.get_device_properties(dev)
    arch=getattr(prop,"gcnArchName","")
    if a.require_gfx803 and (not torch.version.hip or not arch.startswith("gfx803")):
        raise RuntimeError("gfx803 required")
    result=run(dev,getattr(torch,a.dtype))
    result.update(arch=arch,hip=torch.version.hip,torch=torch.__version__)
    print(json.dumps(result,sort_keys=True),flush=True)
if __name__=="__main__":main()
