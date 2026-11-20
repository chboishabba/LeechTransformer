"""GPU-only original/fused routing and full-step benchmark; forward inference only."""
import argparse,json,statistics,time
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from models.x369_shared_head import select_fused_r4
from scripts.check_x369_shared_head_consumer import fused_step

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--width",type=int,default=384)
    p.add_argument("--experts",type=int,default=4)
    p.add_argument("--repeats",type=int,default=7)
    p.add_argument("--iterations",type=int,default=30)
    p.add_argument("--warmup",type=int,default=5)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--require-gfx803",action="store_true")
    a=p.parse_args()
    if min(a.batch,a.width,a.repeats,a.iterations,a.warmup)<1 or a.experts<2 or a.width%3:
        p.error("invalid benchmark shape or count")
    dev=torch.device(a.device)
    if dev.type!="cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    dev=torch.device("cuda",torch.cuda.current_device() if dev.index is None else dev.index)
    prop=torch.cuda.get_device_properties(dev)
    arch=getattr(prop,"gcnArchName","")
    if a.require_gfx803 and (not torch.version.hip or not arch.startswith("gfx803")):
        raise RuntimeError("gfx803 HIP required")
    torch.manual_seed(42)
    m=HistoryOrbitMoE(d_model=a.width,num_experts=a.experts,top_k=2,route_geometry="c2_c3_residual").to(dev).eval()
    x=torch.randn(a.batch,a.width,device=dev)
    h=torch.randn(a.batch,a.experts,a.width,device=dev)
    sync=lambda:torch.cuda.synchronize(dev)
    def measure(fn):
        sync()
        baseline=torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        t=time.perf_counter()
        for _ in range(a.iterations): fn()
        sync()
        return dict(ms=1000*(time.perf_counter()-t)/a.iterations,
                    peak_increment_bytes=torch.cuda.max_memory_allocated(dev)-baseline)
    with torch.no_grad():
        m.c3_residual_weight.normal_(std=.003)
        for scope,orig,fused in (
            ("selection",lambda:m.select_source_experts(x,h),lambda:select_fused_r4(m,x,h)),
            ("full_step",lambda:m.step(x,h),lambda:fused_step(m,x,h))):
            u,v=orig(),fused()
            for left,right in zip(u,v):
                if left.dtype==torch.long: assert torch.equal(left,right)
                else: torch.testing.assert_close(left,right,atol=1e-5,rtol=1e-5)
            del u,v
            for _ in range(a.warmup): orig(); fused()
            sync()
            samples={"original":[],"fused":[]}
            for i in range(a.repeats):
                order=[("original",orig),("fused",fused)]
                if i%2: order.reverse()
                for name,fn in order: samples[name].append(measure(fn))
            summary={name:dict(median_ms=statistics.median(v["ms"] for v in vals),
                         peak_increment_bytes=max(v["peak_increment_bytes"] for v in vals))
                     for name,vals in samples.items()}
            print(json.dumps(dict(scope=scope,device=prop.name,arch=arch,hip=torch.version.hip,
                  torch=torch.__version__,args=vars(a),samples=samples,summary=summary,
                  speedup=summary["original"]["median_ms"]/summary["fused"]["median_ms"],
                  inference_only=True,kernel_count_measured=False)),flush=True)
if __name__=="__main__":main()
