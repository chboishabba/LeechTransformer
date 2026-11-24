"""Pinned original recurrent-module acceptance, not GPU/trained-model evidence."""
import hashlib
from pathlib import Path
from itertools import product
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from models.geometric_paired_history import PairedHistoryMoE
from models.geometric_fair_residual import fair_arm, FairResidualLinear

SOURCE_BLOBS={'history_orbit_moe.py':'a2a250462ce8e663591cd8a8f784fad9bde1b15d','trialectic_wave_prior.py':'0bea25f6128c7979d95949c1dd8638541f40c15b','punctured_nine_wave_prior.py':'d0d890c17ecaf18ea3efa14833405cd116038511','e8_root_prior.py':'6ee8d30d06fd20d3d7168dcd0030c6cc7290d42e','c3_matched_router_control.py':'2765beeaf20442280653c8b696d7a3abccf7de07'}

def source_identity():
    import models.history_orbit_moe as original
    for path,sha in SOURCE_BLOBS.items():
        b=Path(original.__file__).with_name(path).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==sha,path


def fair_check():
    torch.manual_seed(41);count=0
    for paired,mode in product((False,True),('c3','s3','full','entry')):
        base=HistoryOrbitMoE(d_model=6,expert_width=6,route_geometry='c2_c3_residual').double();base.history_cap=1.
        x=torch.randn(5,6,dtype=torch.float64);h=torch.randn((5,4,2,6) if paired else (5,4,6),dtype=torch.float64)/4
        def run(s):
            if paired:
                a=PairedHistoryMoE(s).step(x,h,train_sign=True);return a.branches,a.history,a.ids,a.signs,a.weights
            return s.step(x,h,train_sign=True)
        expected=run(base);arms=[]
        for null,ternary in product((False,True),(False,True)):
            s=fair_arm(base,mode,null,ternary);arms.append(s);out=run(s)
            assert all(torch.equal(a,b) for a,b in zip(expected,out)),(paired,mode,null,ternary)
            (out[0].square().sum()+out[1].square().sum()).backward()
            assert s.orientation.weight.grad is not None and s.orientation.weight.grad.abs().sum()>0
            assert any(isinstance(l,FairResidualLinear) and l.coefficients.grad is not None for l in s.modules())
            count+=1
        assert len(set(sum(p.numel() for p in m.parameters() if p.requires_grad) for m in arms))==1
        for i in (0,2):
            for a,b in zip(arms[i].parameters(),arms[i+1].parameters()):
                assert (a.grad is None)==(b.grad is None)
                if a.grad is not None:assert torch.equal(a.grad,b.grad)
    print('PASS actual original-source common-function/gradient arms',count)

if __name__=='__main__':
    torch.set_num_threads(1);source_identity()
    from scripts.check_lila_history_adapter import main as adapter
    from scripts.check_lila_paired_history import main as paired,original_factory as paired_factory
    from scripts.check_lila_residual_observability import main as residual,original_factory as residual_factory
    adapter();paired(paired_factory);residual(residual_factory);fair_check()
