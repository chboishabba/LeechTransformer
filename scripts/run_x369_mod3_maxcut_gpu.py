"""L6: source-typed nine-point mod3 learning maximum-cut, GPU-first.

Four matched R4 computational arms; additive negative capacity control;
learnable pair-MLP control; exact nonlearned C3 Fourier law positive control.
Fixed horizon; heldout (2,2) never trains or selects a checkpoint.
No original model fingerprint/F5 changes or grokking claim.
"""
import argparse,copy,hashlib,json,math,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from scripts.train_x369_mod3_codec_gpu import Learner
from scripts.audit_x369_typed_mod3_prediction_consumer import (
    TASK_PAIRS,TASK_LABELS,TRAIN_INDICES,HELDOUT_INDEX,typed_mod3_prediction_consumer)
from models.x369_shared_head import select_fused_r4
from models.x369_two_channel_codec import select_two_with_dc
from models.x369_differential_backward import select_vjp_r4

ARMS={"original":None,"shared_head":select_fused_r4,
      "two_dc":select_two_with_dc,"differential_vjp":select_vjp_r4}

class AdditiveLinearControl(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.a=nn.Embedding(3,width);self.b=nn.Embedding(3,width)
        self.readout=nn.Linear(width,3)
    def forward(self,p):
        return self.readout(self.a(p[:,0])+self.b(p[:,1]))

class PairMLPControl(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.a=nn.Embedding(3,width);self.b=nn.Embedding(3,width)
        self.net=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,3))
    def forward(self,p):
        return self.net(torch.cat((self.a(p[:,0]),self.b(p[:,1])),-1))

def exact_fourier_law_logits(p,dtype):
    theta=2*math.pi*(p[:,0].to(dtype)+p[:,1].to(dtype))/3
    centres=2*math.pi*torch.arange(3,device=p.device,dtype=dtype)/3
    return 12*torch.cos(theta[:,None]-centres)

def state_sha(m):
    d=hashlib.sha256()
    for name,v in sorted(m.state_dict().items()):
        t=v.detach().contiguous().cpu()
        for item in (name,str(tuple(t.shape)),str(t.dtype)):
            d.update(item.encode());d.update(b"\\0")
        d.update(t.numpy().tobytes())
    return d.hexdigest()

def classify(rows):
    train=next((v["step"] for v in rows if v["audit"]["training_eight_correct"]),None)
    joint=next((v["step"] for v in rows if v["audit"]["training_eight_correct"] and v["audit"]["heldout_p22_correct"]),None)
    return dict(first_observed_training_fit=train,first_observed_joint_fit=joint,
        delayed_heldout_transition_sampled=bool(train is not None and joint is not None and joint>train),
        grokking_established=False,
        classification="NO_TRAIN_FIT" if train is None else "TRAIN_FIT_ONLY" if joint is None else "JOINT_FIT")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--require-gfx803",action="store_true")
    p.add_argument("--cpu-smoke",action="store_true")
    p.add_argument("--dtype",choices=("float32","float64"),default="float32")
    p.add_argument("--width",type=int,default=384)
    p.add_argument("--experts",type=int,default=4)
    p.add_argument("--steps",type=int,default=2000)
    p.add_argument("--report-every",type=int,default=100)
    p.add_argument("--seeds",type=int,nargs="+",default=[41,42,43])
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--weight-decay",type=float,default=.1)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.width<6 or a.width%3 or a.experts<2 or not a.seeds or len(set(a.seeds))!=len(a.seeds) or min(a.steps,a.report_every)<1 or a.lr<=0 or a.weight_decay<0:p.error("invalid parameters")
    assert tuple(TRAIN_INDICES)==tuple(range(8)) and HELDOUT_INDEX==8
    assert TASK_PAIRS[8]==(2,2) and TASK_LABELS[8]==1
    assert all(TASK_LABELS[i]==sum(TASK_PAIRS[i])%3 for i in range(9))
    if a.cpu_smoke:
        if a.require_gfx803:p.error("gfx803 cannot be CPU smoke")
        device=torch.device("cpu")
    else:
        if not torch.cuda.is_available():raise RuntimeError("GPU required; CPU only with --cpu-smoke")
        device=torch.device("cuda",torch.cuda.current_device())
        prop=torch.cuda.get_device_properties(device)
        if a.require_gfx803 and (not torch.version.hip or not getattr(prop,"gcnArchName","").startswith("gfx803")):raise RuntimeError("gfx803 HIP required")
    dt=getattr(torch,a.dtype)
    pairs=torch.tensor(TASK_PAIRS,device=device,dtype=torch.long)
    labels=torch.tensor(TASK_LABELS,device=device,dtype=torch.long)
    train=torch.tensor(TRAIN_INDICES,device=device,dtype=torch.long)
    law=typed_mod3_prediction_consumer(exact_fourier_law_logits(pairs,dt))
    assert law["full_task_correct"] and law["hard_label_action_defect_count"]==0
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        fp=a.output.open("x",encoding="utf-8")
    else:fp=None
    def emit(v):
        line=json.dumps(v,sort_keys=True,allow_nan=False)
        print(line,flush=True)
        if fp:fp.write(line+"\n");fp.flush()
    try:
        emit(dict(kind="meta",dtype=a.dtype,device=str(device),torch=torch.__version__,hip=torch.version.hip,
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            arch=getattr(torch.cuda.get_device_properties(device),"gcnArchName","") if device.type=="cuda" else None,
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            train_indices=list(TRAIN_INDICES),heldout_index=HELDOUT_INDEX,
            law_positive_control=law,law_is_nonlearned=True,
            checkpoint_selection_by_heldout=False,original_fitted_model=False))
        for seed in a.seeds:
            torch.manual_seed(seed)
            initial=Learner(a.width,a.experts).to(device,dt)
            with torch.no_grad():initial.history.c3_residual_weight.normal_(std=.003)
            initial_hash=state_sha(initial)
            torch.manual_seed(seed+1000003)
            controls={"additive_linear":AdditiveLinearControl(a.width).to(device,dt),
                      "pair_mlp":PairMLPControl(a.width).to(device,dt)}
            work=[(name,copy.deepcopy(initial),selector,True) for name,selector in ARMS.items()]
            work.extend((name,m,None,False) for name,m in controls.items())
            for name,m,selector,is_learner in work:
                if is_learner and state_sha(m)!=initial_hash:raise RuntimeError("initial learner mismatch")
                opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.weight_decay)
                rows=[];m.train()
                if device.type=="cuda":torch.cuda.synchronize(device)
                start=time.perf_counter()
                for step in range(1,a.steps+1):
                    opt.zero_grad(set_to_none=True)
                    logits=m(pairs[train],selector) if is_learner else m(pairs[train])
                    loss=F.cross_entropy(logits,labels[train])
                    if not bool(torch.isfinite(loss)):raise RuntimeError("nonfinite training loss")
                    loss.backward();opt.step()
                    if step==1 or step%a.report_every==0 or step==a.steps:
                        m.eval()
                        with torch.no_grad():
                            all_logits=m(pairs,selector) if is_learner else m(pairs)
                            audit=typed_mod3_prediction_consumer(all_logits)
                            train_ce=float(F.cross_entropy(all_logits[train],labels[train]))
                            predictions=all_logits.argmax(-1).tolist()
                        m.train()
                        if device.type=="cuda":torch.cuda.synchronize(device)
                        row=dict(kind="trajectory",seed=seed,arm=name,step=step,
                            audit=audit,postupdate_train_ce=train_ce,
                            preupdate_train_loss=float(loss.detach()),predictions=predictions,
                            elapsed_seconds=time.perf_counter()-start,
                            learner_initial_state_sha256=initial_hash if is_learner else None)
                        rows.append(row);emit(row)
                emit(dict(kind="summary",seed=seed,arm=name,classification=classify(rows),
                    final=rows[-1]["audit"],horizon=a.steps,
                    source_task="typed_mod3_prediction_consumer",heldout_never_trained=True))
                del opt,m
        emit(dict(kind="completion",status="FIXED_HORIZON_COMPLETE",grokking_claim=False))
    finally:
        if fp:fp.close()

if __name__=="__main__":main()
