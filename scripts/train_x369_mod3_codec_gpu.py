"""GPU-only matched finite mod3 prediction experiment, not original fitted LILA.
Task labels/split/action are owned by the existing typed consumer audit.
No heldout loss, checkpoint selection, or F5 inputs.
"""
import argparse,copy,json,time
import torch
from torch import nn
from torch.nn import functional as F
from models.history_orbit_moe import HistoryOrbitMoE
from models.x369_shared_head import select_fused_r4
from models.x369_two_channel_codec import select_two_with_dc
from models.x369_differential_backward import select_vjp_r4
from scripts.audit_x369_typed_mod3_prediction_consumer import (
    TASK_PAIRS,TASK_LABELS,TRAIN_INDICES,HELDOUT_INDEX,typed_mod3_prediction_consumer)

class Learner(nn.Module):
    def __init__(self,width,experts):
        super().__init__()
        self.embed=nn.Embedding(3,width)
        self.history=HistoryOrbitMoE(d_model=width,num_experts=experts,top_k=2,route_geometry="c2_c3_residual")
        self.readout=nn.Linear(width,3)
    def forward(self,pairs,selector=None):
        # Process a and b sequentially with independent zero initial histories
        # for each example; never carry training examples into the heldout state.
        h=self.embed.weight.new_zeros(pairs.shape[0],self.history.num_experts,self.history.d_model)
        for t in range(2):
            x=self.embed(pairs[:,t])
            if selector is None:
                y,h,ids,signs,w=self.history.step(x,h,train_sign=self.training)
            else:
                ids,w=selector(self.history,x,h)
                idx=ids.unsqueeze(-1).expand(-1,-1,self.history.d_model)
                y,updated,signs=self.history.execute_selected(x,ids,h.gather(1,idx),w,train_sign=self.training)
                h=h.scatter(1,idx,updated)
        return self.readout(y+x)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--require-gfx803",action="store_true")
    p.add_argument("--dtype",choices=("float32","float64"),default="float32")
    p.add_argument("--width",type=int,default=384)
    p.add_argument("--experts",type=int,default=4)
    p.add_argument("--steps",type=int,default=2000)
    p.add_argument("--report-every",type=int,default=100)
    p.add_argument("--seeds",type=int,nargs="+",default=[41,42,43])
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--weight-decay",type=float,default=.1)
    a=p.parse_args()
    if a.width<3 or a.width%3 or a.experts<2 or min(a.steps,a.report_every)<1 or a.lr<=0 or a.weight_decay<0:p.error("bad parameters")
    if not torch.cuda.is_available():raise RuntimeError("GPU required; no CPU fallback")
    dev=torch.device("cuda",torch.cuda.current_device())
    prop=torch.cuda.get_device_properties(dev); arch=getattr(prop,"gcnArchName","")
    if a.require_gfx803 and (not torch.version.hip or not arch.startswith("gfx803")):raise RuntimeError("gfx803 required")
    dt=getattr(torch,a.dtype)
    pairs=torch.tensor(TASK_PAIRS,device=dev); labels=torch.tensor(TASK_LABELS,device=dev)
    train=torch.tensor(TRAIN_INDICES,device=dev)
    assert HELDOUT_INDEX not in TRAIN_INDICES
    arms={"original":None,"shared":select_fused_r4,"two_dc":select_two_with_dc,"vjp":select_vjp_r4}
    print(json.dumps(dict(kind="finite_mod3_prediction",arch=arch,device=prop.name,hip=torch.version.hip,torch=torch.__version__,args=vars(a),train=list(TRAIN_INDICES),heldout=HELDOUT_INDEX)),flush=True)
    for seed in a.seeds:
        torch.manual_seed(seed)
        initial=Learner(a.width,a.experts).to(dev,dt)
        with torch.no_grad():initial.history.c3_residual_weight.normal_(std=.003)
        for name,selector in arms.items():
            model=copy.deepcopy(initial); optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
            first_train=None; first_heldout=None
            torch.cuda.synchronize(dev); start=time.perf_counter()
            for t in range(1,a.steps+1):
                optimizer.zero_grad(set_to_none=True)
                logits=model(pairs[train],selector)
                loss=F.cross_entropy(logits,labels[train])
                if not torch.isfinite(loss):raise RuntimeError("nonfinite train loss")
                loss.backward(); optimizer.step()
                if t==1 or t%a.report_every==0 or t==a.steps:
                    model.eval()
                    with torch.no_grad(): receipt=typed_mod3_prediction_consumer(model(pairs,selector))
                    model.train()
                    if receipt["training_eight_correct"] and first_train is None:first_train=t
                    if receipt["training_eight_correct"] and receipt["heldout_p22_correct"] and first_heldout is None:first_heldout=t
                    torch.cuda.synchronize(dev)
                    print(json.dumps(dict(seed=seed,arm=name,step=t,preupdate_train_loss=float(loss.detach()),audit=receipt,first_sampled_train_fit=first_train,first_sampled_joint_fit=first_heldout,elapsed_seconds=time.perf_counter()-start),allow_nan=False),flush=True)
            del optimizer,model
    print(json.dumps(dict(status="COMPLETED_NOT_A_GROKKING_CLAIM",fixed_horizon=a.steps)),flush=True)
if __name__=="__main__":main()
