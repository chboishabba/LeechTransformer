"""C3 output extension controls. Supplied symmetry is NOT learned recognition."""
import torch

def rotate_rows(logits,shift):
    if logits.ndim!=2 or logits.shape[1]!=3 or shift.shape!=(len(logits),):raise ValueError('invalid C3 carrier')
    ix=(torch.arange(3,device=logits.device)[None,:]-shift[:,None])%3
    return logits.gather(1,ix)

def orbit_forward(model,pairs,orientation=1):
    if orientation not in (-1,1):raise ValueError('signed orientation required')
    # Exactly one forward per canonical representative, all three observed in train.
    b=torch.arange(3,device=pairs.device,dtype=pairs.dtype)
    canonical=torch.stack((torch.zeros_like(b),b),dim=1)
    return rotate_rows(model(canonical).index_select(0,pairs[:,1]),orientation*pairs[:,0])

def training_edges(pairs):
    """Only edges with BOTH endpoints present; never infer missing logits."""
    lookup={tuple(x):i for i,x in enumerate(pairs.detach().cpu().tolist())}
    if len(lookup)!=len(pairs):raise ValueError('duplicate pairs')
    e=[]
    for (a,b),i in lookup.items():
        for dst in (((a+1)%3,b),(a,(b+1)%3)):
            if dst in lookup:e.append((i,lookup[dst]))
    return torch.tensor(e,device=pairs.device,dtype=torch.long).reshape(-1,2)

def edge_loss(logits,edges):
    if not len(edges):raise ValueError('empty train edge set')
    z=logits-logits.mean(-1,keepdim=True)
    return (z[edges[:,1]]-z[edges[:,0]].roll(1,-1)).square().mean()

def transport_metrics(logits,orientation=1):
    """Full nine-point evaluation only, strict label margins and anchored defects."""
    if logits.shape!=(9,3) or not torch.isfinite(logits).all():raise ValueError('canonical finite logits required')
    x=logits.double().reshape(3,3,3)
    y=(torch.arange(3,device=x.device)[:,None]+torch.arange(3,device=x.device)[None,:])%3
    selected=x.gather(2,y[...,None]).squeeze(-1)
    other=x.masked_fill(torch.nn.functional.one_hot(y,3).bool(),-torch.inf).amax(-1)
    margin=selected-other
    edge=(x.roll(-1,0)-x.roll(orientation,-1)).abs().amax(-1)
    # two-edge path (0,2)->(1,2)->(2,2); margin sufficient for intended + action only
    paid=float(edge[0,2]+edge[1,2]);m=float(margin[0,2])
    return dict(margin=margin.tolist(),first_action_max_error=float(edge.max()),second_action_max_error=float((x.roll(-1,1)-x.roll(1,-1)).abs().max()),p22_path_error=paid,p02_margin=m,p22_margin=float(margin[2,2]),intended_margin_certificate=bool(orientation==1 and m>2*paid),strict_all_correct=bool((margin>0).all()),supplied_orientation=orientation)
