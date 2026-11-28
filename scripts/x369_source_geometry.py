"""Opt-in typed ordered source geometry, not a new training campaign.

Original Leech-LILA: Anatolii Kornienko. Reuse original source step and
independently owned task labels. The supplied action register is not a
learned task prediction. The neural readout is a separate consumer.
"""
import torch
from torch import nn
from models.history_orbit_moe import HistoryOrbitMoE
from models.geometric_reflection_fibre import GROUP, act, inv
from models.geometric_orbit_descent import (
    OrbitDescentHistoryMoE, apply_group_indices, batch_where)
from scripts.train_x369_ordered_action import TRIADS, GRAMS, HEADS
from scripts.audit_x369_ordered_signed_task import gram


def literal_controls(ops):
    if ops.ndim!=3 or ops.shape[-1]!=7 or not ((ops==0)|(ops==1)).all():
        raise ValueError("expected original one-hot literal and guard")
    a,b=ops[...,:4],ops[...,4:]
    if not ((a.sum(-1)==1)&(b.sum(-1)==1)).all():
        raise ValueError("one literal and one guard required")
    tag,guard=a.argmax(-1),b.argmax(-1)
    if ((tag>=2)&(guard>1)).any():raise ValueError("invalid original guard")
    # Original positive translation is inverse of the donor's left roll.
    # Original reflection fixes coordinate 0: left-roll^2 * reversal.
    c=torch.where(tag<2,4*((-guard)%3),
        torch.where(tag==2,10*guard,guard))
    return c,tag


def signed_seed(seed,experts,width):
    if seed.ndim!=2 or seed.shape[1]!=9 or not torch.isfinite(seed).all() or not (seed[:,6:]==0).all():
        raise ValueError("original signed seed (M,D,zero padding) required")
    if experts<2 or width<3 or width%3:raise ValueError("complete triads required")
    m,d=seed[:,:3],seed[:,3:6]
    h=seed.new_zeros(len(seed),experts,2,width)
    h[:,0,0,:3]=m+d;h[:,0,1,:3]=m-d
    return h


def label_permutations():
    # Gram is the sign-forgetting image of the original signed triad.
    # The class action must not depend on which preimage is chosen.
    result=[]
    for g in GROUP:
        mean={v:act(g,v) for v in TRIADS}
        signed={v:tuple((-1)**g[1]*z for z in act(g,v)) for v in TRIADS}
        quadratic={}
        for v,w in signed.items():
            k,q=gram(v),gram(w)
            if k in quadratic and quadratic[k]!=q:raise ValueError("Gram action does not descend")
            quadratic[k]=q
        row=[]
        for labels,transform in ((TRIADS,mean),(GRAMS,quadratic),(TRIADS,signed)):
            dst=[labels.index(transform[v]) for v in labels]
            if sorted(dst)!=list(range(len(labels))):raise ValueError("not a label permutation")
            row.append(tuple(dst.index(i) for i in range(len(labels))))
        result.append(tuple(row))
    return tuple(result)


class SourceGeometryOrderedLearner(nn.Module):
    """Ideal-real naturality on (signed seed, conjugation-closed controls).

    Literal tag is invariant, group operation is conjugated. Source step is
    shared, evaluated once in a joint canonical frame, with explicit
    stabilizer projection. No claim of original dense-function equality,
    training convergence, or exact task law for the learned neural heads.
    """
    def __init__(self,width=12,experts=4,*,expert_width=None,typed_consumer=False):
        super().__init__()
        self.width,self.experts=width,experts
        self.expert_width=width if expert_width is None else expert_width
        if not isinstance(self.expert_width,int) or self.expert_width<1:
            raise ValueError('positive integer expert width required')
        self.typed_consumer=bool(typed_consumer)
        source=HistoryOrbitMoE(d_model=width,num_experts=experts,top_k=2,
            expert_width=self.expert_width,route_geometry="c2_c3_residual")
        self.history=OrbitDescentHistoryMoE(source,paired=True)
        self.control_embed=nn.Embedding(4*len(GROUP),width)
        self.readout=nn.Linear(2*(experts+1)*width,sum(HEADS))
        if self.typed_consumer:
            from scripts.x369_typed_prototype_readout import TypedPrototypeReadout
            self.prototype=TypedPrototypeReadout(persistent=False)
        p=label_permutations()
        for i in range(3):
            self.register_buffer(f"label_perm_{i}",torch.tensor([x[i] for x in p]),persistent=False)

    def consume(self,out):
        if self.typed_consumer:
            # Constructive state contract only: consume the actual world
            # recurrence, not the independent known_action_register.
            from scripts.x369_constructive_finite_source import decode_history
            m,d=decode_history(out.history)
            return self.prototype(m,d)
        return self.raw_affine_consume(out)

    def raw_affine_consume(self,out):
        """Original local affine classifier; not admitted by constructive mode."""
        local=self.readout(torch.cat((out.local_output.flatten(1),
            out.local_history.flatten(1)),1)).split(HEADS,-1)
        result=[]
        for j,p in enumerate(local):
            table=getattr(self,f"label_perm_{j}")
            total=torch.zeros_like(p)
            for i in range(len(GROUP)):
                total=total+batch_where(out.stabilizer[:,i],p[:,table[i]],torch.zeros_like(p))
            projected=total/out.stabilizer.sum(1,keepdim=True).to(p.dtype)
            world=torch.zeros_like(p)
            for i,g in enumerate(GROUP):
                world=batch_where(out.world_to_local==i,
                    projected[:,table[GROUP.index(inv(g))]],world)
            result.append(world)
        return tuple(result)

    def rollout_group(self,seed,controls,tags):
        if controls.ndim!=2 or controls.shape!=tags.shape or controls.shape[0]!=len(seed) or not controls.shape[1] or controls.dtype!=torch.long or tags.dtype!=torch.long or not ((tags>=0)&(tags<4)).all():
            raise ValueError("indexed operation and literal tag sequences required")
        h=signed_seed(seed,self.experts,self.width)
        trace=[]
        for t in range(controls.shape[1]):
            tag=tags[:,t]
            out=self.history.step_controlled(controls[:,t],h,
                lambda c:self.control_embed(tag*len(GROUP)+c),train_sign=self.training)
            trace.append(out);h=out.history
        return self.consume(trace[-1]),trace

    def forward(self,seed,ops):
        controls,tags=literal_controls(ops)
        return self.rollout_group(seed,controls,tags)[0]


def known_action_register(seed,controls):
    """Independent supplied transport, not trainable output or label leakage."""
    if seed.ndim!=2 or seed.shape[1]!=9 or not (seed[:,6:]==0).all():
        raise ValueError("signed seed required")
    if controls.ndim!=2 or controls.shape[0]!=len(seed):raise ValueError("operation shape")
    m,d=seed[:,:3],seed[:,3:6]
    for t in range(controls.shape[1]):
        c=controls[:,t]
        m=apply_group_indices(m,c)
        d=apply_group_indices(d,c)*torch.where((c//2)%2==1,-1,1)[:,None]
    a,b,c=d.unbind(-1)
    return m,torch.stack((a*a,b*b,c*c,b*c,c*a,a*b),-1),d
