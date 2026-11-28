"""Retained sophontic reasoning lift over the original neural/SLR source.

Finite geometry mirrors DASHI.Biology.DialecticalSheetSpiralExact and
TrialecticBoundaryFaceNonfactorabilityExact. Actual SLR evidence and
semantic actions remain independently supplied, not predicted by sign.
Original Leech-LILA: Anatolii Kornienko.
"""
from dataclasses import dataclass
from scripts.x369_whole_source_indexed_transport import (
    IndexedJoint, reopen_joint, joint_step, stage13_view)
from scripts.x369_original_receipt_multichart_weld import require

FACES=('reciprocal','coercive','underdetermined')
GEOMETRY=('quarter_turn','reflection','hold')


def trits(x,n):
    require(type(x) is tuple and len(x)==n and all(type(t) is int and t in (-1,0,1) for t in x),
            'typed signed trit carrier')
    return x


@dataclass(frozen=True)
class Spiral:
    sheet: tuple
    height: int
    sense: int

    def __post_init__(self):
        trits(self.sheet,9)
        require(type(self.height) is int and self.height>=0 and
                type(self.sense) is int and self.sense in (-1,1),'historical spiral')


def geometric_move(s,operation):
    require(operation in GEOMETRY,'unowned geometry')
    if operation=='hold':return s
    p=(6,3,0,7,4,1,8,5,2) if operation=='quarter_turn' else (2,1,0,5,4,3,8,7,6)
    return Spiral(tuple(s.sheet[i] for i in p),s.height+1,
                  s.sense if operation=='quarter_turn' else -s.sense)


@dataclass(frozen=True)
class Trialectic:
    edges: tuple
    face: str

    def __post_init__(self):
        trits(self.edges,3)
        require(self.face in FACES,'independent trialectic face')


@dataclass(frozen=True)
class RetainedReasoning:
    joint: IndexedJoint
    spiral: Spiral
    trialectic: Trialectic
    evidence_ref: str


def pack_reasoning(joint,spiral,trialectic,evidence_ref):
    reopen_joint(joint)
    require(isinstance(spiral,Spiral) and isinstance(trialectic,Trialectic) and
            type(evidence_ref) is str and bool(evidence_ref),'source reasoning evidence required')
    return RetainedReasoning(joint,spiral,trialectic,evidence_ref)


def reopen_reasoning(p):
    require(pack_reasoning(p.joint,p.spiral,p.trialectic,p.evidence_ref)==p,'reasoning packet changed')
    return reopen_joint(p.joint),p.spiral,p.trialectic,p.evidence_ref


def reason_step(p,u,operation,step,context_transport,index_transport,semantic_transport):
    """Geometric move and semantic action are independent authorities.

    semantic_transport consumes the actual previous and next contexts,
    source evidence and control; it must independently return typed
    trialectic/evidence. Geometry is not a rule of argument inference.
    """
    require(callable(semantic_transport),'actual governed semantic transport required')
    reopen_reasoning(p)
    moved=geometric_move(p.spiral,operation)
    obs,n=joint_step(p.joint,u,step,context_transport,index_transport)
    face,ev=semantic_transport(p,n,u)
    q=pack_reasoning(n,moved,face,ev)
    reopen_reasoning(q)
    return obs,q


def reasoning_views(p,source_stage,source_pants):
    # The source-owned reader sees reasoning and semantic history, and
    # never reconstructs the trialectic face from the three boundary edges.
    reopen_reasoning(p)
    require(callable(source_stage) and callable(source_pants),'actual source ports required')
    return stage13_view(p.joint,
        lambda h,c,i:source_stage(h,c,i,p.spiral,p.trialectic,p.evidence_ref),
        lambda h,c,i:source_pants(h,c,i,p.spiral,p.trialectic,p.evidence_ref))


def spiral_certificate():
    from itertools import product
    n=0
    for sheet in product((-1,0,1),repeat=9):
        for sense in (-1,1):
            x=Spiral(sheet,0,sense);y=x
            for _ in range(4):y=geometric_move(y,'quarter_turn')
            require(y.sheet==x.sheet and y.height==4 and y.sense==sense,'spiral four-turn')
            y=geometric_move(geometric_move(x,'reflection'),'reflection')
            require(y.sheet==x.sheet and y.height==2 and y.sense==sense,'spiral reflection')
            n+=1
    a=Trialectic((1,1,1),'reciprocal');b=Trialectic((1,1,1),'coercive')
    require(a.edges==b.edges and a!=b,'irreducible face')
    return dict(signed_sheet_sense_cases=n,projected_return_retains_height=True,
                same_boundary_distinct_face=True,source_semantic_authority_created=False,
                actual_learned_reasoning_certified=False)
