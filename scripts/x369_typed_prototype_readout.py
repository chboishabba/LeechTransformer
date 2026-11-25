"""Frame-free typed polynomial readout of a DECLARED decoded (M,D) triad.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
No state decoder, supplied-action register, task label bypass, checkpoint
migration or learned semantic claim. Reuses original task alphabets.
"""
import torch
from torch import nn
from models.geometric_bitlinear import generated_369
from scripts.audit_x369_ordered_signed_task import TRIADS
from scripts.train_x369_ordered_action import GRAMS


class TypedPrototypeReadout(nn.Module):
    """Nearest signed-vector / rank-one Gram prototypes, up to common offsets.
    Input must be independently decoded from the actual recurrent state.
    Ideal-real polynomial equivariance for all finite real triads.
    Exact unique argmax for true ternary triads (including zero).
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("triads",torch.tensor(TRIADS))
        self.register_buffer("grams",torch.tensor(GRAMS))
        self.register_buffer("metric",torch.tensor((1,1,1,2,2,2)))

    def forward(self,mean,residual):
        if (mean.shape!=residual.shape or mean.ndim<1 or mean.shape[-1]!=3
            or not mean.is_floating_point() or mean.dtype!=residual.dtype
            or mean.device!=residual.device):
            raise ValueError("matching real decoded triads required")
        t=self.triads.to(mean);g=self.grams.to(mean);metric=self.metric.to(mean)
        m=2*(mean@t.T)-t.square().sum(-1)
        d=2*(residual@t.T)-t.square().sum(-1)
        # generated_369: (x,y,z,x²,y²,z²,xy,yz,zx).
        # Original independent task gram: (x²,y²,z²,yz,zx,xy).
        q=generated_369(residual)[...,[3,4,5,7,8,6]]
        gram=2*((q*metric)@g.T)-(g.square()*metric).sum(-1)
        return m,gram,d
