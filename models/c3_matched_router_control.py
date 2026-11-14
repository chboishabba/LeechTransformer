"""F3 route-only equal-norm AND equal-displacement control for relative C3.

For each original activation x, the C3 comparator is y=rotate3(x). We
generate a random x' with ||x'||=||x|| and ||x'-x||=||y-x||, to rounding,
so a CE effect is not automatically attributable to perturbation size.
The source-owned activation remains common to all expert routes; only
router feature input is changed. No independent task-label causality is
inferred by this geometric comparator alone.
"""
import torch
from models.trialectic_wave_prior import rotate3


def matched_displacement_random_x(x: torch.Tensor, *, seed: int = 41):
    if x.ndim != 2 or x.shape[-1] < 3 or x.shape[-1] % 3:
        raise ValueError("expected [batch, three-divisible original hidden width]")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError("finite float original activation required")
    xd = x.to(torch.float64)
    target = rotate3(xd)
    sq = (xd * xd).sum(dim=-1, keepdim=True)
    projection = (xd * target).sum(dim=-1, keepdim=True) / sq.clamp_min(1e-100)
    parallel = projection * xd
    perpendicular = target - parallel
    perpendicular_norm = torch.linalg.vector_norm(perpendicular,dim=-1,keepdim=True)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random_direction = torch.randn(x.shape,generator=generator,dtype=torch.float64,device="cpu").to(x.device)
    random_direction = random_direction - (
        (random_direction * xd).sum(dim=-1,keepdim=True)/sq.clamp_min(1e-100)
    ) * xd
    random_norm = torch.linalg.vector_norm(random_direction,dim=-1,keepdim=True)
    if (random_norm[(sq > 1e-100)] < 1e-10).any():
        raise ValueError("ill-conditioned matched random tangent direction")
    result = parallel + perpendicular_norm * random_direction / random_norm.clamp_min(1e-100)
    result = torch.where(sq > 1e-100, result,torch.zeros_like(result))
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite matched displacement control")
    return result.to(x.dtype)
