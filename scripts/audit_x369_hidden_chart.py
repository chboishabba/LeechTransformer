"""Frozen affine task-to-history chart; fit on training only. No latent uniqueness claim."""
import torch


def fit_chart(task,history,rtol=1e-10):
    y,z=task.detach().double().cpu(),history.detach().double().cpu()
    if y.ndim!=2 or z.ndim!=2 or len(y)!=len(z) or not torch.isfinite(y).all() or not torch.isfinite(z).all():raise ValueError('finite aligned matrices required')
    ym,zm=y.mean(0),z.mean(0);yc=y-ym;zc=z-zm
    rank=int(torch.linalg.matrix_rank(yc,rtol=rtol))
    U=torch.linalg.pinv(yc,rtol=rtol)@zc
    urank=int(torch.linalg.matrix_rank(U,rtol=rtol))
    return dict(task_mean=ym,history_mean=zm,U=U,decoder=torch.linalg.pinv(U,rtol=rtol),task_rank=rank,chart_rank=urank,full_rank=rank==urank==y.shape[1])


def chart_metrics(chart,task,history):
    y,z=task.detach().double().cpu(),history.detach().double().cpu()
    yc=y-chart['task_mean'];zc=z-chart['history_mean'];U=chart['U']
    yh=zc@chart['decoder']+chart['task_mean']
    err=yh-y
    return dict(full_rank=chart['full_rank'],task_rmse=float(err.square().mean().sqrt()),mean_rmse=float(err[:,:3].square().mean().sqrt()),signed_rmse=float(err[:,3:].square().mean().sqrt()),
                dc_rmse=float((err.reshape(-1,2,3).sum(-1)/3**.5).square().mean().sqrt()),relative_history_error=float((zc-yc@U).norm()/zc.norm().clamp_min(1e-30)))


def transport_delta(chart,task,other):
    """Additive latent intervention in fitted chart, retain orthogonal residual."""
    if not chart['full_rank']:raise ValueError('rank-deficient chart cannot certify signed action')
    return (other.detach().double().cpu()-task.detach().double().cpu())@chart['U']
