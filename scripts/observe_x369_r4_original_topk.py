"""Capture the ORIGINAL R4 router's actual logits without changing its bytes.

The fitted R4 residual delta binds a geometry SHA256 covering
models/history_orbit_moe.py, among other producer sources. Editing the
model merely to expose logits would invalidate already-issued deltas.

Use a scoped PyTorch dispatcher observation of aten.topk during the
real HistoryOrbitMoE.select_source_experts call. The operation and
source model are unchanged; no surrogate routing is computed.

Only the source's standard router_selection='topk' is supported.
The nonstandard sort fallback requires a distinct explicit contract.
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class _SourceTopKObservation(TorchDispatchMode):
    def __init__(self, batch, experts, k):
        super().__init__()
        self.batch = batch
        self.experts = experts
        self.k = k
        self.captured = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.topk.default:
            kw = kwargs or {}
            tensor = args[0]
            requested_k = args[1] if len(args) > 1 else kw.get("k")
            dimension = args[2] if len(args) > 2 else kw.get("dim", -1)
            if (tensor.ndim == 2 and tuple(tensor.shape) ==
                (self.batch, self.experts) and requested_k == self.k
                and dimension in (-1, 1)):
                # Snapshot the real original logits before topk, without
                # changing source tensor, dtype, device or selection.
                self.captured.append(tensor.detach().clone())
        return func(*args, **(kwargs or {}))


def source_selected_with_logits(moe, x, history):
    """Return original ids, weights, captured logits.

    This is a consumer observer, NOT authentication or an alternative
    router. It refuses unknown fallback selection and unexpected
    dispatch patterns, including missing/duplicate observed topk calls.
    """
    if moe.router_selection != "topk":
        raise ValueError("original R4 source logit observer requires topk")
    if x.ndim != 2 or tuple(history.shape) != (
            x.shape[0], moe.num_experts, moe.d_model):
        raise ValueError("source x/all-E history shape mismatch")
    observation = _SourceTopKObservation(
        x.shape[0], moe.num_experts, moe.top_k)
    with observation:
        ids, weights = moe.select_source_experts(x, history)
    if len(observation.captured) != 1:
        raise RuntimeError(
            "actual original topk logits not uniquely observed")
    logits = observation.captured[0]
    if (tuple(logits.shape) != (x.shape[0], moe.num_experts)
        or logits.device != x.device or logits.dtype != x.dtype
        or ids.shape != (x.shape[0], moe.top_k)
        or weights.shape != ids.shape
        or not bool(torch.isfinite(logits).all())):
        raise RuntimeError("source topk logits/outputs invalid")
    with torch.no_grad():
        values = logits.gather(1, ids)
        reference_weights = torch.softmax(values, dim=-1)
        if not torch.equal(reference_weights, weights):
            raise RuntimeError("original captured logits/weights disagree")
    return ids, weights, logits
