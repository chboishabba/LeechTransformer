"""Original R4 native execution contract (no model-source modification).

CPU is the historical portable fixture mode. gfx803 requires actual ROCm
and the selected device; there is never a silent fallback. Deserialization
and source hashes remain on CPU; actual inference is on the declared device.
"""
import torch


def resolve_device(mode):
    if mode=='cpu':
        return torch.device('cpu')
    if mode!='gfx803':
        raise ValueError('R4 device must be cpu or gfx803')
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('native gfx803 HIP required; no CPU fallback')
    if not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803'):
        raise RuntimeError('native gfx803 device required; no CPU fallback')
    return torch.device('cuda:0')


def runtime(device):
    device=torch.device(device)
    if device.type=='cpu':
        return dict(mode='cpu',torch=str(torch.__version__),hip=None,arch=None)
    if device.type!='cuda' or device.index not in (None,0):
        raise ValueError('unexpected original source device')
    resolve_device('gfx803')
    return dict(mode='gfx803',torch=str(torch.__version__),hip=torch.version.hip,
                arch=torch.cuda.get_device_properties(0).gcnArchName)


def tensor_bytes(t):
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def bitwise_equal(a,b):
    return a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and tensor_bytes(a)==tensor_bytes(b)


def verify_trace_runtime(request,layer_receipt,device):
    actual=runtime(device)
    # Historic CPU receipts had no execution metadata. They are never
    # admitted as proof of a native GPU capture.
    for name,record in (('source trace',request),('layer export',layer_receipt)):
        if record is None:continue
        claimed=record.get('source_execution_runtime')
        if claimed is None:
            if actual['mode']!='cpu':
                raise ValueError(name+' missing original gfx803 execution identity')
        elif claimed!=actual:
            raise ValueError(name+' original execution runtime mismatch')
    return actual
