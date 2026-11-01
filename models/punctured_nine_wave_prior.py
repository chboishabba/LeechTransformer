"""DASHI punctured-nine / wave Fourier prior for a 384D learner.

Exact structural split:
    d=384 = 16*3*(3**2-1) = 8 heads * 2 blocks * 3 lanes * 8.
Each eight is the punctured balanced-ternary plane T^2\{(0,0)}
with four antipodal point pairs (DASHI OggSSPP2BalancedTernaryPuncturedPlaneExact).

An eight-point *spatial puncture* and the eight *non-DC Fourier
coefficients* are distinct vector spaces. The explicit invertible bridge
embeds the punctured amplitudes with origin=0, discards its constant mode,
and uses an orthonormal real Fourier basis on the nine-point group F3^2.
The inverse recovers the punctured values by subtracting the recovered
origin. This is NOT a unitary map of the naïve spatial eight-norm.

The Fourier representation carries an exact mathematical F3^2 translation
action and four real conjugate frequency pairs. Global C2 amplitude
negation, Fourier conjugation under site inversion, and semantic argument
opposition are different transformations. No arbitrary ambient action is
asserted to be a Leech lattice automorphism or an analytic j action.

Historical mathematical ownership: DASHI/Moonshine/
OggSSPP2BalancedTernaryPuncturedPlaneExact.agda;
Foundations/Base369FaceSheetPunctureIncidenceIdentityExact.agda;
Moonshine/C3FourierConjugationExact.agda. Original Leech-LILA
architecture: Anatolii Kornienko (DOI 10.5281/zenodo.18784424).
"""
from __future__ import annotations
import math
import torch

# Exact source-owner ordering, with a distinguished central point.
PUNCTURE_SITES = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (1, 1), (-1, 1), (1, -1),
)
SITES = ((0, 0),) + PUNCTURE_SITES
# One representative from each nonzero character pair {k,-k}.
FREQUENCY_PAIRS = ((1, 0), (0, 1), (1, 1), (1, -1))


def real_fourier9_basis(*, device=None, dtype=torch.float64) -> torch.Tensor:
    """[9,8] orthonormal real basis of the zero-mean 9-sheet subspace."""
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("floating real Fourier dtype required")
    rows=[]
    normalization=math.sqrt(2.0/9.0)
    for x,y in SITES:
        terms=[]
        for k,l in FREQUENCY_PAIRS:
            phase=2.0*math.pi*((k*x+l*y)%3)/3.0
            terms.extend((normalization*math.cos(phase),
                          normalization*math.sin(phase)))
        rows.append(terms)
    return torch.tensor(rows,device=device,dtype=dtype)


def puncture8_to_wave8(x: torch.Tensor) -> torch.Tensor:
    """Explicit invertible pointed-puncture to non-DC-wave coordinate map."""
    if x.shape[-1]!=8:
        raise ValueError("expected eight punctured coordinates")
    zero=torch.zeros_like(x[..., :1])
    full=torch.cat((zero,x),dim=-1)
    basis=real_fourier9_basis(device=x.device,dtype=x.dtype)
    return full @ basis


def wave8_to_puncture8(waves: torch.Tensor) -> torch.Tensor:
    """Undo Fourier rechart AND recover the original punctured values."""
    if waves.shape[-1]!=8:
        raise ValueError("expected eight nonconstant Fourier coordinates")
    basis=real_fourier9_basis(device=waves.device,dtype=waves.dtype)
    centred=waves @ basis.T
    return centred[...,1:] - centred[..., :1]


def wave8_to_nine_zero_mean(waves: torch.Tensor) -> torch.Tensor:
    if waves.shape[-1]!=8:
        raise ValueError("expected eight nonconstant Fourier coordinates")
    basis=real_fourier9_basis(device=waves.device,dtype=waves.dtype)
    return waves @ basis.T


def translate_wave8(waves: torch.Tensor, shift: tuple[int,int]) -> torch.Tensor:
    """The Z3 x Z3 translation representation on four real phase pairs."""
    if waves.shape[-1]!=8:
        raise ValueError("expected eight Fourier coordinates")
    if len(shift)!=2 or not all(isinstance(z,int) for z in shift):
        raise ValueError("translation is a pair of integers modulo three")
    pair=waves.reshape(*waves.shape[:-1],4,2)
    a,b=pair.unbind(-1)
    angles=torch.tensor(
        [2*math.pi*((k*shift[0]+l*shift[1])%3)/3
         for k,l in FREQUENCY_PAIRS],
        device=waves.device,dtype=waves.dtype)
    co,si=torch.cos(angles),torch.sin(angles)
    return torch.stack((co*a-si*b,si*a+co*b),dim=-1).reshape(waves.shape)


def invert_sites_wave8(waves: torch.Tensor) -> torch.Tensor:
    """Spatial inversion u->-u: Fourier cosine fixed, sine reversed."""
    pair=waves.reshape(*waves.shape[:-1],4,2)
    return torch.stack((pair[...,0],-pair[...,1]),dim=-1).reshape(waves.shape)


def punctured_wave_geometry(d_model:int=384,n_heads:int=8) -> dict:
    if d_model%24 or d_model%n_heads or (d_model//n_heads)%24:
        raise ValueError("24D block and head alignment required")
    return {
        "width":d_model,"prime_factors":{"two_exponent":7 if d_model==384 else None,
                                         "three_exponent":1 if d_model==384 else None},
        "blocks24":d_model//24,"heads":n_heads,
        "head_blocks24":(d_model//n_heads)//24,
        "lanes3_per_block24":3,"punctured_points_per_lane":8,
        "punctured_pair_classes_per_lane":4,
        "constant_mode_excluded":True,"complete_nine_reconstruction_requires_centre_gauge":True,
        "puncture_is_not_fourier_dual":True}


def pair_wave_router_features(x:torch.Tensor,h:torch.Tensor) -> torch.Tensor:
    """[... ,2d] C2- and F3^2-translation-invariant expert route features.

    The first half repeats per-frequency energy on each cosine/sine pair.
    The second half carries ordered relational dot/wedge (one each);
    exchanging x,h reverses the wedge but NOT the dot.

    This is a learner prior. These relations are not semantic objections
    or irreducible trialectic face observations.
    """
    if x.shape!=h.shape or x.shape[-1]%24:
        raise ValueError("paired activation/history require matching 24D blocks")
    p=x.reshape(*x.shape[:-1],x.shape[-1]//24,3,4,2)
    q=h.reshape(*h.shape[:-1],h.shape[-1]//24,3,4,2)
    energy=p.square().sum(-1)
    dot=(p*q).sum(-1)
    wedge=p[...,0]*q[...,1]-p[...,1]*q[...,0]
    unary=torch.stack((energy,energy),-1).reshape(x.shape)
    relation=torch.stack((dot,wedge),-1).reshape(x.shape)
    return torch.cat((unary,relation),dim=-1)


def punctured_blocks_to_wave_blocks(x: torch.Tensor) -> torch.Tensor:
    """Rechart each actual eight-position punctured sheet into its dual modes."""
    if x.shape[-1]%8:
        raise ValueError("punctured blocks require a multiple of eight coordinates")
    return puncture8_to_wave8(x.reshape(*x.shape[:-1],-1,8)).reshape(x.shape)


def wave_blocks_to_punctured_blocks(waves: torch.Tensor) -> torch.Tensor:
    """Exact inverse (up to floating rounding) including the removed centre."""
    if waves.shape[-1]%8:
        raise ValueError("wave blocks require a multiple of eight coordinates")
    return wave8_to_puncture8(waves.reshape(*waves.shape[:-1],-1,8)).reshape(waves.shape)


def translate_punctured_blocks(x: torch.Tensor, shift:tuple[int,int]) -> torch.Tensor:
    """Induced nine-site wave translation after fixing puncture centre gauge.

    Reconstructed point amplitudes are re-gauged to origin=0 after each
    translation. This is NOT a literal centre-preserving site permutation.
    """
    waves=punctured_blocks_to_wave_blocks(x)
    transformed=translate_wave8(waves.reshape(*waves.shape[:-1],-1,8),shift)
    return wave_blocks_to_punctured_blocks(transformed.reshape(waves.shape))


def nine_to_wave_and_centre(sheet:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    """Decompose arbitrary 9-site real values into 8 differences + centre.

    The punctured channel measures f(u)-f(0). An independent value at the
    distinguished centre is carried as side state, never silently dropped.
    """
    if sheet.shape[-1]!=9:
        raise ValueError("full sheet must have nine coordinates")
    centre=sheet[...,0:1]
    offsets=sheet[...,1:]-centre
    return puncture8_to_wave8(offsets),centre


def wave_and_centre_to_nine(waves:torch.Tensor,centre:torch.Tensor) -> torch.Tensor:
    """Reopen all nine amplitudes with explicit retained centre state."""
    if waves.shape[-1]!=8 or centre.shape!=waves.shape[:-1]+(1,):
        raise ValueError("centre must be supplied with one coordinate per sheet")
    offsets=wave8_to_puncture8(waves)
    return torch.cat((centre,centre+offsets),dim=-1)
