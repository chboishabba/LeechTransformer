"""Local numerical controls, not fitted LILA, GPU or formal acceptance."""
from itertools import product
import math
import torch
from scripts.x369_phase_geometry import (action_matrices,common_hom,cube_metrics,orbit_spectrum,anchored_phase,mod3_spectrum,cross_time)


def main():
    torch.set_num_threads(1);torch.manual_seed(41);dt=torch.float64
    x=torch.tensor(list(product((-1,0,1),repeat=6)),dtype=dt);lookup={tuple(y.tolist()):i for i,y in enumerate(x)}
    rho=action_matrices();ix={g:torch.tensor([lookup[tuple(y.tolist())] for y in x@a]) for g,a in rho.items()}
    U=torch.eye(6,dtype=dt)+.2*torch.randn(6,6,dtype=dt);A={g:torch.linalg.solve(U,a@U) for g,a in rho.items()};q=x@U
    h=common_hom(A,U);assert h['numerical_nullity']==6 and h['candidate_rank']==6 and h['candidate_relative_residual']<1e-12
    c=cube_metrics(A,q,ix);assert len(c['faces'])==len(c['paths'])==6 and max(c['paths'].values())<1e-12 and max(v for f in c['faces'].values() for v in f.values())<1e-12
    bad=dict(A,J=torch.eye(6,dtype=dt));assert common_hom(bad,U)['candidate_relative_residual']>.1 and max(cube_metrics(bad,q,ix)['paths'].values())>.1
    y=torch.tensor([(a+b)%3 for a,b in product(range(3),repeat=2)]);L=torch.nn.functional.one_hot(y,3).double();m=mod3_spectrum(L);assert abs(m['target_fraction']-1)<1e-14 and abs(m['target_phase'])<1e-14
    m=mod3_spectrum(L.roll(1,-1));assert abs(m['target_fraction']-1)<1e-14 and abs(m['target_phase']-2*math.pi/3)<1e-14
    m=mod3_spectrum(L[:,[0,2,1]]);assert m['conjugate_energy']>.99999 and m['target_fraction']<1e-14
    L[8]=torch.tensor([1.,0.,0.]);m=mod3_spectrum(L);assert abs(m['target_fraction']-19/27)<1e-14
    m=mod3_spectrum(torch.zeros_like(L));assert m['zero_response'] and m['target_phase'] is None and m['target_fraction'] is None
    a=anchored_phase(x,x);assert all(abs(v['anchored_real']-1)<1e-14 and abs(v['anchored_imag'])<1e-14 and v['phase_rmse']==0 for v in a)
    spec=orbit_spectrum(x,ix['R'],x);assert abs(spec['total_energy']-float(x.square().sum(1).mean()))<1e-12
    assert spec['anchored_feature_coefficients'][0]['real'][0]>0
    W=torch.eye(6,dtype=dt)+.1*torch.randn(6,6,dtype=dt);V=torch.eye(6,dtype=dt)
    def chart(U):return {'chart':dict(U=U.tolist(),V=V.tolist(),A={g:torch.linalg.solve(U,a@U).tolist() for g,a in rho.items()},composition_q=(x@U).tolist(),composition_keys=list(range(len(x))))}
    t=cross_time(chart(U),chart(W));assert t['admitted'] and t['composition_transport_rmse']<1e-12 and max(t['generator_square_defect'].values())<1e-12
    assert not cross_time({},chart(W))['admitted']
    print('PASS 729 states; six faces and paths; rank-6 Hom and bad J; five Fourier controls; orbit Parseval; anchored phase; cross-time transport',max(c['paths'].values()),t['composition_transport_rmse'])
if __name__=='__main__':main()
