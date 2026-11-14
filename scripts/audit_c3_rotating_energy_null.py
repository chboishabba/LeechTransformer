"""Cluster-aware C3 rotating-energy null; distinct from a learned-phase proof.

Input: original teacher-input residual tensor [source_rows, tokens, hidden],
captured *without* changing weights. Random signed feature permutations
preserve row norms but break the source coordinate-triad grouping. This is
a test of privileged alignment to that grouping, NOT a full isotropy or
task-level equivariance test. Cluster bootstrap uses whole source rows.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def rotating_energy(residual: np.ndarray) -> np.ndarray:
    x=np.asarray(residual,dtype=np.float64)
    if x.ndim!=3 or x.shape[-1]%3 or not np.isfinite(x).all():
        raise ValueError("finite [rows,tokens,d] residuals, d multiple of 3 required")
    y=x.reshape(x.shape[0],x.shape[1],-1,3)
    norm=np.square(y).sum(axis=(1,2,3))
    if np.any(norm<=0):
        raise ValueError("zero-energy row")
    dc=np.square(y.sum(axis=-1)).sum(axis=(1,2))/3.
    return 1.-dc/norm

def result(residual: np.ndarray, *, seed=41, permutations=1999, bootstraps=1999):
    x=np.asarray(residual,dtype=np.float64)
    fractions=rotating_energy(x)
    rng=np.random.default_rng(seed)
    n,t,d=x.shape
    if n<2 or permutations<19 or bootstraps<19:
        raise ValueError("at least 2 independent rows and 19 null replicates needed")
    weight=np.square(x).sum(axis=(1,2))
    num=fractions*weight
    observed=float(num.sum()/weight.sum())
    null=[]
    for _ in range(permutations):
        shuffle=rng.permutation(d)
        signs=rng.choice((-1.,1.),size=d)
        p=rotating_energy(x[...,shuffle]*signs)
        null.append(float(np.dot(p,weight)/weight.sum()))
    null=np.asarray(null)
    distance=abs(observed-np.mean(null))
    tail=np.count_nonzero(abs(null-np.mean(null))>=distance)
    draws=rng.integers(n,size=(bootstraps,n))
    boot=num[draws].sum(axis=1)/weight[draws].sum(axis=1)
    return {
       "kind":"original_c3_rotating_energy_cluster_and_signed_permutation_null",
       "schema":1,"source_rows":n,"tokens_per_row":t,"hidden_dim":d,
       "observed_rotating_fraction":observed,
       "isotropic_theoretical_fraction":2/3,
       "delta_vs_two_thirds":observed-2/3,
       "row_bootstrap_95pct_interval":np.quantile(boot,[.025,.975]).tolist(),
       "signed_axis_permutation_null_mean":float(np.mean(null)),
       "signed_axis_permutation_null_95pct_interval":np.quantile(null,[.025,.975]).tolist(),
       "signed_axis_permutation_two_sided_p":float((tail+1)/(permutations+1)),
       "seed":seed,"permutations":permutations,"bootstraps":bootstraps,
       "null_only_tests_prescribed_coordinate_triad_alignment":True,
       "full_residual_isotropy_proved":False,
       "C3_task_equivariance_proved":False,
       "trained_C3_circuit_proved":False,
       "note":"Two-thirds is the isotropic expectation, not proof of isotropy. "
              "Coordinate signs/permutations are not independent future test data."
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",default="cpu",
        help="Accepted for Schaka CLI compatibility; finite NumPy null operates on CPU")
    p.add_argument("--residuals",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--permutations",type=int,default=1999)
    p.add_argument("--bootstraps",type=int,default=1999)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse overwrite existing original result")
    x=np.load(a.residuals,allow_pickle=False)
    d=result(x,seed=a.seed,permutations=a.permutations,bootstraps=a.bootstraps)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(d,indent=2,sort_keys=True,allow_nan=False))
if __name__=="__main__":main()
