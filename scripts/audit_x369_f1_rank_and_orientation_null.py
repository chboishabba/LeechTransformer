"""X369-F1: rank-aware original dxd C3 geometry after full hidden rejection.

All calculations use the ACTUAL trained residual matrix and the actual
consecutive-triad cyclic permutation. The Frobenius identities are exact
finite orthogonal-action algebra; the numerical tests use float64.

D = ||C - P C P^T||F / ||C||F. For PSD C:
  0 <= D <= sqrt(2)
  <C,PCP^T>/||C||F² = 1 - D²/2.
For order-three P, Reynolds E=(C+PCP^T+P²CP^T²)/3:
  ||C-E||F/||C||F = D/sqrt(3).
This is NOT an implicit learned gauge or task-defined neural symmetry.

The matched orientation null randomizes each triad's nonidentity
C3-generator direction (+1 vs -1), while every block retains the same
three-cycle eigenvalues. It compares the SPECIFIC globally coherent
coordinate orientation against mixed but spectrally matched order-three
actions; it is not a causal intervention or a universal null theorem.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def c3_full_rank_gauge_diagnostic(x,*,seed=41,orientation_null_draws=199):
    x=np.asarray(x,dtype=np.float64)
    if (x.ndim!=3 or x.shape[0]<3 or x.shape[1]<2
            or x.shape[2]%3 or not np.isfinite(x).all()):
        raise ValueError("need finite [source_rows,tokens,3k] residuals")
    if (type(orientation_null_draws) is not int
            or not 1<=orientation_null_draws<=4096):
        raise ValueError("null draws outside fixed source budget")
    n,t,d=x.shape
    y=x.reshape(-1,d)
    y=y-y.mean(axis=0)
    cov=y.T@y/len(y)
    norm=float(np.linalg.norm(cov))
    if norm<1e-30:
        raise ValueError("zero full covariance")
    triad=np.arange(d).reshape(-1,3)
    p=triad[:,[1,2,0]].reshape(-1)
    p2=p[p]
    shifted=cov[np.ix_(p,p)]
    doubly=cov[np.ix_(p2,p2)]
    selected=float(np.linalg.norm(cov-shifted)/norm)
    projected=(cov+shifted+doubly)/3
    invariant_part=float(np.linalg.norm(projected)/norm)
    noninvariant=float(np.linalg.norm(cov-projected)/norm)
    similarity=float(np.sum(cov*shifted)/(norm*norm))
    expected=1.-selected*selected/2.
    if (abs(similarity-expected)>2e-10
            or abs(noninvariant-selected/np.sqrt(3))>2e-10):
        raise AssertionError("C3 finite Reynolds projection identities violated")
    if similarity < -1e-9 or selected>np.sqrt(2)+1e-9:
        raise AssertionError("original covariance not positive semidefinite")
    participation=(float(np.trace(cov))/norm)**2
    trace=float(np.trace(cov))
    dc_variance=sum(float(cov[np.ix_(triad[j],triad[j])].sum())/3.
                    for j in range(len(triad)))
    dc_variance_share=dc_variance/max(trace,1e-30)
    rotating_variance_share=1.-dc_variance_share
    if dc_variance_share < -1e-10 or dc_variance_share > 1+1e-10:
        raise AssertionError("original covariance violates PSD DC bound")
    result={
        "schema":1,
        "kind":"x369_f1_full_hidden_rank_and_c3_orientation_null",
        "source_rows":n,"tokens_per_row":t,
        "original_hidden_dimension":d,
        "observed_original_c3_full_commutator_defect":selected,
        "coherent_c3_frobenius_overlap":similarity,
        "coherent_c3_frobenius_overlap_from_defect":expected,
        "coherent_c3_reynolds_projection_fraction":invariant_part,
        "coherent_c3_orthogonal_complement_fraction":noninvariant,
        "full_covariance_effective_rank_participation":participation,
        "trivial_dc_fixed_vector_variance_share":dc_variance_share,
        "rotating_two_character_variance_share":rotating_variance_share,
        "rotating_character_variance_is_not_learned_phase_evidence":True,
        "empirical_covariance_rank_upper_bound":min(d,len(y)-1),
        "c3_psd_commutator_ceiling":float(np.sqrt(2.)),
        "coordinate_action_specification":
            "same cyclic shift on every original consecutive hidden triad",
        "orientation_null":
            "random independent +1 or -1 C3 shifts per triad; "
            "every triad receives nonidentity 3-cycle with same eigenphases",
        "source_document_identity_independently_verified":False,
        "c3_semantic_action_inferred":False,
        "learned_c3_automorphism_proved":False,
    }
    if d//3<2:
        result.update({"orientation_null_available":False,
                       "orientation_null_reason":
                           "only one triad; no mixed orientation"})
        return result
    rng=np.random.default_rng(seed)
    minus=triad[:,[2,0,1]]
    plus=triad[:,[1,2,0]]
    null=np.empty(orientation_null_draws)
    for j in range(orientation_null_draws):
        for _ in range(1000):
            sign=rng.integers(0,2,size=len(triad),dtype=np.int8)
            if sign.any() and not sign.all():
                break
        else:
            raise AssertionError("failed to sample a nontrivial orientation")
        permutation=np.where(sign[:,None].astype(bool),plus,minus).reshape(-1)
        null[j]=np.linalg.norm(cov[np.ix_(permutation,permutation)]-cov)/norm
    result.update({
        "orientation_null_available":True,
        "orientation_null_seed":seed,
        "orientation_null_draws":int(orientation_null_draws),
        "orientation_null_commutator_quantiles":
            np.quantile(null,[.025,.5,.975]).tolist(),
        "orientation_null_mean":float(null.mean()),
        "observed_below_mixed_orientation_fraction":
            float(np.mean(selected<=null)),
        "orientation_null_precommitment_is_architectural_not_semantic":True,
    })
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--residuals",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--orientation-null-draws",type=int,default=199)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("refuse overwrite original F1 rank receipt")
    arrays=[np.load(path,allow_pickle=False) for path in a.residuals]
    if any(x.shape[0]!=arrays[0].shape[0] for x in arrays):
        raise ValueError("original document-row count differs across layers")
    result={"schema":1,
            "kind":"x369_f1_original_full_hidden_rank_and_orientation_audit",
            "seed":a.seed,"draws":a.orientation_null_draws,
            "historical_document_independence_unverified":True,
            "source_files":[{"path":str(path),
                "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in a.residuals],
            "layers":[c3_full_rank_gauge_diagnostic(x,seed=a.seed,
                        orientation_null_draws=a.orientation_null_draws)
                      for x in arrays]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,
                                   allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
