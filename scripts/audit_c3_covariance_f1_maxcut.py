"""X369-F1: full hidden covariance plus original-row-cluster uncertainty.

Input: per-layer [independent source rows,tokens,hidden] residual arrays.
A source row, NOT one token or triad, is the bootstrap cluster. Bonferroni
intervals are descriptive and cannot prove population C3 symmetry, isotropy
or a learned neural circuit. Layer row IDs must match.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def check_residual(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=3 or x.shape[0]<3 or x.shape[1]<2 or x.shape[2]%3:
        raise ValueError("requires [>=3 rows,>=2 tokens,3k coordinates]")
    if not np.isfinite(x).all() or not np.square(x).sum(axis=(1,2)).min()>0:
        raise ValueError("nonfinite or zero-energy original row")
    return x

def pooled_metrics(x):
    tri=x.reshape(-1,3)
    mu=tri.mean(0)
    cen=tri-mu
    cov=cen.T@cen/len(tri)
    cycle=[1,2,0]
    norm=max(float(np.linalg.norm(cov)),1e-30)
    return {
        "mean":mu.tolist(),"covariance":cov.tolist(),
        "isotropy_defect":float(np.linalg.norm(cov-np.eye(3)*np.trace(cov)/3)/norm),
        "c3_commutator_defect":float(np.linalg.norm(cov[np.ix_(cycle,cycle)]-cov)/norm),
    }

def full_metrics(x):
    _,_,d=x.shape
    flat=x.reshape(-1,d)
    cen=flat-flat.mean(0)
    cov=cen.T@cen/len(flat)
    norm=max(float(np.linalg.norm(cov)),1e-30)
    cycle=np.arange(d).reshape(-1,3)[:,[1,2,0]].reshape(-1)
    same=np.zeros_like(cov)
    defects=[]
    for off in range(0,d,3):
        block=cov[off:off+3,off:off+3]
        same[off:off+3,off:off+3]=block
        bnorm=max(float(np.linalg.norm(block)),1e-30)
        defects.append(float(np.linalg.norm(block[np.ix_([1,2,0],[1,2,0])]-block)/bnorm))
    return {
        "hidden_dimension":d,
        "full_hidden_covariance_tested":True,
        "full_c3_commutator_defect":float(np.linalg.norm(cov[np.ix_(cycle,cycle)]-cov)/norm),
        "full_isotropy_defect":float(np.linalg.norm(cov-np.eye(d)*np.trace(cov)/d)/norm),
        "off_triad_covariance_frobenius_fraction":float(np.linalg.norm(cov-same)/norm),
        "per_triad_c3_defect_summary":{
            "min":min(defects),"median":float(np.median(defects)),
            "max":max(defects),"num_triads":len(defects),
            "fraction_above_0_1":float(np.mean(np.asarray(defects)>.1))},
        "covariance_rank_upper_bound":min(d,len(flat)-1),
    }

def audit_layers(layers,*,seed=41,bootstraps=1999,familywise_alpha=.05,
                 tolerance=.05):
    if not 0<familywise_alpha<1 or not 0<=tolerance<1:
        raise ValueError("invalid predeclared alpha or tolerance")
    if bootstraps<19:raise ValueError("at least 19 row bootstrap draws")
    xs=[check_residual(x) for x in layers]
    if not xs or any(x.shape[0]!=xs[0].shape[0] for x in xs):
        raise ValueError("each layer must have matching independent source rows")
    n=xs[0].shape[0]
    rng=np.random.default_rng(seed)
    # Share row indices across layers, preserving paired row dependence.
    draws=rng.integers(n,size=(bootstraps,n))
    tail=familywise_alpha/(2*len(xs))
    reports=[]
    for x in xs:
        defects=np.asarray([pooled_metrics(x[idx])["c3_commutator_defect"]
                            for idx in draws])
        interval=np.quantile(defects,[tail,1-tail]).tolist()
        reports.append({
            "rows":n,"tokens_per_row":x.shape[1],"pooled_triads":x.shape[2]//3,
            "observed_pooled":pooled_metrics(x),
            "observed_full_hidden":full_metrics(x),
            "cluster_bootstrap_commutator_interval":interval,
            "empirical_bootstrap_pooled_defect_median":float(np.median(defects)),
            "against_predeclared_tolerance":
                ("lower_bound_exceeds_tolerance" if interval[0]>tolerance
                 else "inconclusive_for_population_symmetry"),
        })
    return {"schema":1,"kind":"original_f1_full_hidden_cluster_covariance_audit",
        "seed":seed,"bootstraps":bootstraps,"familywise_alpha":familywise_alpha,
        "bonferroni_tail_each_side":tail,"relative_defect_tolerance":tolerance,
        "cluster_unit":"independent original source row; NOT token or triad",
        "all_layers_use_identical_row_bootstrap_draws":True,
        "no_population_isotropy_or_c3_symmetry_proof":True,
        "no_neural_learned_c3_mechanism_proof":True,
        "source_rows_are_not_independent_if_document_provenance_unverified":True,
        "layers":reports,
        "note":"Conditional bootstrap intervals are not an equivalence test. "
               "Source document identity/stationarity remain to be audited."}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",default="cpu",help="Schaka compatibility; runs NumPy CPU")
    p.add_argument("--residuals",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--bootstraps",type=int,default=1999)
    p.add_argument("--familywise-alpha",type=float,default=.05)
    p.add_argument("--relative-tolerance",type=float,default=.05)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse overwrite F1 evidence")
    result=audit_layers([np.load(path,allow_pickle=False) for path in a.residuals],
                        seed=a.seed,bootstraps=a.bootstraps,
                        familywise_alpha=a.familywise_alpha,tolerance=a.relative_tolerance)
    result["inputs"]=[{"path":str(path),
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in a.residuals]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
if __name__=="__main__":main()
