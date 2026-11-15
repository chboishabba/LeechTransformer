"""X369-F1 spectral C3 identifiability audit on ORIGINAL residual activations.

A covariance-only search for a hidden orthogonal C3 action is nonidentifying:
an order-three rotation in an eigenspace of two equal eigenvalues is an exact
covariance symmetry, whether or not the network computes any C3 task.

This file audits all one-pair spectral order-three rotations, including the
unrestricted potentially VACUOUS low-variance optimum and constrained
minimum rotating-trace-share choices. For eigenvalue pair (a,b), a 120 degree
rotation on its two eigenvectors, identity elsewhere, has EXACT defect
  D_pair = sqrt(3/2) * |a-b| / ||C||_F.
This proves an achievable covariance-only value; it does NOT bound every
multi-pair action or establish trained source/consumer equivariance.

Checks original row concentration in the leading principal component.
No model/heldout scoring, no checkpoint mutation, no invented task semantics.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np


def spectral_pair_screen(x, *, min_rotating_shares=(0.,.01,.05,.10,.25)):
    x=np.asarray(x,dtype=np.float64)
    if (x.ndim!=3 or x.shape[0]<3 or x.shape[1]<2
            or x.shape[2]<3 or x.shape[2]%3 or not np.isfinite(x).all()):
        raise ValueError("finite original [>=3 rows,>=2 tokens,3k hidden] required")
    shares=tuple(float(t) for t in min_rotating_shares)
    if (not shares or any(not math.isfinite(t) or t<0 or t>1 for t in shares)
            or any(shares[i]>=shares[i+1] for i in range(len(shares)-1))):
        raise ValueError("distinct increasing rotating-variance thresholds in [0,1]")
    rows,tokens,d=x.shape
    flat=x.reshape(-1,d)
    centred=flat-flat.mean(axis=0)
    cov=centred.T@centred/(rows*tokens)
    eig,u=np.linalg.eigh(cov)
    eig=eig[::-1];u=u[:,::-1]
    norm=float(np.linalg.norm(cov))
    trace=float(eig.sum())
    if norm<1e-30 or trace<1e-30:
        raise ValueError("zero covariance cannot establish rotating phase")
    if eig.min() < -1e-9*max(norm,1.):
        raise ValueError("observed covariance not positive semidefinite")
    eig=np.maximum(eig,0.)
    norm=float(np.linalg.norm(eig));trace=float(eig.sum())
    p=np.arange(d).reshape(-1,3)[:,[1,2,0]].reshape(-1)
    native=float(np.linalg.norm(cov[np.ix_(p,p)]-cov)/norm)
    ii,jj=np.triu_indices(d,k=1)
    gap=np.abs(eig[ii]-eig[jj])
    diffs=np.sqrt(1.5)*gap/norm
    share=(eig[ii]+eig[jj])/trace
    results=[]
    for threshold in shares:
        allowed=np.flatnonzero(share+1e-14 >= threshold)
        if not len(allowed):
            results.append({"required_rotating_trace_share":threshold,
                            "pair_exists":False,
                            "note":"no TWO eigenvectors carry this variance fraction"})
            continue
        # Stable tie handling; select the smallest defect then greatest
        # rotating variance if exact defect ties. This is a source-only
        # geometric comparator, never model fitting on a lockbox.
        best=min(allowed,key=lambda k:(float(diffs[k]),-float(share[k]),
                                      int(ii[k]),int(jj[k])))
        results.append({
            "required_rotating_trace_share":threshold,
            "pair_exists":True,
            "eigenvalue_indices_descending":[int(ii[best]),int(jj[best])],
            "pair_eigenvalues":[float(eig[ii[best]]),
                                float(eig[jj[best]])],
            "achievable_single_pair_c3_covariance_defect":float(diffs[best]),
            "actual_rotating_trace_share":float(share[best]),
            "rotating_pair_selection_uses_only_covariance":True,
            "task_orbit_action_verified":False,
            "model_output_consumer_equivariance_verified":False})
    leading=u[:,0]
    projected=centred.reshape(rows,tokens,d)@leading
    row_energy=np.square(projected).sum(axis=1)
    row_fractions=row_energy/row_energy.sum()
    effective_doc_count=float(1/np.square(row_fractions).sum())
    return {
        "schema":1,
        "kind":"x369_f1_original_hidden_spectral_pair_identifiability",
        "hidden_dimension":d,"source_row_count":rows,
        "tokens_per_row":tokens,
        "observed_original_coherent_triads_c3_defect":native,
        "original_covariance_participation_rank":float((trace/norm)**2),
        "leading_eigenvalue_trace_share":float(eig[0]/trace),
        "leading_eigenvalue_frobenius_share":float(eig[0]/norm),
        "largest_document_share_of_top_pc_energy":float(row_fractions.max()),
        "effective_document_count_for_top_pc_energy":effective_doc_count,
        "row_contribution_to_top_pc_energy":row_fractions.tolist(),
        "eigenvalues_descending":eig.tolist(),
        "single_pair_order_three_screens":results,
        "unrestricted_spectral_symmetry_may_be_vacuous":True,
        "one_pair_screen_is_not_global_optimum_for_multi_pair_actions":True,
        "covariance_only_c3_representation_identifiable":False,
        "learned_c3_phase_or_semantic_equivariance_proved":False,
        "source_document_independence_verified":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--residuals",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("refuse overwrite F1 spectral evidence")
    xs=[np.load(q,allow_pickle=False) for q in a.residuals]
    if any(x.shape[0]!=xs[0].shape[0] for x in xs):
        raise ValueError("all layers must have matched source-row counts")
    out={"schema":1,"kind":"x369_f1_original_hidden_spectral_pair_audit",
         "sources":[{"path":str(path),
                     "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in a.residuals],
         "layers":[spectral_pair_screen(x) for x in xs],
         "full_report_is_not_trained_model_c3_mechanism_evidence":True}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,sort_keys=True,indent=2,
                                    allow_nan=False)+"\n")
    print(json.dumps(out,sort_keys=True,indent=2,allow_nan=False))

if __name__=="__main__":main()
