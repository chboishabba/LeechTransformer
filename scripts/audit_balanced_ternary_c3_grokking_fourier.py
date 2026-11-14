"""Finite C3 modular grokking / balanced-ternary phase and opposition audit.

The exact finite task is addition in Z/3Z.  The real Fourier basis
(cos(2*pi*k/3), sin(2*pi*k/3)) yields the bilinear multiplication rule
for addition. The 8-of-9 memorizer from DASHI's Agda donor fits training
and fails (2,2); equivariance defect distinguishes it on two directed
edges.  Balanced residue inversion is *reflection* of the phase plane,
not the central -I of the trained Leech C2 source.

Nothing here claims a neural model trained on a modular task or learned
these circuits.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

def rotate_phase(z):
    x,y=z
    c=-0.5
    s=math.sqrt(3.)/2.
    return (c*x-s*y,s*x+c*y)

def invert_phase(z):
    """C3 automorphism k -> -k is phase conjugation."""
    return (z[0],-z[1])

def central_antipodal(z):
    """Leech C2 is -I on this comparison plane, not residue inversion."""
    return (-z[0],-z[1])

def phase(k):
    t=2.*math.pi*(k%3)/3.
    return math.cos(t),math.sin(t)

def multiply_phase(a,b):
    return (a[0]*b[0]-a[1]*b[1],
            a[0]*b[1]+a[1]*b[0])

def dist(a,b):
    return math.hypot(a[0]-b[0],a[1]-b[1])

def memorizer(a,b):
    return 0 if (a,b)==(2,2) else (a+b)%3

def eq_defect(rule):
    return sum(rule((a+1)%3,b)!=(rule(a,b)+1)%3
               for a in range(3) for b in range(3))

def report():
    residues=(-1,0,1)
    # [-1,0,1] and [2,0,1] denote exactly the same C3 carrier.
    balanced={str(a):a%3 for a in residues}
    rows=[]
    for a in range(3):
        for b in range(3):
            target=(a+b)%3
            fourier=multiply_phase(phase(a),phase(b))
            residual=dist(fourier,phase(target))
            rows.append({"a":a,"b":b,"answer":target,
                "memorizer":memorizer(a,b),"fourier_residual":residual})
    g=phase(1)
    sample=phase(2)
    reflection_conjugates=dist(
        invert_phase(rotate_phase(invert_phase(sample))),
        rotate_phase(rotate_phase(sample)))
    central_commutes=dist(
        central_antipodal(rotate_phase(sample)),
        rotate_phase(central_antipodal(sample)))
    # Central antipodal -I on the phase plane is NOT the C3 inversion
    # reflection; its action sends a unit root to a 6th root instead.
    inversion_difference=dist(central_antipodal(sample),invert_phase(sample))
    # The task's phase inversion is a *relation on residues*; don't
    # confuse its arithmetic -k with generic semantic negation.
    neg_residual=max(dist(invert_phase(phase(a)),phase(-a))
                     for a in range(3))
    max_residual=max(row["fourier_residual"] for row in rows)
    claims={"memorizer_training_fit_8_of_8":all(
            memorizer(a,b)==(a+b)%3 for a in range(3)
            for b in range(3) if (a,b)!=(2,2)),
        "memorizer_heldout_fails":memorizer(2,2)!=(2+2)%3,
        "structural_equivariance_defect":eq_defect(lambda a,b:(a+b)%3),
        "memorizer_equivariance_defect":eq_defect(memorizer),
        "max_fourier_multiplication_residual":max_residual,
        "max_residue_negation_conjugation_residual":neg_residual,
        "reflection_inverts_c3_residual":reflection_conjugates,
        "central_minus_I_commutes_with_c3_residual":central_commutes,
        "central_minus_I_is_not_c3_residue_inversion_distance":inversion_difference}
    if not (claims["memorizer_training_fit_8_of_8"]
            and claims["memorizer_heldout_fails"]
            and claims["structural_equivariance_defect"]==0
            and claims["memorizer_equivariance_defect"]==2
            and max_residual<1e-12 and neg_residual<1e-12
            and reflection_conjugates<1e-12
            and central_commutes<1e-12 and inversion_difference>.5):
        raise AssertionError("finite original C3 Fourier/grokking source-law contract failed")
    return {"schema":1,"kind":"balanced_ternary_c3_grokking_fourier_law",
        "balanced_to_mod3":balanced,"task":"mod3 addition nine pairs",
        "finite_cases":rows,"claims":claims,
        "learned_network_tested":False,
        "sophontic_argument_negation_identified_with_mod3_inversion":False,
        "model_c2_c3_intertwiner_proved":False,
        "donors":["DASHI/Cognition/PNF/ModularFourierGrokkingC3Exact.agda",
                  "DASHI/Cognition/PNF/GrokkingEquivarianceDefectGeometryExact.agda",
                  "DASHI/Cognition/PNF/CyclotomicGrokkingActionLinearizationExact.agda",
                  "DASHI/Biology/BalancedTernaryFourierModeExact.agda"]}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    data=report()
    rendered=json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if a.output:
        if a.output.exists():raise FileExistsError("do not overwrite original phase proof receipt")
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(rendered)
    print(rendered,end="")
if __name__=="__main__":
    main()
