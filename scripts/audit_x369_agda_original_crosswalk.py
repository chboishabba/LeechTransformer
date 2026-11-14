"""Fail-closed original Agda-to-Leech signed nine-site enumeration crosswalk.

The user passes the actual checked-out Agda A00 source by path.
This checks the literal Torus3x3 mapping and original Python SITES order,
not Agda kernel acceptance or a trained neural intertwiner.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
from models.punctured_nine_wave_prior import SITES

SLOT_NAMES=("siteCentre","siteMinusZero","sitePlusZero","siteZeroMinus",
            "siteZeroPlus","siteMinusMinus","sitePlusPlus","siteMinusPlus",
            "sitePlusMinus")
VALUES={"residueMinus":-1,"residueZero":0,"residuePlus":1}

def audit_original_agda_site_crosswalk(source: str):
    if 'FiniteThreeCycleTorusExact' not in source or 'sourceToOriginalTorus' not in source:
        raise ValueError('original Torus carrier source missing')
    order=re.search(r'^sourceSlotOrder\s*:[^\n]*\nsourceSlotOrder\s*=\s*\n(.*?)(?=\n\S)',source,re.M|re.S)
    if not order:raise ValueError('explicit Python signed site order absent')
    found=tuple(re.findall(
        r'\bsite(?:Centre|MinusZero|PlusZero|ZeroMinus|ZeroPlus|MinusMinus|PlusPlus|MinusPlus|PlusMinus)\b',
        order.group(1)))
    if found!=SLOT_NAMES:
        raise ValueError('signed slot order not identical to Leech source')
    matches=re.findall(
        r'^sourceToOriginalTorus\s+(site\w+)\s*=\s*\n\s*Torus\.torusPoint\s+Torus\.(residue\w+)\s+Torus\.(residue\w+)',
        source,re.M)
    if len(matches)!=9 or len(set(m[0] for m in matches))!=9:
        raise ValueError('incomplete or duplicate original torus point mapping')
    try:
        mapped={name:(VALUES[a],VALUES[b]) for name,a,b in matches}
    except KeyError as e:
        raise ValueError('unknown original Residue3 constructor') from e
    if tuple(mapped[name] for name in SLOT_NAMES)!=SITES:
        raise ValueError('Agda original torus point mapping differs from Leech SITES')
    return {'schema':1,
        'kind':'x369_original_agda_to_leech_signed_slot_crosswalk',
        'original_carrier':'FiniteThreeCycleTorusExact.Torus3x3',
        'source_slot_order_exact':True,
        'pointwise_original_torus_map_exact':True,'points_checked':9,
        'source_agda_kernel_accepted':False,
        'trained_original_leech_c3_mechanism_proved':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--agda-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('refuse overwrite crosswalk audit')
    data=a.agda_source.read_bytes()
    result=audit_original_agda_site_crosswalk(data.decode('utf-8'))
    result['agda_a00_source_sha256']=hashlib.sha256(data).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=='__main__':main()
