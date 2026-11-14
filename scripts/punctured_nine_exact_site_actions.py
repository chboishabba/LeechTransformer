"""Bit-exact geometric actions on DASHI's eight punctured F3² sites.

Each 24D block = 3 lanes x 8 noncentral sites. The source geometry:
  OggSSPP2BalancedTernaryPuncturedPlaneExact (eight site enumeration)
  Base369FaceSheetPunctureIncidenceIdentityExact (four opposite pairs)

Actions here are explicit GL(2,F3) site-index permutations fixing origin.
They are NOT site translations (those move the puncture), NOT arbitrary
neural-coordinate antipodes, NOT a demonstrated Leech-lattice group
automorphism, and NOT semantic argumentative opposition.

All float16/float32 storage bits move unchanged; no numerical arithmetic.
"""
from __future__ import annotations

SITES=((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1))

# Row-major 2x2 matrices over F3, all with determinant nonzero mod 3.
ACTIONS={
    "identity":(1,0,0,1),
    "point_neg":(-1,0,0,-1),
    "swap":(0,1,1,0),
    "shear":(1,1,0,1),
    "shear2":(1,-1,0,1),
    "rot3":(0,-1,1,-1),
    "rot3_2":(-1,1,-1,0),
}


def _mod3(value):
    return ((value+1)%3)-1


def site_permutation(action:str)->tuple[int,...]:
    if action not in ACTIONS:
        raise ValueError("unknown punctured-nine geometric action")
    a,b,c,d=ACTIONS[action]
    if (a*d-b*c)%3==0:
        raise ValueError("noninvertible punctured-plane action")
    table=[]
    for x,y in SITES:
        next_site=(_mod3(a*x+b*y),_mod3(c*x+d*y))
        table.append(SITES.index(next_site))
    if set(table)!=set(range(8)):
        raise ValueError("puncture-site map not a permutation")
    return tuple(table)


def permute_punctured_nine_bytes(raw:bytes,itemsize:int,action:str)->bytes:
    """Apply same exact site action in every eight-position subblock."""
    if itemsize not in (2,4) or not isinstance(raw,bytes) or (
        len(raw)%(8*itemsize)):
        raise ValueError("require aligned float16/float32 punctured lanes")
    permutation=site_permutation(action)
    out=bytearray(len(raw))
    lane=8*itemsize
    for start in range(0,len(raw),lane):
        for src,dst in enumerate(permutation):
            out[start+dst*itemsize:start+(dst+1)*itemsize]=(
                raw[start+src*itemsize:start+(src+1)*itemsize])
    return bytes(out)
