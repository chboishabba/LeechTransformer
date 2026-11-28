# X369 LILA — exact affine recurrent signed-residual classification

24 November 2026. Original Leech-LILA architecture attribution: Anatolii Kornienko. DASHI mathematical extension, prior [#1550](https://github.com/chboishabba/dashi_agda/pull/1550) sparse labelled quotient. Inspiration: attached Langlands even/odd mediator and joint gluing, #1549/#120; use only linear algebra here, no arithmetic or Clay claims. Scope: ideal real arithmetic, fixed model, tagged d=6 E=4 TopK=2 coherent R4, three source-legal probes u_j, no history cap or router intervention, fixed tags, arbitrary real payload p_e in R^3. Not a fitted-model result.

## F2 exact theorem for the whole affine family

Reuse the #1550 original strict route (0,j), sign +1, constant positive weights a,b and source paired GELU identity GELU(z)-GELU(-z)=z, giving y_j=a p_0+b p_j. The original affine expert transition receives canon_h, canon_x and y_c, so ignoring the latter two and preserving tags realizes arbitrary selected payload maps p_e -> A_e p_e+c_e, A_e real 3x3, c_e real 3-vector. Inactive states unchanged. Tags and selection remain invariant by construction. No new router/FFN primitives.

Let C=(y_1,y_2,y_3), D=vertical stack of (A_e-I) for e=0..3, W=ker D. Then:

    H ~ H' iff C(H)=C(H') and D p_0=D p'_0.
    dim minimal continuous Euclidean code = 9 + rank D = 12 - dim W.

Proof: C equal iff delta H=K v=(b v,-a v,-a v,-a v). For first probe j, next difference is T_j K v where T_j is block diagonal, with A_0,A_j in selected slots and identity elsewhere. On second probe k != j, delta y_k=ab(A_0-I)v. On k=j, delta y_j=ab(A_0-A_j)v. Since all j and k are admitted, length-at-most-two trace equality forces (A_e-I)v=0 for every e. Conversely v in W implies T_j K v=K v for every j and C K v=0; induction proves all-word equivalence. Affine offsets cancel. The proof uses all labelled words of length at most two from the same initial state (not a claim that one online two-step word reconstructs every state). This pays both directions for arbitrary affine matrices on this domain, including noncommuting and singular choices. It does not assume invertibility of transitions.

Choose a full row rank B from the row space of D (r rows). Code (C,B p_0) is surjective onto R^(9+r): choose p_0 for Bp_0 then each p_j=(y_j-a p_0)/b. Restrict any continuous sufficient encoder to a linear section and it must be injective there. Invariance of domain forbids an injection into lower Euclidean dimension (embed lower target into R^(9+r)). No lower bound on arbitrary discontinuous real encoders. For a finite alphabet, count actual distinguishable classes rather than applying the real rank as a bit bound without proof.

## Constructive codec update without recovering the original history

Let r=Bp_0. Factor A_e-I=L_e B (exists by row-space inclusion). On probe j,

    y_k' = y_k + a L_0 r + a c_0, k != j
    y_j' = A_j y_j + a(L_0-L_j) r + a c_0 + b c_j
    r' = r + B L_0 r + B c_0.

This is exact compressed execution, not only an abstract quotient. Output y_j and known route/weights/sign before update. No arbitrary section or unobserved state reconstruction required. For dense payload dimension m, update on chosen j uses O(m^2 + m*r + r^2 + E*m) arithmetic operations with precomputed matrices/constants, vs storing E*m coordinates; this is an upper bound on this specific codec, not a speedup or universal time lower bound. For the original m=3 case r<=3. Native rounding and physical costs remain separate.

## Signed reflection and mediator (inspiration from attached even/odd descent)

Let J swap payload coordinates 1 and 3, preserving the spatial middle coordinate; P+=(I+J)/2 rank2, P-=(I-J)/2 rank1. This is a reflection on the payload, not a claim of arithmetic Langlands action, and its middle spatial coordinate is not the Fourier DC line. If all A_e commute with J then W decomposes into even and odd common fixed subspaces. All four exact minimal dimensions occur with A_j=I for j>0:

| A_0 | common fixed W | minimal dim |
|---|---|---|
| I | full 3 | 9 |
| J | reflection-even plane | 10 |
| -J | reflection-odd line | 11 |
| 0 | zero | 12 |

J and -J give different signed residuals despite the same present y_j law. If matrices mix parity, splitting the codec independently by parity is not generally sound: for the even-only observer P+, v=(1,0,-1), T=diag(1,0,0), P+v=0 but P+Tv !=0. This is a linear diagnostic, not an assertion that the original full observer is even-only. Retain the joint row space D, not unsigned per-step norms or expanded paths. This is a LILA-specific use of the signed descent insight. 

## F3 quantitative boundary

Define M as the stack of C and C T_j for j=1..3. For affine offset known and removed, two-step labelled observations equal M H. Its kernel is K W by proof above. On the orthogonal complement of ker M, sigma_min^+(M)>0 and ||projection(delta H)||_2 <= ||M delta H||_2 / sigma_min^+(M). This is exact ideal-real inverse stability; actual observation error budgets, route/chart margins, GELU arithmetic and cap are not supplied by it. Near an extra common fixed direction, sigma_min^+ can approach zero; rank alone does not give numerical safety.

## Acceptance and remaining exact obligations

Local SymPy 1.14.0 executed companion script (625 transition quadruples, symbolic two-step block identities, common fixed invariance, all 4 ranks, parity-mixing negative control). Initial symbolic structural equality assertion repaired by expansion before successful run. No Agda kernel, GitHub CI or native GPU run. No original source model or fingerprint change.

| Phase | Deliverable / producer -> consumer | Status |
|---|---|---|
| F2a-c | Source-legal affine tagged family -> exact two-step behavioural quotient and executable reduced recurrence | proved in ideal-real restricted family |
| F2d | General affine switched reachable state -> invariant kernel iteration and constructive codec (including input-dependent selection) | next; fixed tag probes do not cover unrestricted input |
| F3 | strict actual TopK/chart margins, cap/FFN nonlinear guard, calibrated finite precision -> recurrent error | open |
| F4 | signed parity mixing and true learned geometry vs coordinate gauge -> actual task model | partial |
| F5 | independently typed task-action source -> trained prediction correspondence | open |
| F6 | formal kernel and actual fitted checkpoint validation | downstream |

LILA remains the target; no change to Clay or Langlands branches. Preserve original Leech-LILA credit; no F5 lockbox access.
