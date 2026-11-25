# G5a4 — exact stabilizer-stratified signed recurrent codec

25 November 2026. Owner LeechTransformer #5; original Leech-LILA Anatolii Kornienko, DASHI geometric extension. Source inspected at 69b4c238ebdaa77b1f10b58eaab73de22ab4207b: models/geometric_orbit_descent.py, models/geometric_reflection_fibre.py, scripts/x369_source_geometry.py, scripts/train_x369_ordered_action.py, scripts/audit_x369_ordered_signed_task.py. Reuse prior G5 paired/marked fibre, affine signed quotient and residual observability curvature documents and the previously identified DASHI #1550 future trace donor. Do not change original source fingerprints, F5, or other branches.

## 1. Fixed-space construction, not a learned quotient

G = S3 x C2, elements (k,b,a) act on each paired triad via spatial left roll R^k and reflection J^b, branch swap b, global sign (-1)^a. Write n=(0,0,1). On the six slots (branch,coordinate), the unsigned S3 action is free and transitive: rotations cover three coordinates within a branch; reflections exchange branches and cover the other three. It is the regular S3 permutation representation, tensored with the central sign character.

For subgroup K, the Reynolds operator P_K on one paired triad has trace (6/|K|)*(1 - indicator[n in K]); all other group elements have zero trace. P_K^2=P_K, so rank equals trace. If n in K, fixed space zero. Otherwise projection K -> S3 is injective, |K| in {1,2,3,6}, and rank is 6/|K|. Each coordinate orbit has size |K| with no contradictory sign transport.

Construct an exact signed gather/scatter codec: choose one slot per K-orbit; store that scalar. For each other slot choose k taking the representative to that slot, decode with (-1)^a(k). The sign is independent of the choice (free unsigned K action). When n in K all slots must be zero, so no scalar payload is required. Equivalently an orbit sum, divided by orbit length and using signed coefficients, computes the representative of P_K z. This replaces a dense group average by disjoint signed orbit reductions; no claim of measured speed. Zero and signed residual retained; no assumption that D=0, no absolute-value compression.

For E experts and width w divisible by three, paired history has 2 E w scalars before encoding. On a fixed K stratum the payload dimension is 2 E w/|K| when n absent; 0 when n present. For E=4,w=12: 96,48,32,16 or 0. Output plus history has 120,60,40,20 or 0. These counts exclude discrete subgroup/frame metadata, scratch, and rounding. On the full fixed linear space the dimension is minimal among continuous real encodings by a linear section and invariance of domain. Not a minimal behavioural quotient, nor a bound on dimensions of a particular reachable subset.

## 2. Bidirectional indexed codec and actual source seam

For history h choose canonical c=max rho(g)h, mark m with rho(m)c=h and stabilizer H=Stab(c); encode (H,mH,signed orbit representatives). Decode exactly by signed scatter then group action. Mark may be stored as deterministic coset representative; it must NOT be discarded. Store H explicitly or recover it from a checked decoded canonical history. State and control carry distinct actions: history is polar paired, operation transforms by conjugation, literal tag is invariant.

To step: decode the marked state; jointly canonicalize (control,history) as in canonical_controlled; execute original paired source once with local embedding; project y and h' to joint K; reconstruct the world state; re-encode the next history using its own stabilizer and mark. In exact arithmetic inactive histories already fixed under K, so the source's inactive-history identity is compatible. K fixes the projected local successor; its history stabilizer can be larger, then conjugated by the next canonical frame. A new control can break a prior stabilizer, so never freeze the old K across time. This is an exact transport construction, not a proof that the currently implemented floating tensor projection is bitwise fixed. Preserve step/model revision in receipts.

The source currently returns raw/projected local values, decoded world history, local signs, weights, IDs, mark and stabilizer. Exact state storage alone preserves future steps for a fixed deterministic source if all state is decoded, but cannot be claimed to reconstruct previous emitted observations. If those are declared consumers, retain the emitted packet or prove their factorization. NaN/Inf reject; a native lossless codec must validate signed slot relations, including declared signed-zero convention, before gather. A rounded Reynolds sum can violate exact fixed-point relations; fail closed or use orbit-based projection by construction with certified arithmetic. Neither averaging order nor bitwise native equivalence is asserted.

Central sign fixes every conjugation control. Thus h=0 is a true absorbing history for this architecture (all paired projected outputs also zero) in ideal arithmetic. This is a symmetry consequence, not a training failure. Escaping requires an additional nontrivial polar input or seed; do not silently turn an operation embedding into such a polar input. Readout biases can still emit invariant logits.

## 3. Exact terminal consumer on each stabilizer stratum

scripts/x369_source_geometry.py consume: concatenate projected local output/history z, apply learned affine Wz+b, average label permutations over K, then inverse-frame transport. Let Q_K denote the block-diagonal label Reynolds operator, P_K paired-feature Reynolds. Ideal-real local logit map is Q_K W P_K raw_z + Q_K b. For fixed K and world mark, two raw packets yield equal logits iff their difference is in ker(Q_K W P_K). Row basis of this matrix is an exact sufficient terminal linear code, with dimension its rank on the full ambient packet domain (no reachability minimality claim). Different marks require label transport; matching current logits does not guarantee matching subsequent histories, routing, local marks or future outputs. The rank is bounded by min(rank P_K, rank Q_K). The label rank is the count of K-orbits in each of the mean 27, Gram 14, signed 27 classes. For K={e}, 68; order-3 K, 11+6+11=28; order-6 K without n, 7+6+10=23 or 10+6+7=23. For K with n the feature rank is zero, not necessarily the bias. Gram is antipodal; signed head uses reflection-twisted permutation. Never identify their quotients.

For a prospective nonlinear recurrence code q, the exact remaining weld (reuse #1550) is: on the declared reachable domain, for every allowed control and same revision, q(h)=q(h') implies equal declared emitted observations and q(T(h,u))=q(T(h',u)), including guard/legality equivalence if controls depend on state. This one-step closure gives all future words by induction. A counterexample is a same-code pair with different observations now or after a separating continuation. This is the constructive admission criterion; terminal row rank or finite audits do not imply it. The reversible marked codec above satisfies it trivially by decoding equality; further lossy compression requires the actual fitted source and declared consumer.

## 4. Executed reference and remaining gates

Local Python/SymPy exact 6-slot reference derived from the inspected tuple action: enumerated the 16 subgroups of G, checked P_K^2=P_K and rank formula on all 16, calculated label fixed dimensions from the independent ternary/Gram labels. No original Torch source integration, GPU, Agda kernel, CI, fitted checkpoint or native performance run. Mathematical proof does not depend on this enumeration; no broader synthetic sweep admitted.

| Gate | Exact next deliverable | Status |
|---|---|---|
| G5a4a | signed fixed-stratum coordinate codec with mark/subgroup roundtrip | ideal-real construction/proof; local exact projector reference |
| G5a4b | source-native gather/scatter implementation, signed zero and strict fixed-space guard, preserve inactive identity | implementation unpaid; do not label native accepted |
| G5a4c | terminal projected-readout factorization and exact rank criterion | proved at fixed frame/stratum; fitted rank not measured |
| G5a3/F2g | additional fitted reachable nonlinear consumer quotient or separating word | open; do not train to rediscover theorem |
| G6 | numeric source transport, route/order/chart guards, carry and error propagation | open |
| native wiring | existing scripts.check_x369_source_geometry_native, zero optimizer steps | host receipt missing here |
| learning dynamics | extract existing 40 native audits before any new training | raw receipts absent here |
| L7 / 96-arm | reopen only if a declared unresolved consumer admits an actionable discriminator | deferred |

Next implementation should use signed-orbit gather/scatter as an opt-in exact storage/projection path; avoid modifying the original source function or introducing a CPU training lane. Profile native gfx803 only after correctness admission. Original source credit and evidence hierarchy retained.
