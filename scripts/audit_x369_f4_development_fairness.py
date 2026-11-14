"""X369 F4 development-only adjudication, BEFORE opening frozen F5 rows.

The original two-geometry screen compares BEST DEVELOPMENT CEs, but
changed routing features can make two geometries differ at initialization.
For each seed, decompose
  best_C3-best_C2 = (init_C3-init_C2) - (gain_C3-gain_C2)
where gain := init-best. This distinguishes initial representation choice
from router-specific training improvement without consuming F5.

Requires SHA256-bound fresh-split provenance; only reads the development
array digest recorded by the preparation tool. The reserved array is NOT
loaded or inspected; its hash is copied as a promise, not a proof of
independent source documents. No release claim or bootstrap from 2 seeds.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path


def adjudicate_f4_development(screen, split):
    if (screen.get("kind") != "original_bounded_c2_vs_c2c3_router_matched_screen"
            or screen.get("release_admitted") is not False):
        raise ValueError("need original unpromoted two-arm F4 receipt")
    if split.get("kind") != "x369_f4_f6_fresh_real_document_token_split":
        raise ValueError("need original fresh split provenance")
    file_hashes=split.get("output_token_array_sha256")
    docs=split.get("source_document_ids")
    texts=split.get("source_text_sha256")
    if (not isinstance(file_hashes,dict) or
            not isinstance(file_hashes.get("development"),str) or
            not isinstance(file_hashes.get("reserved"),str) or
            file_hashes["development"]!=screen.get("fresh_validation_sha256") or
            not isinstance(docs,dict) or not isinstance(texts,dict)):
        raise ValueError("development source hash not bound to F4 screen")
    n=screen.get("validation_rows")
    ids=docs.get("development")
    raw=texts.get("development")
    if (type(n) is not int or n<3 or
            not isinstance(ids,list) or len(ids)!=n or
            not isinstance(raw,list) or len(raw)!=n or
            len(set(ids))!=n or len(set(raw))!=n or
            any(type(v) is not str or not v for v in ids) or
            any(type(v) is not str or len(v)!=64 for v in raw)):
        raise ValueError("development document identity malformed or duplicate")
    reserved_ids=docs.get("reserved")
    reserved_texts=texts.get("reserved")
    if (not isinstance(reserved_ids,list) or
            not isinstance(reserved_texts,list) or
            len(reserved_ids)!=len(reserved_texts) or
            len(reserved_ids)<1 or
            len(set(ids+reserved_ids))!=n+len(reserved_ids) or
            len(set(raw+reserved_texts))!=n+len(reserved_texts)):
        raise ValueError("development/reserved document identity collision")
    arms=screen.get("arms")
    paired=screen.get("paired")
    if (not isinstance(arms,list) or not isinstance(paired,list)
            or not paired or len(arms)!=2*len(paired)):
        raise ValueError("paired two-arm evidence incomplete")
    steps=screen.get("fit_steps_per_arm")
    if type(steps) is not int or steps<1:
        raise ValueError("bad declared matched steps")
    geometries=("c2_invariant","c2c3_wave")
    by={}
    for arm in arms:
        seed=arm.get("seed")
        geo=arm.get("geometry")
        if (type(seed) is not int or geo not in geometries or
                (seed,geo) in by):
            raise ValueError("duplicate or unexpected source arm")
        if arm.get("steps")!=steps:
            raise ValueError("arm differs in actual training budget")
        init=arm.get("initial_fresh_ce")
        best=arm.get("best_fresh_ce")
        best_step=arm.get("best_step")
        count=arm.get("trained_parameter_count")
        if (not all(type(v) in (int,float) and math.isfinite(v)
                    for v in (init,best)) or
                type(best_step) is not int or not 0<=best_step<=steps or
                type(count) is not int or count<1 or
                best>init+1e-7):
            raise ValueError("nonfinite or inconsistent initial/best CE")
        if (not isinstance(arm.get("candidate_checkpoint_sha256"),str) or
                len(arm["candidate_checkpoint_sha256"])!=64):
            raise ValueError("F4 frozen arm checkpoint SHA missing")
        by[(seed,geo)]=arm
    paired_report=[]
    seen=set()
    for pair in paired:
        seed=pair.get("seed")
        if type(seed) is not int or seed in seen:
            raise ValueError("duplicate or invalid paired seed")
        seen.add(seed)
        if any((seed,g) not in by for g in geometries):
            raise ValueError("matched paired seed arm absent")
        c2=by[(seed,geometries[0])]
        c3=by[(seed,geometries[1])]
        if c2["trained_parameter_count"]!=c3["trained_parameter_count"]:
            raise ValueError("router trained parameter counts do not match")
        c2best=float(c2["best_fresh_ce"])
        c3best=float(c3["best_fresh_ce"])
        delta=c3best-c2best
        if any(type(pair.get(k)) not in (float,int) or
               not math.isfinite(pair[k]) or abs(pair[k]-v)>1e-7
               for k,v in (("c2_best_ce",c2best),("c2c3_best_ce",c3best),
                           ("c2c3_minus_c2",delta))):
            raise ValueError("paired development values disagree with arms")
        c2start=float(c2["initial_fresh_ce"])
        c3start=float(c3["initial_fresh_ce"])
        rawstart=c3start-c2start
        gain2=c2start-c2best
        gain3=c3start-c3best
        train_delta=gain3-gain2
        if abs(delta-(rawstart-train_delta))>1e-10:
            raise AssertionError("development CE decomposition failed")
        paired_report.append({
            "seed":seed,"initial_c3_minus_c2":rawstart,
            "c2_training_gain":gain2,"c3_training_gain":gain3,
            "c3_minus_c2_training_gain":train_delta,
            "best_c3_minus_c2":delta,
            "c3_better_at_initialization":rawstart<0,
            "c3_better_after_development_fit":delta<0})
    if len(seen)!=len(by)//2:
        raise ValueError("unpaired original F4 arm seeds")
    mean=lambda key:sum(float(row[key]) for row in paired_report)/len(paired_report)
    init_mean=mean("initial_c3_minus_c2")
    gain_mean=mean("c3_minus_c2_training_gain")
    best_mean=mean("best_c3_minus_c2")
    selected="c2c3_wave" if best_mean<0 else "c2_invariant"
    return {"schema":1,
            "kind":"x369_f4_development_only_initial_vs_training_gain",
            "selected_on_development_only":selected,
            "frozen_f5_reserved_array_sha256_unopened":
                file_hashes["reserved"],
            "fresh_validation_sha256":file_hashes["development"],
            "original_document_identity_claimed_complete":
                split.get("source_document_identity_inventory_claimed_complete")
                is True,
            "original_document_identity_independently_verified":False,
            "development_rows":n,"seed_count":len(paired_report),
            "paired":paired_report,
            "mean_initial_c3_minus_c2":init_mean,
            "mean_c3_minus_c2_training_gain":gain_mean,
            "mean_best_c3_minus_c2":best_mean,
            "f5_document_independent_release_admitted":False,
            "causal_c3_learning_proved":False,
            "no_F5_rows_loaded_or_scored":True,
            "note":"Initial geometry can change CE before any router updates. "
                   "Separate initial advantage from training gain; F5 remains "
                   "unopened until protocol-level approval."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen-receipt",type=Path,required=True)
    p.add_argument("--fresh-provenance",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("refuse overwrite original F4 fairness decision")
    screen=json.loads(a.screen_receipt.read_text())
    split=json.loads(a.fresh_provenance.read_text())
    out=adjudicate_f4_development(screen,split)
    out["original_f4_receipt_sha256"]=hashlib.sha256(
        a.screen_receipt.read_bytes()).hexdigest()
    out["original_split_provenance_sha256"]=hashlib.sha256(
        a.fresh_provenance.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,sort_keys=True,indent=2,
                                    allow_nan=False)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":main()
