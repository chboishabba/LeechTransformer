"""X369 F4/F6: prepare honest fresh document-disjoint token windows.

Unlike a pure 16-token n-gram check, an original document inventory is
required for a document-disjoint RELEASE claim. Missing historical
provenance permits an explicitly nonadmissible provisional development
sample only. The final set is merely reserved, never evaluated here.

Local JSONL source: {"document_id":"immutable-id","text":"...", "split":
"development"|"reserved"}. The original 2048-token SentencePiece model
and original prior token arrays must be supplied at execution time.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def signatures(a,window):
    a=np.asarray(a)
    if a.ndim!=2 or a.shape[1]<window:
        raise ValueError("integer token arrays with enough context required")
    if not np.issubdtype(a.dtype,np.integer):
        raise ValueError("original arrays must contain token IDs")
    return {tuple(int(z) for z in row[k:k+window])
            for row in a for k in range(a.shape[1]-window+1)}

def validate_prior_inventory(manifest,prior_sha,*,provisional):
    if manifest is None:
        if not provisional:
            raise ValueError("historical document inventory required for release")
        return set(),set(),False
    if manifest.get("kind")!="x369_historical_document_inventory_v1":
        raise ValueError("unexpected historical document manifest kind")
    hashes=manifest.get("source_text_sha256")
    ids=manifest.get("source_document_ids")
    arrays=manifest.get("prior_token_array_sha256")
    if (not isinstance(hashes,list) or not isinstance(ids,list)
            or not isinstance(arrays,list) or not hashes or not ids
            or len(ids)!=len(hashes) or len(set(ids))!=len(ids)
            or len(set(hashes))!=len(hashes)
            or any(not isinstance(v,str) or len(v)!=64 for v in hashes)
            or any(not isinstance(v,str) or not v for v in ids)):
        raise ValueError("malformed historical source-document evidence")
    if not set(prior_sha).issubset(arrays):
        raise ValueError("historical manifest does not bind all prior token arrays")
    complete=manifest.get("complete") is True
    if not complete and not provisional:
        raise ValueError("only complete historical source-document inventory admits release")
    return set(hashes),set(ids),complete

def construct_fresh_splits(documents,tokenizer,prior_arrays,*,length,window,
                           development_rows,reserved_rows,seed,
                           historical_hashes=(),historical_ids=(),
                           historical_inventory_claimed_complete=False,
                           expected_vocab=2048):
    if (min(length,window,development_rows,reserved_rows,expected_vocab)<1
            or window>length or not prior_arrays):
        raise ValueError("invalid original token split contract")
    old_signatures=set()
    for a in prior_arrays:
        a=np.asarray(a)
        if (a.ndim!=2 or a.shape[0]<1 or a.shape[1]!=length or
                not np.issubdtype(a.dtype,np.integer) or a.min()<0 or
                a.max()>=expected_vocab):
            raise ValueError("all prior arrays must match original context width")
        old_signatures.update(signatures(a,window))
    seen_signatures=set(old_signatures)
    seen_hashes=set(historical_hashes)
    seen_ids=set(historical_ids)
    rng=random.Random(seed)
    selected={"development":[],"reserved":[]}
    rejected={"duplicate_document":0,"short_or_invalid":0,
              "prior_or_internal_token_overlap":0,"invalid_split":0}
    for document in documents:
        if not isinstance(document,dict):
            rejected["short_or_invalid"]+=1
            continue
        split=document.get("split")
        if split not in selected:
            rejected["invalid_split"]+=1
            continue
        if len(selected[split])>=(
                development_rows if split=="development" else reserved_rows):
            continue
        doc_id=document.get("document_id")
        raw=document.get("text")
        if not isinstance(doc_id,str) or not doc_id or (
                not isinstance(raw,str) or not raw.strip()):
            rejected["short_or_invalid"]+=1
            continue
        digest=sha(raw.encode("utf-8"))
        if doc_id in seen_ids or digest in seen_hashes:
            rejected["duplicate_document"]+=1
            continue
        tokens=np.asarray(tokenizer.encode(raw),dtype=np.int64)
        if (len(tokens)<length+1 or tokens.min()<0
                or tokens.max()>=expected_vocab):
            rejected["short_or_invalid"]+=1
            continue
        offsets=list(range(len(tokens)-length+1))
        rng.shuffle(offsets)
        chosen=None
        for pos in offsets:
            row=tokens[pos:pos+length]
            grams=signatures(row[None,:],window)
            if not grams.intersection(seen_signatures):
                chosen=(row.copy(),pos,grams)
                break
        if chosen is None:
            rejected["prior_or_internal_token_overlap"]+=1
            continue
        row,pos,grams=chosen
        seen_hashes.add(digest)
        seen_ids.add(doc_id)
        seen_signatures.update(grams)
        selected[split].append({"ids":row,"source_document_id":doc_id,
                                "text_sha256":digest,"token_offset":pos})
        if (len(selected["development"])==development_rows
                and len(selected["reserved"])==reserved_rows):
            break
    required={"development":development_rows,"reserved":reserved_rows}
    for split,count in required.items():
        if len(selected[split])!=count:
            raise ValueError(f"fresh {split} source exhausted: "
                f"{len(selected[split])}/{count}; rejection={rejected}")
    arrays={name:np.stack([rec["ids"] for rec in rows])
            for name,rows in selected.items()}
    return arrays,{
        "schema":1,"kind":"x369_f4_f6_fresh_real_document_token_split",
        "source_document_identity_inventory_claimed_complete":
            bool(historical_inventory_claimed_complete),
        "historical_document_inventory_independently_verified":False,
        "release_document_disjointness_admitted":False,
        "original_token_overlap_window":window,
        "tokenizer_context_length":length,
        "source_document_ids":{
            s:[r["source_document_id"] for r in rows]
            for s,rows in selected.items()},
        "source_text_sha256":{
            s:[r["text_sha256"] for r in rows]
            for s,rows in selected.items()},
        "source_token_offsets":{
            s:[r["token_offset"] for r in rows]
            for s,rows in selected.items()},
        "selected_rows":{s:len(rows) for s,rows in selected.items()},
        "rejected":rejected,
        "reserved_set_was_not_scored_or_fit":True,
        "selected_text_sha256_disjoint":True,
        "selected_token_windows_disjoint_from_all_supplied_prior_arrays":True,
        "note":"A self-reported complete historical inventory is not independent "
               "proof of historical document identity; release remains closed."
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--documents-jsonl",type=Path,required=True)
    p.add_argument("--tokenizer",type=Path,required=True)
    p.add_argument("--prior-ids",type=Path,nargs="+",required=True)
    p.add_argument("--historical-doc-inventory",type=Path)
    p.add_argument("--provisional",action="store_true")
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--length",type=int,default=128)
    p.add_argument("--window",type=int,default=16)
    p.add_argument("--development-rows",type=int,default=32)
    p.add_argument("--reserved-rows",type=int,default=32)
    p.add_argument("--seed",type=int,default=41)
    a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError("refuse overwrite any existing split/lockbox")
    if not a.tokenizer.is_file():
        raise FileNotFoundError("actual original tokenizer model absent")
    import sentencepiece as spm
    tok=spm.SentencePieceProcessor(model_file=str(a.tokenizer))
    if tok.get_piece_size()!=2048:
        raise ValueError("original Leech tokenizer requires 2048 vocab")
    prior=[np.load(path,allow_pickle=False) for path in a.prior_ids]
    prior_sha=[sha(path.read_bytes()) for path in a.prior_ids]
    manifest=(json.loads(a.historical_doc_inventory.read_text())
              if a.historical_doc_inventory is not None else None)
    hashes,ids,complete=validate_prior_inventory(manifest,prior_sha,
                                                   provisional=a.provisional)
    with a.documents_jsonl.open("r",encoding="utf8") as src:
        docs=(json.loads(line) for line in src if line.strip())
        splits,result=construct_fresh_splits(
            docs,tok,prior,length=a.length,window=a.window,
            development_rows=a.development_rows,
            reserved_rows=a.reserved_rows,seed=a.seed,
            historical_hashes=hashes,historical_ids=ids,
            historical_inventory_claimed_complete=complete)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for split,data in splits.items():
        np.save(a.output_dir/f"{split}_ids.npy",data,allow_pickle=False)
    result.update({
        "provisional":bool(a.provisional),
        "original_tokenizer_sha256":sha(a.tokenizer.read_bytes()),
        "input_jsonl_sha256":sha(a.documents_jsonl.read_bytes()),
        "historical_document_inventory_sha256":
            sha(a.historical_doc_inventory.read_bytes())
            if a.historical_doc_inventory else None,
        "prior_token_array_sha256":prior_sha,
        "output_token_array_sha256":{
            s:sha((a.output_dir/f"{s}_ids.npy").read_bytes())
            for s in splits}})
    (a.output_dir/"provenance.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))

if __name__=="__main__":main()
