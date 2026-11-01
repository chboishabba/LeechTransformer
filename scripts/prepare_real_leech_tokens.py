"""Prepare REAL Leech-LILA token-ID train/heldout arrays from original sources.

Uses the checked-in 2048-token SentencePiece model, not a guessed tokenizer.
Supports original notebook TinyStories and original training FineWeb-Edu.
No random/synthetic token fallback. Both splits have disjoint source
records, repeated text is deduplicated across BOTH sides, and sampled
windows contain genuine consecutive tokenizer IDs without padding.

Usage:
 python -m scripts.prepare_real_leech_tokens --source tinystories \
   --train-rows 1024 --heldout-rows 256 --length 128 \
   --output-dir artifacts/real_tinystories

FineWeb-Edu train/heldout follow data_utils/streaming_dataset.py:
reserved first 5000 source records for validation, training from skip(5000).
The default sample-10BT configuration is an affordable *new experiment*
and does not imply it matches the source checkpoint's historical dataset
config. Set --fineweb-config default for the original default configuration.

This script requires access to public Hugging Face datasets. Source IDs,
dataset revision, tokenizer hash and output hashes are saved in metadata.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable,Mapping
import numpy as np


def _sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def _tokenize_document(text:str,tokenizer,length:int,rng:random.Random):
    if not isinstance(text,str) or not text.strip():
        return None
    ids=tokenizer.encode(text)
    if len(ids)<length+1:
        return None  # no padding or manufactured token continuity
    start=rng.randrange(len(ids)-length) if len(ids)>length+1 else 0
    # A full contiguous token window, leaving a next-token continuation
    # available in original document even if capture needs only the input.
    return np.asarray(ids[start:start+length],dtype=np.int64)


def tokenize_distinct_documents(
        train_docs:Iterable[Mapping],eval_docs:Iterable[Mapping],
        tokenizer,*,train_rows:int,heldout_rows:int,length:int,
        seed:int,max_scanned:int=500000):
    if min(train_rows,heldout_rows,length,max_scanned)<1:
        raise ValueError("positive row/length/scan constraints required")
    seen=set()
    output={}
    audit={}
    for split,docs,needed,offset in (
        ("train",train_docs,train_rows,0),
        ("heldout",eval_docs,heldout_rows,0xFACE)):
        rng=random.Random(seed+offset)
        rows=[]
        scanned=0
        seen_duplicates=0
        skipped_short=0
        text_hashes=[]
        for example in docs:
            scanned+=1
            if scanned>max_scanned:
                break
            if not isinstance(example,dict) or not isinstance(example.get("text"),str):
                continue
            text=example["text"]
            digest=_sha(text.encode("utf-8"))
            if digest in seen:
                seen_duplicates+=1
                continue
            window=_tokenize_document(text,tokenizer,length,rng)
            if window is None:
                skipped_short+=1
                continue
            if len(window)!=length:
                raise AssertionError("tokenizer length corruption")
            seen.add(digest)
            rows.append(window)
            text_hashes.append(digest)
            if len(rows)==needed:
                break
        if len(rows)!=needed:
            raise ValueError(
                f"{split} source exhausted: got {len(rows)}/{needed} rows, "
                f"scanned {scanned}, short {skipped_short}, duplicated {seen_duplicates}")
        output[split]=np.stack(rows)
        audit[split]={"rows":len(rows),"source_documents_scanned":scanned,
            "skipped_short":skipped_short,"duplicate_documents":seen_duplicates,
            "first_source_sha256":text_hashes[0],
            "document_hash_set_sha256":_sha("".join(sorted(text_hashes)).encode()),
            "window_array_sha256":_sha(output[split].tobytes())}
    return output,audit


def source_iterators(source:str,*,fineweb_config:str="sample-10BT",
                     dataset_revision:str|None=None,
                     train_text:Path|None=None,
                     heldout_text:Path|None=None):
    if source=="local-text":
        if train_text is None or heldout_text is None or (
                train_text.resolve()==heldout_text.resolve()):
            raise ValueError("two distinct local training/heldout files required")
        def read_docs(path):
            with path.open("r",encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        yield {"text":line.strip()}
        return read_docs(train_text),read_docs(heldout_text),{
            "dataset":"local-text",
            "train_source":str(train_text.resolve()),
            "heldout_source":str(heldout_text.resolve()),
            "disjoint":True}
    # Import only for remote sources, so local-tests do not need datasets.
    from datasets import load_dataset
    revision_args={"revision":dataset_revision} if dataset_revision else {}
    if source=="tinystories":
        train=load_dataset("roneneldan/TinyStories",split="train",
                           streaming=True,**revision_args)
        valid=load_dataset("roneneldan/TinyStories",split="validation",
                           streaming=True,**revision_args)
        meta={"dataset":"roneneldan/TinyStories",
              "train_split":"train","heldout_split":"validation",
              "dataset_revision":dataset_revision}
    elif source=="fineweb-edu":
        dataset=load_dataset("HuggingFaceFW/fineweb-edu",name=fineweb_config,
                             split="train",streaming=True,**revision_args)
        valid=dataset.take(5000)
        train=dataset.skip(5000)
        meta={"dataset":"HuggingFaceFW/fineweb-edu",
              "subset":fineweb_config,"heldout_source":"take(first 5000)",
              "train_source":"skip(first 5000)",
              "dataset_revision":dataset_revision}
    else:
        raise ValueError("unrecognized source")
    return iter(train),iter(valid),meta


def prepare(outdir:Path,*,source:str,train_rows:int,
            heldout_rows:int,length:int,seed:int,
            tokenizer_path:Path|None=None,
            expected_vocab:int=2048,
            fineweb_config:str="sample-10BT",
            dataset_revision:str|None=None,
            train_text:Path|None=None,heldout_text:Path|None=None,
            max_scanned:int=500000):
    import sentencepiece as spm
    root=Path(__file__).resolve().parents[1]
    if tokenizer_path is None:
        tokenizer_path=root/"vocab"/"e8_morpheme_.model"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer missing: {tokenizer_path}")
    tokenizer=spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.get_piece_size()!=expected_vocab:
        raise ValueError("tokenizer vocabulary disagrees with checkpoint")
    train,valid,source_meta=source_iterators(
        source,fineweb_config=fineweb_config,dataset_revision=dataset_revision,
        train_text=train_text,heldout_text=heldout_text)
    arrays,audit=tokenize_distinct_documents(
        train,valid,tokenizer,train_rows=train_rows,heldout_rows=heldout_rows,
        length=length,seed=seed,max_scanned=max_scanned)
    for side in arrays:
        if arrays[side].min()<0 or arrays[side].max()>=expected_vocab:
            raise ValueError("tokenizer generated out-of-vocabulary IDs")
    outdir.mkdir(parents=True,exist_ok=True)
    for side in ("train","heldout"):
        np.save(outdir/f"{side}_ids.npy",arrays[side])
    metadata={
        "real_text_source":True,"not_synthetic":True,
        "source":source_meta,"tokenizer_path":str(tokenizer_path.resolve()),
        "tokenizer_sha256":_sha(tokenizer_path.read_bytes()),
        "tokenizer_vocab":expected_vocab,
        "seed":seed,"length":length,"split_audit":audit,
        "record_separation":"train and heldout use distinct upstream splits or reserved document ranges, and duplicate text SHA256 across both is rejected",
        "warning":"This material is a newly constructed sample from the documented training families, not a byte-for-byte recovery of original checkpoint training windows."
    }
    (outdir/"provenance.json").write_text(json.dumps(metadata,indent=2))
    return metadata


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",choices=("tinystories","fineweb-edu","local-text"),
                    default="tinystories")
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--train-rows",type=int,default=1024)
    ap.add_argument("--heldout-rows",type=int,default=256)
    ap.add_argument("--length",type=int,default=128)
    ap.add_argument("--seed",type=int,default=41)
    ap.add_argument("--tokenizer",type=Path)
    ap.add_argument("--expected-vocab",type=int,default=2048)
    ap.add_argument("--fineweb-config",default="sample-10BT")
    ap.add_argument("--dataset-revision",help="optional immutable source revision")
    ap.add_argument("--train-text",type=Path)
    ap.add_argument("--heldout-text",type=Path)
    ap.add_argument("--max-scanned",type=int,default=500000)
    args=ap.parse_args()
    result=prepare(args.output_dir,source=args.source,
        train_rows=args.train_rows,heldout_rows=args.heldout_rows,
        length=args.length,seed=args.seed,
        tokenizer_path=args.tokenizer,expected_vocab=args.expected_vocab,
        fineweb_config=args.fineweb_config,
        dataset_revision=args.dataset_revision,
        train_text=args.train_text,heldout_text=args.heldout_text,
        max_scanned=args.max_scanned)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
