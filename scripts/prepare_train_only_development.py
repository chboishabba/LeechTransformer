"""Disjoint TRAIN-ONLY fit/development split for diagnostic student sweeps.

Consumes the existing source-audited preparation's one-row-per-distinct
document train_ids.npy. Does NOT touch, load or split heldout_ids.npy.
Preserves original train-split bytes and records row indices and SHA-256.

Do not relabel the development partition as independent final heldout.
Original source preparation: scripts/prepare_real_leech_tokens.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def _sha_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha_file(path:Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_train_only_development(dataset:Path, output:Path, *,
                                  seed:int=41,development_fraction:float=.2):
    if not 0<development_fraction<.5:
        raise ValueError("development fraction must be strictly between 0 and .5")
    metadata_path=dataset/"provenance.json"
    train_path=dataset/"train_ids.npy"
    metadata=json.loads(metadata_path.read_text())
    if (metadata.get("not_random_token_smoke") is not True or
        metadata.get("real_dataset_samples") is not True or
        "duplicate text SHA256" not in metadata.get("record_separation","")):
        raise ValueError("source-audited real preparation required")
    audit=metadata.get("split_audit",{}).get("train",{})
    rows=np.load(train_path,allow_pickle=False)
    if (rows.ndim!=2 or len(rows)<10 or rows.shape[1]<2 or
        not np.issubdtype(rows.dtype,np.integer) or
        audit.get("rows")!=len(rows) or
        _sha_bytes(rows.tobytes())!=audit.get("window_array_sha256")):
        raise ValueError("source train data do not match provenance")
    # prepare_real_leech_tokens uses one distinct document per row, with
    # duplicate text hashes rejected at ingestion. This inherits that
    # SOURCE process claim; the audit lacks individual document hashes.
    count=max(2,int(round(len(rows)*development_fraction)))
    if len(rows)-count<8:
        raise ValueError("insufficient development fit rows")
    perm=np.random.default_rng(seed).permutation(len(rows))
    dev_indices=np.sort(perm[:count])
    fit_indices=np.sort(perm[count:])
    fit=rows[fit_indices].copy()
    dev=rows[dev_indices].copy()
    if set(fit_indices.tolist()) & set(dev_indices.tolist()):
        raise AssertionError("train/dev index overlap")
    if {row.tobytes() for row in fit} & {row.tobytes() for row in dev}:
        raise ValueError("train/dev token windows are duplicated")
    output.mkdir(parents=True,exist_ok=True)
    fit_path=output/"train_only_fit.npy"
    dev_path=output/"train_only_development.npy"
    if fit_path.resolve()==train_path.resolve() or dev_path.resolve()==train_path.resolve():
        raise ValueError("refuse to overwrite original full train split")
    np.save(fit_path,fit)
    np.save(dev_path,dev)
    receipt={
        "scope":"training-only development split, NOT final heldout",
        "origin_provenance_sha256":_sha_file(metadata_path),
        "origin_train_file_sha256":_sha_file(train_path),
        "origin_train_array_sha256":_sha_bytes(rows.tobytes()),
        "origin_one_distinct_document_per_row":
            "inherited from prepare_real_leech_tokens.py source sampling; not independently reconstructed from per-document identifiers",
        "seed":seed,"development_fraction":development_fraction,
        "original_train_rows":len(rows),
        "fit_rows":len(fit),"development_rows":len(dev),
        "fit_indices":fit_indices.tolist(),
        "development_indices":dev_indices.tolist(),
        "fit_array_sha256":_sha_bytes(fit.tobytes()),
        "development_array_sha256":_sha_bytes(dev.tobytes()),
        "fit_file_sha256":_sha_file(fit_path),
        "development_file_sha256":_sha_file(dev_path),
        "source":"original token IDs: no synthetic tokens; no final heldout read",
        "not_final_model_quality_evidence":True}
    (output/"train_only_dev_provenance.json").write_text(
        json.dumps(receipt,indent=2))
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,required=True,
                        help="saved real token-preparation directory")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=41)
    parser.add_argument("--development-fraction",type=float,default=.2)
    args=parser.parse_args()
    report=create_train_only_development(
        args.dataset,args.output,seed=args.seed,
        development_fraction=args.development_fraction)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
