"""X369 R1 native SLR PostgreSQL-only original source handoff (CPU-only).

This CPU stage can finish and emit an independent R1 receipt while RX580 F3
runs. R3 full grammar and R4b need not be finished to read this receipt.
The ONLY parser and source authority is SLR's read-only native PG exporter.
No fixture/legacy rows, invented document identities or unsealed F5 inputs.

Does NOT synthesize a missing production PG database or parser jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_x369_three_repo_physical_maxcut import (
    preflight_original_parser_manifest,
)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def original_jsonl_inventory(path, manifest):
    """Check SLR export output row by row against immutable source manifest.

    A hash-matched genuine SLR exporter output is necessary, not sufficient
    alone, to establish provenance of any historical training documents.
    """
    specs = json.loads(Path(manifest).read_text(encoding="utf-8"))
    by_source = {r["source_revision_ref"]: r for r in specs}
    if len(by_source) != len(specs):
        raise ValueError("source revision duplication in manifest")
    seen = set()
    sha_splits = {}
    counts = {"train": 0, "development": 0}
    doc_ids = set()
    parser_contracts = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            raise ValueError("blank native SLR source output row")
        r = json.loads(line)
        source_ref = r.get("source_revision_ref")
        if source_ref not in by_source or source_ref in seen:
            raise ValueError("extra, duplicate or unknown PG source revision")
        spec = by_source[source_ref]
        if (r.get("schema_version") != "slr.numeric_spacy.leech_grammar.v1"
                or r.get("split") != spec["split"]
                or r.get("slr_parser_run_ref") != spec["parser_run_ref"]
                or r.get("source_sha256") != spec["source_sha256"]
                or (spec.get("document_ref") is not None and
                    r.get("document_id") != spec["document_ref"])
                or r.get("visibility") != "full_context"
                or r.get("source_coordinate_unit") != "unicode_codepoints"
                or r.get("non_promoting_native_parser_projection") is not True
                or r.get("world_resolution_deferred") is not True
                or r.get("parser_observation_is_semantic_authority") is not False):
            raise ValueError("native PG document/identity/schema/authority mismatch")
        raw = r.get("source_text")
        if not isinstance(raw, str) or not raw or hashlib.sha256(
                raw.encode("utf-8")).hexdigest() != spec["source_sha256"]:
            raise ValueError("original canonical PG source text mismatch")
        if (r.get("document_id") in doc_ids or
                not isinstance(r.get("parser_receipt_sha256"), str) or
                len(r["parser_receipt_sha256"]) != 64 or
                not isinstance(r.get("slr_tokens"), list) or
                not r["slr_tokens"]):
            raise ValueError("duplicate document or incomplete PG parser proof")
        seen.add(source_ref)
        doc_ids.add(r["document_id"])
        parser_contracts.add(r.get("parser_contract_sha256"))
        counts[r["split"]] += 1
        old = sha_splits.setdefault(r["source_sha256"], r["split"])
        if old != r["split"]:
            raise ValueError("cross-split original source text duplicate")
    if (seen != set(by_source) or
            min(counts.values()) < 1 or
            len(parser_contracts) != 1 or
            any(not isinstance(s, str) or len(s) != 64 for s in parser_contracts)):
        raise ValueError("native SLR export missing rows/splits/common parser contract")
    return {
        "source_count": len(seen),
        "train_count": counts["train"],
        "development_count": counts["development"],
        "source_document_identity_unique": True,
        "parser_contract_sha256": next(iter(parser_contracts)),
        "all_manifest_source_revisions_present": True,
    }


def _git_output(repo, *args):
    proc = subprocess.run(["git", "-C", str(repo), *args],
                          check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def run(args):
    if args.output_root.exists():
        raise FileExistsError("R1 outputs must use a fresh directory")
    if not args.slr_repo.is_dir() or not args.manifest.is_file():
        raise ValueError("native SLR repository and original manifest required")
    if not args.tokenizer.is_file() or file_sha256(args.tokenizer) != args.tokenizer_sha256:
        raise ValueError("original SentencePiece tokenizer SHA256 not pinned")
    if not os.environ.get("DATABASE_URL"):
        raise ValueError("production SLR DATABASE_URL required in environment only")
    if args.slr_python is None:
        python = sys.executable
    else:
        python = str(args.slr_python)
        if not Path(python).is_file():
            raise ValueError("native SLR Python executable does not exist")
    head = _git_output(args.slr_repo, "rev-parse", "HEAD")
    if head != args.slr_revision or _git_output(args.slr_repo, "status", "--porcelain"):
        raise ValueError("SLR repo not at exact clean pinned native PG exporter revision")
    if not (args.slr_repo / "python/slr_pg_leech_grammar_export.py").is_file():
        raise ValueError("no native SLR PostgreSQL exporter at source revision")
    original = preflight_original_parser_manifest(args.manifest, "slr_native_pg")
    args.output_root.mkdir(parents=True, mode=0o700)
    source = args.output_root / "slr_original_train_dev.jsonl"
    receipt_path = args.output_root / "r1_native_source_receipt.json"
    # Production SLR alone reads from a repeatable-read, read-only DB snapshot.
    # Credentials are inherited through the CPU subprocess environment only,
    # not printed or added to command-line arguments.
    proc = subprocess.run([
        python, "-m", "python.slr_pg_leech_grammar_export",
        "--manifest", str(args.manifest.resolve()),
        "--output", str(source.resolve()),
    ], cwd=args.slr_repo, check=True, capture_output=True, text=True)
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError("native SLR exporter exited without original source data")
    inventory = original_jsonl_inventory(source, args.manifest)
    exporter = json.loads(proc.stdout)
    if (exporter.get("kind") != "slr_native_pg_leech_source_export"
            or exporter.get("output_sha256") != file_sha256(source)
            or exporter.get("source_count") != inventory["source_count"]
            or exporter.get("train_count") != inventory["train_count"]
            or exporter.get("development_count") != inventory["development_count"]
            or exporter.get("F5_source_read") is not False
            or exporter.get("candidate_parser_promoted_to_truth") is not False):
        raise ValueError("SLR native exporter output and audited source disagree")
    receipt = {
        "schema": 1,
        "kind": "x369_r1_native_pg_source_only_v1",
        "parser_authority": "slr_native_pg",
        "slr_git_revision": head,
        "original_manifest_sha256": file_sha256(args.manifest),
        "original_tokenizer_sha256": args.tokenizer_sha256,
        "original_SLR_source_export_sha256": file_sha256(source),
        "original_SLR_source_export_name": source.name,
        "source_inventory": inventory,
        "verified_parser_source_manifest": original,
        "slr_exporter_original_receipt": exporter,
        "F5_raw_source_opened": False,
        "native_PG_readonly_snapshot": True,
        "historical_C2_training_document_disjointness_independently_verified": False,
        "physical_GPU_or_R3_trained": False,
        "source_export_R1_only": True,
    }
    receipt_path.write_text(json.dumps(
        receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "receipt": str(receipt_path),
        "source_jsonl": str(source),
        "source_count": inventory["source_count"],
        "source_sha256": receipt["original_SLR_source_export_sha256"],
        "F5_raw_source_opened": False,
    }, sort_keys=True))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slr-repo", type=Path, required=True)
    p.add_argument("--slr-revision", required=True)
    p.add_argument("--slr-python", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--tokenizer-sha256", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
