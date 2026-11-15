"""R1 standalone release CPU-only manifest + source receipt contracts.

All rows are test fixtures. No PostgreSQL access and no physical release
claim originates from these tests.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.run_x369_r1_native_source_only import original_jsonl_inventory
from scripts.run_x369_r4_three_arm_source_maxcut import validated_r1_summary


class R1NativeOnlyContracts(unittest.TestCase):
    def sample(self, root):
        manifest = []
        output = []
        for i, split in enumerate(("train", "development"), 1):
            raw = "Story %s." % i
            h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            ref = "source:%s" % i
            run = "parser:%s" % i
            manifest.append({
                "source_revision_ref": ref, "parser_run_ref": run,
                "document_ref": "doc:%s" % i,
                "source_sha256": h, "split": split})
            output.append({
                "source_revision_ref": ref, "slr_parser_run_ref": run,
                "document_id": "doc:%s" % i,
                "source_sha256": h, "source_text": raw,
                "split": split, "parser_receipt_sha256": "a"*64,
                "parser_contract_sha256": "b"*64,
                "schema_version": "slr.numeric_spacy.leech_grammar.v1",
                "visibility": "full_context",
                "source_coordinate_unit": "unicode_codepoints",
                "slr_tokens": [{"token_id": 1}],
                "non_promoting_native_parser_projection": True,
                "world_resolution_deferred": True,
                "parser_observation_is_semantic_authority": False})
        m = root/"manifest.json"
        o = root/"out.jsonl"
        m.write_text(json.dumps(manifest))
        o.write_text("\n".join(json.dumps(x) for x in output)+"\n")
        return m,o,manifest,output

    def test_r1_source_inventory_reconciles_all_original_manifest_rows(self):
        with TemporaryDirectory() as d:
            m,o,_,_ = self.sample(Path(d))
            inventory = original_jsonl_inventory(o,m)
            self.assertEqual(inventory["source_count"],2)
            self.assertEqual(inventory["train_count"],1)
            self.assertEqual(inventory["development_count"],1)
            self.assertTrue(inventory["all_manifest_source_revisions_present"])

    def test_reject_wrong_original_sha_duplicate_and_reserved(self):
        with TemporaryDirectory() as d:
            m,o,manifest,output = self.sample(Path(d))
            cases = [
                [output[0],{**output[1],"source_sha256":"0"*64}],
                [output[0],{**output[1],"document_id":"doc:1"}],
                [output[0],{**output[1],"split":"reserved"}],
                [output[0],{**output[1],"source_text":"fabricated"}],
                [output[0]],
            ]
            for bad in cases:
                with self.subTest(bad=bad[-1]):
                    o.write_text("\n".join(json.dumps(x) for x in bad)+"\n")
                    with self.assertRaises(ValueError):
                        original_jsonl_inventory(o,m)

    def test_r4b_accepts_independent_native_r1_only_not_r3(self):
        sha="c"*64
        tok="d"*64
        receipt={
            "schema":1,
            "kind":"x369_r1_native_pg_source_only_v1",
            "parser_authority":"slr_native_pg",
            "F5_raw_source_opened":False,
            "original_tokenizer_sha256":tok,
            "original_SLR_source_export_sha256":sha,
            "source_export_R1_only":True,
            "native_PG_readonly_snapshot":True,
            "slr_exporter_original_receipt":{
                "kind":"slr_native_pg_leech_source_export",
                "output_sha256":sha,"source_count":2,
                "train_count":1,"development_count":1,
                "F5_source_read":False,
                "candidate_parser_promoted_to_truth":False},
            "source_inventory":{
                "source_count":2,"train_count":1,
                "development_count":1,
                "all_manifest_source_revisions_present":True,
                "source_document_identity_unique":True},
            "verified_parser_source_manifest":{
                "source_splits":["development","train"],
                "source_manifest_rows":2,
                "F5_reserved_source_read":False}
        }
        accepted=validated_r1_summary(receipt,jsonl_sha=sha,tokenizer_sha=tok)
        self.assertEqual(accepted["source_manifest_rows"],2)
        for key,val in (
            ("parser_authority","sensiblaw_legacy"),
            ("native_PG_readonly_snapshot",False),
            ("F5_raw_source_opened",True),
            ("source_export_R1_only",False),
            ("original_SLR_source_export_sha256","0"*64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validated_r1_summary(
                    {**receipt,key:val},jsonl_sha=sha,tokenizer_sha=tok)
        with self.assertRaises(ValueError):
            validated_r1_summary(
                {**receipt, "source_inventory":{
                    **receipt["source_inventory"],"source_count":3}},
                jsonl_sha=sha,tokenizer_sha=tok)


if __name__=="__main__":
    unittest.main()
