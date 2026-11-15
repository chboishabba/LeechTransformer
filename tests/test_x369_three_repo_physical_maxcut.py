"""One-command physical X369 three-repo source/grokking pipeline contracts.

A patched process harness verifies orchestration and credential/lockbox
firewall only; it does not execute SLR database, HIP or original TinyStories.
"""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from scripts.run_x369_three_repo_physical_maxcut import (
    validate_preflight,run,run_command)


def setup(tmp,*,with_f3=False):
    root=Path(tmp)
    leech=root/"LeechTransformer"
    slr=root/"SensibLaw"
    leech.mkdir()
    (slr/"scripts").mkdir(parents=True)
    (slr/"scripts"/"export_leech_spacy_grammar.py").write_text("original-source")
    python=slr/"venv"/"bin"/"python"
    python.parent.mkdir(parents=True)
    python.write_text("original-sensiblaw-python")
    ckpt=root/"original.pt";ckpt.write_bytes(b"original-model")
    tok=root/"original.model";tok.write_bytes(b"original-tokenizer")
    manifest=root/"original-manifest.json";manifest.write_text("[]")
    args=SimpleNamespace(sensiblaw_repo=slr,sensiblaw_python=python,
        manifest=manifest,original_checkpoint=ckpt,tokenizer=tok,
        config=None,checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        tokenizer_sha256=hashlib.sha256(tok.read_bytes()).hexdigest(),
        leech_revision="a"*40,sensiblaw_revision="b"*40,
        output_root=root/"output",f3_config=None,
        steps=2048,epochs=20,aux_weight=.2,device="cuda:0")
    if with_f3:
        f3={}
        for key in ("screen_receipt","source_candidate","original_train_ids",
                    "original_heldout_ids","selection_validation_ids",
                    "probe_ids","probe_provenance"):
            file=root/key;file.write_text("F3 pinned source")
            f3[key]=str(file)
        f3["reserved_lockbox_sha256"]="c"*64
        config=root/"f3.json"
        config.write_text(json.dumps(f3))
        args.f3_config=config
    return args,leech,slr


def fake_process(args,leech,slr,seen):
    def call(cmd,**kwargs):
        cmd=list(map(str,cmd))
        if cmd[0]=="git":
            if "rev-parse" in cmd:
                return SimpleNamespace(stdout=(
                    args.leech_revision if cmd[2]==str(leech)
                    else (getattr(args,"slr_revision",None) or
                          args.sensiblaw_revision))+"\n")
            return SimpleNamespace(stdout="")
        seen.append(cmd)
        assert os.environ["DATABASE_URL"] not in " ".join(cmd)
        if "-m" not in cmd:
            raise AssertionError("unexpected physical execution process")
        module=cmd[cmd.index("-m")+1]
        if module in ("scripts.export_leech_spacy_grammar",
                      "python.slr_pg_leech_grammar_export"):
            out=Path(cmd[cmd.index("--output")+1])
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text('{"source":"parser-exported"}\n')
        elif module=="scripts.audit_x369_slr_spacy_grammar_early_epoch":
            out=Path(cmd[cmd.index("--result")+1])
            out.write_text('{"source":"frozen-grammar"}\n')
        elif module=="scripts.train_x369_slr_joint_grokking":
            folder=Path(cmd[cmd.index("--output-dir")+1])
            folder.mkdir(parents=True)
            (folder/"matched_arm_summary.json").write_text(
                '{"source":"five-real-arms"}\n')
        elif module=="scripts.run_x369_f3_two_seed_maxcut":
            folder=Path(cmd[cmd.index("--output-dir")+1])
            folder.mkdir(parents=True)
            (folder/"f3_two_seed_summary.json").write_text(
                '{"source":"f3-two-seeds"}\n')
        else:
            raise AssertionError("unknown physical source producer")
        return SimpleNamespace(returncode=0)
    return call


class X369ThreeRepoPhysicalMaxCut(unittest.TestCase):
    def test_one_maxcut_executes_f3_then_source_then_probe_then_full_training(self):
        with TemporaryDirectory() as tmp:
            args,leech,slr=setup(tmp,with_f3=True)
            seen=[]
            with patch.dict(os.environ,{"DATABASE_URL":"postgres://secret:password@localhost/db"}):
                with patch("scripts.run_x369_three_repo_physical_maxcut.subprocess.run",
                           side_effect=fake_process(args,leech,slr,seen)):
                    out=run(args,leech_repo=leech)
            self.assertEqual([cmd[cmd.index("-m")+1] for cmd in seen],
                ["scripts.run_x369_f3_two_seed_maxcut",
                 "scripts.export_leech_spacy_grammar",
                 "scripts.audit_x369_slr_spacy_grammar_early_epoch",
                 "scripts.train_x369_slr_joint_grokking"])
            self.assertTrue((args.output_root/"physical_maxcut_summary.json").is_file())
            self.assertIn("original_5_arm_grokking_receipt_sha256",out)
            self.assertIn("F3_B_original_two_checkpoint_receipt_sha256",out)
            self.assertFalse(out["F5_raw_source_opened"])
            with patch.dict(os.environ,{"DATABASE_URL":"postgres://secret:password@localhost/db"}):
                with self.assertRaises(FileExistsError):
                    validate_preflight(args,leech_repo=leech)

    def test_native_SLR_PG_canonical_corpus_is_preferred_authority(self):
        with TemporaryDirectory() as tmp:
            args,leech,legacy=setup(tmp,with_f3=False)
            native=Path(tmp)/"slr"
            (native/"python").mkdir(parents=True)
            (native/"python"/"slr_pg_leech_grammar_export.py").write_text(
                "# strict Rust-ingest PostgreSQL source projection")
            native_python=native/"venv"/"bin"/"python"
            native_python.parent.mkdir(parents=True)
            native_python.write_text("# original SLR Python interpreter")
            args.sensiblaw_repo=None
            args.sensiblaw_python=None
            args.sensiblaw_revision=None
            args.slr_repo=native
            args.slr_python=native_python
            args.slr_revision="f"*40
            seen=[]
            with patch.dict(os.environ,{"DATABASE_URL":"postgres://secret@localhost/db"}):
                with patch("scripts.run_x369_three_repo_physical_maxcut.subprocess.run",
                           side_effect=fake_process(args,leech,native,seen)):
                    receipt=run(args,leech_repo=leech)
            self.assertEqual([cmd[cmd.index("-m")+1] for cmd in seen],
                ["python.slr_pg_leech_grammar_export",
                 "scripts.audit_x369_slr_spacy_grammar_early_epoch",
                 "scripts.train_x369_slr_joint_grokking"])
            self.assertEqual(receipt["parser_authority"],"slr_native_pg")
            self.assertEqual(receipt["parser_authority_git_revision"],"f"*40)
            self.assertNotIn("sensiblaw_code_revision",receipt)
            self.assertTrue(receipt["original_SLR_source_export_sha256"])

    def test_rejects_ambiguous_SLR_and_legacy_authority(self):
        with TemporaryDirectory() as tmp:
            args,leech,_=setup(tmp)
            args.slr_repo=Path(tmp)/"native"
            args.slr_python=Path(tmp)/"native-python"
            args.slr_revision="f"*40
            with self.assertRaisesRegex(ValueError,"exactly one"):
                validate_preflight(args,leech_repo=leech)

    def test_source_only_omits_physical_F3_but_still_trains_five_arms(self):
        with TemporaryDirectory() as tmp:
            args,leech,slr=setup(tmp)
            seen=[]
            with patch.dict(os.environ,{"DATABASE_URL":"postgres://secret:password@localhost/db"}):
                with patch("scripts.run_x369_three_repo_physical_maxcut.subprocess.run",
                           side_effect=fake_process(args,leech,slr,seen)):
                    result=run(args,leech_repo=leech)
            self.assertEqual(len(seen),3)
            self.assertNotIn("F3_B_original_two_checkpoint_receipt_sha256",result)

    def test_preflight_rejects_invalid_sources_before_any_database_read(self):
        with TemporaryDirectory() as tmp:
            args,leech,slr=setup(tmp)
            with patch.dict(os.environ,{"DATABASE_URL":""}):
                with self.assertRaisesRegex(ValueError,"DATABASE_URL"):
                    validate_preflight(args,leech_repo=leech)
            args.checkpoint_sha256="0"*64
            with patch.dict(os.environ,{"DATABASE_URL":"postgres://pwd@localhost/db"}):
                with self.assertRaisesRegex(ValueError,"original_checkpoint"):
                    validate_preflight(args,leech_repo=leech)

    def test_secret_not_allowed_in_process_argv(self):
        with patch.dict(os.environ,{"DATABASE_URL":"postgres://secret/db"}):
            with self.assertRaisesRegex(ValueError,"secret"):
                run_command(["python","--database-url","postgres://secret/db"],
                            cwd=Path("."))

if __name__=="__main__":
    unittest.main()
