"""One-command physical X369 F3-B + SLR grammar + real full-Leech grokking.

User-owned machine only: native HIP/RX580, SensibLaw PostgreSQL and immutable
TinyStories sources are NOT accessible to GitHub/this script without the
required paths and DATABASE_URL environment. No new parser; SLR owns source.

Workflow:
  optional F3-B actual seed41/42, same 12 masks, F5 SHA only;
  SLR original train/development numeric PostgreSQL export;
  Leech original 2048-tokenizer source-span validation;
  frozen original-head grammar diagnostic with majority/shuffle controls;
  FIVE same-checkpoint full-backbone 2048+-step grokking arms with
  zero-gated source parity, continuous/resume checkpoints and Gram spectra.

Requires explicit pinned Git revisions, original tokenizer/checkpoint file
SHA256 and clean worktrees BEFORE reading any original source or DB.
Subprocesses receive database credentials by ENV ONLY, never command argv.
No claim of Monster 3B identification, learned C3, or valid F5 evaluation.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fp:
        for data in iter(lambda:fp.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def validate_preflight(args,*,leech_repo=None):
    leech=Path(leech_repo or Path(__file__).resolve().parent.parent)
    slr=args.sensiblaw_repo.resolve()
    original_paths=(args.manifest,args.original_checkpoint,args.tokenizer)
    if args.config is not None:original_paths+= (args.config,)
    for path in original_paths:
        if not path.is_file():raise FileNotFoundError("original source not mounted: "+str(path))
    if not (slr/"scripts/export_leech_spacy_grammar.py").is_file():
        raise FileNotFoundError("SLR exporter not in selected SensibLaw source")
    if not args.sensiblaw_python.is_file():
        raise FileNotFoundError("SLR Python interpreter unavailable")
    if args.output_root.exists():
        raise FileExistsError("refuse previously used X369 maximum-cut output root")
    if not os.environ.get("DATABASE_URL"):
        raise ValueError("DATABASE_URL must be set securely in environment")
    for label,path,digest in (
        ("original_checkpoint",args.original_checkpoint,args.checkpoint_sha256),
        ("tokenizer",args.tokenizer,args.tokenizer_sha256)):
        if sha(path)!=digest:
            raise ValueError(label+" physical SHA256 did not match pinned receipt")
    for name,repo,revision in (
       ("Leech",leech,args.leech_revision),
       ("SensibLaw",slr,args.sensiblaw_revision)):
        actual=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],
                              capture_output=True,text=True,check=True).stdout.strip()
        if actual!=revision:
            raise ValueError(name+" code revision changed since source freeze")
        dirty=subprocess.run(["git","-C",str(repo),"status","--porcelain",
                              "--untracked-files=no"],
                              capture_output=True,text=True,check=True).stdout.strip()
        if dirty:
            raise ValueError(name+" source worktree not pristine")
    if args.f3_config is not None:
        if not args.f3_config.is_file():
            raise FileNotFoundError("F3 original locked source configuration unavailable")
        obj=json.loads(args.f3_config.read_text())
        fields=("screen_receipt","source_candidate","original_train_ids",
                "original_heldout_ids","selection_validation_ids",
                "probe_ids","probe_provenance","reserved_lockbox_sha256")
        if any(k not in obj for k in fields):
            raise ValueError("F3 source manifest lacks immutable source/lockbox SHA")
        if any(not Path(obj[k]).is_file() for k in fields[:-1]):
            raise ValueError("F3 source manifest refers to missing physical file")
        sha0=obj["reserved_lockbox_sha256"]
        if (not isinstance(sha0,str) or len(sha0)!=64 or any(
             c not in "0123456789abcdef" for c in sha0)):
            raise ValueError("F3 F5 SHA receipt invalid; never pass F5 raw source")
        # The reserved F5 bytes/path are explicitly NOT a manifest input.
        if any("f5" in k.lower() and k!="reserved_lockbox_sha256"
               for k in obj):
            raise ValueError("unexpected F5 raw path in physical F3 manifest")
    if (args.steps<1 or args.epochs<1 or args.aux_weight<0 or
        (getattr(args,"target_dev_ce",None) is not None and
         args.target_dev_ce<=0) or args.device not in (
        "cpu","cuda:0")):
        raise ValueError("invalid physical optimizer/host scope")
    return leech,slr


def run_command(args,*,cwd,env=None):
    # Source commands returned for audit must never contain DATABASE_URL.
    if os.environ.get("DATABASE_URL") and any(
         os.environ["DATABASE_URL"] in str(x) for x in args):
        raise ValueError("database secret must not appear in argv")
    subprocess.run(list(map(str,args)),cwd=str(cwd),
                   check=True,env=env or os.environ.copy())


def run(args,*,leech_repo=None):
    leech,slr=validate_preflight(args,leech_repo=leech_repo)
    args.output_root.mkdir(parents=True,exist_ok=False)
    source=args.output_root/"source"/"slr_original_train_dev.jsonl"
    source.parent.mkdir(parents=True)
    results=dict(schema=1,kind="x369_three_repo_physical_maxcut",
       leech_code_revision=args.leech_revision,
       sensiblaw_code_revision=args.sensiblaw_revision,
       original_checkpoint_sha256=args.checkpoint_sha256,
       original_tokenizer_sha256=args.tokenizer_sha256,
       pinned_original_manifest_sha256=sha(args.manifest),
       F5_raw_source_opened=False,
       same_source_five_arm_causal_oracle_distinction=True)
    if args.f3_config is not None:
        obj=json.loads(args.f3_config.read_text())
        cli=[sys.executable,"-m","scripts.run_x369_f3_two_seed_maxcut"]
        for key in ("screen_receipt","source_candidate","original_train_ids",
                    "original_heldout_ids","selection_validation_ids",
                    "probe_ids","probe_provenance"):
            cli+=["--"+key.replace("_","-"),obj[key]]
        cli+=["--reserved-lockbox-sha256",obj["reserved_lockbox_sha256"],
              "--output-dir",args.output_root/"f3_two_seed",
              "--device",args.device]
        if args.device=="cuda:0":
            cli+=["--require-gfx803"]
        for seed in (41,42):
            receipt=obj.get("original_f3_receipt_"+str(seed))
            if receipt is not None:
                cli+=["--original-f3-receipt-"+str(seed),receipt]
        run_command(cli,cwd=leech)
        f3=args.output_root/"f3_two_seed"/"f3_two_seed_summary.json"
        if not f3.exists():raise RuntimeError("F3 original paired receipt not produced")
        results["F3_B_original_two_checkpoint_receipt_sha256"]=sha(f3)

    # SLR holds numeric parser identity, source and grammar labels.
    run_command([
        args.sensiblaw_python,"-m","scripts.export_leech_spacy_grammar",
        "--manifest",args.manifest,"--output",source
    ],cwd=slr)
    if not source.exists():raise RuntimeError("SLR original numeric export absent")
    results["original_SLR_source_export_sha256"]=sha(source)

    def original_cli(module,output,extra=()):
        cli=[sys.executable,"-m",module,
            "--documents-jsonl",source,
            "--tokenizer",args.tokenizer]
        if module.endswith("grammar_early_epoch"):
            cli+=["--original-checkpoint",args.original_checkpoint,
                  "--original-checkpoint-sha256",args.checkpoint_sha256,
                  "--epochs",str(args.epochs),
                  "--device",args.device,
                  "--result",output]
        else:
            cli+=["--original-checkpoint",args.original_checkpoint,
                  "--code-revision",args.leech_revision,
                  "--steps",str(args.steps),
                  "--device",args.device,
                  "--aux-weight",str(args.aux_weight),
                  "--output-dir",output]
        if module.endswith("train_x369_slr_joint_grokking") and (
                getattr(args,"target_dev_ce",None) is not None):
            cli+=["--target-dev-ce",str(args.target_dev_ce)]
        if args.config is not None:cli+=["--config",args.config]
        cli.extend(extra)
        run_command(cli,cwd=leech)
    frozen=args.output_root/"frozen_grammar_probe.json"
    original_cli("scripts.audit_x369_slr_spacy_grammar_early_epoch",frozen)
    if not frozen.exists():raise RuntimeError("original frozen syntax probe missing")
    results["original_frozen_grammar_probe_sha256"]=sha(frozen)
    joined=args.output_root/"long_grokking_five_arms"
    original_cli("scripts.train_x369_slr_joint_grokking",joined)
    complete=joined/"matched_arm_summary.json"
    if not complete.exists():raise RuntimeError("source paired full Leech training missing")
    results["original_5_arm_grokking_receipt_sha256"]=sha(complete)
    final=args.output_root/"physical_maxcut_summary.json"
    final.write_text(json.dumps(results,sort_keys=True,indent=2)+"\n")
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sensiblaw-repo",type=Path,required=True)
    p.add_argument("--sensiblaw-python",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--original-checkpoint",type=Path,required=True)
    p.add_argument("--tokenizer",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--checkpoint-sha256",required=True)
    p.add_argument("--tokenizer-sha256",required=True)
    p.add_argument("--leech-revision",required=True)
    p.add_argument("--sensiblaw-revision",required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--f3-config",type=Path)
    p.add_argument("--steps",type=int,default=2048)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--aux-weight",type=float,default=.2)
    p.add_argument("--target-dev-ce",type=float,
        help="predeclare same-source development CE goal before any fit")
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    run(args)


if __name__=="__main__":
    main()
