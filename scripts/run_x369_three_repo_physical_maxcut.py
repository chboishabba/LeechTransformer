"""One-command physical X369 F3-B + SLR grammar + real full-Leech grokking.

User-owned machine only: native HIP/RX580, production SLR PostgreSQL and immutable
TinyStories sources are NOT accessible to GitHub/this script without the
required paths and DATABASE_URL environment. No new parser; SLR owns source.

Workflow:
  optional F3-B actual seed41/42, same 12 masks, F5 SHA only;
  SLR native ingest.parser_* PG export (legacy SensibLaw optional fallback);
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
import argparse,hashlib,json,os,subprocess,sys,time,signal
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fp:
        for data in iter(lambda:fp.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def parser_authority(args):
    native=getattr(args,"slr_repo",None)
    legacy=getattr(args,"sensiblaw_repo",None)
    if (native is None)==(legacy is None):
        raise ValueError("choose exactly one production SLR or legacy SensibLaw parser authority")
    if native is not None:
        return (native.resolve(),getattr(args,"slr_python",None),
                getattr(args,"slr_revision",None),"slr_native_pg",
                "python/slr_pg_leech_grammar_export.py",
                "python.slr_pg_leech_grammar_export")
    return (legacy.resolve(),getattr(args,"sensiblaw_python",None),
            getattr(args,"sensiblaw_revision",None),"sensiblaw_legacy",
            "scripts/export_leech_spacy_grammar.py",
            "scripts.export_leech_spacy_grammar")



def preflight_original_parser_manifest(manifest,authority):
    """Reject malformed/F5/cross-split source BEFORE even starting F3 GPU.

    Native SLR reopens canonical bytes inside PG; legacy SensibLaw also
    pins external UTF8 source bytes. This is an independent orchestrator
    fence: the exporter still repeats its own strict manifest checks.
    """
    rows=json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not isinstance(rows,list) or not rows:
        raise ValueError("original SLR manifest must contain source rows")
    identities=set()
    source_digests={}
    splits=set()
    for row in rows:
        if not isinstance(row,dict):
            raise ValueError("original source row must be a manifest object")
        split=row.get("split")
        if split not in ("train","development"):
            raise ValueError("reserved/heldout/F5 parser source denied before GPU")
        splits.add(split)
        keys=(("source_revision_ref","parser_run_ref") if
              authority=="slr_native_pg" else
              ("document_ref","run_ref","source_path"))
        if any(not isinstance(row.get(key),str) or not row[key]
               for key in keys):
            raise ValueError("native/legacy parser source identity is incomplete")
        sha_value=row.get("source_sha256")
        if (not isinstance(sha_value,str) or len(sha_value)!=64 or
            any(c not in "0123456789abcdef" for c in sha_value)):
            raise ValueError("exact original UTF8 source SHA256 receipt required")
        identity=(row["source_revision_ref"] if authority=="slr_native_pg"
                  else row["document_ref"])
        if identity in identities:
            raise ValueError("duplicate train/dev original source identity")
        identities.add(identity)
        if sha_value in source_digests and source_digests[sha_value]!=split:
            raise ValueError("same original source bytes on train and development")
        source_digests[sha_value]=split
        if authority=="sensiblaw_legacy":
            path=Path(row["source_path"])
            if not path.is_file() or sha(path)!=sha_value:
                raise ValueError("legacy original source bytes differ from pinned manifest")
            try:path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("original source is not UTF8") from exc
    if splits!={"train","development"}:
        raise ValueError("original disjoint train/development source splits required")
    return dict(source_manifest_rows=len(rows),
        source_splits=sorted(splits),F5_reserved_source_read=False)


def validate_preflight(args,*,leech_repo=None):
    leech=Path(leech_repo or Path(__file__).resolve().parent.parent)
    slr,slr_python,slr_revision,authority,exporter_file,exporter_module=parser_authority(args)
    original_paths=(args.manifest,args.original_checkpoint,args.tokenizer)
    if args.config is not None:original_paths+= (args.config,)
    for path in original_paths:
        if not path.is_file():raise FileNotFoundError("original source not mounted: "+str(path))
    if not (slr/exporter_file).is_file():
        raise FileNotFoundError("native/legacy selected PostgreSQL exporter not in source")
    if slr_python is None or not slr_python.is_file() or not slr_revision:
        raise FileNotFoundError("selected parser authority interpreter/revision unavailable")
    if args.output_root.exists():
        raise FileExistsError("refuse previously used X369 maximum-cut output root")
    if not os.environ.get("DATABASE_URL"):
        raise ValueError("DATABASE_URL must be set securely in environment")
    # An invalid reserved/F5 manifest must not start GPU work while SLR
    # independently rejects the same source in its CPU subprocess.
    preflight_original_parser_manifest(args.manifest,authority)
    for label,path,digest in (
        ("original_checkpoint",args.original_checkpoint,args.checkpoint_sha256),
        ("tokenizer",args.tokenizer,args.tokenizer_sha256)):
        if sha(path)!=digest:
            raise ValueError(label+" physical SHA256 did not match pinned receipt")
    for name,repo,revision in (
       ("Leech",leech,args.leech_revision),
       (authority,slr,slr_revision)):
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
    return leech,slr,slr_python,slr_revision,authority,exporter_module


def run_command(args,*,cwd,env=None):
    # Source commands returned for audit must never contain DATABASE_URL.
    if os.environ.get("DATABASE_URL") and any(
         os.environ["DATABASE_URL"] in str(x) for x in args):
        raise ValueError("database secret must not appear in argv")
    subprocess.run(list(map(str,args)),cwd=str(cwd),
                   check=True,env=env or os.environ.copy())



def source_cpu_environment(threads, *, needs_database):
    """Dedicate bounded CPU workers, never permit a parser to capture HIP."""
    if type(threads) is not int or threads<1:
        raise ValueError("CPU worker thread budget must be positive")
    env=os.environ.copy()
    for name in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES",
                 "ROCR_VISIBLE_DEVICES","GPU_DEVICE_ORDINAL"):
        env[name]=""
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        env[name]=str(threads)
    if not needs_database:
        env.pop("DATABASE_URL",None)
    return env


def gpu_only_environment():
    """A Leech GPU stage does not need the private PostgreSQL credential."""
    env=os.environ.copy()
    env.pop("DATABASE_URL",None)
    return env


def _stage_command(task):
    """Source-pinned stage descriptor, never serialised with environment."""
    name,command,cwd,env=task
    command=tuple(map(str,command))
    secret=os.environ.get("DATABASE_URL")
    if secret and any(secret in x for x in command):
        raise ValueError("database credential in stage argv: "+name)
    if not name or not command or not Path(cwd).is_dir():
        raise ValueError("invalid physical stage descriptor")
    return name,command,str(cwd),env


def _terminate_process_group(process, *, grace_seconds=2.0):
    """Stop all workers of a failed stage, not just its parent Python PID."""
    if process.poll() is not None:
        return
    if os.name=="posix":
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    else:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        if os.name=="posix":
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        else:
            process.kill()
        process.wait()


def run_parallel_stages(tasks, *, mode="overlap", max_stage_seconds=0):
    """Run independent resource-separated stages with true process overlap.

    For dependent stages use separate calls AFTER source receipt barriers.
    On any failure, terminate the peer's process group, never count partial
    artifacts as acceptance. Interval receipts use monotonic clock values
    only for overlap arithmetic (not false exact physical runtime claims).
    """
    tasks=[_stage_command(task) for task in tasks]
    if not tasks or mode not in ("overlap","sequential"):
        raise ValueError("invalid schedule or parallel execution mode")
    if len({name for name,*_ in tasks})!=len(tasks):
        raise ValueError("duplicate stage identity")
    if not isinstance(max_stage_seconds,(int,float)) or max_stage_seconds<0:
        raise ValueError("stage execution timeout must be nonnegative")
    receipt={}
    if mode=="sequential" or len(tasks)==1:
        for name,command,cwd,env in tasks:
            start=time.monotonic()
            run_command(command,cwd=cwd,env=env)
            stop=time.monotonic()
            receipt[name]=dict(stage=name,resource="source_stage",
                exit_code=0,started_monotonic=start,ended_monotonic=stop,
                elapsed_seconds=stop-start)
        return receipt
    processes={}
    start_times={}
    end_times={}
    exits={}
    begun=time.monotonic()
    try:
        # Popen starts each real process *before waiting on either*, so
        # SLR Python/PG and Leech HIP truly overlap.
        for name,command,cwd,env in tasks:
            start_times[name]=time.monotonic()
            processes[name]=subprocess.Popen(command,cwd=cwd,env=env,
                                             start_new_session=True)
        pending=set(processes)
        while pending:
            for name in tuple(pending):
                code=processes[name].poll()
                if code is not None:
                    exits[name]=code
                    end_times[name]=time.monotonic()
                    pending.remove(name)
                    if code!=0:
                        raise RuntimeError(
                            "physical stage "+name+" failed with exit "+str(code))
            if pending and max_stage_seconds and time.monotonic()-begun>max_stage_seconds:
                raise TimeoutError("physical source/GPU stage exceeded stated timeout")
            if pending:time.sleep(.05)
    except BaseException:
        for p in processes.values():
            _terminate_process_group(p)
        raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                _terminate_process_group(p)
    for name in processes:
        start=start_times[name]
        end=end_times[name]
        receipt[name]=dict(stage=name,resource="source_stage",
            exit_code=exits[name],started_monotonic=start,ended_monotonic=end,
            elapsed_seconds=end-start)
    return receipt


def stage_overlap_seconds(receipts, left, right):
    if left not in receipts or right not in receipts:
        return 0.
    a,b=receipts[left],receipts[right]
    return max(0.,min(a["ended_monotonic"],b["ended_monotonic"])-
                    max(a["started_monotonic"],b["started_monotonic"]))




def run_source_dependency_dag(stages, *, mode="overlap", max_stage_seconds=0):
    """Original source-aware DAG: source CPU and C3 GPU overlap, GPU serial.

    SLR completion unlocks CPU frozen grammar immediately. GPU five-arm
    training requires BOTH exact source export and F3 GPU completion, thus
    can overlap CPU grammar but never compete with F3 for RX580 memory.
    Process-group cleanup is mandatory on error/timeout; no partial source
    artifact may satisfy a dependency or count as a scientific receipt.
    """
    if mode not in ("overlap","sequential") or not isinstance(stages,list) or not stages:
        raise ValueError("invalid source dependency DAG schedule")
    if not isinstance(max_stage_seconds,(int,float)) or (
        not (0<=max_stage_seconds<float("inf"))):
        raise ValueError("finite nonnegative DAG stage deadline required")
    indexed={}
    for stage in stages:
        name,cmd,cwd,env=_stage_command((
            stage["name"],stage["command"],stage["cwd"],stage["env"]))
        deps=tuple(stage.get("depends_on",()))
        resource=stage["resource"]
        expected=Path(stage["expected_output"])
        if (name in indexed or name in deps or
            resource not in ("cpu_pg","cpu_probe","gpu","cpu_fit","cpu_f3") or
            not expected.parent.exists() and not expected.parent.parent.exists()):
            raise ValueError("invalid source DAG stage name/resource/output")
        indexed[name]=dict(name=name,command=cmd,cwd=cwd,env=env,
                           depends_on=deps,resource=resource,
                           expected_output=expected)
    if any(dep not in indexed for st in indexed.values()
           for dep in st["depends_on"]):
        raise ValueError("DAG contains unknown source dependency")
    # Fail before physical execution on an impossible dependency cycle.
    reachable=set()
    while True:
        previous=len(reachable)
        for name,st in indexed.items():
            if all(dep in reachable for dep in st["depends_on"]):
                reachable.add(name)
        if len(reachable)==previous:break
    if len(reachable)!=len(indexed):
        raise ValueError("source DAG cyclic; cannot schedule GPU or PostgreSQL")
    pending=set(indexed)
    running={}
    completed={}
    t0=time.monotonic()
    try:
        while pending or running:
            # Poll first: only *successful, verified* outputs unlock children.
            for name,proc in list(running.items()):
                if proc.poll() is None:
                    if (max_stage_seconds and
                        time.monotonic()-proc.x369_started>max_stage_seconds):
                        raise TimeoutError("DAG stage timeout: "+name)
                    continue
                code=proc.poll()
                if code!=0:
                    raise RuntimeError("DAG physical stage "+name+
                                       " failed with exit "+str(code))
                end=time.monotonic()
                st=indexed[name]
                output=st["expected_output"]
                if not output.is_file() or output.stat().st_size==0:
                    raise RuntimeError("DAG completed stage "+name+
                                       " lacks its exact expected source output")
                completed[name]=dict(stage=name,resource=st["resource"],
                    exit_code=0,started_monotonic=proc.x369_started,
                    ended_monotonic=end,elapsed_seconds=end-proc.x369_started,
                    output_sha256=sha(output))
                del running[name]
            for name,st in indexed.items():
                if name not in pending or not all(
                        dep in completed for dep in st["depends_on"]):
                    continue
                if mode=="sequential" and running:
                    continue
                if st["resource"]=="gpu" and any(
                        indexed[x]["resource"]=="gpu" for x in running):
                    continue
                pending.remove(name)
                started=time.monotonic()
                if mode=="sequential":
                    run_command(st["command"],cwd=st["cwd"],env=st["env"])
                    output=st["expected_output"]
                    if not output.is_file() or output.stat().st_size==0:
                        raise RuntimeError("DAG sequential source receipt missing "+name)
                    ended=time.monotonic()
                    completed[name]=dict(stage=name,resource=st["resource"],
                        exit_code=0,started_monotonic=started,
                        ended_monotonic=ended,elapsed_seconds=ended-started,
                        output_sha256=sha(output))
                else:
                    process=subprocess.Popen(st["command"],cwd=st["cwd"],
                        env=st["env"],start_new_session=True)
                    process.x369_started=started
                    running[name]=process
            if running:time.sleep(.05)
        return dict(stages=completed,elapsed_seconds=time.monotonic()-t0,
                    execution_mode=mode,
                    dependency_order_verified=True,
                    mutually_exclusive_GPU=True)
    except BaseException:
        for process in running.values():
            _terminate_process_group(process)
        raise
    finally:
        for process in running.values():
            if process.poll() is None:
                _terminate_process_group(process)

def run(args,*,leech_repo=None):
    leech,slr,slr_python,slr_revision,authority,exporter_module=validate_preflight(
        args,leech_repo=leech_repo)
    args.output_root.mkdir(parents=True,exist_ok=False)
    source=args.output_root/"source"/"slr_original_train_dev.jsonl"
    source.parent.mkdir(parents=True)
    results=dict(schema=1,kind="x369_three_repo_physical_maxcut",
       leech_code_revision=args.leech_revision,
       parser_authority=authority,
       parser_authority_git_revision=slr_revision,
       original_checkpoint_sha256=args.checkpoint_sha256,
       original_tokenizer_sha256=args.tokenizer_sha256,
       pinned_original_manifest_sha256=sha(args.manifest),
       verified_parser_source_manifest=preflight_original_parser_manifest(
           args.manifest,authority),
       F5_raw_source_opened=False,
       same_source_five_arm_causal_oracle_distinction=True)
    mode=getattr(args,"execution_mode","sequential")
    slr_threads=getattr(args,"slr_cpu_threads",2)
    probe_threads=getattr(args,"probe_cpu_threads",2)
    if mode not in ("overlap","sequential") or slr_threads<1 or probe_threads<1:
        raise ValueError("invalid CPU/GPU resource scheduling contract")
    slr_environment=source_cpu_environment(slr_threads,needs_database=True)
    probe_environment=source_cpu_environment(probe_threads,needs_database=False)
    gpu_environment=gpu_only_environment()
    gpu_stage=(args.device=="cuda:0")
    # On an RX580 with one HIP GPU, never schedule two GPU consumers
    # together. F3 GPU overlaps only CPU SLR. Grammar diagnostic on CPU
    # overlaps only GPU training AFTER exact SLR export receipt exists.
    probe_device=("cpu" if mode=="overlap" and gpu_stage
                  else getattr(args,"probe_device","cpu"))
    if mode=="overlap" and gpu_stage and probe_device!="cpu":
        raise ValueError("concurrent RX580 fit requires CPU-only grammar diagnostic")
    if (mode=="overlap" and not gpu_stage and
        getattr(args,"probe_device","cpu")!="cpu"):
        raise ValueError("CPU source-only run cannot launch a GPU parser probe")
    results["scheduling"]=dict(execution_mode=mode,
        cpu_SLR_threads=slr_threads,cpu_probe_threads=probe_threads,
        gpu_device=args.device,cpu_probe_device=probe_device,
        parser_database_url_inherited_by_GPU=False,
        F3_gpu_overlaps_SLR_cpu_only=True,
        five_arm_GPU_overlaps_frozen_probe_CPU_only=True,
        two_simultaneous_GPU_stages=False)
    timeout=getattr(args,"max_stage_seconds",0)
    # First independent DAG layer: CPU SLR/PG and GPU F3 C3 can overlap.
    stage_one=[("slr_native_or_legacy_pg_cpu",[
        slr_python,"-m",exporter_module,
        "--manifest",args.manifest,"--output",source
    ],slr,slr_environment)]
    f3=None
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
        if gpu_stage:
            cli+=["--require-gfx803"]
        for seed in (41,42):
            original_receipt=obj.get("original_f3_receipt_"+str(seed))
            if original_receipt is not None:
                cli+=["--original-f3-receipt-"+str(seed),original_receipt]
        f3=args.output_root/"f3_two_seed"/"f3_two_seed_summary.json"
        stage_one.insert(0,("F3_B_native_hip_gpu",cli,leech,gpu_environment))
    # Outputs are validated by the shared source-aware DAG only after the
    # producer exits successfully. This deliberately avoids holding an
    # unnecessary F3 GPU barrier before starting the CPU grammar probe.
    def original_cli(module,output,extra=(),*,device):
        cli=[sys.executable,"-m",module,
            "--documents-jsonl",source,
            "--tokenizer",args.tokenizer]
        if module.endswith("grammar_early_epoch"):
            cli+=["--original-checkpoint",args.original_checkpoint,
                  "--original-checkpoint-sha256",args.checkpoint_sha256,
                  "--epochs",str(args.epochs),
                  "--device",device,
                  "--result",output]
        else:
            cli+=["--original-checkpoint",args.original_checkpoint,
                  "--code-revision",args.leech_revision,
                  "--steps",str(args.steps),
                  "--device",device,
                  "--aux-weight",str(args.aux_weight),
                  "--output-dir",output]
        if module.endswith("train_x369_slr_joint_grokking") and (
                getattr(args,"target_dev_ce",None) is not None):
            cli+=["--target-dev-ce",str(args.target_dev_ce)]
        if args.config is not None:cli+=["--config",args.config]
        cli.extend(extra)
        return cli
    frozen=args.output_root/"frozen_grammar_probe.json"
    joined=args.output_root/"long_grokking_five_arms"
    stage_two=[
        ("frozen_grammar_CPU",original_cli(
            "scripts.audit_x369_slr_spacy_grammar_early_epoch",
            frozen,device=probe_device),
         leech,probe_environment if probe_device=="cpu" else gpu_environment),
        ("five_arm_original_Leech_GPU",original_cli(
            "scripts.train_x369_slr_joint_grokking",
            joined,device=args.device),
         leech,gpu_environment if gpu_stage else probe_environment)]
    stages=[]
    for name,cli,cwd,env in stage_one:
        if name=="F3_B_native_hip_gpu":
            stages.append(dict(name=name,command=cli,cwd=cwd,env=env,
                depends_on=(),resource="gpu" if gpu_stage else "cpu_f3",
                expected_output=f3))
        else:
            stages.append(dict(name=name,command=cli,cwd=cwd,env=env,
                depends_on=(),resource="cpu_pg",expected_output=source))
    for name,cli,cwd,env in stage_two:
        if name=="frozen_grammar_CPU":
            stages.append(dict(name=name,command=cli,cwd=cwd,env=env,
                depends_on=("slr_native_or_legacy_pg_cpu",),
                resource="cpu_probe" if probe_device=="cpu" else "gpu",
                expected_output=frozen))
        else:
            dependencies=("slr_native_or_legacy_pg_cpu",)
            if f3 is not None:
                dependencies+=("F3_B_native_hip_gpu",)
            stages.append(dict(name=name,command=cli,cwd=cwd,env=env,
                depends_on=dependencies,
                resource="gpu" if gpu_stage else "cpu_fit",
                expected_output=joined/"matched_arm_summary.json"))
    workflow=run_source_dependency_dag(
        stages,mode=mode,max_stage_seconds=timeout)
    receipts=workflow["stages"]
    results["original_SLR_source_export_sha256"]=receipts[
        "slr_native_or_legacy_pg_cpu"]["output_sha256"]
    if f3 is not None:
        results["F3_B_original_two_checkpoint_receipt_sha256"]=receipts[
            "F3_B_native_hip_gpu"]["output_sha256"]
    results["original_frozen_grammar_probe_sha256"]=receipts[
        "frozen_grammar_CPU"]["output_sha256"]
    results["original_5_arm_grokking_receipt_sha256"]=receipts[
        "five_arm_original_Leech_GPU"]["output_sha256"]
    results["stage_one_cpu_SLR_and_GPU_F3_overlap_seconds"]=(
        stage_overlap_seconds(receipts,"F3_B_native_hip_gpu",
                              "slr_native_or_legacy_pg_cpu"))
    results["stage_two_CPU_frozen_grammar_and_GPU_training_overlap_seconds"]=(
        stage_overlap_seconds(receipts,"frozen_grammar_CPU",
                              "five_arm_original_Leech_GPU"))
    results["CPU_probe_overlaps_F3_if_SLR_fast_seconds"]=(
        stage_overlap_seconds(receipts,"frozen_grammar_CPU",
                              "F3_B_native_hip_gpu"))
    results["stage_execution_receipts"]=receipts
    results["DAG_dependency_order_verified"]=workflow["dependency_order_verified"]
    results["single_GPU_exclusivity_verified"]=workflow["mutually_exclusive_GPU"]
    results["physical_critical_path_walltime_seconds"]=workflow["elapsed_seconds"]
    results["all_stage_exit_codes_zero"]=all(
        rec["exit_code"]==0 for rec in receipts.values())
    final=args.output_root/"physical_maxcut_summary.json"
    final.write_text(json.dumps(results,sort_keys=True,indent=2)+"\n")
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slr-repo",type=Path,
        help="preferred Rust SLR native ingest.parser_* PostgreSQL authority")
    p.add_argument("--slr-python",type=Path)
    p.add_argument("--sensiblaw-repo",type=Path,
        help="legacy execution.semantic_parser_* compatibility producer")
    p.add_argument("--sensiblaw-python",type=Path)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--original-checkpoint",type=Path,required=True)
    p.add_argument("--tokenizer",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--checkpoint-sha256",required=True)
    p.add_argument("--tokenizer-sha256",required=True)
    p.add_argument("--leech-revision",required=True)
    p.add_argument("--slr-revision")
    p.add_argument("--sensiblaw-revision")
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--f3-config",type=Path)
    p.add_argument("--steps",type=int,default=2048)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--aux-weight",type=float,default=.2)
    p.add_argument("--target-dev-ce",type=float,
        help="predeclare same-source development CE goal before any fit")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--probe-device",default="cpu",choices=("cpu","cuda:0"),
        help="frozen grammar evaluation; forced CPU when GPU training overlaps")
    p.add_argument("--execution-mode",choices=("overlap","sequential"),
        default="overlap",help="CPU PG/export + HIP F3, then CPU probe + HIP five-arm fit")
    p.add_argument("--slr-cpu-threads",type=int,default=2)
    p.add_argument("--probe-cpu-threads",type=int,default=2)
    p.add_argument("--max-stage-seconds",type=float,default=0,
        help="optional fail-closed timeout for each independent stage pair; 0 means unlimited")
    args=p.parse_args()
    run(args)


if __name__=="__main__":
    main()
