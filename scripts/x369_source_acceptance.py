"""One immutable new-architecture source acceptance; no training or migration.
Original Leech-LILA: Anatolii Kornienko; DASHI extension.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from itertools import product


def digest(data):
    return hashlib.sha256(data).hexdigest()


def choose(rows, n):
    if n < 1 or not rows:
        raise ValueError('positive nonempty sample required')
    return rows if len(rows) <= n else [rows[i*len(rows)//n] for i in range(n)]


def guard_controls(guards):
    a,b,j,n=guards
    if a not in range(3) or b not in range(3) or j not in range(2) or n not in range(2):
        raise ValueError('invalid fixed guards')
    return ((4*((-a)%3),0),(4*((-b)%3),1),(10*j,2),(n,3))


def write_receipt(path, obj):
    # Caller has reserved the output file exclusively. Atomic stage snapshots.
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def source_identity(root, expected):
    def git(*args):
        return subprocess.check_output(('git','-C',str(root),*args),text=True).strip()
    head=git('rev-parse','HEAD')
    if head!=expected or git('status','--porcelain','--untracked-files=no','--','models','scripts'):
        raise ValueError('source revision mismatch or tracked source dirty')
    tracked={p for p in git('ls-files','-z','--','models','scripts').split('\0') if p.endswith('.py')}
    actual={str(p.relative_to(root)) for d in ('models','scripts') for p in (root/d).rglob('*.py')}
    if actual!=tracked or any((root/p).is_symlink() for p in tracked):
        raise ValueError('untracked, missing or symlinked Python source')
    files={p:digest((root/p).read_bytes()) for p in sorted(tracked)}
    return dict(head=head,files=files)


STAGE_ORDER=('admission','packed_trace','linear_decoder','kernel','common_hom','graph','future_consumer','indexed_fibre','interval','factorial','cross_checkpoint','resources')


def stage_plan(a):
    enabled=set(STAGE_ORDER[1:4])
    for name,flag in (('common_hom',a.hom_contexts),('graph',a.graph_budget),('future_consumer',a.future_budget),('indexed_fibre',a.fibre_budget),('interval',a.interval_bounds),('factorial',a.factorial_contexts),('cross_checkpoint',a.previous_source_receipt),('resources',a.resource_rows)):
        if flag:enabled.add(name)
    if a.only_stages is not None:
        requested=set(a.only_stages)
        if len(requested)!=len(a.only_stages) or not requested<=enabled|{'admission'}:
            raise ValueError('duplicate stage or stage missing enabling arguments')
        enabled=requested
    if 'cross_checkpoint' in enabled and 'factorial' not in enabled:
        raise ValueError('cross_checkpoint requires factorial in the same receipt')
    return tuple(s for s in STAGE_ORDER if s in enabled)


def run_stage(receipt,name,fn,save):
    """Execution lifecycle is not a mathematical acceptance verdict."""
    if name not in receipt['requested_stages']:return
    if receipt['stage_lifecycle'][name]!='not_started':raise ValueError('stage already started')
    receipt['stage_lifecycle'][name]='running';save()
    try:receipt['stages'][name]=fn()
    except Exception as e:
        partial=receipt['stages'].get(name)
        receipt['stages'][name]=dict(status='error',error=repr(e),partial=partial)
        receipt['stage_lifecycle'][name]='error';save();raise
    receipt['stage_lifecycle'][name]='returned';save()


def execution_contract(model):
    """Nonpersistent buffers and execution configuration are source identity too."""
    import torch
    h=hashlib.sha256()
    for k,t in sorted(model.named_buffers()):
        a=t.detach().cpu().contiguous()
        h.update(json.dumps((k,str(a.dtype),list(a.shape))).encode()+b'\0')
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    src=model.history.source
    return dict(buffers_sha256=h.hexdigest(),training={k:m.training for k,m in model.named_modules()},
                width=model.width,experts=model.experts,expert_width=model.expert_width,
                paired=model.history.paired,typed_consumer=model.typed_consumer,
                source={k:getattr(src,k,None) for k in ('history_cap','temperature','route_geometry','transition_rank','router_selection','route_probe_action','top_k','num_experts','d_model')})


def read_checkpoint(path,expected_sha256,dtype):
    """Verify bytes before safe deserialization; no model or training provenance claim."""
    import io
    import torch
    if len(expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('expected lowercase SHA-256')
    data=path.read_bytes()
    if digest(data)!=expected_sha256:raise ValueError('checkpoint SHA mismatch')
    state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not state or not all(
        isinstance(k,str) and isinstance(v,torch.Tensor) and v.dtype==dtype and
        v.layout==torch.strided and torch.isfinite(v).all() for k,v in state.items()):
        raise ValueError('expected exact new-architecture raw floating state_dict')
    return state


def translation_provenance(path,expected_sha,checkpoint_sha,source,config,frozen,execution,runtime):
    """Bind translated evidence explicitly; a raw state_dict is not proof of training."""
    if path is None:return dict(origin='unspecified_raw_state',original_new_source_training_authenticated=False)
    data=path.read_bytes()
    if digest(data)!=expected_sha:raise ValueError('translation receipt SHA mismatch')
    r=json.loads(data);canonical=lambda v:json.dumps(v,sort_keys=True,allow_nan=False)
    if (r.get('kind')!='x369_translated_source_candidate' or
        r.get('status') not in ('translated_candidate','translated_candidate_bounded_diagnostics') or
        r.get('candidate_sha256')!=checkpoint_sha or r.get('candidate_model_sha256')!=frozen or
        r.get('source')!=source or canonical(r.get('execution_contract'))!=canonical(execution) or
        canonical(r.get('runtime'))!=canonical(runtime) or canonical(r.get('config'))!=canonical(config) or
        r.get('training_steps')!=0 or r.get('original_new_source_training') is not False or
        r.get('optimizer_migrated') is not False or r.get('release_admitted') is not False):
        raise ValueError('translation candidate identity mismatch')
    return dict(origin='translated_historical_paired',translation_receipt_sha256=expected_sha,
                historical_sha256=r['historical_sha256'],translation_status=r['status'],
                original_new_source_training_authenticated=False,optimizer_migrated=False)


def model_digest(model):
    import torch
    h=hashlib.sha256()
    for k,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous()
        h.update(json.dumps((k,str(a.dtype),list(a.shape))).encode()+b'\0')
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def packed_trace(model, rows):
    import torch
    from scripts.train_x369_ordered_action import encode_inputs
    from scripts.x369_source_geometry import signed_seed,literal_controls
    from models.geometric_reflection_fibre import GROUP
    from models.geometric_stabilizer_codec import encode_history,decode_history,same_bits
    p=next(model.parameters());count=0
    with torch.no_grad():
        for r in rows:
            seed,ops=encode_inputs(r,'signed')
            s=torch.tensor([seed],device=p.device,dtype=p.dtype)
            c,t=literal_controls(torch.tensor([ops],device=p.device,dtype=p.dtype))
            logits,trace=model.rollout_group(s,c,t)
            packet=encode_history(signed_seed(s,model.experts,model.width))
            for j,ref in enumerate(trace):
                tag=t[:,j];h=decode_history(packet)
                expected=signed_seed(s,model.experts,model.width) if j==0 else trace[j-1].history
                if not same_bits(h,expected):raise RuntimeError('packed input differs')
                out=model.history.step_controlled(c[:,j],h,lambda z:model.control_embed(tag*len(GROUP)+z),train_sign=False)
                for name,a in vars(ref).items():
                    b=getattr(out,name)
                    equal=(same_bits(a,b) if a.is_floating_point() else torch.equal(a,b)) if isinstance(a,torch.Tensor) else a==b
                    if not equal:raise RuntimeError('packed trace differs: '+name)
                packet=encode_history(out.history)
                if not same_bits(decode_history(packet),ref.history):raise RuntimeError('packed output differs')
                count+=1
            if not all(same_bits(a,b) for a,b in zip(logits,model.consume(out))):raise RuntimeError('packed logits differ')
    return dict(rows=len(rows),steps=count,bitwise=True,physical_cost_certified=False)


def run(a):
    import torch
    from scripts.x369_source_geometry import SourceGeometryOrderedLearner
    from scripts.audit_x369_joint_action import joint_records,MEANS,task_digest
    from scripts.audit_x369_ordered_signed_task import TRIADS
    from scripts.x369_exact_semantic_decoder import fit,source_records,verify_decoder_family,read_history
    from scripts.x369_origin_ray_feasibility import audit
    from scripts.x369_indexed_residual_closure import explore,native_source
    root=Path(__file__).resolve().parents[1]
    source=source_identity(root,a.source_head)
    if not torch.cuda.is_available() or not torch.version.hip or not getattr(torch.cuda.get_device_properties(0),'gcnArchName','').startswith('gfx803'):
        raise RuntimeError('native gfx803 HIP required; no CPU fallback')
    dt=getattr(torch,a.dtype)
    state=read_checkpoint(a.checkpoint,a.sha256,dt)
    model=SourceGeometryOrderedLearner(a.width,a.experts,expert_width=a.expert_width).to('cuda',dt).eval()
    model.load_state_dict(state,strict=True);model.history.source.history_cap=a.cap
    frozen=model_digest(model);execution=execution_contract(model);plan=stage_plan(a)
    runtime=dict(torch=str(torch.__version__),hip=torch.version.hip,arch=torch.cuda.get_device_properties(0).gcnArchName)
    config=dict(width=a.width,experts=a.experts,expert_width=model.expert_width,cap=a.cap,dtype=a.dtype)
    provenance=translation_provenance(a.translation_receipt,a.translation_receipt_sha256,a.sha256,source,config,frozen,execution,runtime)
    guards=tuple(a.guards);controls=guard_controls(guards)
    rows=[r for r in joint_records() if r['guards']==guards]
    train=choose([r for r in rows if r['split']=='train'],a.rows)
    hold=choose([r for r in rows if r['split']=='composition'],a.rows)
    receipt=dict(kind='new_source_acceptance',checkpoint_sha256=a.sha256,source=source,model_sha256=frozen,
        runtime=runtime,checkpoint_provenance=provenance,
        config=dict(width=a.width,experts=a.experts,expert_width=model.expert_width,cap=a.cap,dtype=a.dtype,guards=guards,controls=controls),
        task_sha256=task_digest(rows),train=train,heldout=hold,stages={},training_steps=0,
        historical_checkpoint_migration=False,training_provenance_authenticated=False,release_admitted=False,
        execution_contract=execution,requested_stages=plan,
        stage_scope={k:getattr(a,k) for k in ('rows','timeout_ms','graph_budget','future_budget','future_scope','future_export','fibre_budget','simultaneous_weld','hom_contexts','factorial_contexts','factorial_pairs','factorial_horizon','resource_rows','resource_repeats')},
        stage_lifecycle={s:'not_started' for s in plan},complete=False,execution_complete=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x'):pass
    def save():
        if model_digest(model)!=frozen or execution_contract(model)!=execution:raise RuntimeError('model or execution contract changed during audit')
        if source_identity(root,a.source_head)!=source:raise RuntimeError('source changed during audit')
        write_receipt(a.output,receipt)
    save()
    def stage(name,fn):run_stage(receipt,name,fn,save)
    stage('admission',lambda:dict(status='admitted',checkpoint_digest_verified=True,
        strict_state_dict_loaded=True,native_gfx803_verified=True,model_sha256=frozen,
        execution_contract=execution,training_provenance_authenticated=False,release_admitted=False))
    stage('packed_trace',lambda:packed_trace(model,train[:3]+hold[:3]))
    def linear():
        records=[];seed,_,_=native_source(model)
        def stream():
            for i,(m,d) in enumerate(product(MEANS,TRIADS)):
                h=torch.frombuffer(bytearray(seed(m,d)),dtype=dt).reshape(a.experts,2,a.width)
                rec=(('seed',i),read_history(h),m+d);records.append(rec);yield rec
            for rec in source_records(model,train):
                records.append(rec);yield rec
        result=fit(stream(),include_family=True)
        if result['consistent']:
            family=verify_decoder_family(source_records(model,hold),result['coefficients'],result['nullspace'])
            result['heldout']=family.pop('canonical')
            result['heldout_family']=family
        else:
            # Preserve exact original values behind every contradiction hash.
            ids={tuple(w) for w in result.get('witnesses',())} | {tuple(w) for w,_,_,_ in result.get('conflict',())}
            result['witness_records']=[dict(id=w,history=[[[str(x) for x in b] for b in p] for p in h],label=y) for w,h,y in records if w in ids]
        return result
    stage('linear_decoder',linear)
    def kernel():
        def progress(v):receipt['stages']['kernel']=v;save()
        # Identity is a legal group control for the conjugation-closed new
        # architecture. Exclusion concerns that broader observer, not the
        # fixed-guard semantic task below.
        return audit(model,controls=((0,0),),timeout_ms=a.timeout_ms,receipt=progress)
    stage('kernel',kernel)
    if a.hom_contexts:
        from scripts.x369_common_hom import native_joint_hom
        stage('common_hom',lambda:native_joint_hom(model,rows,contexts=a.hom_contexts,seconds=a.timeout_ms/1000))
    if a.graph_budget:
        def graph():
            seed,step,action=native_source(model);end=time.monotonic()+a.timeout_ms/1000
            class Budget(Exception):pass
            def bounded(h,u):
                if time.monotonic()>end:raise Budget()
                return step(h,u)
            try:
                out=explore(((seed(m,d),m+d,i) for i,(m,d) in enumerate(product(MEANS,TRIADS))),controls,bounded,action,max_states=a.graph_budget)
            except Budget:return dict(status='time_budget',universal_weld=False)
            table=out.pop('table',None)
            if table is not None:
                out['decoder']=[(h.hex(),z) for h,z in table.items()]
            out['scope']='single fixed guard assignment, all finite words, original joint mean carrier'
            return out
        stage('graph',graph)
    if a.future_budget:
        from scripts.x369_source_future_acceptance import native_future
        selected_controls=controls if a.future_scope=='fixed' else tuple(
            [(4*((-g)%3),tag) for tag in (0,1) for g in range(3)]+
            [(10*g,2) for g in range(2)]+[(g,3) for g in range(2)])
        def future():
            if a.simultaneous_weld:
                from scripts.x369_simultaneous_weld import native_simultaneous
                result=native_simultaneous(model,selected_controls,budget=a.future_budget,seconds=a.timeout_ms/1000,export=a.future_export)
            else:
                result=native_future(model,selected_controls,budget=a.future_budget,seconds=a.timeout_ms/1000,consumer='raw',export_realization=a.future_export)
            result['controls']=selected_controls
            result['scope_kind']=a.future_scope
            return result
        stage('future_consumer',future)
    if a.fibre_budget:
        from scripts.x369_indexed_source_fibre_audit import native_audit
        # Source identity, checkpoint SHA, exact model parameters, original
        # runtime, frozen buffers and receipt lifecycle are verified by
        # this enclosing acceptance owner BEFORE the audit is invoked.
        def indexed_fibre():
            result=native_audit(model,
                controls if a.future_scope=='fixed' else tuple(
                    [(4*((-g)%3),tag) for tag in (0,1) for g in range(3)]
                    +[(10*g,2) for g in range(2)]
                    +[(g,3) for g in range(2)]),
                budget=a.fibre_budget,seconds=a.timeout_ms/1000)
            result.update(
                checkpoint_sha256=a.sha256,
                frozen_model_sha256=frozen,
                execution_buffers_sha256=execution['buffers_sha256'],
                source_head=source['head'],
                source_file_count=len(source['files']),
                source_and_checkpoint_identity_verified_by_parent=True,
                training_provenance_authenticated=False,
                unrestricted_native_source_sufficient=False)
            return result
        stage('indexed_fibre',indexed_fibre)
    if a.interval_bounds:
        from scripts.x369_controlled_interval import audit_native_edge
        def interval():
            raw=a.interval_bounds.read_bytes();v=json.loads(raw)
            if set(v)-{'errors','linear_rounding'} or 'errors' not in v:raise ValueError('invalid external interval hypotheses')
            seed,_,_=native_source(model);r=hold[0];h=seed(r['mean'],r['residual']);u=controls['abjn'.index(r['word'][0])]
            result=audit_native_edge(model,h,u,v['errors'],v.get('linear_rounding'))
            result.update(bounds_sha256=digest(raw),row=r,control=u,scope='one observed first composition-prefix edge; external bounds not authenticated')
            return result
        stage('interval',interval)
    if a.factorial_contexts:
        from scripts.x369_source_factorial_acceptance import source_factorial
        stage('factorial',lambda:source_factorial(model,rows,contexts=a.factorial_contexts,pairs=a.factorial_pairs,horizon=a.factorial_horizon,seconds=a.timeout_ms/1000))
    if a.previous_source_receipt:
        from scripts.x369_source_factorial_acceptance import compare_frozen_phases
        def previous_phase():
            data=a.previous_source_receipt.read_bytes()
            if digest(data)!=a.previous_receipt_sha256:raise ValueError('previous receipt digest mismatch')
            result=compare_frozen_phases(json.loads(data),receipt);result['previous_receipt_sha256']=a.previous_receipt_sha256
            return result
        stage('cross_checkpoint',previous_phase)
    if a.resource_rows:
        from scripts.x369_source_resource_acceptance import resource_audit
        stage('resources',lambda:resource_audit(model,choose(hold,a.resource_rows),repeats=a.resource_repeats,seconds=a.timeout_ms/1000))
    if any(v!='returned' for v in receipt['stage_lifecycle'].values()):raise RuntimeError('requested stage not executed')
    receipt['execution_complete']=receipt['complete']=True;save()


def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--source-head',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expert-width',type=int,help='original intermediate expert width; default equals width');p.add_argument('--width',type=int,default=12);p.add_argument('--experts',type=int,default=4);p.add_argument('--dtype',choices=('float32','float64'),default='float32');p.add_argument('--cap',type=float)
    p.add_argument('--guards',type=int,nargs=4,default=(1,2,1,1));p.add_argument('--rows',type=int,default=24);p.add_argument('--timeout-ms',type=int,default=90000);p.add_argument('--graph-budget',type=int,default=0)
    p.add_argument('--future-budget',type=int,default=0);p.add_argument('--fibre-budget',type=int,default=0,help='original-byte per-consumer factorization graph budget; requires at least all distinct marked seeds');p.add_argument('--future-scope',choices=('fixed','all-literals'),default='all-literals')
    p.add_argument('--future-export',action='store_true',help='emit executable verified quotient only on a closed graph')
    p.add_argument('--hom-contexts',type=int,default=0,help='complete natural contexts per split for independent joint generator fit')
    p.add_argument('--simultaneous-weld',action='store_true',help='semantic decoder, signed defects, original logits and joint future realization on one observed graph')
    p.add_argument('--interval-bounds',type=Path,help='explicit external stage-error hypotheses JSON; no kernel authentication')
    p.add_argument('--factorial-contexts',type=int,default=0);p.add_argument('--factorial-pairs',type=int,default=2);p.add_argument('--factorial-horizon',type=int,default=1)
    p.add_argument('--resource-rows',type=int,default=0);p.add_argument('--resource-repeats',type=int,default=5)
    p.add_argument('--previous-source-receipt',type=Path);p.add_argument('--previous-receipt-sha256')
    p.add_argument('--only-stages',nargs='+',choices=STAGE_ORDER,help='run a declared subset of enabled stages in canonical order')
    p.add_argument('--plan',action='store_true',help='print requested stage plan without reading checkpoint or executing source')
    p.add_argument('--translation-receipt',type=Path);p.add_argument('--translation-receipt-sha256')
    a=p.parse_args(argv)
    if bool(a.translation_receipt)!=bool(a.translation_receipt_sha256):p.error('translation receipt requires SHA')
    if bool(a.previous_source_receipt)!=bool(a.previous_receipt_sha256) or (a.previous_source_receipt and not a.factorial_contexts):p.error('previous receipt requires SHA and factorial contexts')
    if min(a.factorial_contexts,a.resource_rows)<0 or a.factorial_pairs<1 or not 1<=a.factorial_horizon<=3 or a.resource_repeats<2:p.error('invalid factorial/resource scope')
    if a.simultaneous_weld and not a.future_budget:p.error('--simultaneous-weld requires --future-budget')
    if a.hom_contexts<0:p.error('--hom-contexts must be nonnegative')
    if a.future_export and not a.future_budget:p.error('--future-export requires --future-budget')
    if a.future_budget<0 or a.fibre_budget<0 or ((a.future_budget or a.fibre_budget) and a.width<9):p.error('future/fibre audit requires nonnegative budget and width>=9')
    if a.fibre_budget and a.fibre_budget<189:p.error('indexed fibre audit must admit all 189 distinct original public seed histories')
    if a.expert_width is not None and a.expert_width<1:p.error('invalid expert width')
    if a.width<3 or a.width%3 or a.experts<2 or a.rows<1 or not 1<=a.timeout_ms<=90000 or a.graph_budget<0 or (a.cap is not None and not 0<a.cap<float('inf')):p.error('invalid scope')
    try:guard_controls(a.guards);stage_plan(a)
    except ValueError as e:p.error(str(e))
    return a


def main():
    a=parse_args()
    if a.plan:
        print(json.dumps(dict(kind='new_source_acceptance_plan',requested_stages=stage_plan(a),checkpoint=str(a.checkpoint),expected_sha256=a.sha256,source_head=a.source_head,configuration=dict(width=a.width,experts=a.experts,expert_width=a.width if a.expert_width is None else a.expert_width,dtype=a.dtype,cap=a.cap,guards=a.guards),checkpoint_authenticated=False,training_steps=0,release_admitted=False),indent=2));return
    run(a)


if __name__=='__main__':main()
