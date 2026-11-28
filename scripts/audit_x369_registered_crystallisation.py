"""Read-only postprocessing of the registered learning campaign.

No new training or model selection. Retain first-observed event brackets,
actual Fourier and routing trajectory, and matched-seed control comparisons.
This module is deliberately outside the training/registration owner hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.run_x369_registered_learning_campaign import inspect_arm,report,NAMES,SEEDS

ROOT=Path(__file__).resolve().parents[1]


def require(ok,msg):
    if not ok:raise ValueError(msg)


def event(rows,predicate):
    prev=0
    for r in rows:
        if predicate(r):
            return dict(first_observed_step=r['step'],previous_sample_step=prev,
                        onset_between_samples_not_certified=True)
        prev=r['step']
    return None


def trajectory(rows):
    result=[];previous=0
    for row in rows:
        t=row['step'];require(type(t) is int and t>previous,'nonmonotone sampled steps')
        previous=t
        g=row['geometry'];l=g['logit_spectrum']
        pairs=l['frequency_pairs'];require(pairs==[[1,0],[0,1],[1,1],[1,-1]],'original Fourier basis')
        f=l['character_pair_fraction'];a=row['audit']
        routes=[]
        for r in g['actual_router_steps']:
            routes.append(dict(token_step=r['token_step'],entropy=r['selected_entropy_nats'],
                occupancy=r['expert_occupancy'],history_character_fractions=r['history']['character_pair_fraction'],
                history_effective_rank=r['history']['effective_rank'],
                e8_candidate_character_fractions=r.get('e8_candidate_invariant_spectrum',{}).get('character_pair_fraction'),
                puncture_candidate_character_fractions=r.get('punctured_nine_candidate_spectrum',{}).get('character_pair_fraction')))
        result.append(dict(step=t,training_fit=a['training_eight_correct'],
            heldout_correct=a['heldout_p22_correct'],hard_action_defect=a['hard_label_action_defect_count'],
            law_character_fraction=None if f is None else f[2],
            other_character_fraction=None if f is None else sum(f[:2])+f[3],
            centered_energy=l['centered_energy'],effective_rank=l['effective_rank'],
            parseval_absolute_residual=l['parseval_absolute_residual'],
            actual_route_steps=routes))
    fit=event(rows,lambda r:r['audit']['training_eight_correct'])
    joint=event(rows,lambda r:r['audit']['training_eight_correct'] and r['audit']['heldout_p22_correct'])
    delayed=bool(fit and joint and joint['first_observed_step']>fit['first_observed_step'])
    return dict(training_fit=fit,joint_fit=joint,delayed_sampled_joint_fit=delayed,
                sampled_geometry=result,causal_crystallisation_certified=False)


def audit(root):
    path=root/'registration.json';raw=path.read_bytes();plan=json.loads(raw)
    require(plan['arm_order']==list(NAMES) and plan['seed_order']==list(SEEDS) and
            plan['heldout_selection'] is False and len(plan['arms'])==18,'registered scope')
    require([(e['contract']['seed'],e['contract']['arm']) for e in plan['arms']]==
            [(s,a) for s in SEEDS for a in NAMES],'complete registered arm order')
    expected=hashlib.sha256((ROOT/'scripts/run_x369_registered_learning_campaign.py').read_bytes()).hexdigest()
    require(plan['scheduler_sha256']==expected,'scheduler source identity changed')
    for e in plan['arms']:
        require((root/e['name']).is_dir(),'missing registered arm directory')
        for name,sha in e['contract']['source_owners'].items():
            p=(ROOT/name).resolve()
            require(p.is_relative_to(ROOT) and hashlib.sha256(p.read_bytes()).hexdigest()==sha,
                    'registered source identity changed '+name)
    states=[inspect_arm(root,e,verify_all=True) for e in plan['arms']]
    summary=report(states,plan)
    rows={}
    for s in states:
        r=s['last']
        rows[s['entry']['name']]=trajectory(r['sampled_trajectory'] if r else [])
    complete=summary['status']=='FIXED_CAMPAIGN_COMPLETE'
    return dict(status='registered_crystallisation_trajectory_audit',
                registration_sha256=hashlib.sha256(raw).hexdigest(),campaign=summary,
                actual_trajectory_by_arm=rows,terminal_quality_comparison_available=complete,
                p97_reproduction_claimed=False,mechanism_or_causal_superiority_certified=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=audit(a.directory)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(r,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(r['status'],r['campaign']['arms_complete'],r['campaign']['arms_required'])

if __name__=='__main__':main()
