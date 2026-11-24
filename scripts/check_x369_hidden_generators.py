"""Independent generator fit and original-source untrained negative control."""
import json
import torch
from scripts.audit_x369_ordered_signed_task import records
from scripts.audit_x369_hidden_generators import audit_hidden_generators
from scripts.check_x369_hidden_intervention import Toy
from scripts.run_x369_fair_ordered_gpu import FairLearner
from scripts.check_x369_original_source_acceptance import source_identity

if __name__=='__main__':
    torch.set_num_threads(1);torch.manual_seed(41);source_identity();rows=list(records())
    r=audit_hidden_generators(Toy(),rows,'cpu',torch.float64)
    assert r['rank']==3 and max(r['relation_frobenius'].values())<1e-12
    assert r['commutator_frobenius']>1
    for g in r['interventions'].values():
        assert g['nonfixed_pairs']>0 and g['probability_l1_per_row'][1]<1e-12
    print('synthetic',json.dumps(r),flush=True)
    for paired in (False,True):
        torch.manual_seed(41);m=FairLearner(6,4,paired,'dense').double().eval()
        r=audit_hidden_generators(m,rows,'cpu',torch.float64)
        assert not r['learned_same_action_established']
        print('untrained original' if not paired else 'untrained paired original',json.dumps(r),flush=True)
