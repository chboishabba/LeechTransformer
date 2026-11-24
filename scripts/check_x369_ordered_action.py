"""CPU contract checks; fixture only, not original-model acceptance or training."""
from scripts.train_x369_ordered_action import *
from scripts.audit_x369_ordered_action_predictions import audit_predictions,KEYS
from scripts.check_lila_paired_history import Fixture

def main():
    torch.set_num_threads(1);torch.manual_seed(41)
    rows=list(records());A=generator_fit(rows)
    assert all(generator_predict(A,r)==(r['mean_label'],r['gram_label'],r['signed_label']) for r in rows)
    for mode,n in zip(MODES,(115,920,1610,3105)):
        assert len({encode_inputs(r,mode) for r in rows})==n
        for r in rows:
            neg=dict(r,residual=tuple(-x for x in r['residual']))
            assert (encode_inputs(r,mode)==encode_inputs(neg,mode))==(mode!='signed' or not any(r['residual']))
    for geom in ('dense','c3','s3','full'):
        model=OrderedLearner(3,4,True,geom,Fixture()).double()
        rs=[r for r in rows if r['word']=='aj'][:16];s,o,y=batch(rs,'signed','cpu',torch.float64)
        p=model(s,o);loss=sum(NF.cross_entropy(v,y[:,i]) for i,v in enumerate(p));loss.backward()
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        assert model.residual_seed.weight.grad is not None
    logits=[NF.one_hot(torch.tensor([L.index(r[k]) for r in rows]),len(L)).double()*20 for L,k in zip(LABELS,KEYS)]
    good=audit_predictions(rows,logits);bad=audit_predictions(rows,[torch.zeros_like(x) for x in logits])
    assert good['prefix_extension_pairs']==3888 and good['hard_action_defects']==[0]*3
    assert max(good['distribution_l1_mean'])<1e-14
    assert good['both_order_endpoints_correct']==good['ordered_separation_pairs']==[972,864,864]
    assert bad['distribution_l1_mean']==[0.]*3 and bad['both_order_endpoints_correct']==[0]*3
    print('PASS 11664 learned-operator labels; 46656 bottleneck comparisons; 4 paired CPU fixture forward/backward arms; 3888 action pairs; uniform negative control')
if __name__=='__main__':main()
