"""Exhaustive finite control; no neural training or native execution."""
from itertools import product
import torch
from x369_mod3_geometric_control import orbit_forward,training_edges,edge_loss,transport_metrics

def main():
    torch.set_num_threads(1)
    pairs=torch.tensor(list(product(range(3),repeat=2)))
    labels=pairs.sum(1)%3
    edges=training_edges(pairs[:8]);assert len(edges)==14 and int(edges.max())==7
    wrong_ceiling=0;good_count=0
    for anchor in product(range(3),repeat=3):
        w=torch.nn.functional.one_hot(torch.tensor(anchor),3).double()
        def model(p):
            assert (p[:,0]==0).all() and len(p)==3
            return w[p[:,1]]
        z=orbit_forward(model,pairs,1);a=transport_metrics(z)
        assert a['first_action_max_error']==0
        if (z[:8].argmax(-1)==labels[:8]).all():
            assert a['strict_all_correct'] and a['intended_margin_certificate'];good_count+=1
        q=orbit_forward(model,pairs,-1);assert transport_metrics(q,-1)['first_action_max_error']==0
        wrong_ceiling=max(wrong_ceiling,int((q[:8].argmax(-1)==labels[:8]).sum()))
    assert good_count==1 and wrong_ceiling==3
    z=torch.nn.functional.one_hot(labels,3).double();assert edge_loss(z[:8],edges)==0
    z[8]=torch.tensor([1.,0.,0.]);assert edge_loss(z[:8],edges)==0
    assert not transport_metrics(z)['intended_margin_certificate']
    # For a trainable representative table, backward through the orbit gather is valid.
    w=torch.randn(3,3,dtype=torch.float64,requires_grad=True)
    z=orbit_forward(lambda p:w[p[:,1]],pairs[:8],1)
    torch.nn.functional.cross_entropy(z,labels[:8]).backward()
    assert w.grad is not None and torch.isfinite(w.grad).all() and (w.grad.norm(dim=1)>0).all()
    print('PASS 27 representative tables; one correct extension; reversed strict-label ceiling 3/8; 14 train edges; missing-node witness; gradient connectivity')
if __name__=='__main__':main()
