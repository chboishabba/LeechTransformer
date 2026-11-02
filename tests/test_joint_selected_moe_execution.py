"""Model-level BIDI: original learned step == selected-only exact joint dispatch.

The route and normalized weights come from the ORIGINAL local router;
the peer receives one activation and only admitted selected histories.
Inactive histories are retained by the authoritative source. The peer
does not infer an opposite semantic argument or recompute global route.

Random initialized networks are protocol tests, NOT trained MoE quality.
"""
import unittest
import numpy as np
import torch
from models.history_orbit_moe import HistoryOrbitMoE
from scripts.federated_joint_state_quotient import (
    JointState,pack_joint_selected,unpack_joint_selected,
    merge_selected_into_authoritative_state)


class JointSelectedMoEExecutionTests(unittest.TestCase):
    def test_all_router_geometries_exact_selected_execution(self):
        for mode in ("raw","c2_invariant","c2c3_wave","punctured9_wave"):
            with self.subTest(route=mode):
                torch.manual_seed(17)
                m=HistoryOrbitMoE(48,4,2,24,route_geometry=mode,
                                  transition_rank=24).eval()
                x=torch.randn(1,48)
                histories=torch.randn(1,4,48)
                with torch.no_grad():
                    original=m.step(x,histories)
                    original_output,after,ids,signs,weights=original
                    # Packet encoder requires sorted, duplicate-free IDs.
                    order=torch.argsort(ids,dim=1)
                    chosen=ids.gather(1,order)
                    routed_weights=weights.gather(1,order)
                    sel=tuple(int(e) for e in chosen[0].tolist())
                    before=JointState(x[0].numpy().tobytes(),
                        {e:histories[0,e].numpy().tobytes()
                         for e in range(4)})
                    packet=pack_joint_selected(before,selected=sel,
                        expert_count=4,itemsize=4,block_width=24,
                        model_revision="real-model-v1",history_revision="hist-v1")
                    received=unpack_joint_selected(packet,
                        expected_model="real-model-v1",
                        expected_history="hist-v1",expected_selected=sel)
                    remote_x=torch.from_numpy(np.frombuffer(
                        received.activation,dtype="<f4").copy()).view(1,48)
                    remote_h=torch.stack([
                        torch.from_numpy(np.frombuffer(
                            received.histories[e],dtype="<f4").copy())
                        for e in sel],dim=0).unsqueeze(0)
                    # Packet contains NO mixture weights; expert worker
                    # returns k unweighted vectors, source aggregates.
                    partials,remote_after,remote_signs=m.execute_selected(
                        remote_x,chosen,remote_h)
                    # Same expert-index accumulation order as step.
                    source_aggregate=x.new_zeros(x.shape)
                    for expert_id in range(m.num_experts):
                        positions=(chosen==expert_id).nonzero(as_tuple=False)
                        if positions.numel():
                            rows,slots=positions.unbind(1)
                            source_aggregate=source_aggregate.index_add(
                                0,rows,partials[rows,slots]*
                                routed_weights[rows,slots,None])
                    self.assertTrue(torch.allclose(
                        source_aggregate,original_output,rtol=1e-6,atol=1e-6))
                    # Alternative: source can dispatch weights explicitly
                    # and authorize the remote side to aggregate.
                    weighted,_,_=m.execute_selected(
                        remote_x,chosen,remote_h,routed_weights)
                    self.assertTrue(torch.allclose(
                        weighted,original_output,rtol=1e-6,atol=1e-6))
                    self.assertTrue(torch.equal(
                        remote_signs,signs.gather(1,order)))
                    selected_updated=JointState(received.activation,{
                        e:remote_after[0,i].numpy().tobytes()
                        for i,e in enumerate(sel)})
                    merged=merge_selected_into_authoritative_state(
                        before,selected_updated,selected=sel,expert_count=4)
                    all_hist=torch.stack([
                        torch.from_numpy(np.frombuffer(
                            merged.histories[e],dtype="<f4").copy())
                        for e in range(4)],0)
                    self.assertTrue(torch.allclose(
                        all_hist,after[0],atol=1e-6,rtol=1e-6))
                    for e in set(range(4))-set(sel):
                        self.assertTrue(torch.equal(
                            all_hist[e],histories[0,e]))

    def test_multi_token_causal_dispatch_replay_and_inactive_history(self):
        torch.manual_seed(29)
        m=HistoryOrbitMoE(48,4,2,24,route_geometry="punctured9_wave",
                          transition_rank=24).eval()
        tokens=torch.randn(1,6,48)
        initial=torch.randn(1,4,48)
        with torch.no_grad():
            full=m(tokens,initial)
            source_history=initial.clone()
            outputs=[]
            for t in range(tokens.shape[1]):
                x=tokens[:,t]
                reference=m.step(x,source_history)
                _,_,ids,_,weights=reference
                order=torch.argsort(ids,dim=1)
                ordered_ids=ids.gather(1,order)
                ordered_weights=weights.gather(1,order)
                selected=tuple(int(e) for e in ordered_ids[0].tolist())
                authoritative=JointState(x[0].numpy().tobytes(),
                    {e:source_history[0,e].numpy().tobytes() for e in range(4)})
                packet=pack_joint_selected(authoritative,selected=selected,
                    expert_count=4,itemsize=4,block_width=24,
                    model_revision="model-A",history_revision=f"t={t}")
                got=unpack_joint_selected(packet,expected_model="model-A",
                    expected_history=f"t={t}",expected_selected=selected)
                restored_x=torch.from_numpy(np.frombuffer(
                    got.activation,dtype="<f4").copy()).view(1,48)
                restored_hist=torch.stack([
                    torch.from_numpy(np.frombuffer(
                        got.histories[e],dtype="<f4").copy())
                    for e in selected]).view(1,2,48)
                parts,new_hist,_=m.execute_selected(
                    restored_x,ordered_ids,restored_hist)
                total=x.new_zeros(x.shape)
                for e in range(4):
                    matches=(ordered_ids==e).nonzero(as_tuple=False)
                    if matches.numel():
                        rows,slots=matches.unbind(1)
                        total=total.index_add(0,rows,
                            parts[rows,slots]*ordered_weights[rows,slots,None])
                outputs.append(total)
                candidate=JointState(got.activation,
                    {e:new_hist[0,j].numpy().tobytes()
                     for j,e in enumerate(selected)})
                restored=merge_selected_into_authoritative_state(
                    authoritative,candidate,selected=selected,expert_count=4)
                source_history=torch.stack([
                    torch.from_numpy(np.frombuffer(
                        restored.histories[e],dtype="<f4").copy())
                    for e in range(4)]).view(1,4,48)
            self.assertTrue(torch.allclose(
                torch.stack(outputs,dim=1),full[0],atol=1e-5,rtol=1e-5))
            self.assertTrue(torch.allclose(
                source_history,full[1],atol=1e-5,rtol=1e-5))

    def test_remote_requires_explicit_route_and_normalized_weights(self):
        m=HistoryOrbitMoE(48,4,2,24,route_geometry="punctured9_wave")
        x=torch.zeros(1,48);h=torch.zeros(1,2,48);w=torch.ones(1,2)/2
        with self.assertRaises(ValueError):
            m.execute_selected(x,torch.tensor([[1,1]]),h,w)
        with self.assertRaises(ValueError):
            m.execute_selected(x,torch.tensor([[4,1]]),h,w)


if __name__=="__main__":
    unittest.main()
