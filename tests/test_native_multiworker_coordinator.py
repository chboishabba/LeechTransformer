"""Multiworker source reduction and exact ownership/protocol negative controls.

Offline CPU unit tests using genuine HistoryOrbitMoE mathematical operations.
Physical multi-GPU admission must use the existing Schaka 10.1 native launcher.
"""
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

import torch
from models.history_orbit_moe import HistoryOrbitMoE
from scripts.federated_authenticated_session import SessionReceiver
from scripts.federated_joint_state_quotient import JointState
from scripts.native_federated_selected_worker import (
    encode_float,history_revision,session_id,worker_connection)
from scripts.native_multiworker_coordinator import Peer,read_workers,reduce_source_selected


class OwnershipAndMultiworkerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.moe=HistoryOrbitMoE(d_model=24,num_experts=4,top_k=2,
            expert_width=12,route_geometry="c2_invariant").eval()
        self.x=torch.randn(1,24)
        self.h=torch.randn(1,4,24)
        with torch.no_grad():
            self.expected,self.next_h,self.ids,_,self.weights=self.moe.step(
                self.x,self.h,train_sign=False)

    def test_route_only_matches_original_step_without_expert_execution(self):
        with torch.no_grad():
            ids,weights=self.moe.select_source_experts(self.x,self.h)
        self.assertTrue(torch.equal(ids,self.ids))
        self.assertTrue(torch.equal(weights,self.weights))
        for geometry in ("raw","c2_invariant"):
            model=HistoryOrbitMoE(
                d_model=24,num_experts=4,top_k=2,expert_width=12,
                route_geometry=geometry).eval()
            with torch.no_grad():
                _,_,from_step,_,from_step_w=model.step(self.x,self.h)
                routed,routed_weights=model.select_source_experts(self.x,self.h)
            self.assertTrue(torch.equal(from_step,routed))
            self.assertTrue(torch.equal(from_step_w,routed_weights))

    def test_subset_expert_reduction_parity(self):
        got=torch.zeros_like(self.expected[0])
        owners=({0,2},{1,3})
        for owner in owners:
            pairs=[(slot,int(e)) for slot,e in enumerate(self.ids[0].tolist())
                   if e in owner]
            if not pairs:continue
            ids=torch.tensor([[e for _,e in pairs]],dtype=torch.long)
            hist=torch.stack([self.h[0,e] for _,e in pairs]).unsqueeze(0)
            with torch.no_grad():
                output,updated,_=self.moe.execute_selected_subset(
                    self.x,ids,hist)
            for j,(slot,e) in enumerate(pairs):
                got+=output[0,j]*self.weights[0,slot]
                self.assertTrue(torch.allclose(
                    updated[0,j],self.next_h[0,e],atol=1e-6,rtol=1e-6))
        self.assertTrue(torch.allclose(
            got,self.expected[0],atol=1e-6,rtol=1e-6))

    def test_existing_full_selected_signature_unchanged(self):
        with self.assertRaises(ValueError):
            self.moe.execute_selected(self.x,
                self.ids[:,:1],self.h[:,:1,:])
        with torch.no_grad():
            out,h,_=self.moe.execute_selected_subset(self.x,
                self.ids[:,:1],self.h.gather(1,
                    self.ids[:,:1].unsqueeze(-1).expand(-1,-1,24)))
        self.assertEqual(out.shape,(1,1,24))

    def test_source_ownership_manifest_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"workers.json"
            def check(parts):
                path.write_text(json.dumps({"workers":[
                    {"name":str(i),"host":"localhost","port":31000+i,
                     "session":str(i),"key_file":"/not/read/in/parser-%d" % i,
                     "experts":list(ids)} for i,ids in enumerate(parts)]}))
                return read_workers(path,4)
            self.assertEqual(len(check([(0,1),(2,3)])),2)
            with self.assertRaises(ValueError):check([(0,1),(1,2,3)])
            with self.assertRaises(ValueError):check([(0,1),(2,)])
            with self.assertRaises(ValueError):check([(0,1,2,3),()])
            good=check([(0,1),(2,3)])
            def bad_identity(field,value):
                data={"workers":[dict(w) for w in good]}
                data["workers"][1][field]=value
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):read_workers(path,4)
            bad_identity("port",good[0]["port"])
            bad_identity("session",good[0]["session"])
            bad_identity("key_file",good[0]["key_file"])


    def test_reduction_with_sampling_off_is_not_zero_reference(self):
        with torch.no_grad():
            weights={int(e):self.weights[0,j] for j,e in enumerate(self.ids[0].tolist())}
            contributions={}
            updates={}
            for e in weights:
                selected=torch.tensor([[e]],dtype=torch.long)
                out,new_h,_=self.moe.execute_selected_subset(
                    self.x,selected,self.h[:,e:e+1,:])
                contributions[e]=out[0,0]
                updates[e]=new_h[0,0]
            merged,err,herr,checked=reduce_source_selected(
                weights,contributions,updates,d_model=24,device=self.x.device,
                dtype=self.x.dtype,reference=None,expected=None)
            self.assertFalse(checked)
            self.assertIsNone(err)
            self.assertIsNone(herr)
            self.assertTrue(torch.allclose(merged,self.expected[0],atol=1e-6,rtol=1e-6))
            expected_history={e:self.next_h[0,e] for e in updates}
            _,err,herr,checked=reduce_source_selected(
                weights,contributions,updates,d_model=24,device=self.x.device,
                dtype=self.x.dtype,reference=self.expected[0],expected=expected_history,
                atol=1e-6,rtol=1e-6)
            self.assertTrue(checked)
            self.assertLess(err,1e-6)
            self.assertLess(herr,1e-6)
            with self.assertRaises(AssertionError):
                reduce_source_selected(weights,contributions,updates,
                    d_model=24,device=self.x.device,dtype=self.x.dtype,
                    reference=torch.ones_like(self.expected[0]),
                    expected=expected_history,atol=0.,rtol=0.)
            with self.assertRaises(AssertionError):
                reduce_source_selected(weights,{e:v for e,v in contributions.items()
                    if e!=next(iter(contributions))},updates,
                    d_model=24,device=self.x.device,dtype=self.x.dtype)

    def test_worker_owned_experts_reject_duplicate_and_bad_ids(self):
        from scripts.native_federated_selected_worker import parse_owned_experts
        self.assertEqual(parse_owned_experts("0,2",4),{0,2})
        self.assertEqual(parse_owned_experts(None,4),set(range(4)))
        for bad in ("", "0,0","-1","4","0,x","0,,1"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_owned_experts(bad,4)


    def test_32row_two_peer_bounded_source_stream_with_sampled_parity(self):
        """One integrated local protocol check, not two physical GPU hosts."""
        from scripts.native_multiworker_coordinator import reduce_source_selected
        key_a=b"A"*32
        key_b=b"B"*32
        owners=({0,2},{1,3})
        keys=(key_a,key_b)
        peers=[]
        sockets=[]
        threads=[]
        finished=[]
        torch.manual_seed(716)
        try:
            for i,owner in enumerate(owners):
                a,b=socket.socketpair()
                a.settimeout(15)
                b.settimeout(15)
                sockets.extend((a,b))
                session="multi-%d"%i
                def worker(sock=b,owned=owner,k=keys[i],sess=session):
                    try:
                        finished.append(worker_connection(sock,moe=self.moe,
                            key=k,model_rev="model-verified",session=sess,
                            layer=1,max_requests=32,device=torch.device("cpu"),
                            owned_experts=owned))
                    except Exception as exc:
                        finished.append(exc)
                t=threading.Thread(target=worker,daemon=True)
                t.start()
                threads.append(t)
                peers.append(Peer("owner-%d"%i,tuple(sorted(owner)),
                    session,keys[i],a,SessionReceiver(
                        keys[i],session_id("response",session),history_revision(0))))
            verified=0
            for row in range(32):
                x=torch.randn(1,24)
                h=torch.zeros(1,4,24)
                check=(row%8==0)
                with torch.no_grad():
                    if check:
                        reference,next_h,selected,_,weights=self.moe.step(x,h)
                        expected={int(e):next_h[0,int(e)] for e in selected[0].tolist()}
                    else:
                        selected,weights=self.moe.select_source_experts(x,h)
                        reference=None
                        expected=None
                weights_map={int(e):weights[0,j] for j,e in enumerate(selected[0].tolist())}
                joint=JointState(encode_float(x[0]),{
                    e:encode_float(h[0,e]) for e in range(4)})
                sent=[]
                for peer in peers:
                    selected_here=tuple(sorted(set(peer.expert_ids)&set(weights_map)))
                    if not selected_here:
                        continue
                    payload=JointState(joint.activation,{
                        e:joint.histories[e] for e in selected_here})
                    seq=peer.dispatch(payload,selected_here,
                        model="model-verified",layer=1,count=4)
                    sent.append((peer,seq,selected_here))
                contributions={}
                updates={}
                for peer,seq,selected_here in sent:
                    response,changed,_=peer.receive(
                        expected_sequence=seq,model="model-verified",
                        layer=1,selected=selected_here,width=24)
                    for j,e in enumerate(selected_here):
                        contributions[e]=torch.from_numpy(response[j])
                        updates[e]=torch.from_numpy(changed[j])
                _,err,herr,checked=reduce_source_selected(
                    weights_map,contributions,updates,d_model=24,
                    device=x.device,dtype=x.dtype,
                    reference=reference[0] if check else None,
                    expected=expected,atol=1e-5,rtol=1e-5)
                if check:
                    verified+=1
                    self.assertTrue(checked)
                    self.assertLess(err,1e-5)
                    self.assertLess(herr,1e-5)
                else:
                    self.assertFalse(checked)
                    self.assertIsNone(err)
                for peer,seq,_ in sent:
                    peer.acknowledge(seq)
            self.assertEqual(verified,4)
            self.assertTrue(all(p.completed>0 for p in peers))
        finally:
            for sock in sockets:
                try:sock.close()
                except OSError:pass
            for t in threads:
                t.join(timeout=15)
        self.assertEqual(len(finished),2)
        self.assertTrue(all(type(value) is int for value in finished),finished)
        self.assertEqual(sum(finished),sum(peer.completed for peer in peers))

    def test_two_real_authenticated_socketpair_workers(self):
        # The same code runs on genuine TCP sockets on remote GPU workers.
        # socketpair keeps this a source/protocol test, not a network benchmark.
        sockets=[]
        threads=[]
        peers=[]
        finished=[]
        key=b"K"*32
        chosen=self.ids[0].tolist()
        unused=[e for e in range(4) if e not in chosen]
        # Ensure each worker receives a selected expert, regardless of
        # which pair the real source router chose on this deterministic row.
        owners=({chosen[0],unused[0]},{chosen[1],unused[1]})
        try:
            for i,owner in enumerate(owners):
                a,b=socket.socketpair()
                sockets.extend((a,b))
                session="worker%d"%i
                def worker(sock=b,owned=owner,sess=session):
                    try:
                        finished.append(worker_connection(sock,moe=self.moe,
                            key=key,model_rev="m1",session=sess,layer=0,
                            max_requests=1,device=torch.device("cpu"),
                            owned_experts=owned))
                    except Exception as ex:finished.append(ex)
                t=threading.Thread(target=worker,daemon=True)
                t.start()
                threads.append(t)
                peers.append(Peer(str(i),tuple(sorted(owner)),session,key,a,
                    SessionReceiver(key,session_id("response",session),
                        history_revision(0))))
            source=JointState(encode_float(self.x[0]),{
                e:encode_float(self.h[0,e]) for e in range(4)})
            tasks=[]
            for peer in peers:
                selected=tuple(sorted(set(peer.expert_ids)&
                    set(self.ids[0].tolist())))
                if not selected:continue
                seq=peer.dispatch(JointState(source.activation,{
                    e:source.histories[e] for e in selected}),selected,
                    model="m1",layer=0,count=4)
                tasks.append((peer,seq,selected))
            total=torch.zeros_like(self.expected[0])
            updates={}
            for peer,seq,selected in tasks:
                y,h,_=peer.receive(expected_sequence=seq,
                    model="m1",layer=0,selected=selected,width=24)
                for j,e in enumerate(selected):
                    slot=(self.ids[0]==e).nonzero(as_tuple=False)[0,0]
                    total+=torch.from_numpy(y[j])*self.weights[0,slot]
                    updates[e]=torch.from_numpy(h[j])
            self.assertTrue(torch.allclose(
                total,self.expected[0],atol=1e-6,rtol=1e-6))
            for e,updated in updates.items():
                self.assertTrue(torch.allclose(
                    updated,self.next_h[0,e],atol=1e-6,rtol=1e-6))
            for peer,seq,_ in tasks:
                peer.acknowledge(seq)
                self.assertEqual(peer.completed,1)
        finally:
            for s in sockets:
                try:s.close()
                except OSError:pass
            for t in threads:t.join(timeout=5)
        self.assertEqual(len(finished),len(tasks))
        self.assertTrue(all(x==1 for x in finished))


if __name__=="__main__":
    unittest.main()
