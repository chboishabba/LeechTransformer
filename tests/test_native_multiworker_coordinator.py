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
from scripts.native_multiworker_coordinator import Peer,read_workers


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
                     "session":str(i),"key_file":"/not/read/in/parser",
                     "experts":list(ids)} for i,ids in enumerate(parts)]}))
                return read_workers(path,4)
            self.assertEqual(len(check([(0,1),(2,3)])),2)
            with self.assertRaises(ValueError):check([(0,1),(1,2,3)])
            with self.assertRaises(ValueError):check([(0,1),(2,)])
            with self.assertRaises(ValueError):check([(0,1,2,3),()])

    def test_two_real_authenticated_socketpair_workers(self):
        # The same code runs on genuine TCP sockets on remote GPU workers.
        # socketpair keeps this a source/protocol test, not a network benchmark.
        sockets=[]
        threads=[]
        peers=[]
        finished=[]
        key=b"K"*32
        owners=({0,2},{1,3})
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
