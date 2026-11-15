"""CPU protocol contract tests: real frame/HMAC/JQ1/selected expert execution.
These DO NOT demonstrate physical remote GPU throughput, WAN encryption,
recurrent history, durability, or complete model correctness.
"""
import os
import socket
import tempfile
import threading
import unittest

import torch
from models.history_orbit_moe import HistoryOrbitMoE
from scripts.federated_authenticated_session import seal,SessionReceiver
from scripts.federated_joint_state_quotient import JointState
from scripts.native_federated_selected_worker import (
    encode_float,decode_float,encode_message,decode_message,request_wire,
    response_from_worker,send_frame,recv_frame,worker_connection,read_shared_key,
    session_id,history_revision)

class WorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.key=b"A"*32
        self.model="checkpoint-sha256-fixture"
        self.session="sample_session"
        self.moe=HistoryOrbitMoE(d_model=24,num_experts=4,top_k=2,
            expert_width=12,route_geometry="raw").eval()
        self.x=torch.randn(1,24)
        self.h=torch.zeros(1,4,24)
        with torch.no_grad():
            self.original,self.next_h,self.ids,_,self.weights=self.moe.step(self.x,self.h)
        order=torch.argsort(self.ids[0])
        self.selected=tuple(self.ids[0][order].tolist())
        self.sorted_weights=self.weights[0][order]
        self.state=JointState(encode_float(self.x[0]),{
            e:encode_float(self.h[0,e]) for e in self.selected})

    def _request(self,seq,*,model=None,selected=None):
        selected=self.selected if selected is None else selected
        model=self.model if model is None else model
        joint=request_wire(self.state,selected,count=4,model_revision=model,
            history_revision_text=history_revision(seq).decode())
        payload=encode_message({"kind":"selected-expert-request",
            "sequence":seq,"layer":0,"model":model,
            "selected":list(selected)},joint)
        return seal(self.key,session_id("request",self.session),
            seq,history_revision(seq),payload)

    def test_jq1_source_and_selected_gpu_contract_cpu_oracle(self):
        wire=request_wire(self.state,self.selected,count=4,
            model_revision=self.model,history_revision_text="hist-0")
        from scripts.federated_joint_state_quotient import unpack_joint_selected
        opened=unpack_joint_selected(wire,expected_model=self.model,
            expected_history="hist-0",expected_selected=self.selected)
        out,updated=response_from_worker(self.moe,opened,self.selected,
            torch.device("cpu"))
        k=len(self.selected)
        y=torch.stack([decode_float(out[j*24*4:(j+1)*24*4],24,torch.device("cpu"))
                       for j in range(k)])
        reconstructed=(y*self.sorted_weights[:,None]).sum(dim=0)
        self.assertTrue(torch.allclose(reconstructed,self.original[0],atol=1e-6,rtol=1e-6))
        self.assertEqual(tuple(opened.histories),self.selected)

    def test_persistent_tcp_framing_and_authenticated_monotone_replies(self):
        left,right=socket.socketpair()
        finished=[]
        def serve():
            try:
                finished.append(worker_connection(right,moe=self.moe,
                    key=self.key,model_rev=self.model,session=self.session,
                    layer=0,max_requests=2))
            except Exception as exc:finished.append(exc)
        thread=threading.Thread(target=serve,daemon=True)
        thread.start()
        response_receiver=SessionReceiver(self.key,
            session_id("response",self.session),history_revision(0))
        try:
            # Two requests are actually in flight before any reply is read.
            for seq in range(2):
                send_frame(left,self._request(seq))
            for seq in range(2):
                reply,_=recv_frame(left)
                payload=response_receiver.verify(reply)
                meta,raw=decode_message(payload)
                self.assertEqual(meta["sequence"],seq)
                self.assertEqual(meta["selected"],list(self.selected))
                self.assertEqual(len(raw),(2*len(self.selected)*24+len(self.selected))*4)
                response_receiver.commit(history_revision(seq+1))
        finally:
            left.close()
        thread.join(timeout=5)
        self.assertEqual(finished,[2])

    def test_stale_model_refused(self):
        left,right=socket.socketpair()
        failures=[]
        def serve():
            try:
                worker_connection(right,moe=self.moe,key=self.key,
                    model_rev=self.model,session=self.session,layer=0,max_requests=1)
            except ValueError as exc:failures.append(str(exc))
        t=threading.Thread(target=serve,daemon=True)
        t.start()
        try:send_frame(left,self._request(0,model="bad-model"))
        finally:left.close()
        t.join(timeout=5)
        self.assertTrue(failures)

    def test_duplicate_request_replay_refused(self):
        left,right=socket.socketpair()
        failures=[]
        def serve():
            try:worker_connection(right,moe=self.moe,key=self.key,
                model_rev=self.model,session=self.session,layer=0,max_requests=2)
            except ValueError as exc:failures.append(str(exc))
        t=threading.Thread(target=serve,daemon=True)
        t.start()
        try:
            packet=self._request(0)
            send_frame(left,packet)
            recv_frame(left)
            send_frame(left,packet)
        finally:left.close()
        t.join(timeout=5)
        self.assertTrue(any("replay" in v for v in failures))

    def test_tampered_hmac_rejected(self):
        left,right=socket.socketpair()
        failures=[]
        def serve():
            try:worker_connection(right,moe=self.moe,key=self.key,
                model_rev=self.model,session=self.session,layer=0,max_requests=1)
            except ValueError as exc:failures.append(str(exc))
        t=threading.Thread(target=serve,daemon=True)
        t.start()
        try:
            wire=bytearray(self._request(0))
            wire[-1]^=1
            send_frame(left,bytes(wire))
        finally:left.close()
        t.join(timeout=5)
        self.assertTrue(any("authentication" in x for x in failures))

    def test_shared_key_permissions_and_binary_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=os.path.join(tmp,"key")
            with open(p,"wb") as f:f.write(b"\t"+b"A"*31+b"\n")
            os.chmod(p,0o600)
            self.assertEqual(read_shared_key(p),b"\t"+b"A"*31+b"\n")
            os.chmod(p,0o644)
            with self.assertRaises(PermissionError):read_shared_key(p)

if __name__=="__main__":
    unittest.main()
