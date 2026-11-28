"""Device and byte-authority contracts; CPU-only tests do not certify GPU."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from scripts.x369_r4_execution_device import (
    resolve_device,runtime,tensor_bytes,bitwise_equal,verify_trace_runtime)


class R4ExecutionDevice(unittest.TestCase):
    def test_portable_cpu_and_strict_signed_zero(self):
        d=resolve_device('cpu')
        self.assertEqual(d.type,'cpu')
        a=torch.tensor([0.,1.]);b=torch.tensor([-0.,1.])
        self.assertTrue(torch.equal(a,b))
        self.assertFalse(bitwise_equal(a,b))
        self.assertNotEqual(tensor_bytes(a),tensor_bytes(b))
        self.assertTrue(bitwise_equal(a,a.clone()))
        self.assertEqual(verify_trace_runtime({},None,d),runtime(d))
        rec={'source_execution_runtime':runtime(d)}
        self.assertEqual(verify_trace_runtime(rec,rec,d),runtime(d))
        with self.assertRaises(ValueError):
            verify_trace_runtime({'source_execution_runtime':{'mode':'gfx803'}},None,d)

    def test_gfx803_fail_closed(self):
        with patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaises(RuntimeError):resolve_device('gfx803')
        with patch.object(torch.cuda,'is_available',return_value=True), \
             patch.object(torch.version,'hip','test-hip'), \
             patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(gcnArchName='gfx1100')):
            with self.assertRaises(RuntimeError):resolve_device('gfx803')
        with patch.object(torch.cuda,'is_available',return_value=True), \
             patch.object(torch.version,'hip','test-hip'), \
             patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(gcnArchName='gfx803')):
            d=resolve_device('gfx803')
            self.assertEqual(str(d),'cuda:0')
            actual=runtime(d)
            self.assertEqual(actual['mode'],'gfx803')
            with self.assertRaises(ValueError):verify_trace_runtime({},None,d)
            rec={'source_execution_runtime':actual}
            self.assertEqual(verify_trace_runtime(rec,rec,d),actual)
            with self.assertRaises(ValueError):verify_trace_runtime(rec,{},d)
        with self.assertRaises(ValueError):resolve_device('auto')


if __name__=='__main__':
    unittest.main()
