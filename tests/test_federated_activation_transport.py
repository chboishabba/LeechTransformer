import random
import unittest
from scripts.federated_activation_transport import (
    pack_activation,unpack_activation,estimate_transfer_time_s)


class FederatedTransportTests(unittest.TestCase):
    def test_roundtrip_and_adaptive_fallback(self):
        rng=random.Random(123)
        for size in (2,4):
            for raw in (rng.randbytes(24*size*8), b"\0"*(24*size*8)):
                pkt=pack_activation(raw,dtype_size=size,block_width=24,
                                    model_revision="m1",history_revision="h7")
                self.assertEqual(unpack_activation(pkt,expected_model="m1",expected_history="h7"),raw)
                pure=pack_activation(raw,dtype_size=size,block_width=24,
                                     model_revision="m1",history_revision="h7",
                                     permit_compression=False)
                self.assertLessEqual(len(pkt),len(pure))

    def test_fail_closed_reference_and_tamper(self):
        raw=b"\0"*(24*4*4)
        pkt=pack_activation(raw,dtype_size=4,block_width=24,
                            model_revision="m1",history_revision="h1")
        with self.assertRaises(ValueError):
            unpack_activation(pkt,expected_model="m2",expected_history="h1")
        with self.assertRaises(ValueError):
            unpack_activation(pkt,expected_model="m1",expected_history="stale")
        with self.assertRaises(ValueError):
            unpack_activation(pkt,expected_model="m1",expected_history="h1",max_uncompressed=10)
        with self.assertRaises((ValueError,Exception)):
            unpack_activation(pkt[:-1],expected_model="m1",expected_history="h1")

    def test_link_model(self):
        self.assertEqual(estimate_transfer_time_s(1000,1000,0.1,2),1.2)
        with self.assertRaises(ValueError):
            estimate_transfer_time_s(100,0)

if __name__=="__main__":
    unittest.main()
