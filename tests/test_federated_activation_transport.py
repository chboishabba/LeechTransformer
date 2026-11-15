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

    def test_latency_aware_codec_choice(self):
        import json
        import struct
        raw=b"\\0"*(24*4*8)
        opts=dict(dtype_size=4, block_width=24,
                  model_revision="m", history_revision="h")
        best=pack_activation(raw,**opts,bandwidth_bytes_per_s=1000)
        hlen=struct.unpack("<I",best[4:8])[0]
        self.assertIn(json.loads(best[8:8+hlen])["codec"],("zlib","orbit"))
        slow=pack_activation(
            raw,**opts,bandwidth_bytes_per_s=1000000000,
            encode_cost_seconds={"zlib":5.0,"orbit":5.0},
            decode_cost_seconds={"zlib":5.0,"orbit":5.0})
        hlen=struct.unpack("<I",slow[4:8])[0]
        self.assertEqual(json.loads(slow[8:8+hlen])["codec"],"raw")

    def test_nested_orbit_mismatch_rejected(self):
        import hashlib
        import json
        import struct
        from scripts.ieee_sign_orbit_codec import encode_orbit
        raw=b"\\0"*(24*4*3)
        packet=bytearray(pack_activation(raw,dtype_size=4,block_width=24,
                       model_revision="m", history_revision="h",
                       permit_compression=False))
        n=struct.unpack("<I",packet[4:8])[0]
        meta=json.loads(packet[8:8+n])
        meta["codec"]="orbit"
        head=json.dumps(meta,separators=(",",":"),sort_keys=True).encode()
        nested=encode_orbit(raw,4,24)
        wrong=nested[:4]+(48).to_bytes(2,"little")+nested[6:]
        adversarial=b"FAI1"+struct.pack("<I",len(head))+head+wrong
        with self.assertRaises(ValueError):
            unpack_activation(adversarial,expected_model="m",expected_history="h")

    def test_link_model(self):
        self.assertEqual(estimate_transfer_time_s(1000,1000,0.1,2),1.2)
        with self.assertRaises(ValueError):
            estimate_transfer_time_s(100,0)

if __name__=="__main__":
    unittest.main()
