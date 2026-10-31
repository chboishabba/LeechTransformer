"""Byte-exact sign-orbit codec tests, including NaNs and signed zeros."""
import random
import unittest
from scripts.ieee_sign_orbit_codec import encode_orbit,decode_orbit,flip_ieee_signs

class SignOrbitCodecTests(unittest.TestCase):
    def test_roundtrip_patterns(self):
        rng=random.Random(20261031)
        for size in (2,4):
            for width in (24,48,384):
                for blocks in (1,3,12):
                    for pattern in ("random","zero","ones","signedzero"):
                        n=size*width*blocks
                        data=(rng.randbytes(n) if pattern=="random"
                              else b"\x00"*n if pattern=="zero"
                              else b"\xff"*n if pattern=="ones"
                              else bytes([0,0,0,128])*(n//4) if size==4
                              else bytes([0,128])*(n//2))
                        packet=encode_orbit(data,size,24)
                        self.assertEqual(decode_orbit(packet),data)

    def test_sign_flip_involutive(self):
        rng=random.Random(5)
        for size in (2,4):
            raw=rng.randbytes(size*24)
            self.assertEqual(flip_ieee_signs(flip_ieee_signs(raw,size),size),raw)

    def test_corrupt_header_and_trailing_stream_rejected(self):
        raw=bytes(range(96))
        packet=encode_orbit(raw,4,24)
        with self.assertRaises(ValueError):
            decode_orbit(packet[:6]+bytes([2,0])+packet[8:])
        with self.assertRaises(ValueError):
            decode_orbit(packet+b"trailing")
        with self.assertRaises(ValueError):
            encode_orbit(raw,4,65536)
        with self.assertRaises(ValueError):
            encode_orbit(raw,3,24)

    def test_bad_frame_rejected(self):
        with self.assertRaises(ValueError):
            encode_orbit(b"X"*13,4,24)
        with self.assertRaises(ValueError):
            decode_orbit(b"SO1")

if __name__=="__main__":
    unittest.main()
