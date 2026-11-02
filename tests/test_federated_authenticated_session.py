import unittest
from scripts.federated_authenticated_session import seal,SessionReceiver

KEY=b'K'*32
class SessionTests(unittest.TestCase):
    def test_sequential_exact_history(self):
        rx=SessionReceiver(KEY,b'session-A',b'h0')
        p=seal(KEY,b'session-A',0,b'h0',b'activation')
        self.assertEqual(rx.verify(p),b'activation')
        self.assertEqual(rx.next_sequence,0)
        rx.commit(b'h1')
        with self.assertRaises(ValueError): rx.verify(p)
        self.assertEqual(rx.verify(seal(KEY,b'session-A',1,b'h1',b'result')),b'result')
    def test_tamper_cross_session_and_gap(self):
        rx=SessionReceiver(KEY,b'A',b'h0')
        p=seal(KEY,b'A',0,b'h0',b'x')
        for candidate in (p[:-1]+bytes([p[-1]^1]),seal(KEY,b'B',0,b'h0',b'x'),
                          seal(KEY,b'A',1,b'h0',b'x'),seal(KEY,b'A',0,b'old',b'x')):
            with self.assertRaises(ValueError): rx.verify(candidate)
    def test_requires_real_key(self):
        with self.assertRaises(ValueError): seal(b'bad',b'A',0,b'h',b'x')
if __name__=='__main__':unittest.main()
