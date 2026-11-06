"""Authenticated monotonic session envelope for federated model-state messages.

Integrity/authenticity only: HMAC is NOT encryption. No compression of
untrusted and secret data in a shared context. Receiver must establish the
shared key and current revision over a trusted channel beforehand.
"""
from dataclasses import dataclass
import hashlib
import hmac
import struct

MAGIC = b'FAS1'
TAG_BYTES = 32
MAX_SESSION_BYTES = 255
MAX_PAYLOAD_BYTES = 64*1024*1024

def _validate_key(key):
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError('require a pre-shared 256-bit-or-longer MAC key')

def seal(key: bytes, session: bytes, sequence: int, history_revision: bytes,
         payload: bytes) -> bytes:
    _validate_key(key)
    if not (0 < len(session) <= MAX_SESSION_BYTES and
            0 < len(history_revision) <= MAX_SESSION_BYTES):
        raise ValueError('empty or oversized session/history identifier')
    if not 0 <= sequence < 1 << 64:
        raise ValueError('sequence outside u64')
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError('payload exceeds limit')
    body = (MAGIC + bytes([len(session),len(history_revision)]) +
            struct.pack('>QI',sequence,len(payload)) +
            session + history_revision + payload)
    tag = hmac.new(key,body,hashlib.sha256).digest()
    return body + tag

@dataclass
class SessionReceiver:
    key: bytes
    session: bytes
    expected_history_revision: bytes
    next_sequence: int = 0

    def __post_init__(self):
        _validate_key(self.key)
        if not self.session or not self.expected_history_revision:
            raise ValueError('session and history revision required')

    def verify(self, packet: bytes) -> bytes:
        """Verify without changing state. Advance only after successful execution."""
        if len(packet) < 18+TAG_BYTES or packet[:4] != MAGIC:
            raise ValueError('bad envelope')
        n_session,n_history = packet[4],packet[5]
        sequence,body_len=struct.unpack('>QI',packet[6:18])
        expected_len=18+n_session+n_history+body_len+TAG_BYTES
        if (not n_session or not n_history or body_len > MAX_PAYLOAD_BYTES
                or len(packet) != expected_len):
            raise ValueError('bad lengths')
        content, supplied_tag = packet[:-TAG_BYTES], packet[-TAG_BYTES:]
        if not hmac.compare_digest(hmac.new(self.key,content,hashlib.sha256).digest(),supplied_tag):
            raise ValueError('authentication failure')
        start=18
        session=packet[start:start+n_session]
        history=packet[start+n_session:start+n_session+n_history]
        if session != self.session or history != self.expected_history_revision:
            raise ValueError('session/history mismatch')
        if sequence != self.next_sequence:
            raise ValueError('replay, gap or out-of-order message')
        return packet[18+n_session+n_history:-TAG_BYTES]

    def commit(self, new_history_revision: bytes) -> None:
        """Call only after verified payload was executed and durably recorded.

        Application must ensure verify->execute->commit is atomic; this
        in-memory prototype does not survive process crashes and is not an
        exactly-once delivery guarantee.
        """
        if not new_history_revision or len(new_history_revision)>MAX_SESSION_BYTES:
            raise ValueError('invalid new history revision')
        if self.next_sequence == (1 << 64)-1:
            raise ValueError('sequence rollover requires rekey')
        self.next_sequence += 1
        self.expected_history_revision = new_history_revision
