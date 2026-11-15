"""Fail-closed SentencePiece source coordinates across 0.2.1 / 0.2.2.

The immutable protobuf may expose Unicode character positions, while the
serialized or object protobuf may expose UTF-8 byte positions. The raw
protobuf representation is not an offset-unit authority: only original
source surfaces and exact UTF-8 boundaries determine the conversion.
Never detach ImmutableSentencePiece pieces without materializing their
fields: they can borrow memory from the parent protobuf object.
"""
from __future__ import annotations
from types import SimpleNamespace


def original_sentencepiece_pieces(tokenizer, source):
    def snapshot(proto):
        return [SimpleNamespace(id=int(p.id), begin=int(p.begin),
            end=int(p.end), surface=str(p.surface)) for p in proto.pieces]
    try:
        obj=tokenizer.encode(source,out_type="immutable_proto")
        if hasattr(obj,"pieces"):
            return snapshot(obj)
    except (ValueError,TypeError,AttributeError):
        pass
    proto=tokenizer.encode(source,out_type="proto")
    if isinstance(proto,(bytes,bytearray,memoryview)):
        from sentencepiece import sentencepiece_pb2
        parsed=sentencepiece_pb2.SentencePieceText()
        parsed.ParseFromString(bytes(proto))
        return snapshot(parsed)
    if hasattr(proto,"pieces"):
        return snapshot(proto)
    raise ValueError("SentencePiece proto has no compatible piece representation")


def _unicode_spans(original,pieces,unit):
    byte_boundaries={0:0}
    point=0
    for i,c in enumerate(original,1):
        point+=len(c.encode("utf-8"))
        byte_boundaries[point]=i
    spans=[]
    for piece in pieces:
        lo,hi=int(piece.begin),int(piece.end)
        if lo<0 or hi<lo:
            return None
        if unit=="utf8_bytes":
            if lo not in byte_boundaries or hi not in byte_boundaries:
                return None
            lo,hi=byte_boundaries[lo],byte_boundaries[hi]
        elif hi>len(original):
            return None
        surface=getattr(piece,"surface",None)
        if not isinstance(surface,str) or (surface and original[lo:hi]!=surface):
            return None
        spans.append((lo,hi))
    return tuple(spans)


def normalize_sentencepiece_piece_spans(original,pieces):
    """Return exact Unicode spans + detected source unit; reject ambiguity."""
    if not isinstance(original,str) or not original or not pieces:
        raise ValueError("nonempty original SentencePiece source and pieces required")
    characters=_unicode_spans(original,pieces,"unicode_codepoints")
    bytes_=_unicode_spans(original,pieces,"utf8_bytes")
    if characters is None and bytes_ is None:
        raise ValueError("SentencePiece source coordinate/surface mismatch")
    if characters is not None and bytes_ is not None:
        if characters!=bytes_:
            raise ValueError("ambiguous SentencePiece Unicode/UTF8 source coordinates")
        return characters,"equivalent_unicode_utf8"
    if characters is not None:
        return characters,"unicode_codepoints"
    return bytes_,"utf8_bytes"
