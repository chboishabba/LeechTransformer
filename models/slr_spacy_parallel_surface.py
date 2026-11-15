"""Source-aligned SLR numeric spaCy sidecar for original Leech tokens.

SLR owns parser execution and database-local BIGINT symbols. This module
never runs spaCy or assumes a BIGINT is a dense neural vocabulary ID.
Observations are candidate-only. A full-doc observation is NOT declared
prefix-visible for autoregressive prediction.

Expected input is a source-text + tokenizer-offset + SLR numeric export;
physical export/source joining is a separate acceptance gate. Byte offsets
and Unicode character offsets are distinctly tagged. SLR offset contract is
Unicode character coordinates. Missing labels have compact ID 0 on input
and -100 on prediction targets: neither represents a semantic zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SourceSpan:
    begin: int
    end: int

    def __post_init__(self):
        if type(self.begin) is not int or type(self.end) is not int or self.begin < 0 or self.end < self.begin:
            raise ValueError("invalid source span")


@dataclass(frozen=True)
class NumericSpacyToken:
    token_id: int
    span: SourceSpan
    pos_symbol_id: int | None
    dependency_id: int | None
    morph_set_id: int | None
    head_token_id: int | None

    def __post_init__(self):
        if type(self.token_id) is not int or self.token_id < 1:
            raise ValueError("SLR token_id must be positive BIGINT")
        for name in ("pos_symbol_id", "dependency_id", "morph_set_id", "head_token_id"):
            val = getattr(self, name)
            if val is not None and (type(val) is not int or val < 1):
                raise ValueError("invalid numeric SLR " + name)
        if self.span.begin == self.span.end:
            raise ValueError("SLR lexical token cannot have an empty span")


class CompactSymbolVocabulary:
    """Train-only dense encoding of SLR's database-local IDs.

    Input ID 0 means absent/OOV observation, not neutral POS. Target -100
    is ignored, not a grammatical class. Class 0 is reserved for missing.
    """
    def __init__(self, training_symbols: Sequence[int]):
        if any(type(v) is not int or v < 1 for v in training_symbols):
            raise ValueError("only positive numeric symbols are vocabulary items")
        self.lookup = {v: i + 1 for i, v in enumerate(sorted(set(training_symbols)))}

    @property
    def size(self) -> int:
        return len(self.lookup) + 1

    def input_id(self, symbol: int | None) -> int:
        return self.lookup.get(symbol, 0) if symbol is not None else 0

    def target_id(self, symbol: int | None) -> int:
        return self.lookup.get(symbol, -100) if symbol is not None else -100


@dataclass(frozen=True)
class AlignedSpacyDoc:
    source_sha256: str
    parser_contract_sha256: str
    tokenizer_sha256: str
    visibility: str
    weights: torch.Tensor  # [word,subword], normalised per SLR word
    token_ids: tuple[int, ...]
    head_positions: tuple[int, ...]  # index into SLR words; -100 if missing
    source_character_spans: tuple[SourceSpan, ...]

    def word_hidden(self, subword_hidden: torch.Tensor) -> torch.Tensor:
        if subword_hidden.ndim != 2 or subword_hidden.shape[0] != self.weights.shape[1]:
            raise ValueError("subword hidden states do not fit source alignment")
        return self.weights.to(device=subword_hidden.device,
                               dtype=subword_hidden.dtype) @ subword_hidden


def _valid_sha(s: str) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in '0123456789abcdef' for c in s)


def _as_character_spans(text: str, spans: Sequence[SourceSpan], unit: str) -> list[SourceSpan]:
    if unit == "unicode_codepoints":
        if any(s.end > len(text) for s in spans):
            raise ValueError("Unicode source span exceeds original text")
        return list(spans)
    if unit != "utf8_bytes":
        raise ValueError("source subword offset unit must be explicit")
    offsets = {0: 0}
    position = 0
    for i, char in enumerate(text, 1):
        position += len(char.encode('utf-8'))
        offsets[position] = i
    try:
        return [SourceSpan(offsets[s.begin], offsets[s.end]) for s in spans]
    except KeyError as exc:
        raise ValueError("subword offset splits UTF-8 character") from exc


def align_slr_spacy_to_subwords(
    *, source_text: str, source_sha256: str, parser_contract_sha256: str,
    tokenizer_sha256: str, subword_spans: Sequence[SourceSpan],
    slr_tokens: Sequence[NumericSpacyToken], offset_unit: str,
    visibility: str,
    prefix_visible_end_char: int | None = None,
    parser_input_sha256: str | None = None,
) -> AlignedSpacyDoc:
    if visibility not in ("full_context", "prefix"):
        raise ValueError("unqualified parser visibility")
    if not isinstance(source_text, str) or not all(_valid_sha(s) for s in (
        source_sha256, parser_contract_sha256, tokenizer_sha256)):
        raise ValueError("source/parser/tokenizer SHA256 receipts are mandatory")
    if hashlib.sha256(source_text.encode('utf-8')).hexdigest() != source_sha256:
        raise ValueError("SLR parser text does not match original source bytes")
    # A full-context parser is a valid FULL-CONTEXT training teacher.
    # A causal-prefix parser must prove its input was only the prefix,
    # not a complete story whose tags happen to occupy early positions.
    if visibility == "prefix":
        if (type(prefix_visible_end_char) is not int or
                not 0 < prefix_visible_end_char <= len(source_text) or
                not _valid_sha(parser_input_sha256) or
                hashlib.sha256(
                    source_text[:prefix_visible_end_char].encode('utf-8')
                ).hexdigest() != parser_input_sha256):
            raise ValueError("prefix parser visibility requires prefix-only input SHA")
    elif (prefix_visible_end_char is not None or
          (parser_input_sha256 is not None and
           parser_input_sha256 != source_sha256)):
        raise ValueError("full-context parser scope source mismatch")
    spans = _as_character_spans(source_text, subword_spans, offset_unit)
    if visibility == "prefix" and (any(s.end > prefix_visible_end_char
                                      for s in spans) or
                                    any(t.span.end > prefix_visible_end_char
                                        for t in slr_tokens)):
        raise ValueError("future-derived token or parser annotation in prefix")
    if not spans or not slr_tokens:
        raise ValueError("empty source tokens or SLR numeric observations")
    for seq in (spans, [t.span for t in slr_tokens]):
        if any(s.begin < 0 or s.end > len(source_text) for s in seq) or any(
                a.end > b.begin for a, b in zip(seq, seq[1:])):
            raise ValueError("overlapping/out-of-range or unordered original spans")
    token_ids = tuple(t.token_id for t in slr_tokens)
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("duplicate SLR occurrence token ID")
    id_positions = {tok: i for i, tok in enumerate(token_ids)}
    heads = []
    for t in slr_tokens:
        if t.head_token_id is not None and t.head_token_id not in id_positions:
            raise ValueError("cross-sentence or nonexistent original dependency head")
        heads.append(id_positions.get(t.head_token_id, -100))
    weight = torch.zeros((len(slr_tokens), len(spans)), dtype=torch.float64)
    for i, token in enumerate(slr_tokens):
        for j, sub in enumerate(spans):
            overlap = max(0, min(token.span.end, sub.end) - max(token.span.begin, sub.begin))
            if overlap:
                weight[i, j] = overlap
        if weight[i].sum() == 0:
            raise ValueError("SLR token has no attested SentencePiece source overlap")
        # Preserve one word target even if it contains multiple subwords.
        weight[i] /= weight[i].sum()
    return AlignedSpacyDoc(source_sha256,parser_contract_sha256,
                           tokenizer_sha256,visibility,weight,token_ids,
                           tuple(heads),tuple(t.span for t in slr_tokens))



def word_feature_inputs(tokens: Sequence[NumericSpacyToken],
                        vocabs: Mapping[str, CompactSymbolVocabulary],
                        device: str | torch.device = "cpu") -> dict[str,torch.Tensor]:
    """Explicit SLR BIGINT -> train-fitted compact class mapping per word."""
    return {field:torch.tensor([vocabs[source].input_id(getattr(tok,source))
                               for tok in tokens],dtype=torch.long,device=device)
            for field,source in (("pos","pos_symbol_id"),
                                 ("dep","dependency_id"),
                                 ("morph","morph_set_id"))}


class SpacySyntacticProbe(nn.Module):
    """Trainable POS, DEP and morphology-set category and head-arc probe.

    SLR roots use their own source token index, not an invented ROOT token.
    """
    def __init__(self, d_model: int, pos_classes: int,
                 dep_classes: int, morph_set_classes: int):
        super().__init__()
        if any(type(v) is not int or v < 2 for v in
               (d_model,pos_classes,dep_classes,morph_set_classes)):
            raise ValueError("nontrivial native model and compact syntax classes needed")
        self.pos = nn.Linear(d_model,pos_classes)
        self.dep = nn.Linear(d_model,dep_classes)
        self.morph = nn.Linear(d_model,morph_set_classes)
        self.arc_q = nn.Linear(d_model,d_model,bias=False)
        self.arc_k = nn.Linear(d_model,d_model,bias=False)
        self.scale = d_model ** -.5

    def forward(self, word_hidden: torch.Tensor) -> dict[str,torch.Tensor]:
        if word_hidden.ndim != 2 or word_hidden.shape[0] < 1:
            raise ValueError("expected [original SLR word,hidden width]")
        return {"pos":self.pos(word_hidden),
                "dep":self.dep(word_hidden),
                "morph":self.morph(word_hidden),
                "head":(self.arc_q(word_hidden) @ self.arc_k(word_hidden).T)*self.scale}

    @staticmethod
    def loss(logits: Mapping[str,torch.Tensor], targets: Mapping[str,torch.Tensor]) -> torch.Tensor:
        parts = []
        for field in ("pos","dep","morph","head"):
            if field not in logits or field not in targets:
                raise ValueError("missing source target or predicted field: "+field)
            tar = targets[field]
            scores = logits[field]
            if tar.ndim != 1 or tar.shape[0] != scores.shape[0]:
                raise ValueError("source aligned target length differs")
            if tar.dtype != torch.long:
                raise ValueError("grammar target IDs require torch.long")
            valid = tar != -100
            if not valid.any():
                continue
            if (tar[valid] < 1).any() and field != "head":
                raise ValueError("class zero is missing, not a grammatical label")
            if (tar[valid] < 0).any() or (tar[valid] >= scores.shape[1]).any():
                raise ValueError("source labels exceed compact train-fitted output schema")
            parts.append(F.cross_entropy(scores[valid],tar[valid]))
        if not parts:
            raise ValueError("no observed grammatical labels to supervise")
        return torch.stack(parts).mean()


class ZeroGatedParserEmbedding(nn.Module):
    """Opt-in side input, without modifying original Leech checkpoint keys.

    Gate=0 preserves original embedding values and logits exactly.
    """
    def __init__(self,d_model:int,pos_classes:int,dep_classes:int,morph_classes:int):
        super().__init__()
        self.pos = nn.Embedding(pos_classes,d_model,padding_idx=0)
        self.dep = nn.Embedding(dep_classes,d_model,padding_idx=0)
        self.morph = nn.Embedding(morph_classes,d_model,padding_idx=0)
        self.gate = nn.Parameter(torch.zeros(()))

    def inject(self,base:torch.Tensor, aligned_feature_ids:Mapping[str,torch.Tensor]) -> torch.Tensor:
        if base.ndim != 3:
            raise ValueError("original Leech embedding must have B,T,D")
        delta = None
        for field in ("pos","dep","morph"):
            ids = aligned_feature_ids[field]
            if ids.shape != base.shape[:2] or ids.dtype != torch.long or (
                (ids < 0).any() or (ids >= getattr(self,field).num_embeddings).any()
            ):
                raise ValueError("invalid compact parser input shape/class for "+field)
            v = getattr(self,field)(ids)
            delta = v if delta is None else delta+v
        return base + self.gate * delta

    def inject_word_aligned(self, base:torch.Tensor,
                           aligned:AlignedSpacyDoc,
                           word_feature_ids:Mapping[str,torch.Tensor]) -> torch.Tensor:
        """Sparse SLR word -> source subword mixing, no winner-takes-all tag.

        Weights are normalized per source SLR word for word-hidden readout.
        The inverse observation direction is normalized per subword. A
        cross-word token can therefore read both word labels without
        silently picking a made-up single POS class. No label for gaps.
        """
        if base.ndim!=3 or base.shape[0]!=1 or base.shape[1]!=aligned.weights.shape[1]:
            raise ValueError("word-aligned source currently requires one exact original document")
        w=aligned.weights.to(device=base.device,dtype=base.dtype)
        denom=w.sum(dim=0).clamp_min(1e-12).unsqueeze(-1)
        delta=base.new_zeros(base.shape[1],base.shape[2])
        for field in ("pos","dep","morph"):
            ids=word_feature_ids[field]
            table=getattr(self,field)
            if (ids.ndim!=1 or len(ids)!=w.shape[0] or ids.dtype!=torch.long or
                (ids<0).any() or (ids>=table.num_embeddings).any()):
                raise ValueError("original word compact symbol IDs invalid: "+field)
            emb=table(ids.to(device=base.device)).to(dtype=base.dtype)
            delta=delta+(w.T @ emb)/denom
        return base+self.gate*delta.unsqueeze(0)

    def run_original_word_aligned(self,backbone:nn.Module,idx:torch.Tensor,
                                  aligned:AlignedSpacyDoc,
                                  word_feature_ids:Mapping[str,torch.Tensor],**kwargs):
        if not hasattr(backbone,"tok_emb") or not isinstance(backbone.tok_emb,nn.Embedding):
            raise ValueError("requires original Leech token embedding module")
        if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
            raise ValueError("parser side-input KV cache parity not yet accepted")
        def hook(module,input,output):
            return self.inject_word_aligned(output,aligned,word_feature_ids)
        handle=backbone.tok_emb.register_forward_hook(hook)
        try:
            return backbone(idx,**kwargs)
        finally:
            handle.remove()

    def run_original(self,backbone:nn.Module,idx:torch.Tensor,
                     aligned_feature_ids:Mapping[str,torch.Tensor],**kwargs):
        if not hasattr(backbone,"tok_emb") or not isinstance(backbone.tok_emb,nn.Embedding):
            raise ValueError("requires original Leech token embedding module")
        if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
            raise ValueError("parser side-input KV cache parity not yet accepted")
        # Source original tok_emb hooked only within one synchronous call.
        def hook(module,input,output):
            return self.inject(output,aligned_feature_ids)
        handle = backbone.tok_emb.register_forward_hook(hook)
        try:
            return backbone(idx,**kwargs)
        finally:
            handle.remove()
