"""CPU token counting for unannotated PT text using GRIM KTMG v9 vocabularies.

Mirrors raw UniByte: horizontal spacing, LF, fixed numbers, exact pieces,
then float32 Viterbi. Authored atom delimiters are deliberately rejected.
No CUDA, trainer, server, or vocabulary mutation is involved.
"""
from __future__ import annotations

import hashlib
import math
import re
import struct
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vocab_playground import Vocab, VocabPiece, build_trie, viterbi_segment

MARKER = b"\xe2\x96\x81"
ATOM_OPEN = re.compile(r"<(?:INT|FLOAT|STRING|BOOL|ENTITY|TOOL)>")


def numeric_end(text: bytes, pos: int) -> int:
    """NumericTokens.cu grammar, including grouped thousands."""
    i = pos
    if i < len(text) and text[i] in b"+-":
        i += 1
    start = i
    while i < len(text) and 48 <= text[i] <= 57:
        i += 1
    digits = i > start
    if digits and i - start <= 3:
        while i < len(text) and text[i] == 44:
            j = i + 1
            while j < len(text) and 48 <= text[j] <= 57:
                j += 1
            if j - i != 4:
                break
            i = j
    if i + 1 < len(text) and text[i] == 46 and 48 <= text[i + 1] <= 57:
        i += 1
        while i < len(text) and 48 <= text[i] <= 57:
            i += 1
        digits = True
    if not digits:
        return pos
    if i < len(text) and text[i] in b"eE":
        j = i + 1
        if j < len(text) and text[j] in b"+-":
            j += 1
        start = j
        while j < len(text) and 48 <= text[j] <= 57:
            j += 1
        if j > start:
            i = j
    return i


def numeric_count(text: bytes) -> int:
    count = i = 0
    while i < len(text):
        end = i + 1
        if 48 <= text[i] <= 57:
            while end < len(text) and text[end] == text[i]:
                end += 1
        count += (end - i + 3) // 4
        i = end
    return count


class PTTokenizer:
    def __init__(self, path: Path):
        data = path.read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        offset = 0

        def read(fmt):
            nonlocal offset
            size = struct.calcsize(fmt)
            if offset + size > len(data):
                raise ValueError("Truncated vocab.bin")
            value = struct.unpack_from(fmt, data, offset)
            offset += size
            return value

        magic, version, checksum, records, max_len, flags, space = read("<4sHIII3sI")
        if magic != b"KTMG" or version != 9 or max_len != 32 or not flags[0] & 1:
            raise ValueError("Expected canonical-newline KTMG v9 vocab.bin (max piece length 32)")
        special = {0: b"<unk>", 1: b"<pad>", 2: b"<s>", 3: b"</s>"}
        seen_special, seen_text = set(), set()
        regular, exact = [], []
        learned = 0
        for _ in range(records):
            length, = read("<I")
            if not 0 < length <= 32:
                raise ValueError("Invalid vocabulary piece length")
            encoded, = read(f"<{length}s")
            score, token_id, piece_flags = read("<fiB")
            if not math.isfinite(score) or piece_flags & ~1:
                raise ValueError("Invalid vocabulary score/flags")
            if token_id in special:
                if encoded != special[token_id] or token_id in seen_special or piece_flags:
                    raise ValueError("Invalid special vocabulary record")
                seen_special.add(token_id)
                continue
            if token_id != 319 + learned or encoded in seen_text:
                raise ValueError("Invalid learned vocabulary ID or duplicate piece")
            learned += 1
            seen_text.add(encoded)
            piece = VocabPiece(token_id, encoded.decode("utf-8"), encoded, score)
            (exact if piece_flags else regular).append(piece)
        if offset != len(data) or seen_special != set(special) or space != 319 + learned:
            raise ValueError("Vocabulary record count/token-space mismatch")
        self.vocab = Vocab(tuple(regular), {p.token_id: p for p in regular}, space)
        self.trie = build_trie(self.vocab)
        self.exact = build_trie(Vocab(tuple(exact), {}, space))

    @lru_cache(maxsize=8192)
    def _text_count(self, text: bytes) -> int:
        count = start = pos = 0
        while pos < len(text):
            node, end, best = 0, pos, pos
            while end < len(text):
                node = self.exact[node].children.get(text[end])
                if node is None:
                    break
                end += 1
                if self.exact[node].token_id is not None:
                    boundary = (end == len(text) or text[end:].startswith(MARKER)
                                or (text[end] < 128 and not
                                    (chr(text[end]).isalnum() or text[end] == 95)))
                    if not text[pos:].startswith(MARKER) or boundary:
                        best = end
            if best > pos:
                count += len(viterbi_segment(text[start:pos], self.vocab, self.trie)) + 1
                start = pos = best
            else:
                pos += 1
        return count + len(viterbi_segment(text[start:], self.vocab, self.trie))

    def count(self, text: str) -> int:
        if ATOM_OPEN.search(text):
            raise ValueError("Authored atom delimiter in source; needs explicit review")
        if not text:
            return 0
        normalized = MARKER + text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        normalized = normalized.replace(b" ", MARKER).replace(b"\t", MARKER)
        count = normalized.count(b"\n")
        for line in normalized.split(b"\n"):
            start = pos = 0
            while pos < len(line):
                end = numeric_end(line, pos)
                if end > pos:
                    count += self._text_count(line[start:pos]) + numeric_count(line[pos:end])
                    start = pos = end
                else:
                    pos += 1
            count += self._text_count(line[start:])
        return count
