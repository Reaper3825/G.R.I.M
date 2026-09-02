#!/usr/bin/env python3
"""
Decode token IDs from training_data.grmt using the current KTMG vocab.bin.

Token ID Layout (current UniByte tokenizer):
  [0-3]       = Special tokens: <unk>=0, <pad>=1, <s>=2, </s>=3
  [4-259]     = Byte fallback (byte value = token_id - 4)
    [260-305]   = Fixed numeric tokens
    [306-315]   = Typed atom opening/closing boundaries
    [316+]      = Unigram vocabulary pieces (from vocab.bin)

Current vocab.bin format is KTMG v6. The saved record count is the number of
serialized records (4 special-token metadata records + learned unigram pieces),
not the full token-space size. The token-space size is stored separately in the
header and must equal special + bytes + numeric + atoms + learned pieces.

Current training_data.grmt format is GRMT v23. Rows persist atom side channels,
per-sequence AtomTable and sequence-local atom data, row-level Goal metadata,
top-level ConceptBlock known/unknown spans, opaque slot/transition lowering
tables, and variable-arity transition invocations. This script reads the full
current row layout. Typed atom boundaries decode literally; persisted AtomTable
entries remain diagnostic side-channel data.

Usage:
    python decode_token_ids.py
    python decode_token_ids.py --seq 0 5          # Sequences 0 through 4
    python decode_token_ids.py --ids 277 512 36   # Decode specific token IDs
    python decode_token_ids.py --search hello     # Find sequences containing text
    python decode_token_ids.py --raw              # Show raw token IDs alongside text
    python decode_token_ids.py --stats            # Vocabulary and data statistics
"""

import argparse
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# ── Paths (repo-relative defaults) ───────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
VOCAB_BIN = REPO_ROOT / "resources/models/GRIM-text/training/data/vocab.bin"
GRMT_FILE = REPO_ROOT / "resources/models/GRIM-text/training/data/training_data.grmt"

# ── Token layout constants ────────────────────────────────────────────────────

NUM_SPECIAL_TOKENS = 4
SPECIAL_TOKEN_OFFSET = 0
BYTE_TOKEN_OFFSET = NUM_SPECIAL_TOKENS  # 4
BYTE_VOCAB_SIZE = 256
NUMERIC_TOKEN_OFFSET = BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE  # 260
NUMERIC_TOKEN_TEXT = (
    [str(digit) for digit in range(10)]
    + [str(digit) * 2 for digit in range(10)]
    + [str(digit) * 3 for digit in range(10)]
    + [str(digit) * 4 for digit in range(10)]
    + [".", ",", "-", "+", "e", "E"]
)
NUMERIC_TOKEN_END = NUMERIC_TOKEN_OFFSET + len(NUMERIC_TOKEN_TEXT)  # 306
ATOM_TOKEN_OFFSET = NUMERIC_TOKEN_END

SPECIAL_NAMES = {0: "<unk>", 1: "<pad>", 2: "<s>", 3: "</s>"}

ATOM_TYPE_LABELS = {
    0: "INT",
    1: "FLOAT",
    2: "STRING",
    3: "BOOL",
    4: "ENTITY",
}
NUM_ATOM_TYPES = len(ATOM_TYPE_LABELS)

ATOM_TOKEN_END = ATOM_TOKEN_OFFSET + 2 * NUM_ATOM_TYPES  # 316
UNIGRAM_TOKEN_START = ATOM_TOKEN_END
KTMG_VOCAB_VERSION = 6
KTMG_MAX_PIECE_LENGTH = 32
GRMT_MAGIC = 0x474D5254
GRMT_FORMAT_VERSION = 23
ATOM_ENTRY_NONE = 0xFFFFFFFF
PAD_TOKEN_ID = 1


@dataclass(frozen=True)
class GrmtSequenceRecord:
    index: int
    token_ids: list[int]
    token_numeric_values: list[float]
    token_atom_mask: list[int]
    token_atom_flags: list[int]
    atom_texts: list[str]
    execution_active: bool
    prompt_end_pos: int
    prompt_length: int
    target_state_token_ids: list[int] | None
    target_state_span: tuple[int, int] | None
    criteria_span: tuple[int, int] | None
    success_criteria: list[
        tuple[list[int], tuple[int, int], list[int], tuple[int, int]]
    ]
    constraints: list[tuple[list[int], tuple[int, int]]]
    knowns: list[tuple[list[int], tuple[int, int]]]
    unknowns: list[tuple[list[int], tuple[int, int]]]


# ── Binary helpers ───────────────────────────────────────────────────────────

def read_exact(f, size: int, source: str) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise EOFError(f"truncated read from {source}: expected {size} bytes, got {len(data)}")
    return data


def read_u8(f, source: str) -> int:
    return struct.unpack("<B", read_exact(f, 1, source))[0]


def read_u16(f, source: str) -> int:
    return struct.unpack("<H", read_exact(f, 2, source))[0]


def read_u32(f, source: str) -> int:
    return struct.unpack("<I", read_exact(f, 4, source))[0]


def read_i32(f, source: str) -> int:
    return struct.unpack("<i", read_exact(f, 4, source))[0]


def read_f32(f, source: str) -> float:
    return struct.unpack("<f", read_exact(f, 4, source))[0]


def read_i32_array(f, count: int, source: str) -> list[int]:
    if count == 0:
        return []
    return list(struct.unpack(f"<{count}i", read_exact(f, 4 * count, source)))


def read_u32_array(f, count: int, source: str) -> list[int]:
    if count == 0:
        return []
    return list(struct.unpack(f"<{count}I", read_exact(f, 4 * count, source)))


def read_f32_array(f, count: int, source: str) -> list[float]:
    if count == 0:
        return []
    return list(struct.unpack(f"<{count}f", read_exact(f, 4 * count, source)))


def skip_exact(f, size: int, source: str):
    read_exact(f, size, source)


def read_atom_table_texts(f, source: str) -> dict[int, str]:
    has_atom_table = read_u8(f, source)
    if has_atom_table > 1:
        raise ValueError(f"Invalid AtomTable presence flag in {source}: {has_atom_table}")
    if has_atom_table == 0:
        return {}
    if read_exact(f, 4, source) != b"ATMB":
        raise ValueError(f"Bad AtomTable magic in {source}")

    entry_count = read_u32(f, source)
    pool_size = read_u32(f, source)
    raw_text_refs: dict[int, tuple[int, int]] = {}
    for _ in range(entry_count):
        skip_exact(f, 8, source)  # hash
        entry_id = read_u32(f, source)
        skip_exact(f, 8, source)  # type, category, origin, padding
        raw_offset = read_u32(f, source)
        raw_length = read_u32(f, source)
        skip_exact(f, 36, source)  # confidence through reserved_zero
        has_arg_number = read_u8(f, source)
        if has_arg_number:
            skip_exact(f, 82, source)  # fixed AtomNumber fields
            digit_count = read_u32(f, source)
            skip_exact(f, 15 * digit_count, source)
        raw_text_refs[entry_id] = (raw_offset, raw_length)

    skip_exact(f, 8 * entry_count, source)  # exact float payloads
    skip_exact(f, 8 * entry_count, source)  # exact integer payloads
    skip_exact(f, entry_count, source)      # numeric payload kinds
    pool = read_exact(f, pool_size, source)

    texts: dict[int, str] = {}
    for entry_id, (offset, length) in raw_text_refs.items():
        end = offset + length
        if end > len(pool):
            raise ValueError(
                f"AtomTable raw text span is out of range in {source}: "
                f"entry_id={entry_id} span=[{offset},{end}) pool_size={len(pool)}"
            )
        texts[entry_id] = pool[offset:end].decode("utf-8", errors="strict")
    return texts


def read_sequence_local_atom_table(f, source: str) -> list[list[str]]:
    has_table = read_u8(f, source)
    if has_table not in (0, 1):
        raise ValueError(
            f"Invalid SequenceLocalAtomTable presence flag in {source}: {has_table}"
        )
    if has_table == 0:
        return [[] for _ in range(NUM_ATOM_TYPES)]
    if read_exact(f, 4, source) != b"SLAT":
        raise ValueError(f"Bad SequenceLocalAtomTable magic in {source}")
    type_count = read_u32(f, source)
    if type_count != NUM_ATOM_TYPES:
        raise ValueError(
            f"SequenceLocalAtomTable type_count={type_count} in {source}; "
            f"expected {NUM_ATOM_TYPES}"
        )
    values_by_type: list[list[str]] = []
    for _ in range(type_count):
        value_count = read_u32(f, source)
        values = []
        for _ in range(value_count):
            length = read_u32(f, source)
            values.append(read_exact(f, length, source).decode("utf-8", errors="strict"))
        values_by_type.append(values)
    return values_by_type


def read_concept_block_entry_spans(
    f, source: str
) -> list[tuple[list[int], tuple[int, int]]]:
    entries = []
    for _ in range(read_u32(f, source)):
        token_ids = read_i32_array(f, read_u32(f, source), source)
        span = (read_i32(f, source), read_i32(f, source))
        entries.append((token_ids, span))
    return entries


# ── Token layout helpers ─────────────────────────────────────────────────────

def is_special_token_id(token_id: int) -> bool:
    return SPECIAL_TOKEN_OFFSET <= token_id < SPECIAL_TOKEN_OFFSET + NUM_SPECIAL_TOKENS


def is_byte_token_id(token_id: int) -> bool:
    return BYTE_TOKEN_OFFSET <= token_id < NUMERIC_TOKEN_OFFSET


def is_numeric_token_id(token_id: int) -> bool:
    return NUMERIC_TOKEN_OFFSET <= token_id < NUMERIC_TOKEN_END


def is_atom_token_id(token_id: int) -> bool:
    return ATOM_TOKEN_OFFSET <= token_id < ATOM_TOKEN_END


def is_atom_open_token_id(token_id: int) -> bool:
    return ATOM_TOKEN_OFFSET <= token_id < ATOM_TOKEN_OFFSET + NUM_ATOM_TYPES


def atom_type_label_for_token_id(token_id: int) -> str:
    if not is_atom_token_id(token_id):
        raise ValueError(f"token_id={token_id} is outside the atom token range")

    atom_index = token_id - ATOM_TOKEN_OFFSET
    is_close = atom_index >= NUM_ATOM_TYPES
    type_index = atom_index - NUM_ATOM_TYPES if is_close else atom_index
    label = ATOM_TYPE_LABELS.get(type_index)
    if label is None:
        raise ValueError(f"token_id={token_id} does not map to a live atom type")
    return f"</{label}>" if is_close else f"<{label}>"


def format_numeric_value(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6 and abs(value) < 1e15:
        return str(int(rounded))
    return f"{value:.9g}"


def format_hex_byte(byte_value: int) -> str:
    return f"0x{byte_value:02X}"


def summarize_byte_run(byte_values: bytes, max_preview_bytes: int = 16) -> str:
    preview = byte_values[:max_preview_bytes]
    summary = " ".join(format_hex_byte(byte_value) for byte_value in preview)
    if len(byte_values) > max_preview_bytes:
        summary += " ..."
    return summary


def format_token_prefix(token_ids: list[int], token_count: int) -> str:
    return "[" + ", ".join(str(token_id) for token_id in token_ids[:token_count]) + "]"


def append_validated_utf8_byte_run(
    output_parts: list[str],
    byte_values: bytes,
    token_ids: list[int],
    token_start_index: int,
):
    if not byte_values:
        return

    try:
        output_parts.append(byte_values.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as exc:
        bad_byte = byte_values[exc.start]
        bad_token_index = token_start_index + exc.start
        raise ValueError(
            "decode_sequence: byte-token run starting at token index="
            f"{token_start_index} after prior_token_count={token_start_index} "
            f"prior_token_ids={format_token_prefix(token_ids, token_start_index)} "
            f"produced invalid UTF-8 at run byte offset={exc.start} "
            f"(token_id={token_ids[bad_token_index]}, byte={format_hex_byte(bad_byte)}, "
            f"run_bytes=[{summarize_byte_run(byte_values)}])"
        ) from exc


# ── vocab.bin loader (KTMG v4 format) ────────────────────────────────────────

def load_vocab_bin(path: Path) -> dict[int, str]:
    # sourcery skip: extract-method
    """
    Load vocab.bin and build a complete token_id -> text mapping.

    Returns dict mapping every known token ID to its string representation.
    """
    if not path.exists():
        raise FileNotFoundError(f"vocab.bin not found: {path}")

    id_to_text = dict(SPECIAL_NAMES)

    # Byte fallback tokens mirror UniByte::decode: token_id - BYTE_TOKEN_OFFSET
    # yields the raw byte value.
    for byte_value in range(BYTE_VOCAB_SIZE):
        token_id = BYTE_TOKEN_OFFSET + byte_value
        id_to_text[token_id] = bytes([byte_value]).decode("latin-1")

    for numeric_index, text in enumerate(NUMERIC_TOKEN_TEXT):
        id_to_text[NUMERIC_TOKEN_OFFSET + numeric_index] = text

    # Typed atom opening/closing boundary tokens come from TokenLayout.hpp.
    for atom_index in range(2 * NUM_ATOM_TYPES):
        token_id = ATOM_TOKEN_OFFSET + atom_index
        id_to_text[token_id] = atom_type_label_for_token_id(token_id)

    # Special-token metadata + learned unigram pieces from vocab.bin.
    with open(path, "rb") as f:
        source = str(path)
        magic = read_exact(f, 4, source)
        if magic != b"KTMG":
            raise ValueError(f"Bad vocab.bin magic: {magic!r}; expected KTMG")

        version = read_u16(f, source)
        if version != KTMG_VOCAB_VERSION:
            raise ValueError(
                f"Unsupported vocab.bin version {version}; expected KTMG v{KTMG_VOCAB_VERSION}. "
                f"Retrain/pull the current tokenizer artifact."
            )

        _checksum = read_u32(f, source)
        serialized_record_count = read_u32(f, source)
        max_length = read_u32(f, source)
        _flags = read_exact(f, 3, source)
        token_space_size = read_u32(f, source)

        if max_length != KTMG_MAX_PIECE_LENGTH:
            raise ValueError(f"Unexpected KTMG max_piece_length={max_length}; expected {KTMG_MAX_PIECE_LENGTH}")
        if serialized_record_count < NUM_SPECIAL_TOKENS:
            raise ValueError(
                f"KTMG serialized_record_count={serialized_record_count} is smaller than "
                f"NUM_SPECIAL_TOKENS={NUM_SPECIAL_TOKENS}"
            )

        seen_special_ids = set()
        learned_piece_count = 0
        for record_idx in range(serialized_record_count):
            piece_len = read_u32(f, source)
            if piece_len > max_length:
                raise ValueError(
                    f"KTMG record {record_idx} has piece_len={piece_len}, max_length={max_length}"
                )
            text = read_exact(f, piece_len, source).decode("utf-8", errors="strict")
            _score = read_f32(f, source)
            token_id = read_i32(f, source)

            if token_id in SPECIAL_NAMES:
                expected_text = SPECIAL_NAMES[token_id]
                if text != expected_text:
                    raise ValueError(
                        f"KTMG special record mismatch at record {record_idx}: "
                        f"token_id={token_id} text={text!r} expected={expected_text!r}"
                    )
                if token_id in seen_special_ids:
                    raise ValueError(f"KTMG duplicate special token record for token_id={token_id}")
                seen_special_ids.add(token_id)
                continue

            expected_id = UNIGRAM_TOKEN_START + learned_piece_count
            if token_id != expected_id:
                raise ValueError(
                    f"KTMG learned-piece token_id mismatch at record {record_idx}: "
                    f"stored={token_id} expected={expected_id}. Do not patch the vocab header; retrain/pull it."
                )

            id_to_text[token_id] = text
            learned_piece_count += 1

        if seen_special_ids != set(SPECIAL_NAMES):
            raise ValueError(
                f"KTMG special metadata set mismatch: seen={sorted(seen_special_ids)} "
                f"expected={sorted(SPECIAL_NAMES)}"
            )

        expected_token_space_size = UNIGRAM_TOKEN_START + learned_piece_count
        if token_space_size != expected_token_space_size:
            raise ValueError(
                f"KTMG token-space size mismatch: header={token_space_size} "
                f"computed={expected_token_space_size} "
                f"({NUM_SPECIAL_TOKENS} special + {BYTE_VOCAB_SIZE} bytes + "
                f"{len(NUMERIC_TOKEN_TEXT)} numeric + {2 * NUM_ATOM_TYPES} atoms + "
                f"{learned_piece_count} unigrams)"
            )

    return id_to_text


# ── training_data.grmt loader ────────────────────────────────────────────────

def read_grmt_header(path: Path) -> dict:
    """Read and validate the 16-byte current GRMT header."""
    with open(path, "rb") as f:
        magic, version, num_sequences, vocab_size = struct.unpack("<IIII", read_exact(f, 16, str(path)))

    if magic != GRMT_MAGIC:
        raise ValueError(f"Bad GRMT magic in {path}: actual=0x{magic:08X} expected=0x{GRMT_MAGIC:08X}")
    if version != GRMT_FORMAT_VERSION:
        raise ValueError(f"Unsupported GRMT version {version}; expected {GRMT_FORMAT_VERSION}")
    if num_sequences == 0:
        raise ValueError(f"GRMT header reports num_sequences=0: {path}")
    if vocab_size == 0:
        raise ValueError(f"GRMT header reports vocab_size=0: {path}")

    return {
        "magic": magic,
        "magic_hex": hex(magic),
        "version": version,
        "num_sequences": num_sequences,
        "vocab_size": vocab_size,
    }


def read_goal_metadata(
    f,
    source: str,
) -> tuple[
    list[int] | None,
    tuple[int, int] | None,
    tuple[int, int] | None,
    list[tuple[list[int], tuple[int, int], list[int], tuple[int, int]]],
    list[tuple[list[int], tuple[int, int]]],
]:
    has_goal = read_u8(f, source)
    if has_goal not in (0, 1):
        raise ValueError(f"invalid GRMT goal flag in {source}: {has_goal}")
    if has_goal == 0:
        return None, None, None, [], []

    has_target_state = read_u8(f, source)
    if has_target_state not in (0, 1):
        raise ValueError(
            f"invalid GRMT target_state flag in {source}: {has_target_state}"
        )
    target_state_token_ids = None
    target_state_span = None
    if has_target_state:
        target_state_token_ids = read_i32_array(f, read_u32(f, source), source)
        target_state_span = (read_i32(f, source), read_i32(f, source))

    has_success_criteria = read_u8(f, source)
    if has_success_criteria not in (0, 1):
        raise ValueError(
            f"invalid GRMT success_criteria flag in {source}: {has_success_criteria}"
        )
    success_criteria = []
    criteria_span = None
    if has_success_criteria:
        criteria_span = (read_i32(f, source), read_i32(f, source))
        entry_count = read_u32(f, source)
        for _ in range(entry_count):
            criterion = read_i32_array(f, read_u32(f, source), source)
            criterion_span = (read_i32(f, source), read_i32(f, source))
            evidence = read_i32_array(f, read_u32(f, source), source)
            evidence_span = (read_i32(f, source), read_i32(f, source))
            success_criteria.append(
                (criterion, criterion_span, evidence, evidence_span)
            )

    has_constraints = read_u8(f, source)
    if has_constraints not in (0, 1):
        raise ValueError(
            f"invalid GRMT constraints flag in {source}: {has_constraints}"
        )
    constraints = []
    if has_constraints:
        entry_count = read_u32(f, source)
        for _ in range(entry_count):
            token_ids = read_i32_array(f, read_u32(f, source), source)
            span = (read_i32(f, source), read_i32(f, source))
            constraints.append((token_ids, span))

    return (
        target_state_token_ids,
        target_state_span,
        criteria_span,
        success_criteria,
        constraints,
    )


def iter_grmt_sequences(path: Path):
    """Yield decoded GRMT rows using the current persisted row layout.

    GRMT v23 per-sequence layout (must read ALL fields to stay in sync):
      uint32         seq_len
      int32[seq_len] token_ids
      int32[seq_len] targets
      float[seq_len] numeric_values
      uint8[seq_len] atom_mask
      uint32[seq_len] atom_flags
      uint32[seq_len] atom_entry_ids
      uint8 has_atom_table + optional AtomTable payload
      uint32[seq_len] token_local_atom_indices
      uint8 has_local_atom_table + optional SequenceLocalAtomTable payload
      uint8 execution_active, int8 execution_gate_target
      int32 prompt_end_pos, int32 prompt_length
      uint8 has_goal + optional target-state, criterion/evidence, and
          per-constraint token spans
      uint32 known_count + known token IDs/spans
      uint32 unknown_count + unknown token IDs/spans
      int32[seq_len] token_exec_slots
      uint32 compiled_slot_binding_count, then {uint64 SlotId, int32 SlotIndex}
      uint32 compiled_transition_binding_count, then
          {uint64 TransitionId, int32 TransitionIndex}
      uint32 compiled_bootstrap_binding_count, then 12 bytes each
      uint32 transition_target_count, then variable-arity TransitionInvocation
    """
    with open(path, "rb") as f:
        source = str(path)
        header = read_grmt_header(path)
        f.seek(16)
        num_sequences = header["num_sequences"]

        for idx in range(num_sequences):
            row_source = f"{source}#seq{idx}"
            seq_len = read_u32(f, row_source)
            if seq_len == 0:
                raise ValueError(f"GRMT sequence length is zero in {row_source}")
            if seq_len > 1_000_000:
                raise ValueError(f"GRMT sequence length is suspiciously large in {row_source}: {seq_len}")

            tokens = read_i32_array(f, seq_len, row_source)

            skip_exact(f, 4 * seq_len, row_source)                # targets (int32)
            token_numeric_values = read_f32_array(f, seq_len, row_source)
            token_atom_mask = list(read_exact(f, seq_len, row_source))
            token_atom_flags = read_u32_array(f, seq_len, row_source)
            atom_entry_ids = read_u32_array(f, seq_len, row_source)
            atom_table_texts = read_atom_table_texts(f, row_source)
            missing_atom_entry_ids = sorted({
                entry_id for entry_id in atom_entry_ids
                if entry_id != ATOM_ENTRY_NONE and entry_id not in atom_table_texts
            })
            if missing_atom_entry_ids:
                raise ValueError(
                    f"GRMT atom_entry_ids are absent from the AtomTable in {row_source}: "
                    f"{missing_atom_entry_ids}"
                )
            atom_texts = [
                "" if entry_id == ATOM_ENTRY_NONE
                else atom_table_texts.get(entry_id, "")
                for entry_id in atom_entry_ids
            ]
            token_local_atom_indices = read_u32_array(f, seq_len, row_source)
            local_atom_values = read_sequence_local_atom_table(f, row_source)

            for token_index in range(seq_len):
                atom_text = atom_texts[token_index]
                token_id = tokens[token_index]
                token_is_atom = is_atom_token_id(token_id)
                token_is_atom_open = is_atom_open_token_id(token_id)

                if atom_text and not token_is_atom_open:
                    raise ValueError(
                        f"GRMT atom text exists outside an opening atom boundary in {row_source} "
                        f"index={token_index} token_id={token_id}"
                    )
                if token_atom_mask[token_index] != 0 and not token_is_atom_open:
                    raise ValueError(
                        f"GRMT token_atom_mask is set outside an opening atom boundary in {row_source} "
                        f"index={token_index} token_id={token_id} mask={token_atom_mask[token_index]}"
                    )
                if token_is_atom_open and token_atom_mask[token_index] != 1:
                    raise ValueError(
                        f"GRMT atom opening boundary does not have token_atom_mask=1 in {row_source} "
                        f"index={token_index} token_id={token_id}"
                    )
                if token_is_atom and not token_is_atom_open and token_atom_mask[token_index] != 0:
                    raise ValueError(
                        f"GRMT atom closing boundary has metadata in {row_source} "
                        f"index={token_index} token_id={token_id}"
                    )
                local_index = token_local_atom_indices[token_index]
                if token_is_atom_open:
                    atom_type = token_id - ATOM_TOKEN_OFFSET
                    if local_index == ATOM_ENTRY_NONE:
                        raise ValueError(
                            f"GRMT atom opening has no local index in {row_source} "
                            f"index={token_index}"
                        )
                    if local_index >= len(local_atom_values[atom_type]):
                        raise ValueError(
                            f"GRMT local atom index is out of range in {row_source}: "
                            f"type={atom_type} local_index={local_index}"
                        )
                elif local_index != ATOM_ENTRY_NONE:
                    raise ValueError(
                        f"GRMT local atom index exists outside an opening boundary "
                        f"in {row_source} index={token_index}"
                    )

            execution_active = (read_u8(f, row_source) != 0)
            skip_exact(f, 1, row_source)                          # execution_gate_target (int8)
            prompt_end_pos = read_i32(f, row_source)
            prompt_length = read_i32(f, row_source)
            (
                target_state_token_ids,
                target_state_span,
                criteria_span,
                success_criteria,
                constraints,
            ) = read_goal_metadata(f, row_source)
            knowns = read_concept_block_entry_spans(f, row_source)
            unknowns = read_concept_block_entry_spans(f, row_source)
            skip_exact(f, 4 * seq_len, row_source)                # token_exec_slots (int32)

            csb_count = read_u32(f, row_source)
            skip_exact(f, 12 * csb_count, row_source)             # CompiledSlotBinding

            ctb_count = read_u32(f, row_source)
            skip_exact(f, 12 * ctb_count, row_source)             # CompiledTransitionBinding

            cbb_count = read_u32(f, row_source)
            skip_exact(f, 12 * cbb_count, row_source)             # CompiledBootstrapBinding

            transition_target_count = read_u32(f, row_source)
            for _ in range(transition_target_count):
                skip_exact(f, 8, row_source)                      # TransitionId
                argument_count = read_u32(f, row_source)
                skip_exact(f, 8 * argument_count, row_source)     # argument SlotIds
                result_count = read_u32(f, row_source)
                skip_exact(f, 8 * result_count, row_source)       # result SlotIds

            yield GrmtSequenceRecord(
                index=idx,
                token_ids=tokens,
                token_numeric_values=token_numeric_values,
                token_atom_mask=token_atom_mask,
                token_atom_flags=token_atom_flags,
                atom_texts=atom_texts,
                execution_active=execution_active,
                prompt_end_pos=prompt_end_pos,
                prompt_length=prompt_length,
                target_state_token_ids=target_state_token_ids,
                target_state_span=target_state_span,
                criteria_span=criteria_span,
                success_criteria=success_criteria,
                constraints=constraints,
                knowns=knowns,
                unknowns=unknowns,
            )


# ── Decode helpers ────────────────────────────────────────────────────────────

def decode_token(tid: int, vocab: dict[int, str]) -> str:
    """Decode a single token ID to its text representation."""
    return vocab.get(tid, f"<UNK:{tid}>")


def decode_atom_token(
    token_index: int,
    token_id: int,
    vocab: dict[int, str],
    atom_texts: list[str] | None = None,
    token_numeric_values: list[float] | None = None,
    token_atom_mask: list[int] | None = None,
) -> str:
    del token_index, atom_texts, token_numeric_values, token_atom_mask
    return decode_token(token_id, vocab)


def decode_sequence(
    tokens: list[int],
    vocab: dict[int, str],
    atom_texts: list[str] | None = None,
    token_numeric_values: list[float] | None = None,
    token_atom_mask: list[int] | None = None,
) -> str:
    """Decode a full sequence of token IDs to readable text.

    Mirrors UniByte::decode semantics:
    - contiguous byte tokens are buffered as a raw byte run and must be valid UTF-8
    - PAD is skipped
        - typed atom opening/closing boundaries render literally; side channels do
            not replace model-visible in-band content
    """
    token_count = len(tokens)
    if atom_texts is not None and len(atom_texts) != token_count:
        raise ValueError(
            f"decode_sequence: atom_texts length={len(atom_texts)} != token_count={token_count}"
        )
    if token_numeric_values is not None and len(token_numeric_values) != token_count:
        raise ValueError(
            f"decode_sequence: token_numeric_values length={len(token_numeric_values)} != token_count={token_count}"
        )
    if token_atom_mask is not None and len(token_atom_mask) != token_count:
        raise ValueError(
            f"decode_sequence: token_atom_mask length={len(token_atom_mask)} != token_count={token_count}"
        )

    output_parts: list[str] = []
    byte_buffer = bytearray()
    pending_byte_run_start = 0

    def flush_bytes():
        nonlocal byte_buffer
        if byte_buffer:
            append_validated_utf8_byte_run(
                output_parts,
                bytes(byte_buffer),
                tokens,
                pending_byte_run_start,
            )
            byte_buffer = bytearray()

    for token_index, tid in enumerate(tokens):
        if is_byte_token_id(tid):
            if not byte_buffer:
                pending_byte_run_start = token_index
            byte_buffer.append(tid - BYTE_TOKEN_OFFSET)
            continue

        flush_bytes()
        if is_special_token_id(tid):
            if tid != PAD_TOKEN_ID:
                output_parts.append(SPECIAL_NAMES[tid])
            continue

        if is_atom_token_id(tid):
            output_parts.append(
                decode_atom_token(
                    token_index,
                    tid,
                    vocab,
                    atom_texts=atom_texts,
                    token_numeric_values=token_numeric_values,
                    token_atom_mask=token_atom_mask,
                )
            )
            continue

        if tid in vocab:
            output_parts.append(vocab[tid])
        else:
            output_parts.append(f"<UNK:{tid}>")

    flush_bytes()
    text = "".join(output_parts)
    text = text.replace("\u2581", " ")  # ▁ (Unigram space marker) → space
    return text


def token_type_label(tid: int) -> str:
    """Return which region a token ID belongs to."""
    if is_special_token_id(tid):
        return "SPECIAL"
    if is_byte_token_id(tid):
        return "BYTE"
    if is_numeric_token_id(tid):
        return "NUMERIC"
    if is_atom_token_id(tid):
        return "ATOM"
    if tid >= UNIGRAM_TOKEN_START:
        return "UNIGRAM"
    return "INVALID"


def validate_grmt_vocab_pair(header: dict, vocab: dict[int, str], grmt: Path):
    """Fail loudly if the GRMT header and loaded vocab disagree on token-space size."""
    if header["vocab_size"] != len(vocab):
        raise ValueError(
            f"GRMT/vocab token-space mismatch for {grmt}: "
            f"grmt_header={header['vocab_size']} loaded_vocab={len(vocab)}. "
            f"Pull/rebuild vocab.bin and training_data.grmt as one artifact pair."
        )


def parse_cli_token_ids(raw_values: list[str]) -> list[int]:
    """Parse --ids values, accepting either whitespace- or comma-separated integers."""
    parsed_ids: list[int] = []

    for raw_value in raw_values:
        for piece in raw_value.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                parsed_ids.append(int(piece))
            except ValueError as exc:
                raise ValueError(f"Invalid token ID {piece!r} in --ids; expected integers") from exc

    if not parsed_ids:
        raise ValueError("--ids requires at least one integer token ID")

    return parsed_ids


# ── CLI actions ───────────────────────────────────────────────────────────────

def cmd_decode_ids(args, vocab: dict[int, str]):
    """Decode a list of token IDs from the command line."""
    for tid in args.ids:
        text = decode_token(tid, vocab)
        region = token_type_label(tid)
        print(f"  {tid:>6d}  [{region:>7s}]  {text!r}")

    print("\nDecoded sequence:")
    print(f"  {decode_sequence(args.ids, vocab)!r}")


def wrap_text(text: str, width: int = 100) -> str:
    """Word-wrap text at the given width, preserving existing newlines."""
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split(" ")
        current_line = ""
        for word in words:
            if not word:
                continue
            if current_line and len(current_line) + 1 + len(word) > width:
                lines.append(current_line)
                current_line = word
            elif current_line:
                current_line += f" {word}"
            else:
                current_line = word
        if current_line:
            lines.append(current_line)
    return "\n".join(lines)


def cmd_decode_sequences(args, vocab: dict[int, str]):
    """Decode sequences from training_data.grmt."""
    grmt = Path(args.grmt)
    if not grmt.exists():
        raise FileNotFoundError(f"GRMT file not found: {grmt}")

    header = read_grmt_header(grmt)
    validate_grmt_vocab_pair(header, vocab, grmt)
    total = header["num_sequences"]
    start = args.seq[0] if args.seq else 0
    end = args.seq[1] if args.seq and len(args.seq) > 1 else start + 10
    end = min(end, total)

    max_chars = None if args.full else 500

    print(f"GRMT: {total} sequences, vocab_size={header['vocab_size']}")
    print(f"Showing sequences [{start}, {end}):\n")

    for record in iter_grmt_sequences(grmt):
        if record.index < start:
            continue
        if record.index >= end:
            break

        text = decode_sequence(
            record.token_ids,
            vocab,
            atom_texts=record.atom_texts,
            token_numeric_values=record.token_numeric_values,
            token_atom_mask=record.token_atom_mask,
        )

        print(f"{'═' * 80}")
        exec_tag = "exec-active" if record.execution_active else "exec-inactive"
        print(f"  Sequence {record.index}  |  {len(record.token_ids)} tokens  |  {len(text)} chars  |  {exec_tag}")
        if record.prompt_length > 0:
            prompt_start = record.prompt_end_pos - record.prompt_length + 1
            response_start = record.prompt_end_pos + 1
            print(
                f"  Prompt [{prompt_start}, {response_start}) = {record.prompt_length} tokens"
                f"  |  Response [{response_start}, {len(record.token_ids)}) = "
                f"{len(record.token_ids) - response_start} tokens"
            )
        else:
            print(
                f"  Prompt: absent (length={record.prompt_length}, end={record.prompt_end_pos})"
                f"  |  Whole-stream tokens={len(record.token_ids)}"
            )
        print(f"{'═' * 80}")
        if args.raw:
            for i, tid in enumerate(record.token_ids):
                if is_atom_token_id(tid):
                    piece = decode_atom_token(
                        i,
                        tid,
                        vocab,
                        atom_texts=record.atom_texts,
                        token_numeric_values=record.token_numeric_values,
                        token_atom_mask=record.token_atom_mask,
                    )
                    atom_extra = (
                        f"  atom_mask={record.token_atom_mask[i]}"
                        f" atom_flags={record.token_atom_flags[i]}"
                        f" numeric={format_numeric_value(record.token_numeric_values[i])}"
                    )
                else:
                    piece = decode_token(tid, vocab)
                    atom_extra = ""

                region = token_type_label(tid)
                print(f"  [{i:>4d}] {tid:>6d} {region:>7s}  {piece!r}{atom_extra}")
            print()

        display = text
        truncated = False
        if max_chars is not None and len(text) > max_chars:
            display = text[:max_chars]
            truncated = True

        print(wrap_text(display.strip()))
        if truncated:
            assert max_chars is not None
            print(f"\n  ... [{len(text) - max_chars} more chars, use --full to see all]")
        print()


def cmd_search(args, vocab: dict[int, str]):
    """Search sequences for a substring."""
    grmt = Path(args.grmt)
    if not grmt.exists():
        raise FileNotFoundError(f"GRMT file not found: {grmt}")

    header = read_grmt_header(grmt)
    validate_grmt_vocab_pair(header, vocab, grmt)

    query = args.search.lower()
    found = 0
    limit = args.limit

    for record in iter_grmt_sequences(grmt):
        text = decode_sequence(
            record.token_ids,
            vocab,
            atom_texts=record.atom_texts,
            token_numeric_values=record.token_numeric_values,
            token_atom_mask=record.token_atom_mask,
        )
        if query in text.lower():
            print(f"{'═' * 80}")
            print(f"  Sequence {record.index}  |  {len(record.token_ids)} tokens  |  {len(text)} chars")
            print(f"{'═' * 80}")
            print(wrap_text(text.strip()))
            print()
            found += 1
            if found >= limit:
                print(f"(stopped after {limit} matches, use --limit N to show more)")
                break

    if found == 0:
        print(f"No sequences contain '{args.search}'")
    else:
        print(f"Found {found} matching sequence(s).")


def cmd_stats(args, vocab: dict[int, str]):
    """Print vocab and GRMT statistics."""
    grmt = Path(args.grmt)
    if not grmt.exists():
        raise FileNotFoundError(f"GRMT file not found: {grmt}")

    header = read_grmt_header(grmt)
    validate_grmt_vocab_pair(header, vocab, grmt)

    print("═══ Vocabulary ═══")
    n_special = NUM_SPECIAL_TOKENS
    n_byte = BYTE_VOCAB_SIZE
    n_numeric = len(NUMERIC_TOKEN_TEXT)
    n_atom = 2 * NUM_ATOM_TYPES
    n_unigram = len(vocab) - UNIGRAM_TOKEN_START
    print(f"  Total entries : {len(vocab)}")
    print(f"  Special       : {n_special}  (IDs 0-{NUM_SPECIAL_TOKENS - 1})")
    print(f"  Byte fallback : {n_byte}  (IDs {BYTE_TOKEN_OFFSET}-{NUMERIC_TOKEN_OFFSET - 1})")
    print(f"  Fixed numeric : {n_numeric}  (IDs {NUMERIC_TOKEN_OFFSET}-{NUMERIC_TOKEN_END - 1})")
    print(f"  Atom slots    : {n_atom}  (IDs {ATOM_TOKEN_OFFSET}-{ATOM_TOKEN_END - 1})")
    print(f"  Unigram pieces: {n_unigram}  (IDs {UNIGRAM_TOKEN_START}+)")
    print()

    print("═══ Training Data (GRMT) ═══")
    print(f"  Magic       : {header['magic_hex']}")
    print(f"  Version     : {header['version']}")
    print(f"  Sequences   : {header['num_sequences']}")
    print(f"  Vocab size  : {header['vocab_size']}")
    print()

    token_counter = Counter()
    total_tokens = 0
    seq_lengths = []
    unknown_ids = set()
    execution_active_count = 0
    atom_token_count = 0
    prompt_lengths = []
    response_lengths = []
    promptless_count = 0
    invalid_prompt_count = 0
    prompt_over_stride_count = 0
    prompt_at_or_over_window_count = 0
    over_window_count = 0

    for record in iter_grmt_sequences(grmt):
        seq_lengths.append(len(record.token_ids))
        total_tokens += len(record.token_ids)
        if record.execution_active:
            execution_active_count += 1
        if len(record.token_ids) > 1024:
            over_window_count += 1
        if record.prompt_length == 0:
            promptless_count += 1
            if record.prompt_end_pos != -1:
                invalid_prompt_count += 1
        else:
            prompt_start = record.prompt_end_pos - record.prompt_length + 1
            response_start = record.prompt_end_pos + 1
            if (
                record.prompt_length < 0
                or prompt_start < 0
                or record.prompt_end_pos >= len(record.token_ids)
                or response_start > len(record.token_ids)
            ):
                invalid_prompt_count += 1
            else:
                prompt_lengths.append(record.prompt_length)
                response_lengths.append(len(record.token_ids) - response_start)
                if record.prompt_length > 768:
                    prompt_over_stride_count += 1
                if record.prompt_length >= 1024:
                    prompt_at_or_over_window_count += 1
        atom_token_count += sum(1 for tid in record.token_ids if is_atom_token_id(tid))
        for tid in record.token_ids:
            token_counter[tid] += 1
            if tid not in vocab:
                unknown_ids.add(tid)

    print("═══ Sequence Statistics ═══")
    print(f"  Total tokens   : {total_tokens:,}")
    print(f"  Total sequences: {len(seq_lengths):,}")
    print(f"  Exec-active seq: {execution_active_count:,}")
    print(f"  Atom tokens    : {atom_token_count:,}")
    if seq_lengths:
        print(f"  Avg seq length : {sum(seq_lengths) / len(seq_lengths):.1f}")
        print(f"  Min seq length : {min(seq_lengths)}")
        print(f"  Max seq length : {max(seq_lengths)}")
        print(f"  Seq > 1024     : {over_window_count:,}")
    print()

    print("=== Prompt / Response Geometry ===")
    print(f"  Prompt-bearing : {len(prompt_lengths):,}")
    print(f"  Promptless      : {promptless_count:,}")
    print(f"  Invalid spans   : {invalid_prompt_count:,}")
    if prompt_lengths:
        print(
            f"  Prompt tokens   : min={min(prompt_lengths):,} "
            f"avg={sum(prompt_lengths) / len(prompt_lengths):.1f} "
            f"max={max(prompt_lengths):,}"
        )
        print(
            f"  Response tokens : min={min(response_lengths):,} "
            f"avg={sum(response_lengths) / len(response_lengths):.1f} "
            f"max={max(response_lengths):,}"
        )
        print(f"  Prompt > 768    : {prompt_over_stride_count:,}")
        print(f"  Prompt >= 1024  : {prompt_at_or_over_window_count:,}")
    print()

    if unknown_ids:
        print(f"  ⚠ Unknown token IDs ({len(unknown_ids)}): {sorted(unknown_ids)[:20]}")
    else:
        print("  ✓ All token IDs map to known vocab entries")
    print()

    print("═══ Top 20 Most Frequent Tokens ═══")
    for tid, count in token_counter.most_common(20):
        text = decode_token(tid, vocab)
        region = token_type_label(tid)
        pct = 100.0 * count / total_tokens
        print(f"  {tid:>6d} [{region:>7s}]  {count:>8,}  ({pct:5.2f}%)  {text!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Decode GRIM-text token IDs from training_data.grmt using vocab.bin"
    )
    parser.add_argument("--vocab", default=str(VOCAB_BIN),
                        help="Path to vocab.bin")
    parser.add_argument("--grmt", default=str(GRMT_FILE),
                        help="Path to training_data.grmt")
    parser.add_argument("--ids", nargs="+",
                        help="Decode token IDs (e.g. --ids 277 512 36 or --ids 277,512,36)")
    parser.add_argument("--seq", type=int, nargs="*",
                        help="Sequence range: --seq START END (default: first 10)")
    parser.add_argument("--search", type=str,
                        help="Search decoded text for a substring")
    parser.add_argument("--raw", action="store_true",
                        help="Show raw token IDs alongside decoded text")
    parser.add_argument("--stats", action="store_true",
                        help="Print vocabulary and training data statistics")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max results for --search (default: 20)")
    parser.add_argument("--full", action="store_true",
                        help="Show full sequence text (default: truncated to 500 chars)")

    args = parser.parse_args()

    vocab_path = Path(args.vocab)
    print(f"Loading vocab: {vocab_path}")
    vocab = load_vocab_bin(vocab_path)
    print(f"Loaded {len(vocab)} token mappings\n")

    if args.ids:
        args.ids = parse_cli_token_ids(args.ids)
        cmd_decode_ids(args, vocab)
    elif args.search:
        cmd_search(args, vocab)
    elif args.stats:
        cmd_stats(args, vocab)
    else:
        cmd_decode_sequences(args, vocab)


if __name__ == "__main__":
    main()
