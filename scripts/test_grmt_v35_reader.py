"""Host-only regression for GRMT row boundaries after execution metadata removal."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decode_token_ids import GRMT_MAGIC, iter_grmt_sequences


def text(value):
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def row(identity, tokens, spans):
    n = len(tokens)
    result = text(identity) + struct.pack("<I", n)
    result += struct.pack(f"<{n}i", *tokens)
    result += struct.pack(f"<{n}i", *(tokens[1:] + [-1]))
    result += bytes(n * 4) + bytes(n) + bytes(n * 4)
    result += struct.pack(f"<{n}I", *([0xFFFFFFFF] * n))
    result += b"\x00"  # no AtomTable
    result += struct.pack(f"<{n}I", *([0xFFFFFFFF] * n))
    result += b"\x00"  # no sequence-local atom table
    result += struct.pack("<iiI", 0, 1, len(spans))
    for name, parent, begin, end in spans:
        result += text(name) + struct.pack("<iii", parent, begin, end)
    return result


class GrmtReaderTests(unittest.TestCase):
    def test_successive_rows_preserve_targets_identity_and_nested_spans(self):
        spans = [("criteria", -1, 0, 2), ("criterion", 0, 0, 1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.grmt"
            path.write_bytes(struct.pack("<IIII", GRMT_MAGIC, 35, 2, 1024)
                             + text("[]")
                             + row("first", [101, 102], spans)
                             + row("second", [103], []))
            records = list(iter_grmt_sequences(path))
        self.assertEqual([r.concept_block_id for r in records], ["first", "second"])
        self.assertEqual(records[0].targets, [102, -1])
        self.assertEqual(records[0].named_spans, spans)
        self.assertEqual(records[1].token_ids, [103])
        self.assertEqual(records[1].targets, [-1])

    def test_old_format_is_rejected_before_row_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.grmt"
            path.write_bytes(struct.pack("<IIII", GRMT_MAGIC, 34, 1, 1024))
            with self.assertRaisesRegex(ValueError, "expected 35"):
                list(iter_grmt_sequences(path))


if __name__ == "__main__":
    unittest.main()
