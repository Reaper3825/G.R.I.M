import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

import generate_hf_pt as hf
from verify_hf_pt import validate_raw_entry
from test_generate_developer_pt import setup_target


class HFIngestionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Existing PyArrow environment required")
    def test_parquet_pages_preserve_rows_and_reject_changed_shard(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "source.parquet"
            pq.write_table(pa.Table.from_pylist([{"text": f"Document {i}"} for i in range(5)]), path, row_group_size=2)
            spec = dict(path=str(path), sha256=hf.digest(path.read_bytes()), size=path.stat().st_size,
                        dataset="org/dataset", config="config", revision="revision", repository_path="train-0.parquet")
            pages = list(hf.parquet_pages(spec, [3, 0], 2, root / "cache"))
            self.assertEqual([r["row"]["text"] for p in pages for r in p["rows"]],
                             ["Document 3", "Document 4", "Document 0", "Document 1"])
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaises(ValueError):
                list(hf.parquet_pages(spec, [0], 2, root / "cache"))

    def test_prefetch_preserves_sampling_order_and_bounds(self):
        with patch.object(hf, "get_page", side_effect=lambda dataset, config, offset, *rest: offset) as fetch:
            pages = hf.source_pages("dataset", "config", [40, 10, 90, 0, 70, 20, 30, 50], 10, 100, "rev", Path("cache"))
            self.assertEqual(next(pages), 40)
            pages.close()
            self.assertLessEqual(fetch.call_count, 5)
        with patch.object(hf, "get_page", side_effect=lambda dataset, config, offset, *rest: offset):
            self.assertEqual(list(hf.source_pages("dataset", "config", [40, 10, 90], 10, 100, "rev", Path("cache"))), [40, 10, 90])

    def test_raw_pt_contract_rejects_structured_payload(self):
        row = {"id": "example", "format_type": "raw", "raw": "Plain source text", "prompt": "", "explanation": []}
        validate_raw_entry(row)
        for field in ("prompt", "answer", "determine", "define", "execute", "update", "goal", "knowns"):
            with self.assertRaises(ValueError):
                validate_raw_entry(row | {field: "unexpected structured payload"})
        for change in ({"raw": " "}, {"format_type": "chain_of_thought"}):
            with self.assertRaises(ValueError):
                validate_raw_entry(row | change)

    def test_math_domain_matching_has_host_boundaries(self):
        self.assertTrue(hf.math_domain_allowed("https://math.stackexchange.com/questions/1"))
        self.assertTrue(hf.math_domain_allowed("https://math.mit.edu/lesson"))
        self.assertFalse(hf.math_domain_allowed("https://math.stackexchange.com.spam.test/x"))
        self.assertFalse(hf.math_domain_allowed("https://spam.test/?q=mit.edu"))

    def test_rate_limit_waits_and_stops_after_bounded_retries(self):
        error = HTTPError("https://example.com", 429, "rate limited", {"Retry-After": "12"}, None)
        with patch.object(hf.urllib.request, "urlopen", side_effect=error) as fetch:
            with patch.object(hf.time, "sleep") as sleep:
                with self.assertRaises(HTTPError):
                    hf.request_json("https://example.com")
                self.assertEqual(fetch.call_count, 8)
                self.assertEqual(sleep.call_count, 7)
                sleep.assert_called_with(12)

    def test_viewer_revision_mismatch_never_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            with patch.object(hf, "request_json", return_value=({"rows": []}, "stale")):
                with self.assertRaises(ValueError):
                    hf.get_page("org/data", "config", 0, 50, "expected", cache)
            self.assertEqual(list(cache.iterdir()), [])

    def test_cache_retains_revision_and_full_response(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            response = {"rows": [{"row_idx": 15, "row": {"text": "source"}}]}
            with patch.object(hf, "request_json", return_value=(response, "revision")) as fetch:
                self.assertEqual(hf.get_page("org/data", "config", 0, 50, "revision", cache), response)
                self.assertEqual(hf.get_page("org/data", "config", 0, 50, "revision", cache), response)
                self.assertEqual(fetch.call_count, 1)

    def test_rejects_truncated_rows_and_observed_spam(self):
        row = {"text": "A mathematical explanation. " * 40}
        self.assertEqual(hf.rejection(row, ["text"]), "truncated_row")
        self.assertIsNone(hf.rejection(row, []))
        row["text"] += " write a paper online"
        self.assertEqual(hf.rejection(row, []), "essay_spam")
        self.assertEqual(hf.rejection({"text": "# Search by Topic\n" + "topic text " * 100}, []), "topic_index")
        self.assertEqual(hf.rejection({"text": "Web multiplication worksheet. " * 30}, []), "worksheet_seo_text")

    def test_append_routes_distinct_courses_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            setup_target(root)
            for key in ("hf_howto", "hf_math"):
                row = {"id": key, "raw": key + " unique content"}
                self.assertEqual(hf.append_rows(root, "Pre-Trainingv1", [row], key, key)["new_members"], 1)
                self.assertEqual(hf.append_rows(root, "Pre-Trainingv1", [row], key, key)["new_members"], 0)
            registry, curriculum, _ = hf.load_target(root, "Pre-Trainingv1")
            self.assertEqual(len(curriculum["concept_block_ids"]), 3)
            self.assertEqual([c["name"] for c in registry["courses"]], ["Existing", "hf_howto", "hf_math"])

    def test_rejects_fragmented_math_and_language_mismatch(self):
        text = "Mathematics exercise. " * 40 + "\n" + "\n".join(str(i) for i in range(40))
        self.assertEqual(hf.rejection({"text": text}, []), "fragmented_extraction")
        self.assertEqual(hf.rejection({"text": "数学课程" * 150}, []), "non_latin_heavy_text")
        self.assertTrue(hf.PRACTICAL.search("Organize Your Email"))
        self.assertFalse(hf.PRACTICAL.search("Care for a Webkinz Plushie"))


if __name__ == "__main__":
    unittest.main()
