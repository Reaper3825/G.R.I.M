import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import generate_developer_pt as gen
from pt_vocab import PTTokenizer, numeric_end, numeric_count


def write_vocab(path, pieces=()):
    records = [(b"<unk>", 0, 0, 0), (b"<pad>", 0, 1, 0),
               (b"<s>", 0, 2, 0), (b"</s>", 0, 3, 0)]
    records += [(text, score, 319 + i, flags) for i, (text, score, flags) in enumerate(pieces)]
    data = struct.pack("<4sHIII3sI", b"KTMG", 9, 0, len(records), 32, b"\x01\0\0", 319 + len(pieces))
    for text, score, tid, flags in records:
        data += struct.pack("<I", len(text)) + text + struct.pack("<fiB", score, tid, flags)
    path.write_bytes(data)


def setup_target(path):
    course = {"id": "old-course", "name": "Existing", "concept_block_ids": ["old"]}
    cur = {"id": "pt", "name": "Pre-Trainingv1", "training_stage": "pt", "format_as_concept": False,
           "course_ids": ["old-course"], "concept_block_ids": ["old"]}
    reg = {"schema_version": 2, "courses": [course], "curriculums": [cur], "preserved": "yes"}
    (path / "curriculum_registry.json").write_bytes(gen.json_bytes(reg))
    (path / "Pre-Trainingv1.json").write_bytes(gen.json_bytes({"concept_block_ids": ["old"], "extra": 7}))
    (path / "concept_blocks.jsonl").write_text(json.dumps({"id": "old", "raw": "Original code"}) + "\n", encoding="utf-8")


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "vocab.bin"

    def tearDown(self):
        self.temp.cleanup()

    def test_numeric_grammar_and_runs(self):
        for value in (b"-1,000.55e+20", b".123", b"+3", b"1234", b"1,000,000"):
            self.assertEqual(numeric_end(value, 0), len(value))
        self.assertEqual(numeric_end(b"1234,000", 0), 4)
        self.assertEqual(numeric_end(b"1,2345", 0), 1)
        self.assertEqual(numeric_count(b"00000111"), 3)
        self.assertEqual(numeric_count(b"-1.2e+3"), 7)

    def test_current_newline_and_number_layout(self):
        write_vocab(self.path, [(b"\xe2\x96\x81", -1, 0)])
        tokenizer = PTTokenizer(self.path)
        self.assertEqual(tokenizer.count("123"), 4)  # marker + 3 numeric tokens
        self.assertEqual(tokenizer.count("0000"), 2)
        self.assertEqual(tokenizer.count("a\r\nb\rc\n"), 7)
        self.assertEqual(tokenizer.count("\t"), 2)
        self.assertEqual(tokenizer.count(""), 0)
        with self.assertRaises(ValueError):
            tokenizer.count("<INT>42</INT>")

    def test_exact_pieces_require_lexical_boundary(self):
        write_vocab(self.path, [(b"\xe2\x96\x81", -1, 0), ("▁hello".encode(), -10000, 1)])
        tokenizer = PTTokenizer(self.path)
        self.assertEqual(tokenizer.count("hello"), 1)
        self.assertEqual(tokenizer.count("hello!"), 2)
        self.assertEqual(tokenizer.count("helloworld"), 11)
        self.assertEqual(tokenizer.count("hello_"), 7)

    def test_viterbi_uses_score_not_greedy_longest(self):
        write_vocab(self.path, [(b"\xe2\x96\x81", -1, 0), (b"a", -1, 0), (b"aa", -10, 0)])
        self.assertEqual(PTTokenizer(self.path).count("aa"), 3)

    def test_reject_old_or_truncated_vocab(self):
        write_vocab(self.path)
        data = self.path.read_bytes()
        self.path.write_bytes(data[:4] + b"\x04\0" + data[6:])
        with self.assertRaises(ValueError):
            PTTokenizer(self.path)
        self.path.write_bytes(data[:-1])
        with self.assertRaises(ValueError):
            PTTokenizer(self.path)

    def test_chunks_preserve_source_and_boundaries(self):
        write_vocab(self.path, [(b"\xe2\x96\x81", -1, 0)])
        tokenizer = PTTokenizer(self.path)
        text = "Title\n\n" + "paragraph with Unicode Ω and numbers 3.14\n" * 20
        chunks = list(gen.chunk_document(text, tokenizer, 100))
        self.assertEqual("".join(c for c, _, _ in chunks), text)
        for chunk, offset, rejection in chunks:
            self.assertIsNone(rejection)
            self.assertEqual(text[offset:offset + len(chunk)], chunk)
            self.assertLessEqual(tokenizer.count(chunk) + 2, 100)

    def test_fences_not_cut_or_accepted_unclosed(self):
        write_vocab(self.path)
        tokenizer = PTTokenizer(self.path)
        for text in ("```python\n" + "print('x')\n" * 30 + "```\n", "```\nx\n"):
            chunks = list(gen.chunk_document(text, tokenizer, 64))
            self.assertEqual(chunks[0][2], "oversized_or_unclosed_code_fence")


class AppendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        setup_target(self.root)
        self.row = {"id": "new", "raw": "def test():\n    assert True", "format_type": "raw"}

    def tearDown(self):
        self.temp.cleanup()

    def test_append_membership_and_idempotence(self):
        original = (self.root / "concept_blocks.jsonl").read_bytes()
        self.assertEqual(gen.append_rows(self.root, "Pre-Trainingv1", [self.row])["new_members"], 1)
        self.assertTrue((self.root / "concept_blocks.jsonl").read_bytes().startswith(original))
        registry, cur, manifest = gen.load_target(self.root, "Pre-Trainingv1")
        self.assertEqual(cur["concept_block_ids"], ["old", "new"])
        self.assertEqual(registry["preserved"], "yes")
        self.assertEqual(manifest["extra"], 7)
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        self.assertEqual(gen.append_rows(self.root, "Pre-Trainingv1", [self.row])["new_rows"], 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_duplicate_content_does_not_create_row(self):
        duplicate = {"id": "different", "raw": "Original code"}
        self.assertEqual(gen.append_rows(self.root, "Pre-Trainingv1", [duplicate])["new_members"], 0)

    def test_failed_metadata_write_rolls_back(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        real = gen.atomic_bytes
        failed = False
        def fail_once(path, data):
            nonlocal failed
            if path.name == "Pre-Trainingv1.json" and not failed:
                failed = True
                raise OSError("simulated disk failure")
            return real(path, data)
        with patch.object(gen, "atomic_bytes", side_effect=fail_once):
            with self.assertRaises(OSError):
                gen.append_rows(self.root, "Pre-Trainingv1", [self.row])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_corrupt_corpus_is_not_modified(self):
        corpus = self.root / "concept_blocks.jsonl"
        with corpus.open("a") as f:
            f.write("not json\n")
        before = corpus.read_bytes()
        with self.assertRaises(ValueError):
            gen.append_rows(self.root, "Pre-Trainingv1", [self.row])
        self.assertEqual(corpus.read_bytes(), before)

    def test_membership_mismatch_rejected(self):
        (self.root / "Pre-Trainingv1.json").write_text('{"concept_block_ids": []}')
        with self.assertRaises(ValueError):
            gen.append_rows(self.root, "Pre-Trainingv1", [self.row])

    def test_crash_recovery_accepts_partial_tail(self):
        corpus = self.root / "concept_blocks.jsonl"
        size = corpus.stat().st_size
        metadata = {}
        for name in ("curriculum_registry.json", "Pre-Trainingv1.json"):
            before = (self.root / name).read_text()
            metadata[name] = {"before": before, "after": before}
        tx = {"size": size, "append": "partial append\n", "metadata": metadata}
        (self.root / ".developer_pt_transaction.json").write_bytes(gen.json_bytes(tx))
        with corpus.open("ab") as f:
            f.write(b"partial")
        gen.recover(self.root)
        self.assertEqual(corpus.stat().st_size, size)
        self.assertFalse((self.root / ".developer_pt_transaction.json").exists())

    def test_recovery_refuses_unrelated_edits(self):
        corpus = self.root / "concept_blocks.jsonl"
        tx = {"size": corpus.stat().st_size, "append": "expected", "metadata": {}}
        (self.root / ".developer_pt_transaction.json").write_bytes(gen.json_bytes(tx))
        with corpus.open("ab") as f:
            f.write(b"someone else's data")
        before = corpus.read_bytes()
        with self.assertRaises(ValueError):
            gen.recover(self.root)
        self.assertEqual(corpus.read_bytes(), before)


class ExtractionTests(unittest.TestCase):
    def test_rst_includes_preserve_selected_code(self):
        source = 'Intro\n\n.. literalinclude:: example.py\n   :language: python\n   :start-after: # begin\n   :end-before: # end\n\nConclusion\n'
        def load(path, parent):
            self.assertEqual(path, "example.py")
            return '# begin\ndef f():\n    return 42\n# end\n', path
        rendered = gen.render_rst(source, "guide.rst", load)
        self.assertIn('```python\ndef f():\n    return 42\n```', rendered)
        self.assertIn("Conclusion", rendered)
        self.assertNotIn("literalinclude", rendered)

    def test_rst_code_and_raw_html(self):
        source = '.. code-block:: python\n\n    x = 1\n    print(x)\n\n.. raw:: html\n\n    <details>\n\nProse\n'
        rendered = gen.render_rst(source, "x.rst", None)
        self.assertIn('```python\nx = 1\nprint(x)\n```', rendered)
        self.assertNotIn("<details>", rendered)
        self.assertIn("Prose", rendered)

    def test_unsupported_include_selectors_fail(self):
        with self.assertRaises(ValueError):
            gen.render_rst('.. literalinclude:: x.py\n   :pyobject: f\n', 'x.rst', None)

    def test_cleanup_preserves_indentation(self):
        text = '---\ntitle: Example\n---\n# Title\n\n```python\ndef f():\n    return 1\n```\n'
        rendered = gen.clean_document(text)
        self.assertIn('    return 1', rendered)
        self.assertFalse(rendered.startswith('---'))

    def test_github_paths_with_spaces_are_escaped(self):
        sha = "a" * 40
        source = {"id": "example", "repository": "org/repo", "ref": "main", "topic": "tools",
                  "license_paths": ["LICENSE"], "include": ["docs/*.md"]}
        urls = []
        def fake_fetch(url, cache, **kwargs):
            urls.append(url)
            if "/commits/" in url:
                return json.dumps({"sha": sha}).encode()
            if "/git/trees/" in url:
                return json.dumps({"tree": [{"type": "blob", "path": "docs/a b.md"}]}).encode()
            return b"Example text"
        with patch.object(gen, "fetch", side_effect=fake_fetch):
            docs = list(gen.github_documents(source, Path("unused"), 1))
        self.assertTrue(any("a%20b.md" in url for url in urls))
        self.assertIn("a%20b.md", docs[0]["source_url"])


if __name__ == "__main__":
    unittest.main()
