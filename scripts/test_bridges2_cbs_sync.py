"""Exercise launcher CBS preflight with real flatc, without SSH or training."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from build_concept_blocks_flatbuffer import default_flatc


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
FLATC = default_flatc(ROOT)


@unittest.skipUnless(BASH and FLATC.is_file(), "requires existing bash and vcpkg flatc")
class CbsSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="grim cbs test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        # Execute the real argument parsing and preflight, stopping before SSH.
        source = (ROOT / "scripts/run_train_on_bridges2.sh").read_text(encoding="utf-8")
        prefix, marker, _ = source.partition("# SSH target: bridges2 or bridges2.psc.edu")
        self.assertTrue(marker)
        self.launcher = scripts / "run_train_on_bridges2.sh"
        self.launcher.write_text(prefix + '\necho PREFLIGHT_COMPLETE\n', encoding="utf-8", newline="\n")
        # Use the real converter and schema; keep all generated data in the fixture.
        builder = ROOT / "scripts/build_concept_blocks_flatbuffer.py"
        schema = ROOT / "DataCollection/concept_block.fbs"
        (scripts / builder.name).write_text(
            "import runpy, sys\n"
            f"sys.argv.extend(['--flatc', {str(FLATC)!r}, '--schema', {str(schema)!r}])\n"
            f"runpy.run_path({str(builder)!r}, run_name='__main__')\n",
            encoding="utf-8",
        )
        self.data = self.root / "resources/models/GRIM-text/training/data"
        self.data.mkdir(parents=True)
        self.jsonl = self.data / "concept_blocks.jsonl"
        self.fb = self.data / "concept_blocks.fb"
        self.jsonl.write_text(json.dumps({"id": "fresh", "prompt": "Fresh question?"}) + "\n", encoding="utf-8")
        self.fb.write_bytes(b"stale flatbuffer")
        # Explicitly reproduce misleading transfer timestamps.
        os.utime(self.fb, (2000000000, 2000000000))
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GRIM_BRIDGES2_")}

    def launch(self, *args):
        return subprocess.run([BASH, self.launcher.as_posix(), *args],
                              env=self.env, capture_output=True, text=True, timeout=30)

    def test_all_cbs_entrypoints_rebuild_even_when_fb_is_newer(self):
        for args, extra_env in [
            (("--time", "4:30:00", "--sync", "cbs", "crs"), {}),
            (("--sync-cbs",), {}),
            (("--sync-all",), {}),
            ((), {"GRIM_BRIDGES2_SYNC_CBS": "1"}),
        ]:
            with self.subTest(args=args, env=extra_env):
                self.fb.write_bytes(b"stale flatbuffer")
                os.utime(self.fb, (2000000000, 2000000000))
                self.env.update(extra_env)
                result = self.launch(*args)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PREFLIGHT_COMPLETE", result.stdout)
                self.assertEqual(self.fb.read_bytes()[4:8], b"GRCB")
                subprocess.run([str(FLATC), "--json", "--strict-json", "-o", str(self.data),
                                str(ROOT / "DataCollection/concept_block.fbs"), "--", str(self.fb)],
                               check=True, capture_output=True)
                decoded = json.loads((self.data / "concept_blocks.json").read_text(encoding="utf-8"))
                self.assertEqual(decoded["blocks"][0]["id"], "fresh")
                self.assertEqual(decoded["blocks"][0]["prompt"], "Fresh question?")
                for key in extra_env:
                    self.env.pop(key)

    def test_invalid_jsonl_aborts_and_preserves_previous_fb(self):
        self.jsonl.write_text('{invalid json\n', encoding="utf-8")
        result = self.launch("--sync", "cbs")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("PREFLIGHT_COMPLETE", result.stdout)
        self.assertIn("refusing to upload stale data", result.stderr)
        self.assertEqual(self.fb.read_bytes(), b"stale flatbuffer")

    def test_missing_jsonl_aborts(self):
        self.jsonl.unlink()
        result = self.launch("--sync-cbs")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires local JSONL", result.stderr)

    def test_no_cbs_and_pull_modes_leave_fb_untouched(self):
        for args in [("--sync", "crs"), ("--sync-cbs", "--pull-logs"),
                     ("--sync-cbs", "--pull-vocab")]:
            with self.subTest(args=args):
                result = self.launch(*args)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.fb.read_bytes(), b"stale flatbuffer")


if __name__ == "__main__":
    unittest.main()
