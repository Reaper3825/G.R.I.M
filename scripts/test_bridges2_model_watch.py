"""Focused watcher tests: use a local Python process as the SSH transport."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from watch_models_bridges2 import InstanceLock, Watcher, local_path, protected_checkpoints, sftp_quote, error_message


class LocalWatcher(Watcher):
    def command(self, source, *args):
        return [sys.executable, "-c", source, self.remote_root, *args]

    def transfer_via_sftp(self, relative, temporary):
        shutil.copyfile(Path(self.remote_root) / relative, temporary)


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / "model_store"
        self.local = self.root / "local"
        self.remote.mkdir()
        self.local.mkdir()
        self.state = self.root / "cache" / "state.json"
        self.watcher = LocalWatcher("test", str(self.remote), self.local, self.state)

    def write_remote(self, name, data):
        path = self.remote / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_stable_files_download_and_restart_skips_completed_files(self):
        self.write_remote("model/checkpoint.grimckpt", b"checkpoint")
        self.write_remote("model/checkpoint.opt", b"optimizer")
        self.write_remote("model/adapter.grimlorackpt", b"adapter")
        self.watcher.poll()
        self.assertEqual(list(self.local.iterdir()), [])
        self.watcher.poll()
        self.assertEqual((self.local / "model/checkpoint.opt").read_bytes(), b"optimizer")
        self.assertEqual(len(json.loads(self.state.read_text())), 3)
        restarted = LocalWatcher("test", str(self.remote), self.local, self.state)
        with patch.object(restarted, "download", side_effect=AssertionError("Redownload")):
            restarted.poll()
            restarted.poll()

    def test_changing_file_waits_for_another_stable_poll(self):
        source = self.write_remote("model/checkpoint.opt", b"partial")
        self.watcher.poll()
        source.write_bytes(b"complete optimizer state")
        self.watcher.poll()
        self.assertFalse((self.local / "model/checkpoint.opt").exists())
        self.watcher.poll()
        self.assertEqual((self.local / "model/checkpoint.opt").read_bytes(), source.read_bytes())

    def test_failed_transfer_preserves_existing_file_and_retries(self):
        self.write_remote("model/checkpoint.grimckpt", b"new checkpoint")
        target = self.local / "model/checkpoint.grimckpt"
        target.parent.mkdir()
        target.write_bytes(b"old checkpoint")
        self.watcher.poll()
        def fail_stream(relative, temporary):
            Path(temporary).write_bytes(b"partial")
            raise RuntimeError("Interrupted SFTP download")

        with patch.object(self.watcher, "transfer_via_sftp", side_effect=fail_stream):
            self.watcher.poll()
        self.assertEqual(target.read_bytes(), b"old checkpoint")
        self.assertEqual(list(target.parent.glob("*.partial")), [])
        self.watcher.poll()
        self.assertEqual(target.read_bytes(), b"new checkpoint")

    def test_source_replaced_after_inventory_is_rejected(self):
        source = self.write_remote("checkpoint.grimckpt", b"old")
        signature = self.watcher.inventory()["checkpoint.grimckpt"]
        replacement = self.write_remote("replacement.tmp", b"new")
        replacement.replace(source)
        with self.assertRaises(subprocess.CalledProcessError):
            self.watcher.download("checkpoint.grimckpt", signature)
        self.assertFalse((self.local / "checkpoint.grimckpt").exists())
        self.assertEqual(list(self.local.iterdir()), [])

    def test_temporary_files_ignored_and_remote_deletion_keeps_local_copy(self):
        for name in ("model/save.tmp", "model/save.id.tmp", "model/save.partial", ".hidden/file", "model/.lock"):
            self.write_remote(name, b"unfinished")
        source = self.write_remote("model/checkpoint.grimckpt", b"checkpoint")
        self.assertEqual(list(self.watcher.inventory()), ["model/checkpoint.grimckpt"])
        self.watcher.poll()
        self.watcher.poll()
        source.unlink()
        self.watcher.poll()
        self.assertEqual((self.local / "model/checkpoint.grimckpt").read_bytes(), b"checkpoint")

    def test_git_managed_configs_are_never_downloaded_or_overwritten(self):
        names = ("model.grimcfg", "model_config.json", "Router_configuration.json", "settings.yaml", "settings.toml")
        (self.local / "model").mkdir()
        for name in names:
            self.write_remote("model/" + name, b"remote configuration")
            (self.local / "model" / name).write_bytes(b"local git configuration")
        self.write_remote("model/checkpoint_PT_epoch_1.grimckpt", b"checkpoint")
        self.watcher.prune_remote = True
        self.watcher.poll()
        self.watcher.poll()
        for name in names:
            self.assertEqual((self.local / "model" / name).read_bytes(), b"local git configuration")
            self.assertTrue((self.remote / "model" / name).exists())
            self.assertNotIn("model/" + name, self.watcher.state)

    def test_source_changed_during_stream_never_publishes_download(self):
        self.write_remote("checkpoint.grimckpt", b"checkpoint bytes")
        target = self.local / "checkpoint.grimckpt"
        target.write_bytes(b"previous local checkpoint")
        signature = self.watcher.inventory()["checkpoint.grimckpt"]
        original = self.watcher.transfer_via_sftp

        def change_during_stream(relative, temporary):
            original(relative, temporary)
            path = self.remote / relative
            s = path.stat()
            os.utime(path, ns=(s.st_atime_ns, s.st_mtime_ns + 1000000000))

        with patch.object(self.watcher, "transfer_via_sftp", side_effect=change_during_stream):
            with self.assertRaises(subprocess.CalledProcessError):
                self.watcher.download("checkpoint.grimckpt", signature)
        self.assertEqual(target.read_bytes(), b"previous local checkpoint")
        self.assertEqual(list(self.local.glob("*.partial")), [])

    def test_unsafe_paths_rejected(self):
        for name in ("../outside", "/absolute", "C:/drive", "model/file:ads", "model\\file", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                local_path(self.local, name)

    def test_lock_excludes_duplicate_watcher_and_releases_on_exit(self):
        lock = self.root / "cache" / "watch.lock"
        with InstanceLock(lock):
            with self.assertRaises(RuntimeError):
                with InstanceLock(lock):
                    self.fail("Duplicate lock acquired")
        with InstanceLock(lock):
            pass

    def test_real_ssh_command_quotes_remote_paths(self):
        watcher = Watcher("user@host", "/repo with 'quotes'/models", self.local, self.state)
        command = watcher.command("print('hello')", "model/file with spaces", "[1, 2]")
        import shlex
        remote_args = shlex.split(command[-1])
        self.assertEqual(remote_args[-3:], [watcher.remote_root, "model/file with spaces", "[1, 2]"])
        self.assertIn("BatchMode=yes", command)

    def test_control_ssh_and_data_sftp_use_separate_hosts(self):
        watcher = Watcher("user@data", "/remote/model_store", self.local, self.state, control_host="bridges2")
        self.assertEqual(watcher.command("print('metadata')")[-2], "bridges2")
        with patch("watch_models_bridges2.subprocess.Popen") as start:
            process = start.return_value
            process.returncode = 0
            watcher.transfer_via_sftp("a/file with spaces.grimckpt", str(self.local / "target"))
            self.assertEqual(start.call_args.args[0][-1], "user@data")
            batch = process.communicate.call_args.args[0].decode()
            self.assertIn('get "/remote/model_store/a/file with spaces.grimckpt"', batch)
            self.assertNotIn("python", str(start.call_args.args[0]))

    def test_sftp_paths_quote_globs_and_reject_batch_injection(self):
        self.assertEqual(sftp_quote('/a/file [1] "x"'), '"/a/file \\[1\\] \\"x\\""')
        with self.assertRaises(ValueError):
            sftp_quote("file\nrm /other")

    def test_errors_show_stderr_without_encoded_command(self):
        error = subprocess.CalledProcessError(255, ["ssh", "huge base64 command"], stderr=b"Login denied: python3 is not an allowed command\n")
        message = error_message(error)
        self.assertIn("Login denied", message)
        self.assertNotIn("base64", message)

    def test_vocab_sync_only_downloads_vocab_and_never_removes_remote_files(self):
        data = self.root / "data"
        data.mkdir()
        (data / "vocab.bin").write_bytes(b"binary vocabulary")
        (data / "vocab.txt").write_bytes(b"text vocabulary")
        (data / "training_data.grmt").write_bytes(b"large corpus excluded")
        (data / "checkpoint_PT_epoch_1.grimckpt").write_bytes(b"never prune training data")
        vocab = LocalWatcher("test", str(data), self.local, self.state, include_files=("vocab.bin", "vocab.txt"))
        vocab.poll()
        vocab.poll()
        self.assertEqual(set(p.name for p in self.local.iterdir()), {"vocab.bin", "vocab.txt"})
        self.assertEqual((self.local / "vocab.bin").read_bytes(), b"binary vocabulary")
        self.assertTrue((data / "vocab.bin").exists())
        self.assertTrue((data / "vocab.txt").exists())
        (data / "vocab.bin").write_bytes(b"updated binary vocabulary")
        vocab.poll()
        self.assertEqual((self.local / "vocab.bin").read_bytes(), b"binary vocabulary")
        vocab.poll()
        self.assertEqual((self.local / "vocab.bin").read_bytes(), b"updated binary vocabulary")
        self.assertTrue((data / "vocab.bin").exists())

    def test_vocab_cannot_enable_pruning_and_missing_vocab_is_optional(self):
        with self.assertRaises(ValueError):
            LocalWatcher("test", str(self.remote), self.local, self.state, prune_remote=True,
                         include_files=("vocab.bin", "vocab.txt"))
        vocab = LocalWatcher("test", str(self.remote), self.local, self.state, include_files=("vocab.bin", "vocab.txt"))
        self.assertEqual(vocab.inventory(), {})

    def checkpoint_pair(self, directory, stage, epoch, mtime):
        prefix = f"{directory}/checkpoint_{stage}_epoch_{epoch}"
        for suffix, data in ((".grimckpt", b"checkpoint"), (".opt", b"optimizer")):
            path = self.write_remote(prefix + suffix, data)
            os.utime(path, ns=(mtime, mtime))
        return prefix

    def test_prune_keeps_latest_pt_sft_and_sidecars_per_directory(self):
        self.watcher.prune_remote = True
        old = []
        retained = []
        for directory in ("model_a/checkpoints", "model_b/checkpoints"):
            for stage in ("PT", "SFT"):
                old.append(self.checkpoint_pair(directory, stage, 1, 1000000000))
                retained.append(self.checkpoint_pair(directory, stage, 2, 2000000000))
        config = self.write_remote("model_a/model.grimcfg", b"required configuration")
        self.watcher.poll()
        self.assertTrue(all((self.remote / (name + ".grimckpt")).exists() for name in old))
        self.watcher.poll()
        for name in old:
            for suffix in (".grimckpt", ".opt"):
                self.assertFalse((self.remote / (name + suffix)).exists())
                self.assertTrue((self.local / (name + suffix)).exists())
        for name in retained:
            for suffix in (".grimckpt", ".opt"):
                self.assertTrue((self.remote / (name + suffix)).exists())
        self.assertTrue(config.exists())

    def test_latest_ranking_uses_mtime_then_numeric_epoch(self):
        snapshot = {
            "a/checkpoint_PT_epoch_9.grimckpt": [10, 100, 0, 1],
            "a/checkpoint_PT_epoch_10.grimckpt": [10, 100, 0, 2],
            "a/checkpoint_SFT_final.grimckpt": [10, 200, 0, 3],
            "a/checkpoint_SFT_epoch_99.grimckpt": [10, 100, 0, 4],
            "a/checkpoint_PT_epoch_11.grimckpt": [0, 300, 0, 5],
        }
        protected = protected_checkpoints(snapshot)
        self.assertIn("a/checkpoint_PT_epoch_10.grimckpt", protected)
        self.assertIn("a/checkpoint_SFT_final.opt", protected)
        self.assertNotIn("a/checkpoint_PT_epoch_9.grimckpt", protected)
        self.assertNotIn("a/checkpoint_PT_epoch_11.grimckpt", protected)

    def prepare_prune(self):
        old = self.checkpoint_pair("model", "PT", 1, 1000000000)
        new = self.checkpoint_pair("model", "PT", 2, 2000000000)
        self.watcher.poll()
        self.watcher.poll()
        self.watcher.prune_remote = True
        return old, new

    def test_local_corruption_with_preserved_metadata_blocks_remote_delete(self):
        old, _ = self.prepare_prune()
        target = self.local / (old + ".grimckpt")
        s = target.stat()
        target.write_bytes(b"corruption")  # same size as b'checkpoint'
        os.utime(target, ns=(s.st_atime_ns, s.st_mtime_ns))
        self.watcher.poll()
        self.assertTrue(self.watcher.prune_failed)
        self.assertTrue((self.remote / (old + ".grimckpt")).exists())
        self.assertTrue((self.remote / (old + ".opt")).exists())

    def test_new_checkpoint_not_archived_yet_blocks_pruning(self):
        old, _ = self.prepare_prune()
        self.checkpoint_pair("model", "PT", 3, 3000000000)
        self.watcher.poll()
        self.assertTrue((self.remote / (old + ".grimckpt")).exists())
        self.watcher.poll()
        self.assertFalse((self.remote / (old + ".grimckpt")).exists())

    def test_cleanup_rechecks_retention_on_remote(self):
        old, new = self.prepare_prune()
        original = self.watcher.command

        def change_retention(source, *args):
            if "request = json.load(sys.stdin)" in source:
                (self.remote / (new + ".grimckpt")).unlink()
            return original(source, *args)

        with patch.object(self.watcher, "command", side_effect=change_retention):
            self.watcher.poll()
        self.assertTrue((self.remote / (old + ".grimckpt")).exists())
        self.assertTrue((self.remote / (old + ".opt")).exists())
        self.assertTrue(self.watcher.prune_failed)

    def test_failed_sidecar_download_blocks_pruning_entire_pair(self):
        old = self.checkpoint_pair("model", "PT", 1, 1000000000)
        self.checkpoint_pair("model", "PT", 2, 2000000000)
        self.watcher.prune_remote = True
        original = self.watcher.download

        def fail_optimizer(name, signature):
            if name == old + ".opt":
                raise RuntimeError("Simulated failed download")
            return original(name, signature)

        self.watcher.poll()
        with patch.object(self.watcher, "download", side_effect=fail_optimizer):
            self.watcher.poll()
        self.assertTrue((self.remote / (old + ".grimckpt")).exists())
        self.assertTrue((self.remote / (old + ".opt")).exists())
        self.watcher.poll()
        self.assertFalse((self.remote / (old + ".grimckpt")).exists())

    def test_remote_change_after_plan_blocks_cleanup(self):
        old, _ = self.prepare_prune()
        original = self.watcher.command

        def change_source(source, *args):
            if "request = json.load(sys.stdin)" in source:
                path = self.remote / (old + ".grimckpt")
                path.write_bytes(b"newer contents")
            return original(source, *args)

        with patch.object(self.watcher, "command", side_effect=change_source):
            self.watcher.poll()
        self.assertTrue((self.remote / (old + ".grimckpt")).exists())
        self.assertTrue((self.remote / (old + ".opt")).exists())
        self.assertTrue(self.watcher.prune_failed)


if __name__ == "__main__":
    unittest.main()
