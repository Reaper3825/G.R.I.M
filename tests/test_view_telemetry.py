"""CSV compatibility regressions for retired telemetry and live retrieval loss."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import matplotlib
import pandas as pd

matplotlib.use("Agg")
spec = importlib.util.spec_from_file_location(
    "view_telemetry", Path(__file__).resolve().parents[1] / "view_telemetry.py"
)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class TelemetryCsvTests(unittest.TestCase):
    def read_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            return viewer.read_telemetry_csv(path)

    def test_indexed_legacy_rows_are_filtered_and_live_loss_is_restored(self):
        rows = [
            {"global_step": 1, "level": 0, "stream_idx": i,
             "stream_name": "unknown", "raw_observation": 0.25}
            for i in range(94)
        ]
        result = self.read_rows(rows)
        self.assertEqual(len(result), 50)
        self.assertEqual(
            result.loc[result.stream_idx == 62, "stream_name"].tolist(),
            ["local_atom_retrieval_loss"],
        )
        self.assertEqual(len(viewer.pivot_streams(result)), 50)

    def test_named_legacy_streams_without_indices_are_filtered(self):
        names = ["loss", "text_loss", "local_atom_retrieval_loss",
                 "exec_grad_norm", "exec_op_accuracy", "execution_loss",
                 "eb_read_gate_mean", "reserved_14", "mtp_loss",
                 "sb_atom_embed_rms", "selector_loss", "latent_preset_loss"]
        result = self.read_rows([
            {"global_step": 1, "level": 0, "stream_name": name}
            for name in names
        ])
        self.assertEqual(result.stream_name.tolist(), names[:3])

    def test_historical_mtp_slot_does_not_conflict_with_retrieval_name(self):
        result = self.read_rows([
            {"global_step": 1, "level": 0, "stream_idx": 62,
             "stream_name": "mtp_loss"},
            {"global_step": 2, "level": 0, "stream_idx": 62,
             "stream_name": "local_atom_retrieval_loss"},
        ])
        self.assertEqual(result.global_step.tolist(), [2])

    def test_live_stream_mismatch_still_fails(self):
        with self.assertRaisesRegex(ValueError, "stream_idx/name mismatch"):
            self.read_rows([
                {"global_step": 1, "level": 0, "stream_idx": 62,
                 "stream_name": "text_loss"},
            ])


if __name__ == "__main__":
    unittest.main()
