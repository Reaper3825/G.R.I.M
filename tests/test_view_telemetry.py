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

    def test_new_fixed_names_and_dynamic_names(self):
        result = self.read_rows([
            {"global_step": 1, "level": 0, "stream_idx": index,
             "stream_name": "unknown", "raw_observation": 1}
            for index in range(94, 142)
        ] + [{"global_step": 1, "level": 0, "stream_idx": 142,
              "stream_name": "layer7_attention_residual_gate_multiplier_mean", "raw_observation": 1}])
        self.assertEqual(result.stream_name.iloc[0], "gradient_global_sum_sq_pre")
        self.assertEqual(result.stream_name.iloc[8], "top_grad_1_group_index")
        self.assertEqual(result.stream_name.iloc[28], "top_energy_1_group_index")
        self.assertEqual(result.stream_name.iloc[-2], "top_energy_5_parameter_count")
        self.assertEqual(result.stream_name.iloc[-1], "layer7_attention_residual_gate_multiplier_mean")

    def capture_fixture(self):
        rows = []
        for step in (10, 20, 50):
            values = {"gradient_global_sum_sq_pre": 0.01, "gradient_clipped": 0,
                      "gradient_global_rms_post": 0.001, "gradient_clip_scale": 1,
                      "attention_residual_gate_energy_pct": 0.02,
                      "gradient_capture_optimizer_iteration": step // 10}
            for name, magnitude in (("layer0_wo_weight", 0.001),
                                    ("layer7_attention_residual_gate_bias", 0.1)):
                for metric, value in zip(viewer.GROUP_METRICS, (magnitude, 0.005, 50, 1024)):
                    values[f"{name}_grad_{metric}"] = value
            for ranking in ("grad", "energy"):
                for rank in range(1, 6):
                    for field, value in zip(("group_index", "grad_rms", "energy_pct", "parameter_count"),
                                            (rank + step, 0.01, 5, 1024)):
                        values[f"top_{ranking}_{rank}_{field}"] = value
            if step != 20:
                values.update({"gate_forward_sample_global_step": step - 1,
                               "gate_forward_sample_token_count": 1024,
                               "layer7_attention_residual_gate_multiplier_mean": 1.1,
                               "layer7_attention_residual_gate_multiplier_std": 0.1,
                               "layer7_attention_residual_gate_bias_grad_postclip_mean": -0.01})
            rows.extend({"global_step": step, "stream_name": name, "level": 0,
                         "raw_observation": value} for name, value in values.items())
        return viewer.pivot_streams(pd.DataFrame(rows))

    def test_capture_alignment_keeps_sparse_gaps_and_stable_names(self):
        table = viewer.capture_tables(self.capture_fixture())
        self.assertEqual(table.index.tolist(), [10, 20, 50])
        self.assertTrue(pd.isna(table.loc[20, "layer7_attention_residual_gate_multiplier_mean"]))
        self.assertEqual(table.top_grad_1_group_index.tolist(), [11, 21, 51])
        energy = viewer.group_metric_table(table, "energy_pct")
        self.assertEqual(energy.columns.tolist(), ["layer0_wo_weight", "layer7_attention_residual_gate_bias"])
        self.assertEqual(energy.layer0_wo_weight.tolist(), [50, 50, 50])

    def test_heatmap_preserves_missing_observation_column(self):
        fig, ax = viewer.plt.subplots()
        frame = pd.DataFrame({"Layer 7": [1, float("nan"), 2]}, index=[9, 20, 49])
        viewer.capture_heatmap(fig, ax, frame, "Gate")
        image = ax.images[0].get_array()
        self.assertEqual(image.shape, (1, 3))
        self.assertTrue(image.mask[0, 1])
        viewer.plt.close(fig)

    def test_offline_ranks_and_concentration_use_all_named_groups(self):
        rows = {}
        for i in range(7):
            rows[f"group{i}_grad_rms_pre"] = [7 - i, 1]
            rows[f"group{i}_grad_energy_pct"] = [i + 1, 10]
        table = pd.DataFrame(rows, index=[10, 20])
        ranks, concentration = viewer.offline_group_comparisons(table)
        self.assertEqual(ranks["rms_rank"].loc[10].tolist(), list(range(1, 8)))
        self.assertEqual(ranks["energy_rank"].loc[10].tolist(), list(range(7, 0, -1)))
        self.assertEqual(ranks["rms_rank"].loc[20].tolist(), list(range(1, 8)))
        self.assertEqual(concentration.loc[10, "grad"], 15)
        self.assertEqual(concentration.loc[10, "energy"], 25)
        table.loc[20, "group6_grad_energy_pct"] = float("nan")
        _, incomplete = viewer.offline_group_comparisons(table)
        self.assertTrue(incomplete.loc[20].isna().all())

    def test_capture_without_training_rankings(self):
        streams = {name: frame for name, frame in self.capture_fixture().items()
                   if not name.startswith("top_")}
        with tempfile.TemporaryDirectory() as directory:
            outputs = viewer.generate_capture_figures(str(Path(directory) / "unranked.csv"), streams)
            self.assertEqual(len(outputs), 5)
            self.assertTrue(all(Path(path).stat().st_size > 1000 for path in outputs))

    def test_capture_plot_smoke_and_legacy_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(viewer.generate_capture_figures(str(Path(directory) / "old.csv"), {}), [])
            outputs = viewer.generate_capture_figures(str(Path(directory) / "new.csv"), self.capture_fixture())
            self.assertEqual(len(outputs), 7)
            for path in outputs:
                self.assertGreater(Path(path).stat().st_size, 1000)
            self.assertEqual(viewer.plt.get_fignums(), [])

    def test_tied_energy_ratio_uses_raw_energy_and_deduplicated_counts(self):
        table = pd.DataFrame({
            "embedding_lm_head_tied_grad_sum_sq_pre": [6, 0, 6, 6],
            "embedding_lm_head_tied_grad_parameter_count": [20, 20, 20, 20],
            "other_grad_sum_sq_pre": [4, 0, 4, 4],
            "other_grad_parameter_count": [80, 80, float("nan"), 80],
            "gradient_global_sum_sq_pre": [10, 0, 10, 20],
            # Deliberately inconsistent derived percentage: use raw energy instead.
            "embedding_lm_head_tied_grad_energy_pct": [99, 99, 99, 99],
        }, index=[10, 20, 30, 40])
        count_pct, energy_pct, ratio, tied = viewer.offline_gradient_energy_density(table)
        self.assertEqual(count_pct.loc[10, "embedding_lm_head_tied"], 20)
        self.assertEqual(energy_pct.loc[10, "embedding_lm_head_tied"], 60)
        self.assertEqual(ratio.loc[10, "embedding_lm_head_tied"], 3)
        self.assertEqual(ratio.loc[10, "other"], 0.5)
        self.assertEqual(tied.loc[10, "total_parameter_count"], 100)
        self.assertTrue(ratio.loc[[20, 30, 40]].isna().all().all())
        self.assertTrue(pd.isna(tied.loc[30, "total_parameter_count"]))
        # An older CSV without raw counts cannot supply this denominator.
        historical = table.drop(columns=[name for name in table if name.endswith("parameter_count")])
        _, _, legacy_ratio, _ = viewer.offline_gradient_energy_density(historical)
        self.assertTrue(legacy_ratio.isna().all().all())

    def test_tied_ratio_plot_and_offline_csv_export(self):
        streams = {name.replace("layer0_wo_weight", "embedding_lm_head_tied"): frame
                   for name, frame in self.capture_fixture().items() if not name.startswith("top_")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tied.csv"
            outputs = viewer.generate_capture_figures(str(path), streams)
            self.assertEqual(len(outputs), 6)
            folder = Path(directory) / "tied_gradient_gate"
            tied = pd.read_csv(folder / "tied_gradient_ratio.csv")
            self.assertEqual(tied.tied_gradient_energy_ratio.tolist(), [1, 1, 1])
            self.assertEqual(tied.tied_parameter_count_pct.tolist(), [50, 50, 50])
            groups = pd.read_csv(folder / "group_gradient_energy.csv")
            self.assertEqual(len(groups), 6)
            self.assertEqual(set(groups.group_name), {"embedding_lm_head_tied", "layer7_attention_residual_gate_bias"})


if __name__ == "__main__":
    unittest.main()
