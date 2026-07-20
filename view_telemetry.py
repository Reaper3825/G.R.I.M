#!/usr/bin/env python3
"""Simple TelemetryLattice CSV pattern viewer for GRIM-text training runs."""

import sys
import glob
import os
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


REQUIRED_TELEMETRY_COLUMNS = {"global_step", "stream_name", "level"}
TELEMETRY_COLUMN_ALIASES = {
    "step": "global_step",
    "global step": "global_step",
    "stream": "stream_name",
    "stream id": "stream_idx",
    "observation": "raw_observation",
    "raw observation": "raw_observation",
}
TELEMETRY_STREAM_NAMES_BY_INDEX = {
    0: "loss",
    1: "grad_norm_mean",
    2: "grad_norm_max",
    3: "learning_rate",
    4: "tokens_per_batch",
    5: "rho_final",
    6: "rho_growth",
    7: "rho_worst_delta",
    8: "h_rms_growth",
    9: "adam_bc2_v_convergence",
    10: "adam_signal_dominance",
    11: "adam_cumulative_disp",
    12: "adam_disruption_emb",
    13: "adam_inv_bc2_amp",
    14: "exec_grad_norm",
    15: "exec_grad_ratio",
    16: "exec_selection_entropy",
    17: "exec_op_entropy",
    18: "exec_div_clamp_rate",
    19: "exec_max_p_write",
    20: "exec_active_ratio",
    21: "eb_inject_gate",
    22: "eb_read_gate_mean",
    23: "eb_inject_weight_norm",
    24: "eb_read_weight_norm",
    25: "eb_loss_frac",
    27: "pbm_alibi_slope_rms",
    28: "pbm_alibi_eff_bias_max",
    29: "pbm_rope_inv_freq_rms",
    30: "pbm_batch_max_seq_len",
    31: "rho_raw_avg_abs_dot",
    32: "rho_raw_avg_norm_prod",
    33: "rho_raw_h_rms_min",
    34: "rho_raw_h_rms_max",
    35: "rms_gamma_pre_attn_rms",
    36: "rms_gamma_pre_ffn_rms",
    37: "rms_gamma_final_rms",
    38: "rho_raw_rms_spread",
    39: "hw_cos_rms",
    40: "hw_cos_signed_mean",
    41: "hw_cos_abs_max",
    42: "hw_hbar_wbar_cos",
    43: "hw_h_dc_mean",
    44: "hw_h_dc_abs_max",
    45: "unigram_dir_cos_abs_mean",
    46: "unigram_dir_cos_signed_mean",
    47: "lm_head_w_rms_rms",
    48: "init_tie_cfg",
    49: "init_tie_ptrs_same",
    50: "init_tie_grads_same",
    51: "init_lm_owns_weights",
    52: "init_opt_groups_total",
    53: "init_opt_groups_emb",
    54: "init_opt_groups_lm",
    55: "rho_raw_avg_signed_dot",
    56: "rho_centered_avg_abs_dot",
    57: "rho_mean_vector_rms",
    58: "rho_atom_only",
    59: "rho_nonatom_only",
    60: "optimizer_iteration",
    61: "text_loss",
    68: "execution_loss",
    69: "exec_loss_gate_ce_raw",
    70: "exec_loss_stop_ce_raw",
    71: "exec_loss_op_ce_raw",
    72: "exec_loss_arg1_ce_raw",
    73: "exec_loss_arg2_ce_raw",
    74: "exec_loss_write_ce_raw",
    75: "exec_loss_div_pre_norm",
    76: "exec_loss_entropy_contribution",
    77: "exec_loss_gate_contribution",
    78: "exec_loss_stop_contribution",
    79: "exec_loss_op_contribution",
    80: "exec_loss_arg1_contribution",
    81: "exec_loss_arg2_contribution",
    82: "exec_loss_write_contribution",
    83: "exec_loss_div_contribution",
    84: "exec_loss_reconstructed",
    85: "exec_loss_residual",
    86: "exec_gate_accuracy",
    87: "exec_stop_accuracy",
    88: "exec_op_accuracy",
    89: "exec_arg1_accuracy",
    90: "exec_arg2_accuracy",
    91: "exec_write_accuracy",
    92: "exec_teacher_forced_ratio",
    93: "exec_loss_scalar_term_count",
}

DEAD_TELEMETRY_STREAM_INDICES = {26, 62, 63, 64, 65, 66, 67}
DEAD_TELEMETRY_STREAM_NAMES = {
    "mtp_loss_frac",
    "sb_atom_embed_rms",
    "mtp_loss",
    "selector_loss",
    "latent_preset_loss",
    "latent_preset_traj_loss",
    "latent_preset_delta_loss",
    "latent_preset_gate_loss",
}

TELEMETRY_IDENTIFIER_COLUMNS = {"stream_idx", "stream_name", "level", "stride"}
TELEMETRY_ZERO_REFERENCE_FIELDS = {
    "p",
    "delta_mu",
    "delta_sigma",
    "delta_bar",
    "delta_raw",
    "delta_bar_raw",
    "outlier_raw",
    "hw_cos_signed_mean",
    "hw_hbar_wbar_cos",
    "hw_h_dc_mean",
    "unigram_dir_cos_signed_mean",
    "rho_raw_avg_signed_dot",
    "exec_loss_residual",
}
TELEMETRY_UNIT_INTERVAL_FIELDS = {
    "r_out",
    "ell_out",
    "exec_active_ratio",
    "eb_inject_gate",
    "eb_read_gate_mean",
    "eb_loss_frac",
    "init_tie_cfg",
    "init_tie_ptrs_same",
    "init_tie_grads_same",
    "init_lm_owns_weights",
    "exec_gate_accuracy",
    "exec_stop_accuracy",
    "exec_op_accuracy",
    "exec_arg1_accuracy",
    "exec_arg2_accuracy",
    "exec_write_accuracy",
    "exec_teacher_forced_ratio",
}
TELEMETRY_FIELD_LABELS = {
    "raw_observation": "raw_observation",
    "mu": "mu (running mean)",
    "m2": "m2",
    "sigma": "sigma",
    "sigma_tilde": "sigma_tilde (normalized volatility)",
    "mu_a": "mu_a (anchor mean)",
    "sigma_a": "sigma_a (anchor std)",
    "delta_mu": "delta_mu (mean drift vs anchor)",
    "delta_sigma": "delta_sigma (volatility drift vs anchor)",
    "v_sigma": "v_sigma (meta-volatility)",
    "sigma_prev": "sigma_prev",
    "sigma_jump": "sigma_jump",
    "delta_bar": "delta_bar (directional trend)",
    "p": "p (directional bias)",
    "mu_prev": "mu_prev",
    "delta_raw": "delta_raw",
    "delta_bar_raw": "delta_bar_raw",
    "r_out": "r_out (outlier frequency)",
    "ell_out": "ell_out (outlier persistence)",
    "mu_ex": "mu_ex (excess severity)",
    "outlier_raw": "outlier_raw (raw cutoff gap)",
    "k_out": "k_out (adaptive threshold multiplier)",
    "c_out": "c_out (adaptive cutoff)",
    "step_count": "step_count",
}


def output_root(path):
    return os.path.splitext(path)[0]


def atlas_output_dir(path):
    return output_root(path) + "_atlas"


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_figure(fig, path, *, close=False):
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    if close:
        plt.close(fig)


def sanitize_filename_component(value):
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return cleaned or "stream"


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def pretty_field_name(field):
    return TELEMETRY_FIELD_LABELS.get(field, field)


def ordered_fields(columns):
    prioritized = [field for field in TELEMETRY_FIELD_LABELS if field in columns]
    remainder = sorted(field for field in columns if field not in TELEMETRY_FIELD_LABELS)
    return [*prioritized, *remainder]


def numeric_series(series):
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def looks_booleanish(series):
    values = numeric_series(series).dropna()
    if values.empty:
        return False
    rounded = np.round(values)
    return np.all(np.abs(values - rounded) < 1e-6) and set(rounded.astype(int).tolist()).issubset({0, 1})


def should_use_log_scale(series):
    values = numeric_series(series).dropna()
    if values.empty or (values <= 0).any():
        return False
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum <= 0 or maximum <= 0:
        return False
    return (maximum / minimum) >= 1000.0


def plottable_numeric_columns(df):
    return [
        column for column in df.columns
        if column not in TELEMETRY_IDENTIFIER_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def stream_sort_key(item):
    name, stream_df = item
    if "stream_idx" in stream_df.columns:
        indices = numeric_series(stream_df["stream_idx"]).dropna()
        if not indices.empty:
            return (0, int(indices.iloc[0]), name)
    return (1, name, name)


def sorted_stream_items(streams):
    return sorted(streams.items(), key=stream_sort_key)


def stream_display_name(name, stream_df):
    if "stream_idx" in stream_df.columns:
        indices = numeric_series(stream_df["stream_idx"]).dropna()
        if not indices.empty:
            return f"#{int(indices.iloc[0]):02d} {name}"
    return name


def stream_output_name(name, stream_df, fallback_index):
    if "stream_idx" in stream_df.columns:
        indices = numeric_series(stream_df["stream_idx"]).dropna()
        if not indices.empty:
            return f"{int(indices.iloc[0]):02d}_{sanitize_filename_component(name)}"
    return f"{fallback_index:02d}_{sanitize_filename_component(name)}"


def stream_color(stream_df, fallback_index):
    cmap = plt.get_cmap("tab20")
    if "stream_idx" in stream_df.columns:
        indices = numeric_series(stream_df["stream_idx"]).dropna()
        if not indices.empty:
            return cmap(int(indices.iloc[0]) % 20)
    return cmap(fallback_index % 20)


def plot_field_panel(ax, stream_df, field, *, color):
    x = stream_df.index
    series = numeric_series(stream_df[field])
    finite_values = series.dropna()

    if finite_values.empty:
        ax.text(0.5, 0.5, "no finite data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9)
        ax.set_title(pretty_field_name(field), fontsize=9)
        ax.grid(True, alpha=0.2)
        return

    if field == "raw_observation":
        plot_raw_and_smooth(ax, x, series,
                            color=color,
                            raw_label="raw_observation",
                            smooth_label="raw_observation sm20",
                            raw_alpha=0.22,
                            smooth_linewidth=1.5)
        if "mu" in stream_df.columns:
            ax.plot(x, numeric_series(stream_df["mu"]), linewidth=1.0,
                    linestyle="--", color="tab:orange", label="mu")
        if "mu_a" in stream_df.columns:
            ax.plot(x, numeric_series(stream_df["mu_a"]), linewidth=1.0,
                    linestyle=":", color="tab:green", label="mu_a")
        if "c_out" in stream_df.columns:
            ax.plot(x, numeric_series(stream_df["c_out"]), linewidth=0.9,
                    linestyle="-.", color="tab:red", label="c_out")
        if "mu" in stream_df.columns and "sigma" in stream_df.columns:
            mu = numeric_series(stream_df["mu"])
            sigma = numeric_series(stream_df["sigma"])
            ax.fill_between(x, (mu - sigma).to_numpy(), (mu + sigma).to_numpy(),
                            alpha=0.10, color=color)
        ax.legend(fontsize=6, ncol=2)
    elif looks_booleanish(series):
        ax.step(x, series, where="post", linewidth=1.4, color=color)
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.plot(x, series, alpha=0.22, linewidth=0.55, color=color)
        if finite_values.nunique() > 1:
            ax.plot(x, smooth(series), linewidth=1.35, color=color)
        else:
            ax.plot(x, series, linewidth=1.35, color=color)

    if field in TELEMETRY_ZERO_REFERENCE_FIELDS:
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.7)
    if field in TELEMETRY_UNIT_INTERVAL_FIELDS and not looks_booleanish(series):
        ax.set_ylim(-0.05, 1.05)
    if should_use_log_scale(series):
        ax.set_yscale("log")

    ax.set_title(pretty_field_name(field), fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_raw_stream_panel(ax, name, stream_df, *, color):
    x = stream_df.index
    raw = numeric_series(stream_df["raw_observation"])
    plot_raw_and_smooth(ax, x, raw,
                        color=color,
                        raw_label="raw_observation",
                        smooth_label="raw_observation sm20",
                        raw_alpha=0.22,
                        smooth_linewidth=1.5)

    if "mu" in stream_df.columns:
        ax.plot(x, numeric_series(stream_df["mu"]), linewidth=1.0,
                linestyle="--", color="tab:orange", label="mu")
    if "mu_a" in stream_df.columns:
        ax.plot(x, numeric_series(stream_df["mu_a"]), linewidth=1.0,
                linestyle=":", color="tab:green", label="mu_a")
    if "c_out" in stream_df.columns:
        ax.plot(x, numeric_series(stream_df["c_out"]), linewidth=0.9,
                linestyle="-.", color="tab:red", label="c_out")
    if "mu" in stream_df.columns and "sigma" in stream_df.columns:
        mu = numeric_series(stream_df["mu"])
        sigma = numeric_series(stream_df["sigma"])
        ax.fill_between(x, (mu - sigma).to_numpy(), (mu + sigma).to_numpy(),
                        alpha=0.10, color=color)
    if should_use_log_scale(raw):
        ax.set_yscale("log")

    ax.set_title(stream_display_name(name, stream_df), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)


def generate_raw_stream_atlas(path, streams):
    output_dir = ensure_output_dir(atlas_output_dir(path))
    stream_items = sorted_stream_items(streams)
    per_page = 12
    cols = 3
    rows = 4

    for page_index, stream_chunk in enumerate(chunked(stream_items, per_page), start=1):
        fig, axes = plt.subplots(rows, cols, figsize=(18, 16), constrained_layout=True)
        fig.suptitle("Telemetry raw_observation atlas", fontsize=14, fontweight="bold")
        axes = np.atleast_1d(axes).ravel()

        for local_index, ((name, stream_df), ax) in enumerate(zip(stream_chunk, axes)):
            color = stream_color(stream_df, local_index)
            plot_raw_stream_panel(ax, name, stream_df, color=color)
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("global_step")

        for ax in axes[len(stream_chunk):]:
            ax.axis("off")

        save_figure(fig, os.path.join(output_dir, f"raw_streams_page_{page_index:02d}.png"), close=True)

    return output_dir


def generate_stream_detail_pages(path, streams):
    output_dir = ensure_output_dir(atlas_output_dir(path))

    for stream_number, (name, stream_df) in enumerate(sorted_stream_items(streams), start=1):
        fields = ordered_fields(plottable_numeric_columns(stream_df))
        if not fields:
            continue

        cols = 4
        rows = math.ceil(len(fields) / cols)
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(18, max(8, rows * 3.0)),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        color = stream_color(stream_df, stream_number - 1)
        fig.suptitle(
            f"{stream_display_name(name, stream_df)} — all numeric telemetry fields",
            fontsize=14,
            fontweight="bold",
        )

        for ax, field in zip(axes, fields):
            plot_field_panel(ax, stream_df, field, color=color)
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("global_step")

        for ax in axes[len(fields):]:
            ax.axis("off")

        file_name = f"stream_detail_{stream_output_name(name, stream_df, stream_number)}.png"
        save_figure(fig, os.path.join(output_dir, file_name), close=True)

    return output_dir


def d_model_for_baseline():
    raw_value = os.environ.get("GRIM_TEXT_D_MODEL", "768")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("GRIM_TEXT_D_MODEL must be an integer") from exc
    if value <= 0:
        raise ValueError("GRIM_TEXT_D_MODEL must be positive")
    return value


def find_default_telemetry_csv():
    """Find the most relevant telemetry CSV in the training logs directory."""
    log_dir = os.path.join(os.path.dirname(__file__),
                           "resources", "models", "GRIM-text", "training", "logs")

    candidates = [
        os.path.join(log_dir, "latest_run.csv"),
        os.path.join(log_dir, "telemetary_latest.csv"),
        os.path.join(log_dir, "telemetry_latest.csv"),
        os.path.join(log_dir, "telemetry_*.csv"),
        os.path.join(log_dir, "telemetry_*.Csv"),
        os.path.join(log_dir, "telemetary_*.csv"),
        os.path.join(log_dir, "telemetary_*.Csv"),
    ]

    csvs = []
    for candidate in candidates:
        if any(ch in candidate for ch in "*?["):
            csvs.extend(glob.glob(candidate))
        elif os.path.exists(candidate):
            csvs.append(candidate)

    if not csvs:
        return None

    unique_csvs = list({os.path.realpath(path): path for path in csvs}.values())
    return next(
        (path for path in sorted(unique_csvs, key=os.path.getmtime, reverse=True) if telemetry_csv_looks_valid(path)),
        None,
    )


def telemetry_csv_looks_valid(path):
    try:
        if os.path.getsize(path) <= 0:
            return False
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except (OSError, StopIteration):
        return False

    cleaned_header = {
        TELEMETRY_COLUMN_ALIASES.get(column.lstrip("\ufeff").strip().lower(), column.lstrip("\ufeff").strip())
        for column in header
    }
    return REQUIRED_TELEMETRY_COLUMNS.issubset(cleaned_header)


def canonicalize_telemetry_columns(df):
    rename_map = {}
    for column in df.columns:
        cleaned = column.lstrip("\ufeff").strip()
        cleaned = TELEMETRY_COLUMN_ALIASES.get(cleaned.lower(), cleaned)
        if cleaned != column:
            rename_map[column] = cleaned

    if rename_map:
        df = df.rename(columns=rename_map)

    if missing := sorted(REQUIRED_TELEMETRY_COLUMNS.difference(df.columns)):
        raise KeyError(
            "Telemetry CSV is missing required columns: "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(map(str, df.columns))
        )

    return df


def stream_index_to_name(value):
    if pd.isna(value):
        return None
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"Telemetry stream_idx is not an integer: {value}")
    stream_idx = int(numeric)
    return TELEMETRY_STREAM_NAMES_BY_INDEX.get(stream_idx)


def is_stream_name_compatible(stream_idx, current_name, expected_name):
    return current_name == expected_name


def repair_telemetry_stream_names(df):
    """Fill new/legacy CSV stream names from stream_idx when logger wrote unknown."""
    if "stream_idx" not in df.columns:
        return df

    repaired = df.copy()
    indices = pd.to_numeric(repaired["stream_idx"], errors="coerce")
    indexed_names = indices.map(stream_index_to_name)
    current_names = repaired["stream_name"].astype(str).str.strip()
    lower_names = current_names.str.lower()

    missing_names = indexed_names.notna() & ((current_names == "") | (lower_names == "unknown"))
    repaired.loc[missing_names, "stream_name"] = indexed_names[missing_names]

    compatibility_ok = pd.Series([
        is_stream_name_compatible(int(idx), current, expected)
        if pd.notna(idx) and expected is not None else True
        for idx, current, expected in zip(indices, current_names, indexed_names)
    ], index=repaired.index)

    mismatched_names = indexed_names.notna() & ~missing_names & ~compatibility_ok
    if mismatched_names.any():
        examples = repaired.loc[mismatched_names, ["stream_idx", "stream_name"]].head(8)
        expected = indexed_names[mismatched_names].head(8).tolist()
        details = "; ".join(
            f"idx {row.stream_idx}: csv='{row.stream_name}', expected='{expected[i]}'"
            for i, row in enumerate(examples.itertuples(index=False))
        )
        raise ValueError(f"Telemetry CSV stream_idx/name mismatch: {details}")

    return repaired


def read_telemetry_csv(path):
    if os.path.getsize(path) <= 0:
        raise ValueError(f"Telemetry CSV is empty: {path}")
    df = pd.read_csv(path)
    df = canonicalize_telemetry_columns(df)
    df = repair_telemetry_stream_names(df)
    dead_rows = df["stream_name"].isin(DEAD_TELEMETRY_STREAM_NAMES)
    if "stream_idx" in df.columns:
        indices = pd.to_numeric(df["stream_idx"], errors="coerce")
        dead_rows |= indices.isin(DEAD_TELEMETRY_STREAM_INDICES)
    return df[~dead_rows].copy()


def load_telemetry(path):
    df = read_telemetry_csv(path)
    # Keep only level-0 (raw per-step) observations
    df = df[df["level"] == 0].copy()
    return df

def pivot_streams(df):
    """Pivot so each stream becomes a column keyed by global_step."""
    streams = {}
    for name in df["stream_name"].unique():
        sub = df[df["stream_name"] == name].sort_values(by="global_step")
        streams[name] = sub.set_index("global_step")
    return streams

def smooth(series, window=20):
    return series.rolling(window, min_periods=1).mean()


def plot_raw_and_smooth(ax, x, series, window=20, *, color,
            raw_label="_nolegend_", smooth_label=None,
            raw_alpha=0.2, raw_linewidth=0.5,
            smooth_alpha=1.0, smooth_linewidth=1.5,
            raw_linestyle="-", smooth_linestyle="-"):
    ax.plot(x, series,
        alpha=raw_alpha,
        linewidth=raw_linewidth,
        color=color,
        linestyle=raw_linestyle,
        label=raw_label)
    ax.plot(x, smooth(series, window),
        alpha=smooth_alpha,
        linewidth=smooth_linewidth,
        color=color,
        linestyle=smooth_linestyle,
        label=smooth_label)

def main():
    # --- find CSV ---
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_default_telemetry_csv()
        if path is None:
            print("No telemetry CSV found. Pass path as argument.")
            sys.exit(1)
        print(f"Using: {path}")

    df = load_telemetry(path)
    streams = pivot_streams(df)

    loss = streams.get("loss")
    grad_mean = streams.get("grad_norm_mean")
    grad_max = streams.get("grad_norm_max")
    lr = streams.get("learning_rate")

    # --- Figure 1: Core training curves ---
    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    fig.suptitle("GRIM-text Telemetry", fontsize=14, fontweight="bold")
    gs = GridSpec(3, 2, figure=fig)

    # 1a) Loss raw + smoothed
    ax = fig.add_subplot(gs[0, 0])
    if loss is not None:
        ax.plot(loss.index, loss["raw_observation"], alpha=0.3, linewidth=0.5, label="raw")
        ax.plot(loss.index, smooth(loss["raw_observation"]), linewidth=1.5, label="smooth-20")
        ax.plot(loss.index, loss["mu"], linewidth=1, linestyle="--", label="μ (running)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1b) Loss delta from consecutive raw observations
    ax = fig.add_subplot(gs[0, 1])
    if loss is not None:
        delta = loss["raw_observation"].diff()
        plot_raw_and_smooth(ax, loss.index, delta, window=50,
                            color="tab:red",
                            raw_label="raw Δ loss",
                            smooth_label="smooth Δ loss",
                            smooth_linewidth=1)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Δ Loss")
    ax.set_title("Raw loss delta (x_t - x_{t-1})")
    ax.grid(True, alpha=0.3)

    # 2a) Gradient norms
    ax = fig.add_subplot(gs[1, 0])
    if grad_mean is not None:
        ax.plot(grad_mean.index, grad_mean["raw_observation"], alpha=0.3, linewidth=0.5, label="mean (raw)")
        ax.plot(grad_mean.index, smooth(grad_mean["raw_observation"]), linewidth=1.5, label="mean (smooth)")
    if grad_max is not None:
        plot_raw_and_smooth(ax, grad_max.index, grad_max["raw_observation"],
                            color="tab:orange",
                            raw_label="max (raw)",
                            smooth_label="max (smooth)",
                            raw_alpha=0.2,
                            smooth_linewidth=1,
                            smooth_linestyle="--")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient norms")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # 2b) Learning rate schedule
    ax = fig.add_subplot(gs[1, 1])
    if lr is not None:
        ax.plot(lr.index, lr["raw_observation"], linewidth=1.5, color="tab:green")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR schedule")
    ax.grid(True, alpha=0.3)

    # 3a) Loss vs grad_norm scatter (relationship)
    ax = fig.add_subplot(gs[2, 0])
    if loss is not None and grad_mean is not None:
        common = loss.index.intersection(grad_mean.index)
        l = loss.loc[common, "raw_observation"].values
        g = grad_mean.loc[common, "raw_observation"].values
        colors = np.arange(len(common))
        sc = ax.scatter(g, l, c=colors, cmap="viridis", s=4, alpha=0.5)
        plt.colorbar(sc, ax=ax, label="step")
    ax.set_xlabel("Grad Norm Mean")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Grad Norm (color=step)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # 3b) TelemetryLattice directional bias (p) and adaptive threshold (k_out)
    ax = fig.add_subplot(gs[2, 1])
    if loss is not None:
        ax2 = ax.twinx()
        plot_raw_and_smooth(ax, loss.index, loss["p"], window=10,
                            color="tab:orange",
                            raw_label="p (raw)",
                            smooth_label="p (directional bias)",
                            smooth_linewidth=1)
        plot_raw_and_smooth(ax2, loss.index, loss["k_out"], window=10,
                            color="tab:purple",
                            raw_label="k_out (raw)",
                            smooth_label="k_out (adaptive threshold)",
                            smooth_linewidth=1)
        ax.set_ylabel("p (directional bias)", color="tab:orange")
        ax2.set_ylabel("k_out (adaptive threshold)", color="tab:purple")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Lattice signals: directional bias & adaptive threshold")
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.splitext(path)[0] + "_patterns.png", dpi=150)
    print(f"Saved: {os.path.splitext(path)[0]}_patterns.png")

    # --- Figure 2: Rho & hidden-state streams ---
    rho_final = streams.get("rho_final")
    rho_growth = streams.get("rho_growth")
    rho_worst = streams.get("rho_worst_delta")
    rho_atom_only = streams.get("rho_atom_only")
    rho_nonatom_only = streams.get("rho_nonatom_only")
    h_rms = streams.get("h_rms_growth")
    tpb = streams.get("tokens_per_batch")
    rho_plot_streams = [
        ("rho_final", rho_final, "tab:blue"),
        ("rho_growth", rho_growth, "tab:orange"),
        ("rho_worst_delta", rho_worst, "tab:red"),
        ("rho_atom_only", rho_atom_only, "tab:green"),
        ("rho_nonatom_only", rho_nonatom_only, "tab:purple"),
    ]

    fig2 = plt.figure(figsize=(16, 14), constrained_layout=True)
    fig2.suptitle("GRIM-text Telemetry — Rho & Hidden State", fontsize=14, fontweight="bold")
    gs2 = GridSpec(3, 2, figure=fig2)

    # 2-1a) Rho streams overlaid
    ax = fig2.add_subplot(gs2[0, 0])
    for name, s, color in rho_plot_streams:
        if s is not None:
            ax.plot(s.index, s["raw_observation"], alpha=0.2, linewidth=0.5, color=color)
            ax.plot(s.index, smooth(s["raw_observation"]), linewidth=1.5, label=name, color=color)
    ax.set_ylabel("Rho")
    ax.set_title("Rho metrics")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2-1b) Rho running mean (mu) comparison
    ax = fig2.add_subplot(gs2[0, 1])
    for name, s, color in rho_plot_streams:
        if s is not None:
            ax.plot(s.index, s["mu"], linewidth=1.2, label=f"{name} μ", color=color)
            sig = s["sigma"].values.astype(float)
            mu = s["mu"].values
            ax.fill_between(s.index, mu - sig, mu + sig, alpha=0.1, color=color)
    ax.set_ylabel("μ ± σ")
    ax.set_title("Rho running statistics")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2-2a) h_rms_growth
    ax = fig2.add_subplot(gs2[1, 0])
    if h_rms is not None:
        ax.plot(h_rms.index, h_rms["raw_observation"], alpha=0.2, linewidth=0.5, color="tab:purple")
        ax.plot(h_rms.index, smooth(h_rms["raw_observation"]), linewidth=1.5, label="smooth-20", color="tab:purple")
        ax.plot(h_rms.index, h_rms["mu"], linewidth=1, linestyle="--", label="μ", color="tab:cyan")
    ax.set_ylabel("h_rms_growth")
    ax.set_title("Hidden state RMS growth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2-2b) tokens_per_batch
    ax = fig2.add_subplot(gs2[1, 1])
    if tpb is not None:
        ax.plot(tpb.index, tpb["raw_observation"], linewidth=1.5, color="tab:green")
    ax.set_ylabel("Tokens per batch")
    ax.set_title("Tokens per batch")
    ax.grid(True, alpha=0.3)

    # 2-3a) Rho directional bias (p) across streams
    ax = fig2.add_subplot(gs2[2, 0])
    for name, s, color in [*rho_plot_streams, ("h_rms_growth", h_rms, "tab:brown")]:
        if s is not None:
            plot_raw_and_smooth(ax, s.index, s["p"], window=10,
                                color=color,
                                raw_label="_nolegend_",
                                smooth_label=f"{name}",
                                raw_alpha=0.12,
                                smooth_linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("p (directional bias)")
    ax.set_title("Directional bias across streams")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2-3b) Adaptive threshold (k_out) across streams
    ax = fig2.add_subplot(gs2[2, 1])
    for name, s, color in [("loss", loss, "tab:blue"),
                            ("rho_final", rho_final, "tab:orange"),
                            ("grad_norm_mean", grad_mean, "tab:green"),
                            ("h_rms_growth", h_rms, "tab:purple")]:
        if s is not None:
            plot_raw_and_smooth(ax, s.index, s["k_out"], window=10,
                                color=color,
                                raw_label="_nolegend_",
                                smooth_label=f"{name}",
                                raw_alpha=0.12,
                                smooth_linewidth=1.2)
    ax.set_ylabel("k_out (adaptive threshold)")
    ax.set_title("Adaptive threshold across streams")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig2.savefig(os.path.splitext(path)[0] + "_rho_hidden.png", dpi=150)
    print(f"Saved: {os.path.splitext(path)[0]}_rho_hidden.png")

    # --- Figure 3: Lattice internals for loss stream ---
    fig3 = plt.figure(figsize=(16, 18), constrained_layout=True)
    fig3.suptitle("GRIM-text Telemetry — Lattice Internals (loss)", fontsize=14, fontweight="bold")
    gs3 = GridSpec(4, 2, figure=fig3)

    # 3-1a) sigma & sigma_tilde
    ax = fig3.add_subplot(gs3[0, 0])
    if loss is not None:
        ax.plot(loss.index, loss["sigma"], linewidth=1.2, label="σ", color="tab:blue")
        ax.plot(loss.index, loss["sigma_tilde"], linewidth=1.2, label="σ̃ (normalized)", color="tab:orange")
        ax.plot(loss.index, loss["sigma_a"], linewidth=1, linestyle="--", label="σ_a (anchor)", color="tab:green")
    ax.set_ylabel("σ values")
    ax.set_title("Volatility: σ, σ̃, σ_a")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3-1b) delta_mu & delta_sigma
    ax = fig3.add_subplot(gs3[0, 1])
    if loss is not None:
        plot_raw_and_smooth(ax, loss.index, loss["delta_mu"], window=20,
                            color="tab:blue",
                            raw_label="Δμ raw",
                            smooth_label="Δμ",
                            smooth_linewidth=1.2)
        ax2 = ax.twinx()
        plot_raw_and_smooth(ax2, loss.index, loss["delta_sigma"], window=20,
                            color="tab:red",
                            raw_label="Δσ raw",
                            smooth_label="Δσ",
                            smooth_linewidth=1.2)
        ax.set_ylabel("Δμ", color="tab:blue")
        ax2.set_ylabel("Δσ", color="tab:red")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Rate of change: Δμ, Δσ")
    ax.grid(True, alpha=0.3)

    # 3-2a) v_sigma (meta-volatility)
    ax = fig3.add_subplot(gs3[1, 0])
    if loss is not None:
        plot_raw_and_smooth(ax, loss.index, loss["v_sigma"], window=20,
                            color="tab:red",
                            raw_label="v_σ raw",
                            smooth_label="v_σ")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("v_σ")
    ax.set_title("Meta-volatility (v_sigma)")
    ax.grid(True, alpha=0.3)

    # 3-2b) delta_bar (normalized trend EMA)
    ax = fig3.add_subplot(gs3[1, 1])
    if loss is not None:
        ax.plot(loss.index, loss["delta_bar"], linewidth=1.2, color="tab:cyan")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("δ̄")
    ax.set_title("Normalized trend EMA (delta_bar)")
    ax.grid(True, alpha=0.3)

    # 3-3a) r_out (outlier frequency) & ell_out (outlier persistence)
    ax = fig3.add_subplot(gs3[2, 0])
    if loss is not None:
        plot_raw_and_smooth(ax, loss.index, loss["r_out"], window=10,
                            color="tab:blue",
                            raw_label="r raw",
                            smooth_label="r_out (outlier freq)",
                            smooth_linewidth=1.2)
        ax2 = ax.twinx()
        plot_raw_and_smooth(ax2, loss.index, loss["ell_out"], window=10,
                            color="tab:orange",
                            raw_label="ℓ raw",
                            smooth_label="ell_out (persistence)",
                            smooth_linewidth=1.2)
        ax.set_ylabel("r_out (outlier freq)", color="tab:blue")
        ax2.set_ylabel("ell_out (persistence)", color="tab:orange")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Lattice outputs: outlier frequency & persistence")
    ax.grid(True, alpha=0.3)

    # 3-3b) mu_ex (excess severity) & c_out (adaptive cutoff)
    ax = fig3.add_subplot(gs3[2, 1])
    if loss is not None:
        plot_raw_and_smooth(ax, loss.index, loss["mu_ex"], window=10,
                            color="tab:purple",
                            raw_label="μ_ex raw",
                            smooth_label="mu_ex (excess severity)",
                            smooth_linewidth=1.2)
        ax2 = ax.twinx()
        plot_raw_and_smooth(ax2, loss.index, loss["c_out"], window=10,
                            color="tab:green",
                            raw_label="c_out raw",
                            smooth_label="c_out (adaptive cutoff)",
                            smooth_linewidth=1.2)
        ax.set_ylabel("μ_ex", color="tab:purple")
        ax2.set_ylabel("c_out (adaptive cutoff)", color="tab:green")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Lattice outputs: excess severity & adaptive cutoff")
    ax.grid(True, alpha=0.3)

    # 3-4a) mu_a (anchor mean) vs mu (running mean)
    ax = fig3.add_subplot(gs3[3, 0])
    if loss is not None:
        ax.plot(loss.index, loss["mu"], linewidth=1.2, label="μ (running)", color="tab:blue")
        ax.plot(loss.index, loss["mu_a"], linewidth=1.2, label="μ_a (anchor)", color="tab:orange")
        ax.plot(loss.index, loss["mu_prev"], linewidth=0.8, linestyle="--", label="μ_prev", color="tab:gray")
    ax.set_ylabel("Mean estimates")
    ax.set_title("Mean tracking: μ, μ_a, μ_prev")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3-4b) sigma_prev & m2
    ax = fig3.add_subplot(gs3[3, 1])
    if loss is not None:
        ax.plot(loss.index, loss["sigma_prev"], linewidth=1.2, label="σ_prev", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(loss.index, loss["m2"], linewidth=1.2, label="m2 (variance·n)", color="tab:red")
        ax.set_ylabel("σ_prev", color="tab:blue")
        ax2.set_ylabel("m2", color="tab:red")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Variance tracking: σ_prev & m2")
    ax.grid(True, alpha=0.3)

    fig3.savefig(os.path.splitext(path)[0] + "_lattice.png", dpi=150)
    print(f"Saved: {os.path.splitext(path)[0]}_lattice.png")

    # --- Figure 4: Multi-level lattice view ---
    df_all = read_telemetry_csv(path)
    levels = sorted(df_all["level"].unique())
    if len(levels) > 1:
        fig4, axes = plt.subplots(len(levels), 1, figsize=(14, 3 * len(levels)),
                                   sharex=True, constrained_layout=True)
        fig4.suptitle("Loss across TelemetryLattice levels", fontsize=13, fontweight="bold")
        if len(levels) == 1:
            axes = [axes]
        for i, lvl in enumerate(levels):
            sub = df_all[(df_all["level"] == lvl) & (df_all["stream_name"] == "loss")].copy()
            sub = sub.iloc[np.argsort(sub["global_step"].to_numpy())]
            stride = sub["stride"].iloc[0] if len(sub) > 0 else "?"
            axes[i].plot(sub["global_step"], sub["mu"], linewidth=1.2, label="μ")
            if "sigma" in sub.columns:
                mu = sub["mu"].values
                sig = sub["sigma"].values.astype(float)
                axes[i].fill_between(sub["global_step"], mu - sig, mu + sig, alpha=0.15, label="±σ")
            axes[i].set_ylabel(f"L{lvl} (stride {stride})")
            axes[i].legend(fontsize=7)
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel("global_step")
        fig4.savefig(os.path.splitext(path)[0] + "_levels.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_levels.png")

    # --- Figure 5: Adam Warmup Causation ---
    adam_bc2 = streams.get("adam_bc2_v_convergence")
    adam_sig_dom = streams.get("adam_signal_dominance")
    adam_cum_disp = streams.get("adam_cumulative_disp")
    adam_disrupt = streams.get("adam_disruption_emb")
    adam_inv_bc2 = streams.get("adam_inv_bc2_amp")

    has_adam = any(s is not None for s in [adam_bc2, adam_sig_dom, adam_cum_disp, adam_disrupt, adam_inv_bc2])
    if has_adam:
        fig5 = plt.figure(figsize=(16, 18), constrained_layout=True)
        fig5.suptitle("GRIM-text Telemetry — Adam Warmup Causation", fontsize=14, fontweight="bold")
        gs5 = GridSpec(3, 2, figure=fig5)

        # ln(V) reference for loss plots
        # Detect vocab size from loss data if available
        ln_v = math.log(10000)  # ln(vocab_size)

        # 5-1a) Loss + bc2 overlay — THE causal relationship
        ax = fig5.add_subplot(gs5[0, 0])
        if loss is not None:
            ax.plot(loss.index, loss["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:blue")
            ax.plot(loss.index, smooth(loss["raw_observation"]), linewidth=1.5, color="tab:blue", label="loss")
            ax.axhline(ln_v, color="gray", linewidth=1, linestyle=":", label=f"ln(V)={ln_v:.2f}")
        ax.set_ylabel("Loss", color="tab:blue")
        if adam_bc2 is not None:
            ax2 = ax.twinx()
            ax2.plot(adam_bc2.index, adam_bc2["raw_observation"], linewidth=2, color="tab:red", label="bc₂ (v convergence)")
            ax2.axhline(0.5, color="tab:red", linewidth=0.8, linestyle="--", alpha=0.5)
            ax2.set_ylabel("bc₂ = 1 − β₂^(t+1)", color="tab:red")
            ax2.set_ylim(0, 1.05)
            ax2.legend(loc="center right", fontsize=8)
        ax.axvline(693, color="tab:red", linewidth=0.8, linestyle="--", alpha=0.4, label="β₂ half-life (693)")
        ax.set_title("Loss vs β₂ Convergence (causal overlay)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 5-1b) Signal dominance — predicts loss peak
        ax = fig5.add_subplot(gs5[0, 1])
        if adam_sig_dom is not None:
            sig_dom = adam_sig_dom
            x = sig_dom.index.to_numpy()
            vals = sig_dom["raw_observation"].to_numpy()
            ax.plot(x, vals, linewidth=1.5, color="tab:green", label="signal dominance")
            ax.axhline(1.0, color="tab:red", linewidth=1.5, linestyle="--", label="crossover = 1.0")
            ax.fill_between(x, 0, 1, where=vals < 1, alpha=0.1, color="tab:red", label="destroying (< 1)")
            ax.fill_between(x, 1, vals, where=vals >= 1, alpha=0.1, color="tab:green", label="learning (≥ 1)")
        ax.set_ylabel("bc₂ / (1 − bc₂)")
        ax.set_title("Signal Dominance (>1 = learning, <1 = destroying)")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 5-2a) Cumulative displacement + LR
        ax = fig5.add_subplot(gs5[1, 0])
        if adam_cum_disp is not None:
            ax.plot(adam_cum_disp.index, adam_cum_disp["raw_observation"], linewidth=1.5, color="tab:purple", label="Σlr(t)")
        ax.set_ylabel("Cumulative Σlr(t)", color="tab:purple")
        if lr is not None:
            ax2 = ax.twinx()
            ax2.plot(lr.index, lr["raw_observation"], linewidth=1, color="tab:green", alpha=0.6, label="lr(t)")
            ax2.set_ylabel("LR", color="tab:green")
            ax2.legend(loc="center right", fontsize=8)
        ax.set_title("Cumulative Displacement & LR Schedule")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 5-2b) Disruption in Xavier units
        ax = fig5.add_subplot(gs5[1, 1])
        if adam_disrupt is not None:
            plot_raw_and_smooth(ax, adam_disrupt.index, adam_disrupt["raw_observation"],
                                color="tab:orange",
                                raw_label="displacement / Xavier scale (raw)",
                                smooth_label="displacement / Xavier scale (smooth-20)",
                                raw_alpha=0.22,
                                smooth_linewidth=1.5)
            ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label="1× Xavier")
        ax.set_ylabel("Xavier Embedding Scale Units")
        ax.set_title("Weight Disruption vs Xavier Init Scale")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 5-3a) inv_bc2 amplification (log scale)
        ax = fig5.add_subplot(gs5[2, 0])
        if adam_inv_bc2 is not None:
            ax.plot(adam_inv_bc2.index, adam_inv_bc2["raw_observation"], linewidth=1.5, color="tab:brown", label="1/(1−β₂^(t+1))")
        ax.set_ylabel("v Bias Correction Amplification")
        ax.set_title("Adam v-Estimate Inflation Factor")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 5-3b) Combined: loss + disruption + signal_dominance (normalized)
        ax = fig5.add_subplot(gs5[2, 1])
        if loss is not None:
            plot_raw_and_smooth(ax, loss.index, loss["raw_observation"],
                                color="tab:blue",
                                raw_label="loss (raw)",
                                smooth_label="loss (smoothed)")
            ax.axhline(ln_v, color="gray", linewidth=1, linestyle=":", alpha=0.5)
        ax.set_ylabel("Loss", color="tab:blue")
        if adam_disrupt is not None and adam_bc2 is not None:
            ax2 = ax.twinx()
            ax2.plot(adam_disrupt.index, adam_disrupt["raw_observation"], linewidth=1.2, color="tab:orange", alpha=0.8, label="disruption (Xavier×)")
            ax2.plot(adam_bc2.index, adam_bc2["raw_observation"] * adam_disrupt["raw_observation"].max(), linewidth=1.2, color="tab:red", alpha=0.8, linestyle="--", label="bc₂ (scaled)")
            ax2.set_ylabel("Disruption / Scaled bc₂", color="tab:orange")
            ax2.legend(loc="center right", fontsize=8)
        ax.axvline(693, color="tab:red", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_title("Combined Causation Overview")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig5.savefig(os.path.splitext(path)[0] + "_adam_causation.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_adam_causation.png")
    else:
        missing = [
            name for name, stream in [
                ("adam_bc2_v_convergence", adam_bc2),
                ("adam_signal_dominance", adam_sig_dom),
                ("adam_cumulative_disp", adam_cum_disp),
                ("adam_disruption_emb", adam_disrupt),
                ("adam_inv_bc2_amp", adam_inv_bc2),
            ] if stream is None
        ]
        print("Adam causation figure skipped: missing streams:", ", ".join(missing))

    # --- Figure 6: Execution Block Health ---
    exec_grad_norm = streams.get("exec_grad_norm")
    exec_grad_ratio = streams.get("exec_grad_ratio")
    exec_sel_ent = streams.get("exec_selection_entropy")
    exec_op_ent = streams.get("exec_op_entropy")
    exec_div_clamp = streams.get("exec_div_clamp_rate")
    exec_max_pw = streams.get("exec_max_p_write")
    exec_active = streams.get("exec_active_ratio")

    has_exec = any(s is not None for s in [exec_grad_norm, exec_grad_ratio, exec_sel_ent,
                                            exec_op_ent, exec_div_clamp, exec_max_pw, exec_active])
    if has_exec:
        fig6 = plt.figure(figsize=(16, 14), constrained_layout=True)
        fig6.suptitle("GRIM-text Telemetry — Execution Block Health", fontsize=14, fontweight="bold")
        gs6 = GridSpec(3, 2, figure=fig6)

        # 6-1a) Exec Grad Norm
        ax = fig6.add_subplot(gs6[0, 0])
        if exec_grad_norm is not None:
            ax.plot(exec_grad_norm.index, exec_grad_norm["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:blue")
            ax.plot(exec_grad_norm.index, smooth(exec_grad_norm["raw_observation"]), linewidth=1.5, color="tab:blue", label="exec grad RMS")
        if grad_mean is not None:
            plot_raw_and_smooth(ax, grad_mean.index, grad_mean["raw_observation"],
                                color="tab:gray",
                                raw_label="_nolegend_",
                                smooth_label="total grad RMS",
                                raw_alpha=0.12,
                                smooth_alpha=0.6,
                                smooth_linewidth=1,
                                smooth_linestyle="--")
        ax.set_ylabel("Gradient RMS")
        ax.set_title("Exec Block Gradient Norm")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 6-1b) Exec Grad Ratio (exec/encoder)
        ax = fig6.add_subplot(gs6[0, 1])
        if exec_grad_ratio is not None:
            ax.plot(exec_grad_ratio.index, exec_grad_ratio["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:red")
            ax.plot(exec_grad_ratio.index, smooth(exec_grad_ratio["raw_observation"]), linewidth=1.5, color="tab:red", label="exec/encoder ratio")
            ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label="parity (1.0)")
        ax.set_ylabel("Ratio")
        ax.set_title("Exec/Encoder Gradient Ratio (dying → 0)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 6-2a) Selection Entropy (sharpening = learning)
        ax = fig6.add_subplot(gs6[1, 0])
        if exec_sel_ent is not None:
            ax.plot(exec_sel_ent.index, exec_sel_ent["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:purple")
            ax.plot(exec_sel_ent.index, smooth(exec_sel_ent["raw_observation"]), linewidth=1.5, color="tab:purple", label="selection entropy")
        if exec_op_ent is not None:
            plot_raw_and_smooth(ax, exec_op_ent.index, exec_op_ent["raw_observation"],
                                color="tab:orange",
                                raw_label="_nolegend_",
                                smooth_label="op entropy",
                                raw_alpha=0.15,
                                smooth_linewidth=1.2,
                                smooth_linestyle="--")
        ax.set_ylabel("Entropy (nats)")
        ax.set_title("Selection Entropy (↓ = sharpening)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 6-2b) Op Entropy momentum (p) — is it trending down?
        ax = fig6.add_subplot(gs6[1, 1])
        if exec_sel_ent is not None:
            plot_raw_and_smooth(ax, exec_sel_ent.index, exec_sel_ent["p"], window=10,
                                color="tab:purple",
                                raw_label="_nolegend_",
                                smooth_label="selection entropy p",
                                raw_alpha=0.15,
                                smooth_linewidth=1.2)
        if exec_op_ent is not None:
            plot_raw_and_smooth(ax, exec_op_ent.index, exec_op_ent["p"], window=10,
                                color="tab:orange",
                                raw_label="_nolegend_",
                                smooth_label="op entropy p",
                                raw_alpha=0.15,
                                smooth_linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("p (directional bias)")
        ax.set_title("Entropy directional bias (< 0 = sharpening)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 6-3a) Div Clamp Rate + Max P Write
        ax = fig6.add_subplot(gs6[2, 0])
        if exec_div_clamp is not None:
            plot_raw_and_smooth(ax, exec_div_clamp.index, exec_div_clamp["raw_observation"],
                                color="tab:red",
                                raw_label="div clamp raw",
                                smooth_label="div clamp rate")
        ax.set_ylabel("Rate", color="tab:red")
        if exec_max_pw is not None:
            ax2 = ax.twinx()
            plot_raw_and_smooth(ax2, exec_max_pw.index, exec_max_pw["raw_observation"],
                                color="tab:green",
                                raw_label="max p(write) raw",
                                smooth_label="max p(write)")
            ax2.set_ylabel("max p(write)", color="tab:green")
            ax2.legend(loc="center right", fontsize=8)
        ax.set_title("Div Clamp Rate & Write Concentration")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 6-3b) Active Rows Ratio
        ax = fig6.add_subplot(gs6[2, 1])
        if exec_active is not None:
            ax.plot(exec_active.index, exec_active["raw_observation"], linewidth=1.5, color="tab:cyan", label="active ratio")
            ax.fill_between(exec_active.index, 0, exec_active["raw_observation"].values, alpha=0.15, color="tab:cyan")
        ax.set_ylabel("Fraction")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Execution-Active Rows / Batch Size")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig6.savefig(os.path.splitext(path)[0] + "_exec_block.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_exec_block.png")
    else:
        print("Execution block figure skipped: no exec streams in CSV")

    # --- Figure 6b: Execution Loss Objective Decomposition ---
    exec_loss = streams.get("execution_loss")
    exec_loss_reconstructed = streams.get("exec_loss_reconstructed")
    exec_loss_residual = streams.get("exec_loss_residual")
    exec_raw_ce_streams = [
        ("gate", streams.get("exec_loss_gate_ce_raw"), "tab:blue"),
        ("stop", streams.get("exec_loss_stop_ce_raw"), "tab:orange"),
        ("op", streams.get("exec_loss_op_ce_raw"), "tab:green"),
        ("arg1", streams.get("exec_loss_arg1_ce_raw"), "tab:red"),
        ("arg2", streams.get("exec_loss_arg2_ce_raw"), "tab:purple"),
        ("write", streams.get("exec_loss_write_ce_raw"), "tab:brown"),
    ]
    exec_contribution_streams = [
        ("gate", streams.get("exec_loss_gate_contribution"), "tab:blue"),
        ("stop", streams.get("exec_loss_stop_contribution"), "tab:orange"),
        ("op", streams.get("exec_loss_op_contribution"), "tab:green"),
        ("arg1", streams.get("exec_loss_arg1_contribution"), "tab:red"),
        ("arg2", streams.get("exec_loss_arg2_contribution"), "tab:purple"),
        ("write", streams.get("exec_loss_write_contribution"), "tab:brown"),
        ("division", streams.get("exec_loss_div_contribution"), "tab:pink"),
        ("entropy", streams.get("exec_loss_entropy_contribution"), "tab:gray"),
    ]
    exec_accuracy_streams = [
        ("gate", streams.get("exec_gate_accuracy"), "tab:blue"),
        ("stop", streams.get("exec_stop_accuracy"), "tab:orange"),
        ("op", streams.get("exec_op_accuracy"), "tab:green"),
        ("arg1", streams.get("exec_arg1_accuracy"), "tab:red"),
        ("arg2", streams.get("exec_arg2_accuracy"), "tab:purple"),
        ("write", streams.get("exec_write_accuracy"), "tab:brown"),
    ]
    exec_div_pre_norm = streams.get("exec_loss_div_pre_norm")
    exec_div_contribution = streams.get("exec_loss_div_contribution")
    exec_entropy_contribution = streams.get("exec_loss_entropy_contribution")
    exec_teacher_forced = streams.get("exec_teacher_forced_ratio")
    exec_scalar_terms = streams.get("exec_loss_scalar_term_count")

    exec_loss_diag_streams = [
        exec_loss_reconstructed,
        exec_loss_residual,
        exec_div_pre_norm,
        exec_teacher_forced,
        exec_scalar_terms,
        *(stream for _, stream, _ in exec_raw_ce_streams),
        *(stream for _, stream, _ in exec_contribution_streams),
        *(stream for _, stream, _ in exec_accuracy_streams),
    ]
    if any(stream is not None for stream in exec_loss_diag_streams):
        fig6b = plt.figure(figsize=(18, 16), constrained_layout=True)
        fig6b.suptitle(
            "GRIM-text Telemetry - Execution Loss Objective Decomposition",
            fontsize=14,
            fontweight="bold",
        )
        gs6b = GridSpec(3, 2, figure=fig6b)

        # Aggregate loss must close against the independent reconstruction.
        ax = fig6b.add_subplot(gs6b[0, 0])
        for label, stream, color, linestyle in [
            ("aggregate execution_loss", exec_loss, "tab:blue", "-"),
            ("reconstructed", exec_loss_reconstructed, "tab:orange", "--"),
        ]:
            if stream is not None:
                plot_raw_and_smooth(
                    ax, stream.index, stream["raw_observation"],
                    color=color,
                    raw_label="_nolegend_",
                    smooth_label=label,
                    raw_alpha=0.10,
                    smooth_linewidth=1.5,
                    smooth_linestyle=linestyle,
                )
        ax.set_ylabel("Loss")
        ax.set_title("Aggregate vs Independent Reconstruction")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8, loc="upper left")
        if exec_loss_residual is not None:
            ax2 = ax.twinx()
            plot_raw_and_smooth(
                ax2, exec_loss_residual.index, exec_loss_residual["raw_observation"],
                color="tab:red",
                raw_label="_nolegend_",
                smooth_label="closure residual",
                raw_alpha=0.12,
                smooth_linewidth=1.2,
            )
            ax2.axhline(0.0, color="tab:red", linewidth=0.6, linestyle=":", alpha=0.6)
            ax2.set_ylabel("Residual", color="tab:red")
            ax2.legend(fontsize=8, loc="upper right")

        # Unweighted CE shows which prediction heads are actually learning.
        ax = fig6b.add_subplot(gs6b[0, 1])
        for label, stream, color in exec_raw_ce_streams:
            if stream is not None:
                plot_raw_and_smooth(
                    ax, stream.index, stream["raw_observation"],
                    color=color,
                    raw_label="_nolegend_",
                    smooth_label=label,
                    raw_alpha=0.08,
                    smooth_linewidth=1.25,
                )
        ax.set_ylabel("Cross-entropy")
        ax.set_title("Raw CE by Execution Head")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8, ncol=2)

        # These are the terms after weights and scalar-term normalization.
        ax = fig6b.add_subplot(gs6b[1, 0])
        for label, stream, color in exec_contribution_streams:
            if stream is not None:
                plot_raw_and_smooth(
                    ax, stream.index, stream["raw_observation"],
                    color=color,
                    raw_label="_nolegend_",
                    smooth_label=label,
                    raw_alpha=0.07,
                    smooth_linewidth=1.2,
                )
        ax.axhline(0.0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_ylabel("Contribution to execution_loss")
        ax.set_title("Weighted, Normalized Objective Contributions")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8, ncol=2)

        ax = fig6b.add_subplot(gs6b[1, 1])
        for label, stream, color in exec_accuracy_streams:
            if stream is not None:
                plot_raw_and_smooth(
                    ax, stream.index, stream["raw_observation"],
                    color=color,
                    raw_label="_nolegend_",
                    smooth_label=label,
                    raw_alpha=0.08,
                    smooth_linewidth=1.25,
                )
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Top-1 accuracy")
        ax.set_title("Execution Head Accuracy")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8, ncol=2)

        # Compare the raw division penalty with what survives normalization.
        ax = fig6b.add_subplot(gs6b[2, 0])
        for label, stream, color in [
            ("division pre-normalization", exec_div_pre_norm, "tab:red"),
            ("division contribution", exec_div_contribution, "tab:pink"),
            ("entropy contribution", exec_entropy_contribution, "tab:gray"),
        ]:
            if stream is not None:
                plot_raw_and_smooth(
                    ax, stream.index, stream["raw_observation"],
                    color=color,
                    raw_label="_nolegend_",
                    smooth_label=label,
                    raw_alpha=0.10,
                    smooth_linewidth=1.3,
                )
        ax.axhline(0.0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_ylabel("Loss units")
        ax.set_title("Penalty and Entropy Terms")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8)

        ax = fig6b.add_subplot(gs6b[2, 1])
        if exec_teacher_forced is not None:
            plot_raw_and_smooth(
                ax, exec_teacher_forced.index, exec_teacher_forced["raw_observation"],
                color="tab:cyan",
                raw_label="_nolegend_",
                smooth_label="teacher-forced ratio",
                raw_alpha=0.10,
                smooth_linewidth=1.4,
            )
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Teacher-forced fraction", color="tab:cyan")
        ax.set_title("Supervision Mix and Scalar-Term Denominator")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8, loc="upper left")
        if exec_scalar_terms is not None:
            ax2 = ax.twinx()
            plot_raw_and_smooth(
                ax2, exec_scalar_terms.index, exec_scalar_terms["raw_observation"],
                color="tab:olive",
                raw_label="_nolegend_",
                smooth_label="scalar term count",
                raw_alpha=0.10,
                smooth_linewidth=1.3,
            )
            ax2.set_ylabel("Scalar loss terms", color="tab:olive")
            ax2.legend(fontsize=8, loc="upper right")

        execution_loss_path = output_root(path) + "_exec_loss.png"
        save_figure(fig6b, execution_loss_path)
    else:
        print("Execution loss decomposition figure skipped: no diagnostic streams in CSV")

    # --- Figure 7: EB Injection Diagnostics + Explicit Loss Composition ---
    eb_inject_gate = streams.get("eb_inject_gate")
    eb_read_gate = streams.get("eb_read_gate_mean")
    eb_inject_wnorm = streams.get("eb_inject_weight_norm")
    eb_read_wnorm = streams.get("eb_read_weight_norm")
    eb_loss_frac = streams.get("eb_loss_frac")

    has_inject_diag = any(s is not None for s in [eb_inject_gate, eb_read_gate, eb_inject_wnorm,
                                                   eb_read_wnorm, eb_loss_frac,
                                                   rho_atom_only, rho_nonatom_only])
    if has_inject_diag:
        fig7 = plt.figure(figsize=(16, 14), constrained_layout=True)
        fig7.suptitle("GRIM-text Telemetry — EB Injection Diagnostics & Loss Composition", fontsize=14, fontweight="bold")
        gs7 = GridSpec(3, 2, figure=fig7)

        # 7-1a) Gate values: inject gate + read gate mean
        ax = fig7.add_subplot(gs7[0, 0])
        if eb_inject_gate is not None:
            ax.plot(eb_inject_gate.index, eb_inject_gate["raw_observation"], alpha=0.25, linewidth=0.5, color="tab:blue")
            ax.plot(eb_inject_gate.index, smooth(eb_inject_gate["raw_observation"]), linewidth=1.5, color="tab:blue", label="inject gate σ")
        if eb_read_gate is not None:
            ax.plot(eb_read_gate.index, eb_read_gate["raw_observation"], alpha=0.25, linewidth=0.5, color="tab:red")
            ax.plot(eb_read_gate.index, smooth(eb_read_gate["raw_observation"]), linewidth=1.5, color="tab:red", label="read gate mean σ")
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label="init (0.5)")
        ax.set_ylabel("Gate Value (sigmoid)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("EB Gate Values (↓ = model suppressing EB)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 7-1b) Gate momentum (p values) — trending?
        ax = fig7.add_subplot(gs7[0, 1])
        if eb_inject_gate is not None and "p" in eb_inject_gate.columns:
            plot_raw_and_smooth(ax, eb_inject_gate.index, eb_inject_gate["p"], window=10,
                                color="tab:blue",
                                raw_label="inject gate p raw",
                                smooth_label="inject gate p",
                                smooth_linewidth=1.2)
        if eb_read_gate is not None and "p" in eb_read_gate.columns:
            plot_raw_and_smooth(ax, eb_read_gate.index, eb_read_gate["p"], window=10,
                                color="tab:red",
                                raw_label="read gate p raw",
                                smooth_label="read gate p",
                                smooth_linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("p (directional bias)")
        ax.set_title("Gate directional bias (< 0 = closing gate)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 7-2a) Gate weight norms (learning signal for gates)
        ax = fig7.add_subplot(gs7[1, 0])
        if eb_inject_wnorm is not None:
            ax.plot(eb_inject_wnorm.index, eb_inject_wnorm["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:blue")
            ax.plot(eb_inject_wnorm.index, smooth(eb_inject_wnorm["raw_observation"]), linewidth=1.5, color="tab:blue", label="w_inject_gate RMS")
        if eb_read_wnorm is not None:
            ax.plot(eb_read_wnorm.index, eb_read_wnorm["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:red")
            ax.plot(eb_read_wnorm.index, smooth(eb_read_wnorm["raw_observation"]), linewidth=1.5, color="tab:red", label="W_gate_read RMS")
        ax.set_ylabel("Weight RMS")
        ax.set_title("Gate Weight Norms (0 at init → growing = learning)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 7-2b) EB Loss Fraction
        ax = fig7.add_subplot(gs7[1, 1])
        if eb_loss_frac is not None:
            ax.plot(eb_loss_frac.index, eb_loss_frac["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:orange")
            ax.plot(eb_loss_frac.index, smooth(eb_loss_frac["raw_observation"]), linewidth=1.5, color="tab:orange", label="execution_loss / total_loss")
        ax.set_ylabel("Fraction")
        ax.set_title("Execution Loss Fraction")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 7-3a) Atom/non-atom rho split
        ax = fig7.add_subplot(gs7[2, 0])
        if rho_atom_only is not None:
            plot_raw_and_smooth(ax, rho_atom_only.index, rho_atom_only["raw_observation"],
                                color="tab:orange", raw_label="_nolegend_",
                                smooth_label="rho_atom_only")
        if rho_nonatom_only is not None:
            plot_raw_and_smooth(ax, rho_nonatom_only.index, rho_nonatom_only["raw_observation"],
                                color="tab:purple", raw_label="_nolegend_",
                                smooth_label="rho_nonatom_only")
        ax.set_ylabel("Rho")
        ax.set_title("Atom vs Non-Atom Rho")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 7-3b) Execution loss fraction vs total loss
        ax = fig7.add_subplot(gs7[2, 1])
        if eb_loss_frac is not None:
            plot_raw_and_smooth(ax, eb_loss_frac.index, eb_loss_frac["raw_observation"],
                                color="tab:orange", raw_label="_nolegend_",
                                smooth_label="execution_loss / total_loss",
                                raw_alpha=0.14, smooth_linewidth=1.3)
            ax.set_ylabel("Fraction")
            ax.set_title("Execution Loss Fraction")
            ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig7.savefig(os.path.splitext(path)[0] + "_eb_injection.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_eb_injection.png")
    else:
        print("EB injection diagnostics figure skipped: no injection streams in CSV")

    # --- Figure 8: PBM (Positional Bias Method) Diagnostics ---
    pbm_slope_rms = streams.get("pbm_alibi_slope_rms")
    pbm_eff_bias = streams.get("pbm_alibi_eff_bias_max")
    pbm_rope_rms = streams.get("pbm_rope_inv_freq_rms")
    pbm_seq_len = streams.get("pbm_batch_max_seq_len")

    has_pbm = any(s is not None for s in [pbm_slope_rms, pbm_eff_bias, pbm_rope_rms, pbm_seq_len])
    if has_pbm:
        fig8 = plt.figure(figsize=(16, 10), constrained_layout=True)
        fig8.suptitle("GRIM-text Telemetry — PBM Positional Bias Diagnostics", fontsize=14, fontweight="bold")
        gs8 = GridSpec(2, 2, figure=fig8)

        # 8-1a) ALiBi slope RMS (should be constant — corruption detector)
        ax = fig8.add_subplot(gs8[0, 0])
        if pbm_slope_rms is not None:
            ax.plot(pbm_slope_rms.index, pbm_slope_rms["raw_observation"], linewidth=1.5, color="tab:blue", label="ALiBi slope RMS")
        ax.set_ylabel("RMS")
        ax.set_title("ALiBi Slope RMS (should be constant)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 8-1b) RoPE inv_freq RMS (should be constant — corruption detector)
        ax = fig8.add_subplot(gs8[0, 1])
        if pbm_rope_rms is not None:
            ax.plot(pbm_rope_rms.index, pbm_rope_rms["raw_observation"], linewidth=1.5, color="tab:orange", label="RoPE inv_freq RMS")
        ax.set_ylabel("RMS")
        ax.set_title("RoPE Inverse Frequency RMS (should be constant)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 8-2a) Effective bias max (max|slope| × seq_len) — varies per batch
        ax = fig8.add_subplot(gs8[1, 0])
        if pbm_eff_bias is not None:
            ax.plot(pbm_eff_bias.index, pbm_eff_bias["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:red")
            ax.plot(pbm_eff_bias.index, smooth(pbm_eff_bias["raw_observation"]), linewidth=1.5, color="tab:red", label="max|slope| × seq_len")
        ax.set_ylabel("Effective Bias Magnitude")
        ax.set_title("ALiBi Effective Bias Max (distant-token suppression)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 8-2b) Batch max sequence length + effective bias overlay
        ax = fig8.add_subplot(gs8[1, 1])
        if pbm_seq_len is not None:
            ax.plot(pbm_seq_len.index, pbm_seq_len["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:green")
            ax.plot(pbm_seq_len.index, smooth(pbm_seq_len["raw_observation"]), linewidth=1.5, color="tab:green", label="batch max_seq_len")
        ax.set_ylabel("Sequence Length", color="tab:green")
        if pbm_eff_bias is not None:
            ax2 = ax.twinx()
            plot_raw_and_smooth(ax2, pbm_eff_bias.index, pbm_eff_bias["raw_observation"],
                                color="tab:red",
                                raw_label="_nolegend_",
                                smooth_label="eff. bias max",
                                raw_alpha=0.15,
                                smooth_alpha=0.7,
                                smooth_linewidth=1.2)
            ax2.set_ylabel("Effective Bias", color="tab:red")
            ax2.legend(loc="center right", fontsize=8)
        ax.set_title("Batch Sequence Length & Effective Bias")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig8.savefig(os.path.splitext(path)[0] + "_pbm.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_pbm.png")
    else:
        print("PBM diagnostics figure skipped: no PBM streams in CSV")

    # --- Figure 9: Rho Raw Decomposition ---
    rho_raw_dot = streams.get("rho_raw_avg_abs_dot")
    rho_raw_norm = streams.get("rho_raw_avg_norm_prod")
    rho_raw_hmin = streams.get("rho_raw_h_rms_min")
    rho_raw_hmax = streams.get("rho_raw_h_rms_max")
    rho_raw_spread = streams.get("rho_raw_rms_spread")
    rho_raw_signed = streams.get("rho_raw_avg_signed_dot")
    rho_centered_abs = streams.get("rho_centered_avg_abs_dot")
    rho_mean_vector_rms = streams.get("rho_mean_vector_rms")

    has_rho_raw = any(s is not None for s in [rho_raw_dot, rho_raw_norm, rho_raw_hmin, rho_raw_hmax, rho_raw_spread])
    if has_rho_raw:
        fig9 = plt.figure(figsize=(16, 14), constrained_layout=True)
        fig9.suptitle("GRIM-text Telemetry — Rho Raw Decomposition", fontsize=14, fontweight="bold")
        gs9 = GridSpec(3, 2, figure=fig9)

        # 9-1a) avg|dot(h_i,h_j)| — alignment numerator
        ax = fig9.add_subplot(gs9[0, 0])
        if rho_raw_dot is not None:
            ax.plot(rho_raw_dot.index, rho_raw_dot["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:blue")
            ax.plot(rho_raw_dot.index, smooth(rho_raw_dot["raw_observation"]), linewidth=1.5, color="tab:blue", label="avg|dot(h_i,h_j)|")
        ax.set_ylabel("Average Absolute Dot Product")
        ax.set_title("Alignment Numerator (↑ = more aligned)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 9-1b) avg(||h_i||·||h_j||·d) — normalization denominator
        ax = fig9.add_subplot(gs9[0, 1])
        if rho_raw_norm is not None:
            ax.plot(rho_raw_norm.index, rho_raw_norm["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:orange")
            ax.plot(rho_raw_norm.index, smooth(rho_raw_norm["raw_observation"]), linewidth=1.5, color="tab:orange", label="avg(‖h_i‖·‖h_j‖·d)")
        ax.set_ylabel("Normalization Denominator")
        ax.set_title("Normalization Denominator (scale factor)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 9-2a) h_rms min/max range — collapse/explosion detector
        ax = fig9.add_subplot(gs9[1, 0])
        if rho_raw_hmin is not None:
            plot_raw_and_smooth(ax, rho_raw_hmin.index, rho_raw_hmin["raw_observation"],
                                color="tab:green",
                                raw_label="_nolegend_",
                                smooth_label="h_rms min")
        if rho_raw_hmax is not None:
            plot_raw_and_smooth(ax, rho_raw_hmax.index, rho_raw_hmax["raw_observation"],
                                color="tab:red",
                                raw_label="_nolegend_",
                                smooth_label="h_rms max")
        if rho_raw_hmin is not None and rho_raw_hmax is not None:
            common = rho_raw_hmin.index.intersection(rho_raw_hmax.index)
            ax.fill_between(common,
                            smooth(rho_raw_hmin.loc[common, "raw_observation"]),
                            smooth(rho_raw_hmax.loc[common, "raw_observation"]),
                            alpha=0.15, color="tab:gray")
        ax.set_ylabel("h_rms")
        ax.set_title("Per-Position h_rms Range (min/max)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 9-2b) Rho raw dot/norm ratio + rho_final overlay
        ax = fig9.add_subplot(gs9[1, 1])
        if rho_raw_dot is not None and rho_raw_norm is not None:
            common = rho_raw_dot.index.intersection(rho_raw_norm.index)
            dot_vals = rho_raw_dot.loc[common, "raw_observation"].values
            norm_vals = rho_raw_norm.loc[common, "raw_observation"].values
            ratio = np.where(norm_vals > 1e-12, dot_vals / norm_vals, 0.0)
            ratio_series = pd.Series(ratio, index=common)
            plot_raw_and_smooth(ax, common, ratio_series,
                                color="tab:purple",
                                raw_label="_nolegend_",
                                smooth_label="dot/norm ratio")
        if rho_final is not None:
            plot_raw_and_smooth(ax, rho_final.index, rho_final["raw_observation"],
                                color="tab:blue",
                                raw_label="_nolegend_",
                                smooth_label="rho_final",
                                raw_alpha=0.15,
                                smooth_alpha=0.7,
                                smooth_linewidth=1.2,
                                smooth_linestyle="--")
        ax.set_ylabel("Ratio / ρ")
        ax.set_title("Dot/Norm Ratio vs ρ_final (should track)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 9-3a) rho_raw_rms_spread (rms_max / rms_min) — denominator-collapse detector
        ax = fig9.add_subplot(gs9[2, 0])
        if rho_raw_spread is not None:
            ax.plot(rho_raw_spread.index, rho_raw_spread["raw_observation"],
                    alpha=0.3, linewidth=0.5, color="tab:red")
            ax.plot(rho_raw_spread.index, smooth(rho_raw_spread["raw_observation"]),
                    linewidth=1.5, color="tab:red", label="rms_max / rms_min")
            ax.axhline(1.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.6,
                       label="warning (1.5×)")
            ax.axhline(2.0, color="tab:orange", linewidth=0.8, linestyle="--", alpha=0.6,
                       label="anomaly (2.0×)")
            ax.axhline(4.0, color="tab:red", linewidth=0.8, linestyle="--", alpha=0.6,
                       label="bifurcation (4.0×)")
        ax.set_ylabel("rms spread (max/min)")
        ax.set_title("ρ Denominator Collapse Detector (>2× = warning)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # 9-3b) Spread vs rho_final overlay (denominator collapse → spurious ρ)
        ax = fig9.add_subplot(gs9[2, 1])
        if rho_raw_spread is not None:
            plot_raw_and_smooth(ax, rho_raw_spread.index, rho_raw_spread["raw_observation"],
                                color="tab:red",
                                raw_label="_nolegend_",
                                smooth_label="rms spread")
        ax.set_ylabel("rms spread", color="tab:red")
        if rho_final is not None:
            ax2 = ax.twinx()
            plot_raw_and_smooth(ax2, rho_final.index, rho_final["raw_observation"],
                                color="tab:blue",
                                raw_label="_nolegend_",
                                smooth_label="ρ_final",
                                raw_alpha=0.15,
                                smooth_alpha=0.7,
                                smooth_linewidth=1.2)
            ax2.set_ylabel("ρ_final", color="tab:blue")
            ax2.legend(loc="center right", fontsize=8)
        ax.set_title("rms_spread vs ρ_final (collapse drives spurious ρ)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig9.savefig(os.path.splitext(path)[0] + "_rho_raw.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_rho_raw.png")
    else:
        print("Rho raw decomposition figure skipped: no rho raw streams in CSV")

    # --- Figure 9b: Rho Centering & Signed-Dot Diagnostics ---
    # Streams 55-57 separate coherent mean-vector drift from pairwise collapse:
    #   avg_signed_dot = avg_{i<j}(h_i · h_j)
    #   centered_avg_abs_dot = avg_{i<j}|((h_i-μ) · (h_j-μ))|
    #   mean_vector_rms = sqrt(mean_d(μ_d^2))
    has_rho_centering = any(s is not None for s in [rho_raw_signed, rho_centered_abs, rho_mean_vector_rms])
    if has_rho_centering:
        fig9b = plt.figure(figsize=(16, 12), constrained_layout=True)
        fig9b.suptitle("GRIM-text Telemetry — Rho Centering & Signed Dot", fontsize=14, fontweight="bold")
        gs9b = GridSpec(2, 2, figure=fig9b)

        # 9b-1a) Signed raw pairwise dot: detects coherent same-direction drift.
        ax = fig9b.add_subplot(gs9b[0, 0])
        if rho_raw_signed is not None:
            ax.plot(rho_raw_signed.index, rho_raw_signed["raw_observation"],
                    alpha=0.3, linewidth=0.5, color="tab:purple")
            ax.plot(rho_raw_signed.index, smooth(rho_raw_signed["raw_observation"]),
                    linewidth=1.5, color="tab:purple", label="avg signed dot")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_ylabel("avg signed dot")
        ax.set_title("Raw Signed Pairwise Dot: avg₍ᵢ<ⱼ₎(hᵢ·hⱼ)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # 9b-1b) Centered absolute pairwise dot: subtracts the layer mean first.
        ax = fig9b.add_subplot(gs9b[0, 1])
        if rho_centered_abs is not None:
            ax.plot(rho_centered_abs.index, rho_centered_abs["raw_observation"],
                    alpha=0.3, linewidth=0.5, color="tab:green")
            ax.plot(rho_centered_abs.index, smooth(rho_centered_abs["raw_observation"]),
                    linewidth=1.5, color="tab:green", label="centered avg |dot|")
        if rho_raw_dot is not None:
            plot_raw_and_smooth(ax, rho_raw_dot.index, rho_raw_dot["raw_observation"],
                    color="tab:blue",
                    raw_label="_nolegend_",
                    smooth_label="raw avg |dot|",
                    raw_alpha=0.12,
                    smooth_alpha=0.7,
                    smooth_linewidth=1.1,
                    smooth_linestyle="--")
        ax.set_ylabel("avg absolute dot")
        ax.set_title("Centered Pairwise Dot After h̃ᵢ = hᵢ − μ")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # 9b-2a) Mean hidden vector RMS: direct DC/mean-vector magnitude.
        ax = fig9b.add_subplot(gs9b[1, 0])
        if rho_mean_vector_rms is not None:
            ax.plot(rho_mean_vector_rms.index, rho_mean_vector_rms["raw_observation"],
                    alpha=0.3, linewidth=0.5, color="tab:red")
            ax.plot(rho_mean_vector_rms.index, smooth(rho_mean_vector_rms["raw_observation"]),
                    linewidth=1.5, color="tab:red", label="mean vector RMS")
        ax.set_ylabel("sqrt(mean_d(μ_d²))")
        ax.set_title("Mean Hidden Vector RMS")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # 9b-2b) Signed vs centered comparison: separates mean drift from residual alignment.
        ax = fig9b.add_subplot(gs9b[1, 1])
        if rho_raw_signed is not None:
            plot_raw_and_smooth(ax, rho_raw_signed.index, rho_raw_signed["raw_observation"],
                                color="tab:purple",
                                raw_label="_nolegend_",
                                smooth_label="avg signed dot")
        if rho_centered_abs is not None:
            plot_raw_and_smooth(ax, rho_centered_abs.index, rho_centered_abs["raw_observation"],
                                color="tab:green",
                                raw_label="_nolegend_",
                                smooth_label="centered avg |dot|")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_ylabel("dot diagnostic")
        ax.set_title("Mean-Drift vs Residual Pairwise Alignment")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        fig9b.savefig(os.path.splitext(path)[0] + "_rho_centering.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_rho_centering.png")
    else:
        print("Rho centering figure skipped: no rho signed/centered streams in CSV")

    # --- Figure 10: RMSNorm Gamma Tracking ---
    gamma_pre_attn = streams.get("rms_gamma_pre_attn_rms")
    gamma_pre_ffn = streams.get("rms_gamma_pre_ffn_rms")
    gamma_final = streams.get("rms_gamma_final_rms")

    has_gamma = any(s is not None for s in [gamma_pre_attn, gamma_pre_ffn, gamma_final])
    if has_gamma:
        fig10 = plt.figure(figsize=(16, 10), constrained_layout=True)
        fig10.suptitle("GRIM-text Telemetry — RMSNorm Learned Gamma", fontsize=14, fontweight="bold")
        gs10 = GridSpec(2, 2, figure=fig10)

        # 10-1a) All gamma RMS values overlaid
        ax = fig10.add_subplot(gs10[0, 0])
        if gamma_pre_attn is not None:
            ax.plot(gamma_pre_attn.index, gamma_pre_attn["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:blue")
            ax.plot(gamma_pre_attn.index, smooth(gamma_pre_attn["raw_observation"]), linewidth=1.5, color="tab:blue", label="γ₁ pre-attn (mean)")
        if gamma_pre_ffn is not None:
            ax.plot(gamma_pre_ffn.index, gamma_pre_ffn["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:orange")
            ax.plot(gamma_pre_ffn.index, smooth(gamma_pre_ffn["raw_observation"]), linewidth=1.5, color="tab:orange", label="γ₂ pre-FFN (mean)")
        if gamma_final is not None:
            ax.plot(gamma_final.index, gamma_final["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:green")
            ax.plot(gamma_final.index, smooth(gamma_final["raw_observation"]), linewidth=1.5, color="tab:green", label="γ_final (LM head)")
        ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.5, label="init (1.0)")
        ax.set_ylabel("Gamma RMS")
        ax.set_title("RMSNorm Gamma RMS (init=1.0, drift → learning)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 10-1b) Gamma deviation from init (% change)
        ax = fig10.add_subplot(gs10[0, 1])
        for name, s, color in [("γ₁ pre-attn", gamma_pre_attn, "tab:blue"),
                                ("γ₂ pre-FFN", gamma_pre_ffn, "tab:orange"),
                                ("γ_final", gamma_final, "tab:green")]:
            if s is not None:
                pct_change = (s["raw_observation"] - 1.0) * 100.0
                plot_raw_and_smooth(ax, s.index, pct_change,
                                    color=color,
                                    raw_label="_nolegend_",
                                    smooth_label=name)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("% Deviation from Init")
        ax.set_title("Gamma Drift from Initialization")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 10-2a) Gamma momentum (p) — is gamma trending?
        ax = fig10.add_subplot(gs10[1, 0])
        for name, s, color in [("γ₁ pre-attn", gamma_pre_attn, "tab:blue"),
                                ("γ₂ pre-FFN", gamma_pre_ffn, "tab:orange"),
                                ("γ_final", gamma_final, "tab:green")]:
            if s is not None and "p" in s.columns:
                plot_raw_and_smooth(ax, s.index, s["p"], window=10,
                                    color=color,
                                    raw_label="_nolegend_",
                                    smooth_label=f"{name} p",
                                    raw_alpha=0.15,
                                    smooth_linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("p (directional bias)")
        ax.set_title("Gamma directional bias (trending direction)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 10-2b) Gamma vs loss overlay — causal relationship
        ax = fig10.add_subplot(gs10[1, 1])
        if gamma_pre_attn is not None:
            plot_raw_and_smooth(ax, gamma_pre_attn.index, gamma_pre_attn["raw_observation"],
                                color="tab:blue",
                                raw_label="_nolegend_",
                                smooth_label="γ₁ pre-attn",
                                raw_alpha=0.15,
                                smooth_linewidth=1.2)
        if gamma_pre_ffn is not None:
            plot_raw_and_smooth(ax, gamma_pre_ffn.index, gamma_pre_ffn["raw_observation"],
                                color="tab:orange",
                                raw_label="_nolegend_",
                                smooth_label="γ₂ pre-FFN",
                                raw_alpha=0.15,
                                smooth_linewidth=1.2)
        if gamma_final is not None:
            plot_raw_and_smooth(ax, gamma_final.index, gamma_final["raw_observation"],
                                color="tab:green",
                                raw_label="_nolegend_",
                                smooth_label="γ_final",
                                raw_alpha=0.15,
                                smooth_linewidth=1.2)
        ax.set_ylabel("Gamma RMS", color="tab:blue")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.3)
        if loss is not None:
            ax2 = ax.twinx()
            plot_raw_and_smooth(ax2, loss.index, loss["raw_observation"],
                                color="tab:gray",
                                raw_label="_nolegend_",
                                smooth_label="loss (sm20)",
                                raw_alpha=0.12,
                                smooth_alpha=0.5,
                                smooth_linewidth=1)
            ax2.set_ylabel("Loss", color="tab:gray")
            ax2.legend(loc="center right", fontsize=8)
        ax.set_title("Gamma vs Loss (correlation check)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig10.savefig(os.path.splitext(path)[0] + "_rms_gamma.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_rms_gamma.png")
    else:
        print("RMS gamma figure skipped: no gamma streams in CSV")

    # --- Figure 11: h↔W Alignment (LM-head leak channel) ---
    # Streams 39-44. Detects coherent drift in W rows along h direction caused by
    # grad_W[v] += (p_v − y_v)·h_t accumulating across batches → logit_std inflation
    # without any change in ||h|| or ||W||.
    #
    # Random baseline: RMS(cos(h, W_v)) ≈ 1/sqrt(d_model). At d=768 → 0.0361.
    # Correlation correction: logit_std_ratio² ≈ 1 + d_model · cos_hW_rms²
    #   so ratio=2.31 ⇒ cos_rms ≈ sqrt(2.31²−1)/sqrt(768) ≈ 0.075 (≈2× baseline).
    hw_cos_rms = streams.get("hw_cos_rms")
    hw_cos_signed = streams.get("hw_cos_signed_mean")
    hw_cos_amax = streams.get("hw_cos_abs_max")
    hw_hbar_wbar = streams.get("hw_hbar_wbar_cos")
    hw_h_dc_mean = streams.get("hw_h_dc_mean")
    hw_h_dc_amax = streams.get("hw_h_dc_abs_max")

    has_hw = any(s is not None for s in [hw_cos_rms, hw_cos_signed, hw_cos_amax,
                                          hw_hbar_wbar, hw_h_dc_mean, hw_h_dc_amax])
    if has_hw:
        # Infer d_model from random-cosine baseline: this is the data-independent
        # reference line we plot. Default d=768; override via env var if needed.
        d_model_baseline = d_model_for_baseline()
        cos_random_baseline = 1.0 / math.sqrt(float(d_model_baseline))

        fig11 = plt.figure(figsize=(16, 14), constrained_layout=True)
        fig11.suptitle("GRIM-text Telemetry — h↔W Alignment (LM-head Leak Channel)",
                       fontsize=14, fontweight="bold")
        gs11 = GridSpec(3, 2, figure=fig11)

        # 11-1a) cos_hW_rms vs random baseline + 2× warning band
        ax = fig11.add_subplot(gs11[0, 0])
        if hw_cos_rms is not None:
            ax.plot(hw_cos_rms.index, hw_cos_rms["raw_observation"],
                    alpha=0.3, linewidth=0.5, color="tab:red")
            ax.plot(hw_cos_rms.index, smooth(hw_cos_rms["raw_observation"]),
                    linewidth=1.5, color="tab:red", label="RMS cos(h, W_v)")
        ax.axhline(cos_random_baseline, color="gray", linewidth=1, linestyle="--",
                   alpha=0.7, label=f"random (1/√d={cos_random_baseline:.4f})")
        ax.axhline(2 * cos_random_baseline, color="tab:orange", linewidth=0.8,
                   linestyle=":", alpha=0.7, label="2× baseline (warning)")
        ax.set_ylabel("RMS cosine")
        ax.set_title("h↔W Alignment RMS (primary leak metric)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # 11-1b) cos_signed_mean — DC channel (rank-1 leak)
        ax = fig11.add_subplot(gs11[0, 1])
        if hw_cos_signed is not None:
            ax.plot(hw_cos_signed.index, hw_cos_signed["raw_observation"],
                    alpha=0.3, linewidth=0.5, color="tab:purple")
            ax.plot(hw_cos_signed.index, smooth(hw_cos_signed["raw_observation"]),
                    linewidth=1.5, color="tab:purple", label="signed mean cos(h, W_v)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Signed mean cosine")
        ax.set_title("Signed Mean Cosine (DC-leak channel; should ≈ 0)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 11-2a) Worst-case row alignment
        ax = fig11.add_subplot(gs11[1, 0])
        if hw_cos_amax is not None:
            ax.plot(hw_cos_amax.index, hw_cos_amax["raw_observation"],
                    alpha=0.3, linewidth=0.5, color="tab:brown")
            ax.plot(hw_cos_amax.index, smooth(hw_cos_amax["raw_observation"]),
                    linewidth=1.5, color="tab:brown", label="max |cos(h, W_v)|")
        ax.axhline(cos_random_baseline, color="gray", linewidth=0.8, linestyle="--",
                   alpha=0.5, label=f"1/√d={cos_random_baseline:.4f}")
        ax.set_ylabel("|cos|")
        ax.set_ylim(0, 1)
        ax.set_title("Worst-Case Row Alignment (single-token leak)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 11-2b) Rank-1 explicit DC channel: cos(h̄, W̄)
        ax = fig11.add_subplot(gs11[1, 1])
        if hw_hbar_wbar is not None:
            ax.plot(hw_hbar_wbar.index, hw_hbar_wbar["raw_observation"],
                    alpha=0.3, linewidth=0.5, color="tab:cyan")
            ax.plot(hw_hbar_wbar.index, smooth(hw_hbar_wbar["raw_observation"]),
                    linewidth=1.5, color="tab:cyan", label="cos(mean_t h, mean_v W)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("cos(h̄, W̄)")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("Rank-1 DC Coupling (Σ_t(p−y)·h aggregated leak)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 11-3a) h DC component: mean and abs-max
        ax = fig11.add_subplot(gs11[2, 0])
        if hw_h_dc_mean is not None:
            plot_raw_and_smooth(ax, hw_h_dc_mean.index, hw_h_dc_mean["raw_observation"],
                                color="tab:blue",
                                raw_label="_nolegend_",
                                smooth_label="mean_t (1/d)Σ_d h")
        if hw_h_dc_amax is not None:
            plot_raw_and_smooth(ax, hw_h_dc_amax.index, hw_h_dc_amax["raw_observation"],
                                color="tab:red",
                                raw_label="_nolegend_",
                                smooth_label="max_t |(1/d)Σ_d h|")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("DC offset")
        ax.set_title("Hidden-State DC Component (column-centering check)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 11-3b) Predicted logit_std inflation factor from cos_rms.
        # Theory: logit_std² = h_rms²·d·W_rms_rms²·(1 + d·cos_rms²)
        # so the inflation ratio is sqrt(1 + d · cos_rms²).
        ax = fig11.add_subplot(gs11[2, 1])
        if hw_cos_rms is not None:
            cos_vals = hw_cos_rms["raw_observation"].astype(float).values
            d_f = float(d_model_baseline)
            inflation = np.sqrt(np.maximum(0.0, 1.0 + d_f * (cos_vals * cos_vals)))
            inflation_series = pd.Series(inflation, index=hw_cos_rms.index)
            plot_raw_and_smooth(ax, hw_cos_rms.index, inflation_series,
                    color="tab:orange",
                    raw_label="_nolegend_",
                    smooth_label="√(1 + d·cos_rms²)")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--",
                   alpha=0.7, label="no leak (1.0)")
        ax.axhline(2.0, color="tab:red", linewidth=0.8, linestyle="--",
                   alpha=0.7, label="2× inflation (alarm)")
        ax.set_ylabel("Predicted logit_std / expected")
        ax.set_title(f"Predicted Logit-Std Inflation from Alignment (d={d_model_baseline})")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        fig11.savefig(os.path.splitext(path)[0] + "_hw_alignment.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_hw_alignment.png")
    else:
        print("h↔W alignment figure skipped: no hw_cos_* streams in CSV")

    # --- Figure 12: Unigram direction, LM-head scale, and init invariants ---
    unigram_abs = streams.get("unigram_dir_cos_abs_mean")
    unigram_signed = streams.get("unigram_dir_cos_signed_mean")
    lm_head_w_rms = streams.get("lm_head_w_rms_rms")
    init_tie_cfg = streams.get("init_tie_cfg")
    init_tie_ptrs_same = streams.get("init_tie_ptrs_same")
    init_tie_grads_same = streams.get("init_tie_grads_same")
    init_lm_owns_weights = streams.get("init_lm_owns_weights")
    init_opt_groups_total = streams.get("init_opt_groups_total")
    init_opt_groups_emb = streams.get("init_opt_groups_emb")
    init_opt_groups_lm = streams.get("init_opt_groups_lm")

    init_bool_streams = [
        ("tie cfg", init_tie_cfg, "tab:blue"),
        ("weight ptrs same", init_tie_ptrs_same, "tab:green"),
        ("grad ptrs same", init_tie_grads_same, "tab:purple"),
        ("LM owns weights", init_lm_owns_weights, "tab:red"),
    ]
    init_count_streams = [
        ("groups total", init_opt_groups_total, "tab:blue"),
        ("embedding groups", init_opt_groups_emb, "tab:green"),
        ("LM groups", init_opt_groups_lm, "tab:red"),
    ]
    has_unigram_lm_init = any(s is not None for s in [
        unigram_abs, unigram_signed, lm_head_w_rms,
        init_tie_cfg, init_tie_ptrs_same, init_tie_grads_same, init_lm_owns_weights,
        init_opt_groups_total, init_opt_groups_emb, init_opt_groups_lm,
    ])
    if has_unigram_lm_init:
        d_model_baseline = d_model_for_baseline()
        unigram_random_abs = math.sqrt(2.0 / (math.pi * float(d_model_baseline)))
        cos_random_baseline = 1.0 / math.sqrt(float(d_model_baseline))

        fig12 = plt.figure(figsize=(16, 14), constrained_layout=True)
        fig12.suptitle("GRIM-text Telemetry — Unigram Direction & Init Invariants",
                       fontsize=14, fontweight="bold")
        gs12 = GridSpec(3, 2, figure=fig12)

        # 12-1a) Unigram-frequency direction absolute alignment.
        ax = fig12.add_subplot(gs12[0, 0])
        if unigram_abs is not None:
            ax.plot(unigram_abs.index, unigram_abs["raw_observation"],
                    alpha=0.3, linewidth=0.5, color="tab:orange")
            ax.plot(unigram_abs.index, smooth(unigram_abs["raw_observation"]),
                    linewidth=1.5, color="tab:orange", label="mean |cos(h, e_uf)|")
        ax.axhline(unigram_random_abs, color="gray", linewidth=1, linestyle="--",
                   alpha=0.7, label=f"random |cos|≈{unigram_random_abs:.4f}")
        ax.axhline(2.0 * unigram_random_abs, color="tab:red", linewidth=0.8,
                   linestyle=":", alpha=0.7, label="2× random (warning)")
        ax.set_ylabel("Mean absolute cosine")
        ax.set_title("Unigram-Frequency Direction Alignment")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # 12-1b) Signed unigram direction alignment.
        ax = fig12.add_subplot(gs12[0, 1])
        if unigram_signed is not None:
            ax.plot(unigram_signed.index, unigram_signed["raw_observation"],
                    alpha=0.3, linewidth=0.5, color="tab:purple")
            ax.plot(unigram_signed.index, smooth(unigram_signed["raw_observation"]),
                    linewidth=1.5, color="tab:purple", label="mean cos(h, e_uf)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axhline(unigram_random_abs, color="gray", linewidth=0.8, linestyle=":",
                   alpha=0.5, label="±random |cos|")
        ax.axhline(-unigram_random_abs, color="gray", linewidth=0.8, linestyle=":",
                   alpha=0.5)
        ax.set_ylabel("Signed mean cosine")
        ax.set_title("Signed Unigram Direction Bias (should hover near 0)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # 12-2a) Raw LM-head W RMS term from logit-scale equation.
        ax = fig12.add_subplot(gs12[1, 0])
        if lm_head_w_rms is not None:
            ax.plot(lm_head_w_rms.index, lm_head_w_rms["raw_observation"],
                    alpha=0.3, linewidth=0.5, color="tab:green")
            ax.plot(lm_head_w_rms.index, smooth(lm_head_w_rms["raw_observation"]),
                    linewidth=1.5, color="tab:green", label="W_rms_rms")
            initial_w_rms = float(lm_head_w_rms["raw_observation"].iloc[0])
            ax.axhline(initial_w_rms, color="gray", linewidth=0.8, linestyle="--",
                       alpha=0.6, label=f"initial={initial_w_rms:.6f}")
        ax.set_ylabel("W_rms_rms")
        ax.set_title("LM-Head Weight RMS Term: logit_std ≈ √d · h_rms · W_rms_rms")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # 12-2b) Alignment comparison: h↔W vs unigram direction.
        ax = fig12.add_subplot(gs12[1, 1])
        if hw_cos_rms is not None:
            plot_raw_and_smooth(ax, hw_cos_rms.index, hw_cos_rms["raw_observation"],
                                color="tab:red",
                                raw_label="_nolegend_",
                                smooth_label="RMS cos(h, W_v)",
                                raw_alpha=0.15,
                                smooth_linewidth=1.4)
        if unigram_abs is not None:
            plot_raw_and_smooth(ax, unigram_abs.index, unigram_abs["raw_observation"],
                                color="tab:orange",
                                raw_label="_nolegend_",
                                smooth_label="mean |cos(h, e_uf)|",
                                raw_alpha=0.15,
                                smooth_linewidth=1.4)
        ax.axhline(cos_random_baseline, color="tab:red", linewidth=0.8, linestyle="--",
                   alpha=0.5, label=f"RMS random={cos_random_baseline:.4f}")
        ax.axhline(unigram_random_abs, color="tab:orange", linewidth=0.8, linestyle=":",
                   alpha=0.7, label=f"abs random={unigram_random_abs:.4f}")
        ax.set_ylabel("Cosine magnitude")
        ax.set_title("Competing Collapse Channels")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # 12-3a) Init-time boolean invariants repaired from stream_idx 48-51.
        ax = fig12.add_subplot(gs12[2, 0])
        for name, s, color in init_bool_streams:
            if s is not None:
                ax.step(s.index, s["raw_observation"], where="post",
                        linewidth=1.4, color=color, label=name)
        ax.set_ylabel("0 / 1")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Init Invariants: Tied Embedding / LM-Head Ownership")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # 12-3b) Init-time optimizer group counts repaired from stream_idx 52-54.
        ax = fig12.add_subplot(gs12[2, 1])
        for name, s, color in init_count_streams:
            if s is not None:
                ax.step(s.index, s["raw_observation"], where="post",
                        linewidth=1.4, color=color, label=name)
        ax.set_ylabel("Count")
        ax.set_title("Init Invariants: Optimizer Group Counts")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        fig12.savefig(os.path.splitext(path)[0] + "_unigram_lm_init.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_unigram_lm_init.png")
    else:
        print("Unigram/LM-head/init figure skipped: no streams in CSV")

    # --- Figure 13: Raw loss decomposition ---
    primary_loss_streams = [
        ("total", loss, "tab:blue"),
        ("text", streams.get("text_loss"), "tab:green"),
    ]
    auxiliary_loss_streams = [
        ("execution", streams.get("execution_loss"), "tab:purple"),
    ]
    has_loss_components = any(
        stream is not None
        for _, stream, _ in [*primary_loss_streams[1:], *auxiliary_loss_streams]
    )
    if has_loss_components:
        fig13, axes13 = plt.subplots(2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
        fig13.suptitle("GRIM-text Telemetry - Raw Loss Decomposition", fontsize=14, fontweight="bold")

        for label, stream, color in primary_loss_streams:
            if stream is not None:
                plot_raw_and_smooth(
                    axes13[0], stream.index, stream["raw_observation"],
                    color=color,
                    raw_label="_nolegend_",
                    smooth_label=label,
                    raw_alpha=0.12,
                    smooth_linewidth=1.4,
                )
        axes13[0].set_ylabel("Loss")
        axes13[0].set_title("Primary Objectives")
        axes13[0].legend(fontsize=8)
        axes13[0].grid(True, alpha=0.3)

        for label, stream, color in auxiliary_loss_streams:
            if stream is not None:
                plot_raw_and_smooth(
                    axes13[1], stream.index, stream["raw_observation"],
                    color=color,
                    raw_label="_nolegend_",
                    smooth_label=label,
                    raw_alpha=0.12,
                    smooth_linewidth=1.4,
                )
        axes13[1].set_xlabel("global_step")
        axes13[1].set_ylabel("Loss")
        axes13[1].set_title("Latent and Execution Objectives")
        axes13[1].legend(fontsize=8)
        axes13[1].grid(True, alpha=0.3)

        loss_components_path = os.path.splitext(path)[0] + "_loss_components.png"
        save_figure(fig13, loss_components_path)
    else:
        print("Loss component figure skipped: no raw component streams in CSV")

    atlas_dir = generate_raw_stream_atlas(path, streams)
    generate_stream_detail_pages(path, streams)
    print(f"Saved comprehensive telemetry atlas under: {atlas_dir}")

    plt.show()

if __name__ == "__main__":
    main()
