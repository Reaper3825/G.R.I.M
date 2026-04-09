#!/usr/bin/env python3
"""Simple TelemetryLattice CSV pattern viewer for GRIM-text training runs."""

import sys
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def load_telemetry(path):
    df = pd.read_csv(path)
    # Keep only level-0 (raw per-step) observations
    df = df[df["level"] == 0].copy()
    return df

def pivot_streams(df):
    """Pivot so each stream becomes a column keyed by global_step."""
    streams = {}
    for name in df["stream_name"].unique():
        sub = df[df["stream_name"] == name].sort_values("global_step")
        streams[name] = sub.set_index("global_step")
    return streams

def smooth(series, window=20):
    return series.rolling(window, min_periods=1).mean()

def main():
    # --- find CSV ---
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        log_dir = os.path.join(os.path.dirname(__file__),
                               "resources", "models", "GRIM-text", "training", "logs")
        csvs = sorted(glob.glob(os.path.join(log_dir, "telemetry_*.Csv")))
        if not csvs:
            print("No telemetry CSV found. Pass path as argument.")
            sys.exit(1)
        path = csvs[-1]  # most recent
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

    # 1b) Loss rate-of-change (delta_mu from lattice)
    ax = fig.add_subplot(gs[0, 1])
    if loss is not None:
        delta = loss["raw_observation"].diff()
        ax.plot(loss.index, smooth(delta, 50), linewidth=1, color="tab:red")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Δ Loss (smoothed)")
    ax.set_title("Loss change rate")
    ax.grid(True, alpha=0.3)

    # 2a) Gradient norms
    ax = fig.add_subplot(gs[1, 0])
    if grad_mean is not None:
        ax.plot(grad_mean.index, grad_mean["raw_observation"], alpha=0.3, linewidth=0.5, label="mean (raw)")
        ax.plot(grad_mean.index, smooth(grad_mean["raw_observation"]), linewidth=1.5, label="mean (smooth)")
    if grad_max is not None:
        ax.plot(grad_max.index, smooth(grad_max["raw_observation"]), linewidth=1, linestyle="--", label="max (smooth)")
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

    # 3b) TelemetryLattice anomaly score (p) and curvature (k_out)
    ax = fig.add_subplot(gs[2, 1])
    if loss is not None:
        ax2 = ax.twinx()
        ax.plot(loss.index, smooth(loss["p"], 10), linewidth=1, color="tab:orange", label="p (momentum)")
        ax2.plot(loss.index, smooth(loss["k_out"], 10), linewidth=1, color="tab:purple", label="k (curvature)")
        ax.set_ylabel("p (momentum)", color="tab:orange")
        ax2.set_ylabel("k (curvature)", color="tab:purple")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Lattice signals: momentum & curvature")
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.splitext(path)[0] + "_patterns.png", dpi=150)
    print(f"Saved: {os.path.splitext(path)[0]}_patterns.png")

    # --- Figure 2: Multi-level lattice view ---
    df_all = pd.read_csv(path)
    levels = sorted(df_all["level"].unique())
    if len(levels) > 1:
        fig2, axes = plt.subplots(len(levels), 1, figsize=(14, 3 * len(levels)),
                                   sharex=True, constrained_layout=True)
        fig2.suptitle("Loss across TelemetryLattice levels", fontsize=13, fontweight="bold")
        if len(levels) == 1:
            axes = [axes]
        for i, lvl in enumerate(levels):
            sub = df_all[(df_all["level"] == lvl) & (df_all["stream_name"] == "loss")]
            sub = sub.sort_values("global_step")
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
        fig2.savefig(os.path.splitext(path)[0] + "_levels.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_levels.png")

    plt.show()

if __name__ == "__main__":
    main()
