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

    # --- Figure 2: Rho & hidden-state streams ---
    rho_final = streams.get("rho_final")
    rho_growth = streams.get("rho_growth")
    rho_worst = streams.get("rho_worst_delta")
    h_rms = streams.get("h_rms_growth")
    tpb = streams.get("tokens_per_batch")

    fig2 = plt.figure(figsize=(16, 14), constrained_layout=True)
    fig2.suptitle("GRIM-text Telemetry — Rho & Hidden State", fontsize=14, fontweight="bold")
    gs2 = GridSpec(3, 2, figure=fig2)

    # 2-1a) Rho streams overlaid
    ax = fig2.add_subplot(gs2[0, 0])
    for name, s, color in [("rho_final", rho_final, "tab:blue"),
                            ("rho_growth", rho_growth, "tab:orange"),
                            ("rho_worst_delta", rho_worst, "tab:red")]:
        if s is not None:
            ax.plot(s.index, s["raw_observation"], alpha=0.2, linewidth=0.5, color=color)
            ax.plot(s.index, smooth(s["raw_observation"]), linewidth=1.5, label=name, color=color)
    ax.set_ylabel("Rho")
    ax.set_title("Rho metrics")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2-1b) Rho running mean (mu) comparison
    ax = fig2.add_subplot(gs2[0, 1])
    for name, s, color in [("rho_final", rho_final, "tab:blue"),
                            ("rho_growth", rho_growth, "tab:orange"),
                            ("rho_worst_delta", rho_worst, "tab:red")]:
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

    # 2-3a) Rho momentum (p) across streams
    ax = fig2.add_subplot(gs2[2, 0])
    for name, s, color in [("rho_final", rho_final, "tab:blue"),
                            ("rho_growth", rho_growth, "tab:orange"),
                            ("rho_worst_delta", rho_worst, "tab:red"),
                            ("h_rms_growth", h_rms, "tab:purple")]:
        if s is not None:
            ax.plot(s.index, smooth(s["p"], 10), linewidth=1.2, label=f"{name}", color=color)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("p (momentum)")
    ax.set_title("Momentum across streams")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2-3b) Curvature (k_out) across streams
    ax = fig2.add_subplot(gs2[2, 1])
    for name, s, color in [("loss", loss, "tab:blue"),
                            ("rho_final", rho_final, "tab:orange"),
                            ("grad_norm_mean", grad_mean, "tab:green"),
                            ("h_rms_growth", h_rms, "tab:purple")]:
        if s is not None:
            ax.plot(s.index, smooth(s["k_out"], 10), linewidth=1.2, label=f"{name}", color=color)
    ax.set_ylabel("k (curvature)")
    ax.set_title("Curvature across streams")
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
        ax.plot(loss.index, loss["sigma_tilde"], linewidth=1.2, label="σ̃ (adapted)", color="tab:orange")
        ax.plot(loss.index, loss["sigma_a"], linewidth=1, linestyle="--", label="σ_a", color="tab:green")
    ax.set_ylabel("σ values")
    ax.set_title("Volatility: σ, σ̃, σ_a")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3-1b) delta_mu & delta_sigma
    ax = fig3.add_subplot(gs3[0, 1])
    if loss is not None:
        ax.plot(loss.index, smooth(loss["delta_mu"], 20), linewidth=1.2, label="Δμ", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(loss.index, smooth(loss["delta_sigma"], 20), linewidth=1.2, label="Δσ", color="tab:red")
        ax.set_ylabel("Δμ", color="tab:blue")
        ax2.set_ylabel("Δσ", color="tab:red")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Rate of change: Δμ, Δσ")
    ax.grid(True, alpha=0.3)

    # 3-2a) v_sigma (sigma velocity)
    ax = fig3.add_subplot(gs3[1, 0])
    if loss is not None:
        ax.plot(loss.index, smooth(loss["v_sigma"], 20), linewidth=1.2, color="tab:red")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("v_σ")
    ax.set_title("Sigma velocity (v_sigma)")
    ax.grid(True, alpha=0.3)

    # 3-2b) delta_bar (smoothed delta)
    ax = fig3.add_subplot(gs3[1, 1])
    if loss is not None:
        ax.plot(loss.index, loss["delta_bar"], linewidth=1.2, color="tab:cyan")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("δ̄")
    ax.set_title("Smoothed delta (delta_bar)")
    ax.grid(True, alpha=0.3)

    # 3-3a) r_out (regime) & ell_out (level)
    ax = fig3.add_subplot(gs3[2, 0])
    if loss is not None:
        ax.plot(loss.index, smooth(loss["r_out"], 10), linewidth=1.2, label="r (regime)", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(loss.index, smooth(loss["ell_out"], 10), linewidth=1.2, label="ℓ (level)", color="tab:orange")
        ax.set_ylabel("r (regime)", color="tab:blue")
        ax2.set_ylabel("ℓ (level)", color="tab:orange")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Lattice outputs: regime & level")
    ax.grid(True, alpha=0.3)

    # 3-3b) mu_ex (excess mean) & c_out (confidence)
    ax = fig3.add_subplot(gs3[2, 1])
    if loss is not None:
        ax.plot(loss.index, smooth(loss["mu_ex"], 10), linewidth=1.2, label="μ_ex (excess)", color="tab:purple")
        ax2 = ax.twinx()
        ax2.plot(loss.index, smooth(loss["c_out"], 10), linewidth=1.2, label="c (confidence)", color="tab:green")
        ax.set_ylabel("μ_ex", color="tab:purple")
        ax2.set_ylabel("c (confidence)", color="tab:green")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Lattice outputs: excess mean & confidence")
    ax.grid(True, alpha=0.3)

    # 3-4a) mu_a (adapted mean) vs mu (running mean)
    ax = fig3.add_subplot(gs3[3, 0])
    if loss is not None:
        ax.plot(loss.index, loss["mu"], linewidth=1.2, label="μ (running)", color="tab:blue")
        ax.plot(loss.index, loss["mu_a"], linewidth=1.2, label="μ_a (adapted)", color="tab:orange")
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
    df_all = pd.read_csv(path)
    levels = sorted(df_all["level"].unique())
    if len(levels) > 1:
        fig4, axes = plt.subplots(len(levels), 1, figsize=(14, 3 * len(levels)),
                                   sharex=True, constrained_layout=True)
        fig4.suptitle("Loss across TelemetryLattice levels", fontsize=13, fontweight="bold")
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
        fig4.savefig(os.path.splitext(path)[0] + "_levels.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_levels.png")

    plt.show()

if __name__ == "__main__":
    main()
