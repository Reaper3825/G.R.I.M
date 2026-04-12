#!/usr/bin/env python3
"""Simple TelemetryLattice CSV pattern viewer for GRIM-text training runs."""

import sys
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def find_default_telemetry_csv():
    """Find the most relevant telemetry CSV in the training logs directory."""
    log_dir = os.path.join(os.path.dirname(__file__),
                           "resources", "models", "GRIM-text", "training", "logs")

    candidates = [
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
    return max(unique_csvs, key=os.path.getmtime)

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
        import math
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
            vals = adam_sig_dom["raw_observation"].clip(upper=50)  # clip for readability
            ax.plot(adam_sig_dom.index, vals, linewidth=1.5, color="tab:green", label="signal dominance")
            ax.axhline(1.0, color="tab:red", linewidth=1.5, linestyle="--", label="crossover = 1.0")
            ax.fill_between(adam_sig_dom.index, 0, 1, where=vals < 1, alpha=0.1, color="tab:red", label="destroying (< 1)")
            ax.fill_between(adam_sig_dom.index, 1, vals, where=vals >= 1, alpha=0.1, color="tab:green", label="learning (≥ 1)")
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
            ax.plot(adam_disrupt.index, adam_disrupt["raw_observation"], linewidth=1.5, color="tab:orange", label="displacement / Xavier scale")
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
            ax.plot(loss.index, smooth(loss["raw_observation"]), linewidth=1.5, color="tab:blue", label="loss (smoothed)")
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
            ax.plot(grad_mean.index, smooth(grad_mean["raw_observation"]), linewidth=1, linestyle="--", color="tab:gray", alpha=0.6, label="total grad RMS")
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
            ax.plot(exec_op_ent.index, smooth(exec_op_ent["raw_observation"]), linewidth=1.2, color="tab:orange", linestyle="--", label="op entropy")
        ax.set_ylabel("Entropy (nats)")
        ax.set_title("Selection Entropy (↓ = sharpening)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 6-2b) Op Entropy momentum (p) — is it trending down?
        ax = fig6.add_subplot(gs6[1, 1])
        if exec_sel_ent is not None:
            ax.plot(exec_sel_ent.index, smooth(exec_sel_ent["p"], 10), linewidth=1.2, color="tab:purple", label="selection entropy p")
        if exec_op_ent is not None:
            ax.plot(exec_op_ent.index, smooth(exec_op_ent["p"], 10), linewidth=1.2, color="tab:orange", label="op entropy p")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("p (momentum)")
        ax.set_title("Entropy Momentum (< 0 = sharpening)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 6-3a) Div Clamp Rate + Max P Write
        ax = fig6.add_subplot(gs6[2, 0])
        if exec_div_clamp is not None:
            ax.plot(exec_div_clamp.index, smooth(exec_div_clamp["raw_observation"]), linewidth=1.5, color="tab:red", label="div clamp rate")
        ax.set_ylabel("Rate", color="tab:red")
        if exec_max_pw is not None:
            ax2 = ax.twinx()
            ax2.plot(exec_max_pw.index, smooth(exec_max_pw["raw_observation"]), linewidth=1.5, color="tab:green", label="max p(write)")
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

    # --- Figure 7: EB Injection Diagnostics ---
    eb_inject_gate = streams.get("eb_inject_gate")
    eb_read_gate = streams.get("eb_read_gate_mean")
    eb_inject_wnorm = streams.get("eb_inject_weight_norm")
    eb_read_wnorm = streams.get("eb_read_weight_norm")
    eb_loss_frac = streams.get("eb_loss_frac")
    sb_atom_rms = streams.get("sb_atom_embed_rms")

    has_inject_diag = any(s is not None for s in [eb_inject_gate, eb_read_gate, eb_inject_wnorm,
                                                   eb_read_wnorm, eb_loss_frac, sb_atom_rms])
    if has_inject_diag:
        fig7 = plt.figure(figsize=(16, 14), constrained_layout=True)
        fig7.suptitle("GRIM-text Telemetry — EB Injection Diagnostics", fontsize=14, fontweight="bold")
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
            ax.plot(eb_inject_gate.index, smooth(eb_inject_gate["p"], 10), linewidth=1.2, color="tab:blue", label="inject gate p")
        if eb_read_gate is not None and "p" in eb_read_gate.columns:
            ax.plot(eb_read_gate.index, smooth(eb_read_gate["p"], 10), linewidth=1.2, color="tab:red", label="read gate p")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("p (momentum)")
        ax.set_title("Gate Momentum (< 0 = closing gate)")
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
            ax.plot(eb_loss_frac.index, smooth(eb_loss_frac["raw_observation"]), linewidth=1.5, color="tab:orange", label="exec_loss / total_loss")
        ax.set_ylabel("Fraction")
        ax.set_title("EB Auxiliary Loss Fraction")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 7-3a) SB Atom Embed RMS
        ax = fig7.add_subplot(gs7[2, 0])
        if sb_atom_rms is not None:
            ax.plot(sb_atom_rms.index, sb_atom_rms["raw_observation"], alpha=0.3, linewidth=0.5, color="tab:green")
            ax.plot(sb_atom_rms.index, smooth(sb_atom_rms["raw_observation"]), linewidth=1.5, color="tab:green", label="atom_type_embeddings RMS")
        ax.set_ylabel("RMS")
        ax.set_title("ScratchBlock Atom Embedding Scale")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 7-3b) Combined: gates overlaid with loss
        ax = fig7.add_subplot(gs7[2, 1])
        if eb_inject_gate is not None:
            ax.plot(eb_inject_gate.index, smooth(eb_inject_gate["raw_observation"]), linewidth=1.2, color="tab:blue", label="inject gate")
        if eb_read_gate is not None:
            ax.plot(eb_read_gate.index, smooth(eb_read_gate["raw_observation"]), linewidth=1.2, color="tab:red", label="read gate")
        ax.set_ylabel("Gate Value", color="tab:blue")
        ax.set_ylim(-0.05, 1.05)
        if loss is not None:
            ax2 = ax.twinx()
            ax2.plot(loss.index, smooth(loss["raw_observation"]), linewidth=1, color="tab:gray", alpha=0.5, label="loss (sm20)")
            ax2.set_ylabel("Loss", color="tab:gray")
            ax2.legend(loc="center right", fontsize=8)
        ax.set_title("Gates vs Loss (correlation = causation hypothesis)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig7.savefig(os.path.splitext(path)[0] + "_eb_injection.png", dpi=150)
        print(f"Saved: {os.path.splitext(path)[0]}_eb_injection.png")
    else:
        print("EB injection diagnostics figure skipped: no injection streams in CSV")

    plt.show()

if __name__ == "__main__":
    main()
