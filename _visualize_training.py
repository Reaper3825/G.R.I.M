"""GRIM-text training visualization for the active ExecutionActive log."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

matplotlib.use("Agg")  # Non-interactive backend

BASE = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = BASE / "resources/models/GRIM-text/training/logs/ExecutionActive.log"

LOSS_RE = re.compile(
    r"\[LossStats\] batch=(?P<batch>\d+) loss_mean=(?P<loss>[0-9.eE+-]+).*?valid_tokens=(?P<valid>\d+)"
)
GRAD_RE = re.compile(
    r"\[GradTrace\] COMPONENTS\(rms\): emb_lm_tied=(?P<emb>[0-9.eE+-]+) "
    r"attn=(?P<attn>[0-9.eE+-]+) ffn=(?P<ffn>[0-9.eE+-]+) "
    r"rmsnorm=(?P<rmsnorm>[0-9.eE+-]+) tied=\w+ sb=(?P<sb>[0-9.eE+-]+)"
)
LOGIT_RE = re.compile(
    r"\[LogitSignal\] batch=(?P<batch>\d+) .*?top2_margin=(?P<margin>[0-9.eE+-]+) "
    r"argmax_top1_frac=(?P<top1>[0-9.eE+-]+) .*?unique_argmax=(?P<unique>\d+) "
    r"top_argmax=\[(?P<top>[^\]]*)\]"
)
TOP_TOKEN_RE = re.compile(r"tok(?P<token>\d+):(?P<count>\d+)")
SESSION_RE = re.compile(r"Session ID:\s*(?P<session>\d+)")
VOCAB_RE = re.compile(r"Loaded\s+(?P<vocab>\d+)\s+tokens")
MODEL_RE = re.compile(
    r"Model architecture: d_model=(?P<d_model>\d+), num_layers=(?P<num_layers>\d+), "
    r"num_heads=(?P<num_heads>\d+), d_ff=(?P<d_ff>\d+), max_seq_len=(?P<max_seq_len>\d+)"
)
PLANNED_BATCHES_RE = re.compile(r"Created\s+(?P<planned_batches>\d+)\s+dynamic batches")
RHO_ROW_RE = re.compile(
    r"^\s*(?P<layer>emb|L\d+)\s+(?P<rho>[0-9.]+)\s+(?:[+\-][0-9.]+|—)\s+(?P<hrms>[0-9.]+)\s+"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render GRIM-text diagnostic charts from ExecutionActive.log."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to the GRIM-text ExecutionActive.log file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE,
        help="Directory for generated _viz CSV files and chart PNGs.",
    )
    return parser.parse_args()


def smooth(y: np.ndarray, window: int = 50) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="valid")


def smooth_x(x: np.ndarray, window: int = 50) -> np.ndarray:
    if len(x) < window:
        return x
    return x[window // 2 : window // 2 + len(x) - window + 1]


def format_csv_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.10g}"


def write_csv(path: Path, rows: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(",".join(format_csv_value(float(value)) for value in row) + "\n")


def require_metadata(metadata: dict[str, int | None], key: str) -> int:
    value = metadata.get(key)
    if value is None:
        raise RuntimeError(f"Missing required metadata '{key}' in {DEFAULT_LOG_PATH}")
    return int(value)


def parse_execution_log(log_path: Path) -> tuple[dict[str, int | None], dict[str, np.ndarray]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Execution log not found: {log_path}")

    metadata: dict[str, int | None] = {
        "session_id": None,
        "vocab_size": None,
        "d_model": None,
        "num_layers": None,
        "num_heads": None,
        "d_ff": None,
        "max_seq_len": None,
        "planned_batches": None,
    }

    loss_rows: list[list[float]] = []
    grad_rows: list[list[float]] = []
    collapse_rows: list[list[float]] = []
    logit_rows: list[list[float]] = []
    rho_rows: list[list[float]] = []

    current_batch: int | None = None
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        if metadata["session_id"] is None:
            session_match = SESSION_RE.search(line)
            if session_match:
                metadata["session_id"] = int(session_match.group("session"))

        if metadata["vocab_size"] is None:
            vocab_match = VOCAB_RE.search(line)
            if vocab_match:
                metadata["vocab_size"] = int(vocab_match.group("vocab"))

        if metadata["d_model"] is None:
            model_match = MODEL_RE.search(line)
            if model_match:
                for key, value in model_match.groupdict().items():
                    metadata[key] = int(value)

        if metadata["planned_batches"] is None:
            planned_match = PLANNED_BATCHES_RE.search(line)
            if planned_match:
                metadata["planned_batches"] = int(planned_match.group("planned_batches"))

        loss_match = LOSS_RE.search(line)
        if loss_match:
            current_batch = int(loss_match.group("batch"))
            loss_rows.append(
                [
                    float(current_batch),
                    float(loss_match.group("loss")),
                    float(loss_match.group("valid")),
                ]
            )
            i += 1
            continue

        grad_match = GRAD_RE.search(line)
        if grad_match:
            if current_batch is None:
                raise RuntimeError(
                    "Encountered gradient component log before establishing a batch number"
                )
            grad_rows.append(
                [
                    float(current_batch),
                    float(grad_match.group("emb")),
                    float(grad_match.group("attn")),
                    float(grad_match.group("ffn")),
                    float(grad_match.group("rmsnorm")),
                    float(grad_match.group("sb")),
                ]
            )
            i += 1
            continue

        logit_match = LOGIT_RE.search(line)
        if logit_match:
            current_batch = int(logit_match.group("batch"))
            top2_margin = float(logit_match.group("margin"))
            top1_frac = float(logit_match.group("top1"))
            unique_argmax = float(logit_match.group("unique"))
            top_entries = TOP_TOKEN_RE.findall(logit_match.group("top"))
            if not top_entries:
                raise RuntimeError(
                    f"Unable to parse top_argmax entries for batch {current_batch}"
                )

            dominant_token = int(top_entries[0][0])
            dominant_count = int(top_entries[0][1])
            if top1_frac <= 0.0:
                raise RuntimeError(
                    f"argmax_top1_frac must be > 0 for batch {current_batch}, got {top1_frac}"
                )
            total_samples = int(round(dominant_count / top1_frac))
            if total_samples <= 0:
                raise RuntimeError(
                    f"Derived non-positive dominant-token sample count for batch {current_batch}"
                )

            logit_rows.append(
                [
                    float(current_batch),
                    top2_margin,
                    top1_frac,
                    unique_argmax,
                ]
            )
            collapse_rows.append(
                [
                    float(current_batch),
                    float(dominant_token),
                    float(dominant_count),
                    float(total_samples),
                    top1_frac * 100.0,
                ]
            )
            i += 1
            continue

        if "[RHO_BUILDUP_EQUATION]" in line:
            if current_batch is None:
                raise RuntimeError(
                    "Encountered a rho block before establishing a batch number"
                )

            layer_values: dict[str, tuple[float, float]] = {}
            j = i + 1
            while j < len(lines):
                rho_match = RHO_ROW_RE.match(lines[j])
                if rho_match:
                    layer_name = rho_match.group("layer")
                    layer_values[layer_name] = (
                        float(rho_match.group("rho")),
                        float(rho_match.group("hrms")),
                    )
                    j += 1
                    continue
                if lines[j].strip().startswith("SUMMARY:"):
                    break
                if lines[j].startswith("["):
                    break
                j += 1

            all_layer_names = ["emb"] + [f"L{i}" for i in range(12)]
            missing_layers = [layer for layer in all_layer_names if layer not in layer_values]
            if missing_layers:
                raise RuntimeError(
                    f"Missing rho rows {missing_layers} for batch {current_batch}"
                )

            row: list[float] = [float(current_batch)]
            for layer_name in all_layer_names:
                row.append(layer_values[layer_name][0])  # rho
                row.append(layer_values[layer_name][1])  # hrms
            rho_rows.append(row)
            i = j + 1
            continue

        i += 1

    if not loss_rows:
        raise RuntimeError(f"No [LossStats] rows found in {log_path}")
    if not grad_rows:
        raise RuntimeError(f"No [GradTrace] COMPONENTS(rms) rows found in {log_path}")
    if not collapse_rows:
        raise RuntimeError(f"No dominant-token collapse rows found in {log_path}")
    if not logit_rows:
        raise RuntimeError(f"No [LogitSignal] rows found in {log_path}")
    if not rho_rows:
        raise RuntimeError(f"No rho blocks found in {log_path}")

    arrays = {
        "loss": np.asarray(loss_rows, dtype=float),
        "grad": np.asarray(grad_rows, dtype=float),
        "collapse": np.asarray(collapse_rows, dtype=float),
        "logit": np.asarray(logit_rows, dtype=float),
        "rho": np.asarray(rho_rows, dtype=float),
    }
    return metadata, arrays


def save_intermediate_csvs(output_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    csv_map = {
        "_viz_loss.csv": arrays["loss"],
        "_viz_grad.csv": arrays["grad"],
        "_viz_collapse.csv": arrays["collapse"],
        "_viz_logitsignal.csv": arrays["logit"],
        "_viz_rho.csv": arrays["rho"],
    }
    for name, rows in csv_map.items():
        write_csv(output_dir / name, rows)


def render_charts(
    output_dir: Path, metadata: dict[str, int | None], arrays: dict[str, np.ndarray]
) -> None:
    session_id = require_metadata(metadata, "session_id")
    vocab_size = require_metadata(metadata, "vocab_size")
    d_model = require_metadata(metadata, "d_model")
    num_layers = require_metadata(metadata, "num_layers")
    num_heads = require_metadata(metadata, "num_heads")

    loss_data = arrays["loss"]
    grad_data = arrays["grad"]
    collapse_data = arrays["collapse"]
    logit_data = arrays["logit"]
    rho_data = arrays["rho"]

    batches = loss_data[:, 0]
    losses = loss_data[:, 1]
    gb = grad_data[:, 0]
    emb_g = grad_data[:, 1]
    attn_g = grad_data[:, 2]
    ffn_g = grad_data[:, 3]
    rmsn_g = grad_data[:, 4]
    sb_g = grad_data[:, 5]
    cb = collapse_data[:, 0]
    ctok = collapse_data[:, 1]
    cpct = collapse_data[:, 4]
    lb = logit_data[:, 0]
    margin = logit_data[:, 1]
    unique = logit_data[:, 3]
    rb = rho_data[:, 0]
    # All-layer columns: batch, emb_rho, emb_hrms, L0_rho, L0_hrms, ..., L11_rho, L11_hrms
    all_layer_names = ["emb"] + [f"L{i}" for i in range(12)]
    # rho column for layer i is at index 1 + 2*i, hrms at 1 + 2*i + 1
    emb_rho = rho_data[:, 1]
    emb_hrms = rho_data[:, 2]
    l0_rho = rho_data[:, 3]
    l0_hrms = rho_data[:, 4]
    l11_rho = rho_data[:, 25]  # L11 = index 12 → col 1+2*12=25
    l11_hrms = rho_data[:, 26]

    random_baseline = math.log(vocab_size)
    dominant_token_mode = Counter(ctok.astype(int)).most_common(1)[0][0]
    collapse_baseline = 100.0 / vocab_size
    parsed_batches = int(
        max(
            loss_data[-1, 0],
            grad_data[-1, 0],
            collapse_data[-1, 0],
            logit_data[-1, 0],
            rho_data[-1, 0],
        )
    )
    planned_batches = metadata.get("planned_batches")

    loss_ymin = max(0.0, min(float(losses.min()), random_baseline) - 0.5)
    loss_ymax = max(float(losses.max()), random_baseline) + 0.5
    ratio = emb_g / (np.sqrt(attn_g**2 + ffn_g**2 + rmsn_g**2 + sb_g**2) + 1e-12)
    rho_ymax = min(
        1.0,
        max(0.6, float(np.max(np.concatenate([emb_rho, l0_rho, l11_rho]))) * 1.08),
    )
    collapse_ymax = min(100.0, max(15.0, float(cpct.max()) * 1.05))
    unique_ymax = max(10.0, float(unique.max()) + 5.0)

    # ═══════════════════════════════════════════════════════════════
    # CHART 1: Training Loss
    # ═══════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(batches, losses, alpha=0.15, color="steelblue", linewidth=0.5, label="Raw loss")
    ax1.plot(
        smooth_x(batches),
        smooth(losses),
        color="steelblue",
        linewidth=2,
        label="Smoothed (50-batch)",
    )
    ax1.axhline(
        y=random_baseline,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Random baseline ln({vocab_size})={random_baseline:.2f}",
    )
    ax1.set_xlabel("Batch", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.set_title(
        "CHART 1: Training Loss — Current Run",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(loss_ymin, loss_ymax)
    fig1.tight_layout()
    fig1.savefig(str(output_dir / "chart1_loss.png"), dpi=150)
    plt.close(fig1)

    # ═══════════════════════════════════════════════════════════════
    # CHART 2: Gradient Components Over Time
    # ═══════════════════════════════════════════════════════════════
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax2a.semilogy(
        smooth_x(gb),
        smooth(emb_g),
        linewidth=2,
        label="emb/lm_head (tied)",
        color="#e74c3c",
    )
    ax2a.semilogy(smooth_x(gb), smooth(ffn_g), linewidth=2, label="FFN", color="#3498db")
    ax2a.semilogy(
        smooth_x(gb), smooth(rmsn_g), linewidth=2, label="RMSNorm", color="#2ecc71"
    )
    ax2a.semilogy(
        smooth_x(gb), smooth(attn_g), linewidth=2, label="Attention", color="#9b59b6"
    )
    ax2a.semilogy(
        smooth_x(gb), smooth(sb_g), linewidth=2, label="ScratchBlock", color="#f39c12"
    )
    ax2a.set_ylabel("Gradient RMS (log scale)", fontsize=12)
    ax2a.set_title(
        "CHART 2a: Gradient Components — Who Is Learning?",
        fontsize=14,
        fontweight="bold",
    )
    ax2a.legend(fontsize=9, loc="upper right")
    ax2a.grid(True, alpha=0.3)

    ax2b.plot(smooth_x(gb), smooth(ratio), linewidth=2, color="#e74c3c")
    ax2b.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Balanced (ratio=1)")
    ax2b.fill_between(smooth_x(gb), 0, smooth(ratio), alpha=0.1, color="#e74c3c")
    ax2b.set_xlabel("Batch", fontsize=12)
    ax2b.set_ylabel("emb_grad / encoder_grad", fontsize=12)
    ax2b.set_title(
        "CHART 2b: Gradient Imbalance — Embedding vs Encoder",
        fontsize=14,
        fontweight="bold",
    )
    ax2b.legend(fontsize=10)
    ax2b.grid(True, alpha=0.3)
    ax2b.set_ylim(0, max(10.0, float(np.percentile(ratio, 99)) * 1.05))
    fig2.tight_layout()
    fig2.savefig(str(output_dir / "chart2_gradients.png"), dpi=150)
    plt.close(fig2)

    # ═══════════════════════════════════════════════════════════════
    # CHART 3: Dominant-token collapse
    # ═══════════════════════════════════════════════════════════════
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax3a.scatter(cb, cpct, s=8, alpha=0.4, color="#e74c3c", zorder=2)
    if len(cb) > 20:
        ax3a.plot(
            smooth_x(cb, 20),
            smooth(cpct, 20),
            linewidth=2.5,
            color="darkred",
            label="Smoothed (20-sample)",
            zorder=3,
        )
    ax3a.axhline(
        y=collapse_baseline,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Single-token uniform prob ({collapse_baseline:.3f}%)",
    )
    ax3a.axhline(y=10.0, color="orange", linestyle="--", alpha=0.5, label="Concern threshold (10%)")
    ax3a.set_ylabel("Dominant Token %", fontsize=12)
    ax3a.set_title(
        "CHART 3a: Token Collapse — One Token Dominates Predictions",
        fontsize=14,
        fontweight="bold",
    )
    ax3a.legend(fontsize=10)
    ax3a.grid(True, alpha=0.3)
    ax3a.set_ylim(0, collapse_ymax)

    ax3b.scatter(cb, ctok, s=8, alpha=0.5, color="#3498db")
    dominant_mask = ctok == dominant_token_mode
    if np.any(dominant_mask):
        ax3b.scatter(
            cb[dominant_mask],
            ctok[dominant_mask],
            s=15,
            alpha=0.7,
            color="red",
            label=f"Token {dominant_token_mode}",
            zorder=3,
        )
        ax3b.legend(fontsize=10)
    ax3b.set_xlabel("Batch", fontsize=12)
    ax3b.set_ylabel("Dominant Token ID", fontsize=12)
    ax3b.set_title("CHART 3b: Which Token Is Dominant?", fontsize=14, fontweight="bold")
    ax3b.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(str(output_dir / "chart3_collapse.png"), dpi=150)
    plt.close(fig3)

    # ═══════════════════════════════════════════════════════════════
    # CHART 4: Logit quality
    # ═══════════════════════════════════════════════════════════════
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax4a.plot(lb, margin, alpha=0.15, color="#3498db", linewidth=0.5)
    ax4a.plot(
        smooth_x(lb), smooth(margin), linewidth=2, color="#3498db", label="top2 margin (smoothed)"
    )
    ax4a.set_ylabel("Top-2 Logit Margin", fontsize=12)
    ax4a.set_title(
        "CHART 4a: Prediction Confidence — How Sure Is the Model?",
        fontsize=14,
        fontweight="bold",
    )
    ax4a.legend(fontsize=10)
    ax4a.grid(True, alpha=0.3)

    ax4b.plot(lb, unique, alpha=0.15, color="#2ecc71", linewidth=0.5)
    ax4b.plot(
        smooth_x(lb),
        smooth(unique),
        linewidth=2,
        color="#2ecc71",
        label="unique argmax tokens (smoothed)",
    )
    ax4b.axhline(
        y=float(unique[0]),
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Batch 1 baseline ({int(unique[0])})",
    )
    ax4b.set_xlabel("Batch", fontsize=12)
    ax4b.set_ylabel("Unique Argmax Tokens", fontsize=12)
    ax4b.set_title(
        "CHART 4b: Vocabulary Diversity — How Many Tokens Does Model Use?",
        fontsize=14,
        fontweight="bold",
    )
    ax4b.legend(fontsize=10)
    ax4b.grid(True, alpha=0.3)
    ax4b.set_ylim(0, unique_ymax)
    fig4.tight_layout()
    fig4.savefig(str(output_dir / "chart4_logit_quality.png"), dpi=150)
    plt.close(fig4)

    # ═══════════════════════════════════════════════════════════════
    # CHART 5: Directional similarity (rho)
    # ═══════════════════════════════════════════════════════════════
    fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax5a.plot(
        smooth_x(rb, 30), smooth(emb_rho, 30), linewidth=2, label="ρ(emb) — embedding", color="#3498db"
    )
    ax5a.plot(
        smooth_x(rb, 30), smooth(l0_rho, 30), linewidth=2, label="ρ(L0) — first encoder", color="#e74c3c"
    )
    ax5a.plot(
        smooth_x(rb, 30), smooth(l11_rho, 30), linewidth=2, label="ρ(L11) — last encoder", color="#e67e22"
    )
    ax5a.set_ylabel("ρ = avg|cos(h_i, h_j)|", fontsize=12)
    ax5a.set_title(
        "CHART 5a: Directional Similarity ρ — Are Hidden States Aligning?",
        fontsize=14,
        fontweight="bold",
    )
    ax5a.legend(fontsize=10)
    ax5a.grid(True, alpha=0.3)
    ax5a.set_ylim(0, rho_ymax)

    ax5b.semilogy(smooth_x(rb, 30), smooth(emb_hrms, 30), linewidth=2, label="h_rms(emb)", color="#3498db")
    ax5b.semilogy(smooth_x(rb, 30), smooth(l0_hrms, 30), linewidth=2, label="h_rms(L0)", color="#e74c3c")
    ax5b.semilogy(smooth_x(rb, 30), smooth(l11_hrms, 30), linewidth=2, label="h_rms(L11)", color="#e67e22")
    ax5b.set_xlabel("Batch", fontsize=12)
    ax5b.set_ylabel("h_rms (log scale)", fontsize=12)
    ax5b.set_title("CHART 5b: Activation Magnitude Per Layer", fontsize=14, fontweight="bold")
    ax5b.legend(fontsize=10)
    ax5b.grid(True, alpha=0.3)
    fig5.tight_layout()
    fig5.savefig(str(output_dir / "chart5_rho_direction.png"), dpi=150)
    plt.close(fig5)

    # ═══════════════════════════════════════════════════════════════
    # CHART 6: Combined dashboard
    # ═══════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════
    # CHART 7: All-layer ρ and h_rms profile
    # ═══════════════════════════════════════════════════════════════
    cmap = plt.get_cmap("coolwarm")
    layer_colors = [cmap(k / 12) for k in range(13)]

    fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for k, layer_name in enumerate(all_layer_names):
        rho_col = rho_data[:, 1 + 2 * k]
        ax7a.plot(
            smooth_x(rb, 30),
            smooth(rho_col, 30),
            linewidth=1.6,
            label=layer_name,
            color=layer_colors[k],
            alpha=0.85,
        )
    ax7a.set_ylabel("ρ = avg|cos(h_i, h_j)|", fontsize=12)
    ax7a.set_title(
        "CHART 7a: Per-Layer Directional Similarity ρ(l)",
        fontsize=14,
        fontweight="bold",
    )
    ax7a.legend(fontsize=8, ncol=4, loc="upper right")
    ax7a.grid(True, alpha=0.3)
    ax7a.set_ylim(0, rho_ymax)

    for k, layer_name in enumerate(all_layer_names):
        hrms_col = rho_data[:, 1 + 2 * k + 1]
        ax7b.semilogy(
            smooth_x(rb, 30),
            smooth(hrms_col, 30),
            linewidth=1.6,
            label=layer_name,
            color=layer_colors[k],
            alpha=0.85,
        )
    ax7b.set_xlabel("Batch", fontsize=12)
    ax7b.set_ylabel("h_rms (log scale)", fontsize=12)
    ax7b.set_title(
        "CHART 7b: Per-Layer Activation Magnitude h_rms",
        fontsize=14,
        fontweight="bold",
    )
    ax7b.legend(fontsize=8, ncol=4, loc="upper left")
    ax7b.grid(True, alpha=0.3)
    fig7.tight_layout()
    fig7.savefig(str(output_dir / "chart7_all_layer_rho.png"), dpi=150)
    plt.close(fig7)

    fig6 = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig6, hspace=0.35, wspace=0.25)

    ax6a = fig6.add_subplot(gs[0, 0])
    ax6a.plot(smooth_x(batches), smooth(losses), linewidth=2, color="steelblue")
    ax6a.axhline(y=random_baseline, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax6a.set_title("Loss", fontsize=12, fontweight="bold")
    ax6a.set_ylabel("CE Loss")
    ax6a.grid(True, alpha=0.3)
    ax6a.set_ylim(loss_ymin, loss_ymax)

    ax6b = fig6.add_subplot(gs[0, 1])
    if len(cb) > 20:
        ax6b.plot(smooth_x(cb, 20), smooth(cpct, 20), linewidth=2, color="#e74c3c")
    else:
        ax6b.plot(cb, cpct, linewidth=2, color="#e74c3c")
    ax6b.axhline(y=10.0, color="orange", linestyle="--", alpha=0.5)
    ax6b.set_title(f"Token {dominant_token_mode} Dominance %", fontsize=12, fontweight="bold")
    ax6b.set_ylabel("% argmax")
    ax6b.grid(True, alpha=0.3)
    ax6b.set_ylim(0, collapse_ymax)

    ax6c = fig6.add_subplot(gs[1, 0])
    ax6c.semilogy(smooth_x(gb), smooth(emb_g), linewidth=2, label="emb", color="#e74c3c")
    ax6c.semilogy(smooth_x(gb), smooth(ffn_g), linewidth=2, label="FFN", color="#3498db")
    ax6c.semilogy(smooth_x(gb), smooth(sb_g), linewidth=2, label="SB", color="#f39c12")
    ax6c.set_title("Gradient RMS by Component", fontsize=12, fontweight="bold")
    ax6c.set_ylabel("Grad RMS (log)")
    ax6c.legend(fontsize=8)
    ax6c.grid(True, alpha=0.3)

    ax6d = fig6.add_subplot(gs[1, 1])
    ax6d.plot(smooth_x(lb), smooth(unique), linewidth=2, color="#2ecc71")
    ax6d.axhline(y=float(unique[0]), color="gray", linestyle="--", alpha=0.5)
    ax6d.set_title("Unique Argmax Tokens", fontsize=12, fontweight="bold")
    ax6d.set_ylabel("Count")
    ax6d.grid(True, alpha=0.3)
    ax6d.set_ylim(0, unique_ymax)

    ax6e = fig6.add_subplot(gs[2, 0])
    ax6e.plot(smooth_x(rb, 30), smooth(l0_rho, 30), linewidth=2, label="ρ(L0)", color="#e74c3c")
    ax6e.plot(smooth_x(rb, 30), smooth(l11_rho, 30), linewidth=2, label="ρ(L11)", color="#e67e22")
    ax6e.plot(smooth_x(rb, 30), smooth(emb_rho, 30), linewidth=2, label="ρ(emb)", color="#3498db")
    ax6e.set_title("Directional Similarity ρ", fontsize=12, fontweight="bold")
    ax6e.set_ylabel("ρ")
    ax6e.set_xlabel("Batch")
    ax6e.legend(fontsize=8)
    ax6e.grid(True, alpha=0.3)
    ax6e.set_ylim(0, rho_ymax)

    ax6f = fig6.add_subplot(gs[2, 1])
    ax6f.semilogy(smooth_x(rb, 30), smooth(l11_hrms, 30), linewidth=2, label="h_rms(L11)", color="#e67e22")
    ax6f.semilogy(smooth_x(rb, 30), smooth(l0_hrms, 30), linewidth=2, label="h_rms(L0)", color="#e74c3c")
    ax6f.set_title("Activation Scale (h_rms)", fontsize=12, fontweight="bold")
    ax6f.set_ylabel("h_rms (log)")
    ax6f.set_xlabel("Batch")
    ax6f.legend(fontsize=8)
    ax6f.grid(True, alpha=0.3)

    summary = (
        f"GRIM-text Training Dashboard — Session {session_id}\n"
        f"{parsed_batches} batches parsed"
    )
    if planned_batches is not None:
        summary += f" / {planned_batches} planned"
    summary += f", vocab={vocab_size}, d_model={d_model}, layers={num_layers}, heads={num_heads}"
    fig6.suptitle(summary, fontsize=14, fontweight="bold", y=0.99)
    fig6.savefig(str(output_dir / "chart6_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig6)


def main() -> None:
    args = parse_args()
    log_path = args.log.resolve()
    output_dir = args.output_dir.resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    metadata, arrays = parse_execution_log(log_path)
    save_intermediate_csvs(output_dir, arrays)
    render_charts(output_dir, metadata, arrays)

    print(f"✓ Parsed current run from {log_path}")
    print(f"✓ Wrote refreshed _viz CSV files to {output_dir}")
    print(f"✓ Saved chart1_loss.png to {output_dir}")
    print(f"✓ Saved chart2_gradients.png to {output_dir}")
    print(f"✓ Saved chart3_collapse.png to {output_dir}")
    print(f"✓ Saved chart4_logit_quality.png to {output_dir}")
    print(f"✓ Saved chart5_rho_direction.png to {output_dir}")
    print(f"✓ Saved chart6_dashboard.png to {output_dir}")
    print(f"✓ Saved chart7_all_layer_rho.png to {output_dir}")


if __name__ == "__main__":
    main()
