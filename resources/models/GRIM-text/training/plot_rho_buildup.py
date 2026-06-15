#!/usr/bin/env python3
"""Plot RHO_BUILDUP_EQUATION layer metrics from GRIM-text training logs.

Parses each RHO_BUILDUP_EQUATION / RHO_BUILDUP_EQUATION_PRE_BACKWARD block and
writes one time-series PNG per (layer, metric) pair under the output directory.

Default log target: the most recently modified training_*.log in ./logs/.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs"

RHO_TAGS = (
    "RHO_BUILDUP_EQUATION_PRE_BACKWARD",
    "RHO_BUILDUP_EQUATION",
)

METRICS: tuple[tuple[str, str], ...] = (
    ("rho", "ρ(l)"),
    ("delta_rho", "Δρ(vs)"),
    ("h_rms_avg", "h_rms_avg"),
    ("avg_abs_dot", "avg_abs_dot"),
    ("avg_signed_dot", "avg_signed_dot"),
    ("raw_dot_sum", "raw_dot_sum"),
    ("centered_avg_abs_dot", "centered_avg_abs_dot"),
    ("mean_rms", "mean_rms"),
    ("mu_avg", "mu_avg"),
    ("mu_scale", "mu_scale"),
)

PHASE_LABELS = {
    "PRE": "post_forward_pre_backward",
    "POST": "post_backward",
}

LAYER_ORDER = ["emb"] + [f"L{i}" for i in range(32)] + ["lm_in"]


def find_latest_training_log(log_dir: Path) -> Path:
    candidates = [
        p
        for p in log_dir.glob("training_*.log")
        if re.fullmatch(r"training_\d+\.log", p.name)
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No training_<session_id>.log files found in {log_dir}"
        )
    return candidates[-1]


def fnum(text: str) -> float | None:
    value = text.strip().strip("x").strip("%")
    value = value.replace("+", "")
    if not value or value in {"—", "\u2014", "-"}:
        return None
    try:
        parsed = float(value)
        if not np.isfinite(parsed):
            return None
        return parsed
    except ValueError:
        return None


def parse_delta_rho(token: str) -> float | None:
    token = token.strip()
    if token in {"—", "\u2014", "-"}:
        return None
    match = re.match(r"^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", token)
    if not match:
        return None
    return fnum(match.group(1))


def is_layer_label(token: str) -> bool:
    if token in ("emb", "lm_in"):
        return True
    return len(token) >= 2 and token[0] == "L" and token[1:].isdigit()


def is_timestamp_line(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped.startswith("["):
        return False
    end = stripped.find("]")
    if end <= 1:
        return False
    candidate = stripped[1:end]
    if len(candidate) < 19:
        return False
    return (
        candidate[4] == "-"
        and candidate[7] == "-"
        and candidate[10] == " "
        and candidate[13] == ":"
        and candidate[16] == ":"
    )


def rho_tag_in_line(line: str) -> str | None:
    for tag in RHO_TAGS:
        if tag in line and "phase=" in line:
            return tag
    return None


@dataclass
class RhoBlock:
    tag: str
    phase: str
    batch: int | None
    step: int | None
    loss: float | None
    line_no: int
    layers: dict[str, dict[str, float | None]] = field(default_factory=dict)


def parse_rho_blocks(path: Path, *, include_pre: bool = True) -> list[RhoBlock]:
    blocks: list[RhoBlock] = []
    cur_batch: int | None = None
    cur_step: int | None = None
    cur_loss: float | None = None

    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        if "EXPLICIT_TRAINING_FORWARD_COMPLETE" in line and "batch=" in line:
            try:
                cur_batch = int(line.split("batch=")[1].split()[0])
            except (IndexError, ValueError):
                pass

        if "[Step " in line and "loss=" in line:
            try:
                cur_step = int(line.split("[Step ")[1].split("]")[0])
                cur_loss = fnum(line.split("loss=")[1].split()[0])
            except (IndexError, ValueError):
                pass

        tag = rho_tag_in_line(line)
        if tag is None:
            i += 1
            continue

        phase = "PRE" if "PRE_BACKWARD" in tag else "POST"
        if phase == "PRE" and not include_pre:
            i += 1
            continue

        block = RhoBlock(
            tag=tag,
            phase=phase,
            batch=cur_batch,
            step=cur_step,
            loss=cur_loss,
            line_no=i + 1,
        )

        j = i + 1
        while j < n:
            lj = lines[j]
            if is_timestamp_line(lj) or rho_tag_in_line(lj) is not None:
                break

            stripped = lj.strip()
            if not stripped or stripped.startswith("─"):
                j += 1
                continue

            toks = stripped.split()
            if toks and is_layer_label(toks[0]):
                label = toks[0]
                try:
                    rho = fnum(toks[1])
                    delta_token = toks[2]
                    if delta_token in {"—", "\u2014"}:
                        idx = 3
                    else:
                        idx = 3
                    delta_rho = parse_delta_rho(delta_token)
                    hrms = fnum(toks[idx])
                    abs_dot = fnum(toks[idx + 1])
                    signed = fnum(toks[idx + 2])
                    rawsum = fnum(toks[idx + 3])
                    centered = fnum(toks[idx + 4])
                    meanrms = fnum(toks[idx + 5])
                    muavg = fnum(toks[idx + 6])
                    muscale = fnum(toks[idx + 7])
                    block.layers[label] = {
                        "rho": rho,
                        "delta_rho": delta_rho,
                        "h_rms_avg": hrms,
                        "avg_abs_dot": abs_dot,
                        "avg_signed_dot": signed,
                        "raw_dot_sum": rawsum,
                        "centered_avg_abs_dot": centered,
                        "mean_rms": meanrms,
                        "mu_avg": muavg,
                        "mu_scale": muscale,
                    }
                except (IndexError, ValueError):
                    pass

            j += 1

        if block.layers:
            blocks.append(block)
        i = j

    return blocks


def layer_sort_key(layer: str) -> tuple[int, str]:
    if layer == "emb":
        return (0, layer)
    if layer == "lm_in":
        return (2, layer)
    if layer.startswith("L") and layer[1:].isdigit():
        return (1, f"{int(layer[1:]):04d}")
    return (3, layer)


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 2:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def collect_series(
    blocks: list[RhoBlock],
    layer: str,
    metric: str,
    phase: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[RhoBlock]]:
    xs: list[float] = []
    ys: list[float] = []
    used_blocks: list[RhoBlock] = []

    for idx, block in enumerate(blocks):
        if phase is not None and block.phase != phase:
            continue
        row = block.layers.get(layer)
        if not row:
            continue
        value = row.get(metric)
        if value is None:
            continue
        x = block.batch if block.batch is not None else idx
        xs.append(float(x))
        ys.append(float(value))
        used_blocks.append(block)

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), used_blocks


def plot_layer_metric(
    blocks: list[RhoBlock],
    layer: str,
    metric_key: str,
    metric_label: str,
    out_path: Path,
    *,
    smooth_window: int,
    log_name: str,
) -> bool:
    phases_present = sorted({b.phase for b in blocks if layer in b.layers})
    if not phases_present:
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for phase in phases_present:
        xs, ys, _ = collect_series(blocks, layer, metric_key, phase=phase)
        if len(xs) == 0:
            continue
        color = "tab:blue" if phase == "PRE" else "tab:orange"
        label = PHASE_LABELS.get(phase, phase)
        ax.plot(xs, ys, alpha=0.25, linewidth=0.8, color=color)
        if smooth_window > 1 and len(ys) >= 3:
            ys_smooth = smooth_series(ys, smooth_window)
            ax.plot(xs, ys_smooth, linewidth=1.8, color=color, label=f"{label} (smooth)")
        else:
            ax.plot(xs, ys, linewidth=1.5, color=color, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_title(f"{metric_label} — {layer}\n{log_name}")
    ax.set_xlabel("batch")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def discover_layers(blocks: list[RhoBlock]) -> list[str]:
    found = {layer for block in blocks for layer in block.layers}
    return sorted(found, key=layer_sort_key)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot per-layer RHO_BUILDUP_EQUATION metrics from training logs.",
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default=None,
        help="Training log path (default: latest training_*.log in ./logs/)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory for PNG output (default: <log_stem>_rho_plots next to the log)",
    )
    parser.add_argument(
        "--post-only",
        action="store_true",
        help="Only parse RHO_BUILDUP_EQUATION (skip PRE_BACKWARD blocks)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Moving-average window for smoothed overlay (0 disables smoothing)",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help="Optional layer subset (e.g. emb L1 lm_in). Default: all layers seen in log.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric keys subset. Default: all table columns.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    log_path = Path(args.log_file) if args.log_file else find_latest_training_log(DEFAULT_LOG_DIR)
    if not log_path.is_file():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return 1

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else log_path.parent / f"{log_path.stem}_rho_plots"
    )

    blocks = parse_rho_blocks(log_path, include_pre=not args.post_only)
    if not blocks:
        print(f"No RHO_BUILDUP blocks found in {log_path}", file=sys.stderr)
        return 1

    layers = args.layers if args.layers else discover_layers(blocks)
    metric_items = [
        (key, label)
        for key, label in METRICS
        if args.metrics is None or key in args.metrics
    ]
    if args.metrics and not metric_items:
        print("No valid metrics selected.", file=sys.stderr)
        return 1

    print(f"Log: {log_path}")
    print(
        f"Blocks: {len(blocks)} "
        f"(PRE={sum(1 for b in blocks if b.phase == 'PRE')}, "
        f"POST={sum(1 for b in blocks if b.phase == 'POST')})"
    )
    print(f"Layers: {', '.join(layers)}")
    print(f"Output: {output_dir}")

    written = 0
    for layer in layers:
        for metric_key, metric_label in metric_items:
            out_path = output_dir / layer / f"{metric_key}.png"
            if plot_layer_metric(
                blocks,
                layer,
                metric_key,
                metric_label,
                out_path,
                smooth_window=max(0, args.smooth),
                log_name=log_path.name,
            ):
                written += 1

    print(f"Wrote {written} plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
