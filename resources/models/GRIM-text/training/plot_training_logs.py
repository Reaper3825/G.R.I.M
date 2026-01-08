import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt

FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
LOSS_STATS_RE = re.compile(rf"\[LossStats\] batch=(\d+) loss_mean=({FLOAT_RE})")
LOSS_FALLBACK_RE = re.compile(rf"\[GradTrace\] POST-FORWARD loss=({FLOAT_RE})")
BATCH_RE = re.compile(r"\[GradTrace\] BATCH_INFO batch=(\d+)")
GRAD_NORM_RE = re.compile(rf"\[GradTrace\] POST-GRADNORM preclip=({FLOAT_RE})")
COMP_RE = re.compile(
    rf"\[GradTrace\] COMPUTED COMPONENTS: total=({FLOAT_RE}) "
    rf"emb_lm_tied=({FLOAT_RE}) attn=({FLOAT_RE}) ffn=({FLOAT_RE}) rms=({FLOAT_RE})"
)

@dataclass
class Series:
    name: str
    loss: List[Tuple[int, float]]
    grad_norm: List[Tuple[int, float]]
    components: List[Tuple[int, float, float, float, float, float]]


def parse_log(path: str) -> Series:
    loss = []
    grad_norm = []
    components = []
    batch_idx = 0
    fallback_batch = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = BATCH_RE.search(line)
            if m:
                batch_idx = int(m.group(1))
                fallback_batch = batch_idx
                continue

            m = LOSS_STATS_RE.search(line)
            if m:
                batch_idx = int(m.group(1))
                fallback_batch = batch_idx
                loss.append((batch_idx, float(m.group(2))))
                continue

            m = LOSS_FALLBACK_RE.search(line)
            if m:
                if batch_idx == 0:
                    fallback_batch += 1
                    batch_idx = fallback_batch
                loss.append((batch_idx, float(m.group(1))))
                continue

            m = GRAD_NORM_RE.search(line)
            if m:
                grad_norm.append((batch_idx, float(m.group(1))))
                continue

            m = COMP_RE.search(line)
            if m:
                components.append(
                    (
                        batch_idx,
                        float(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                        float(m.group(4)),
                        float(m.group(5)),
                    )
                )
                continue

    name = os.path.basename(path)
    return Series(name=name, loss=loss, grad_norm=grad_norm, components=components)


def filter_series(series: Series, min_loss: float) -> Series:
    if min_loss <= 0:
        return series
    loss = [(b, v) for b, v in series.loss if v >= min_loss]
    return Series(name=series.name, loss=loss, grad_norm=series.grad_norm, components=series.components)


def make_x(values: List[Tuple[int, float]], mode: str) -> List[float]:
    if mode == "batch":
        return [v[0] for v in values]
    if mode == "normalized":
        if len(values) <= 1:
            return [0.0 for _ in values]
        return [idx / (len(values) - 1) for idx in range(len(values))]
    return list(range(1, len(values) + 1))


def rolling_std(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return [0.0 for _ in values]
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        mean = sum(chunk) / len(chunk)
        var = sum((v - mean) ** 2 for v in chunk) / len(chunk)
        out.append(var ** 0.5)
    return out


def trim_initial_zeros(values: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    if not values:
        return values
    start = 0
    while start < len(values) and values[start][1] == 0.0:
        start += 1
    return values[start:] if start < len(values) else values


def plot_series(
    series: List[Series],
    output: str,
    components_breakdown: bool,
    x_mode: str,
    min_points: int,
    log_grad: bool,
    log_components: bool,
    std_window: int,
    loss_min: float,
    loss_max: float,
    log_loss: bool,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    for s in series:
        loss_values = trim_initial_zeros(s.loss)
        if len(loss_values) >= min_points:
            x = make_x(loss_values, x_mode)
            y = [v[1] for v in loss_values]
            axes[0].plot(x, y, label=s.name, linewidth=0.9, alpha=0.75)
            if std_window > 1:
                std_y = rolling_std(y, std_window)
                axes[3].plot(x, std_y, label=s.name, linewidth=0.9, alpha=0.75)

        if len(s.grad_norm) >= min_points:
            x = make_x(s.grad_norm, x_mode)
            y = [v[1] for v in s.grad_norm]
            axes[1].plot(x, y, label=s.name, linewidth=0.9, alpha=0.75)

        if len(s.components) >= min_points:
            x = make_x([(v[0], v[1]) for v in s.components], x_mode)
            total = [v[1] for v in s.components]
            axes[2].plot(x, total, label=f"{s.name} total", linewidth=0.9, alpha=0.75)
            if components_breakdown:
                emb = [v[2] for v in s.components]
                attn = [v[3] for v in s.components]
                ffn = [v[4] for v in s.components]
                axes[2].plot(x, emb, label=f"{s.name} emb", linewidth=0.6, alpha=0.5)
                axes[2].plot(x, attn, label=f"{s.name} attn", linewidth=0.6, alpha=0.5)
                axes[2].plot(x, ffn, label=f"{s.name} ffn", linewidth=0.6, alpha=0.5)

    axes[0].set_title("Loss (loss_mean)")
    axes[0].set_ylabel("loss_mean")
    axes[0].grid(True, alpha=0.2)
    if log_loss:
        axes[0].set_yscale("symlog", linthresh=1.0)
    if loss_max > 0.0:
        axes[0].set_ylim(loss_min, loss_max)

    axes[1].set_title("Grad Norm (preclip)")
    axes[1].set_ylabel("grad_norm")
    axes[1].grid(True, alpha=0.2)
    if log_grad:
        axes[1].set_yscale("symlog", linthresh=1.0)

    axes[2].set_title("Computed Components")
    axes[2].set_ylabel("component value")
    axes[2].grid(True, alpha=0.2)
    if log_components:
        axes[2].set_yscale("symlog", linthresh=1.0)

    axes[3].set_title("Loss Rolling Std")
    axes[3].set_ylabel("loss_std")
    axes[3].grid(True, alpha=0.2)
    axes[3].set_ylim(0.0, 50.0)

    axes[3].set_xlabel("batch" if x_mode == "batch" else "step")

    for ax in axes:
        ax.legend(fontsize=7, ncol=1, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(output, dpi=150)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot training logs (loss/grad/components)")
    parser.add_argument(
        "--glob",
        default="resources/models/GRIM-text/training/logs/training_*.log",
        help="Glob for log files",
    )
    parser.add_argument(
        "--output",
        default="resources/models/GRIM-text/training/logs/training_plot.png",
        help="Output image path",
    )
    parser.add_argument(
        "--components-breakdown",
        action="store_true",
        help="Plot component breakdown (emb/attn/ffn) per log",
    )
    parser.add_argument(
        "--x-mode",
        choices=["batch", "index", "normalized"],
        default="normalized",
        help="X axis mode (batch id, index per series, or normalized)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=50,
        help="Minimum points required to plot a series for a metric",
    )
    parser.add_argument(
        "--log-grad",
        action="store_true",
        help="Use symlog scale for grad norm plot",
    )
    parser.add_argument(
        "--log-components",
        action="store_true",
        help="Use symlog scale for components plot",
    )
    parser.add_argument(
        "--log-loss",
        action="store_true",
        help="Use symlog scale for loss plot",
    )
    parser.add_argument(
        "--loss-min",
        type=float,
        default=0.0,
        help="Minimum y value for loss plot (ignored if loss-max <= 0)",
    )
    parser.add_argument(
        "--loss-max",
        type=float,
        default=50.0,
        help="Maximum y value for loss plot (<=0 disables clamp)",
    )
    parser.add_argument(
        "--std-window",
        type=int,
        default=25,
        help="Rolling window for loss std plot",
    )
    parser.add_argument(
        "--min-loss",
        type=float,
        default=1e-6,
        help="Drop loss_mean values below this threshold",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"No logs matched {args.glob}")
        return 1

    series = [filter_series(parse_log(p), args.min_loss) for p in paths]
    plot_series(
        series,
        args.output,
        args.components_breakdown,
        args.x_mode,
        args.min_points,
        args.log_grad,
        args.log_components,
        args.std_window,
        args.loss_min,
        args.loss_max,
        args.log_loss,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
