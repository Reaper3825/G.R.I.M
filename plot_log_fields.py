#!/usr/bin/env python3
"""Plot numeric fields from GRIM logs with Matplotlib.

The plain-text log parser deliberately uses string operations only: no regex.
It extracts numeric ``key=value`` fields from lines such as:

    [2026-05-18 01:05:26] [LossStats] batch=1 loss_mean=9.3049

It also supports multiline equation/tape logs, JSONL metric files, and
CSV/TelemetryLattice CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib


def maybe_use_headless_backend(show: bool) -> None:
    if not show:
        matplotlib.use("Agg")


def default_log_path() -> Path:
    root = Path(__file__).resolve().parent
    candidates = [
        root / "resources" / "models" / "GRIM-text" / "training" / "logs" / "latest_run.log",
        root / "resources" / "models" / "GRIM-text" / "training" / "logs" / "latest_run.csv",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError("No default log found; pass a log path explicitly")


def strip_value_decorations(text: str) -> str:
    value = text.strip().strip(",;")
    value = value.strip("\"'")

    while value and value[0] in "([{":
        value = value[1:].strip()
    while value and value[-1] in ")]},;":
        value = value[:-1].strip()

    if len(value) > 1 and value[-1] in "%xX" and value[-2].isdigit():
        value = value[:-1]

    return value


def to_float_or_none(raw_value: object) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return 1.0 if raw_value else 0.0
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        return value if math.isfinite(value) else None

    value = strip_value_decorations(str(raw_value))
    if not value:
        return None
    if "[" in value or "]" in value or ":" in value:
        return None

    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def normalize_key(raw_key: str) -> str:
    key = raw_key.strip().strip(",;:()[]{}")
    key = key.replace("/", "_per_")
    key = key.replace("-", "_")
    key = key.replace(".", "_")
    key = key.replace(" ", "_")
    key = key.replace("(", "_")
    key = key.replace(")", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


def normalize_label(raw_label: str) -> str:
    label = raw_label.strip().lower()
    label = label.replace("/", "_per_")
    pieces: list[str] = []
    for char in label:
        if char.isalnum() or char == "_":
            pieces.append(char)
        else:
            pieces.append("_")
    normalized = "".join(pieces)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def parse_key_value_token(token: str) -> tuple[str, float] | None:
    key, separator, value = token.partition("=")
    if not separator:
        return None
    key = normalize_key(key)
    if not key:
        return None
    parsed_value = to_float_or_none(value)
    return None if parsed_value is None else (key, parsed_value)


def parse_key_values_from_text(text: str) -> dict[str, float]:
    fields: dict[str, float] = {}
    for token in text.replace("\t", " ").split():
        parsed = parse_key_value_token(token)
        if parsed is None:
            continue
        key, value = parsed
        fields[key] = value
    return fields


def numeric_values_from_text(text: str) -> list[float]:
    values: list[float] = []
    spaced = text.replace("/", " / ").replace(",", " ").replace("\t", " ")
    for token in spaced.split():
        if "=" in token:
            continue
        value = to_float_or_none(token)
        if value is not None:
            values.append(value)
    return values


def prefix_fields(prefix: str, fields: dict[str, float]) -> dict[str, float]:
    if not prefix:
        return fields
    return {f"{prefix}_{key}": value for key, value in fields.items()}


def parse_colon_value_line(text: str) -> dict[str, float]:
    label, separator, value_text = text.partition(":")
    if not separator:
        return {}

    prefix = normalize_label(label)
    if key_value_fields := parse_key_values_from_text(value_text):
        return prefix_fields(prefix, key_value_fields)

    values = numeric_values_from_text(value_text)
    if not values or not prefix:
        return {}

    full_head_marker = "full/head" in label.lower()
    if full_head_marker and len(values) >= 2:
        base_prefix = prefix.replace("full_per_head", "").strip("_")
        return {
            f"{base_prefix}_full": values[0],
            f"{base_prefix}_head": values[1],
        }
    if len(values) == 1:
        return {prefix: values[0]}
    return {f"{prefix}_{index + 1}": value for index, value in enumerate(values)}


def parse_continuation_fields(text: str) -> dict[str, float]:
    stripped = text.strip()
    if not stripped:
        return {}
    if ":" in stripped:
        if fields := parse_colon_value_line(stripped):
            return fields
    return parse_key_values_from_text(stripped)


def split_bracket_prefix(text: str) -> tuple[str | None, str]:
    stripped = text.strip()
    if not stripped.startswith("["):
        return None, stripped
    end = stripped.find("]")
    if end <= 0:
        return None, stripped
    return stripped[1:end].strip(), stripped[end + 1 :].strip()


def split_all_bracket_prefixes(text: str) -> tuple[list[str], str]:
    prefixes: list[str] = []
    rest = text.strip()
    while rest.startswith("["):
        end = rest.find("]")
        if end <= 0:
            break
        prefixes.append(rest[1:end].strip())
        rest = rest[end + 1 :].strip()
    return prefixes, rest


def bracket_looks_like_timestamp(value: str) -> bool:
    if len(value) < 10:
        return False
    if value[4:5] != "-" or value[7:8] != "-":
        return False
    return value[:4].isdigit() and value[5:7].isdigit() and value[8:10].isdigit()


def normalize_tag(raw_tag: str | None) -> tuple[str, dict[str, float]]:
    if raw_tag is None or not raw_tag.strip():
        return "untagged", {}

    tag = raw_tag.strip()
    tag_fields: dict[str, float] = {}
    pieces = tag.split()
    if len(pieces) == 2 and pieces[0] == "Step":
        step_value = to_float_or_none(pieces[1])
        if step_value is not None:
            tag_fields["step"] = step_value
            return "Step", tag_fields
    return tag, tag_fields


def tag_from_rest(rest: str) -> str | None:
    left, separator, _right = rest.partition(":")
    if not separator:
        return None
    for piece in left.split():
        cleaned = piece.strip().strip(",;:()[]{}")
        if cleaned.endswith("_EQUATION"):
            return cleaned
    return None


def extract_log_prefix(line: str, current_tag: str) -> tuple[str, dict[str, float], str]:
    prefixes, rest = split_all_bracket_prefixes(line)
    tag_text: str | None = None

    equation_tag = tag_from_rest(rest)
    if equation_tag is not None:
        tag_text = equation_tag
    elif prefixes:
        non_timestamp_prefixes = [prefix for prefix in prefixes if not bracket_looks_like_timestamp(prefix)]
        tag_text = non_timestamp_prefixes[0] if non_timestamp_prefixes else current_tag
    else:
        tag_text = current_tag

    tag, tag_fields = normalize_tag(tag_text)
    return tag, tag_fields, rest


def line_starts_record(line: str) -> bool:
    return line.strip().startswith("[")


def choose_x(fields: dict[str, float], current_batch: float | None, row_index: int) -> tuple[float, str]:
    if "global_step" in fields:
        return fields["global_step"], "global_step"
    if "step" in fields:
        return fields["step"], "step"
    if "s" in fields:
        return fields["s"], "step (s)"
    if "batch" in fields:
        return fields["batch"], "batch"
    if current_batch is not None:
        return current_batch, "batch"
    return float(row_index), "row"


def row_has_numeric_fields(row: dict[str, object]) -> bool:
    for key, value in row.items():
        if key.startswith("__"):
            continue
        if to_float_or_none(value) is not None:
            return True
    return False


def parse_plain_log(path: Path, max_lines: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current_tag = "untagged"
    current_batch: float | None = None
    current_step: float | None = None
    current_row: dict[str, object] | None = None

    def flush_current_row() -> None:
        nonlocal current_row
        if current_row is not None and row_has_numeric_fields(current_row):
            numeric_fields: dict[str, float] = {}
            for key, value in current_row.items():
                parsed_value = to_float_or_none(value)
                if parsed_value is not None:
                    numeric_fields[key] = parsed_value
            x_value, x_name = choose_x(
                numeric_fields,
                None,
                len(rows),
            )
            current_row["__x"] = x_value
            current_row["__x_name"] = x_name
            rows.append(current_row)
        current_row = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_lines is not None and max_lines > 0 and line_number > max_lines:
                break

            if not line.strip():
                continue

            if not line_starts_record(line):
                continuation_fields = parse_continuation_fields(line)
                if continuation_fields and current_row is not None:
                    current_row.update(continuation_fields)
                elif continuation_fields:
                    tag = current_tag or "untagged"
                    current_row = {
                        "__tag": tag,
                        "__line": line_number,
                        "__x": float(len(rows)),
                        "__x_name": "row",
                    }
                    current_row.update(continuation_fields)
                continue

            flush_current_row()

            tag, tag_fields, rest = extract_log_prefix(line, current_tag)
            if tag != "untagged":
                current_tag = tag

            fields = dict(tag_fields)
            fields.update(parse_key_values_from_text(rest))
            if not fields:
                continue

            if "batch" in fields:
                current_batch = fields["batch"]
            if "step" in fields:
                current_step = fields["step"]

            x_value, x_name = choose_x(fields, None, len(rows))
            current_row = {
                "__tag": current_tag,
                "__line": line_number,
                "__x": x_value,
                "__x_name": x_name,
            }
            current_row.update(fields)

    flush_current_row()

    return rows


def flatten_dict(data: dict[str, object], prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def parse_jsonl(path: Path, max_lines: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_lines is not None and max_lines > 0 and line_number > max_lines:
                break
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if not isinstance(data, dict):
                continue
            flattened = flatten_dict(data)
            numeric = {normalize_key(key): value for key, value in flattened.items() if to_float_or_none(value) is not None}
            x_value, x_name = choose_x({key: to_float_or_none(value) or 0.0 for key, value in numeric.items()}, None, len(rows))
            row: dict[str, object] = {
                "__tag": "jsonl",
                "__line": line_number,
                "__x": x_value,
                "__x_name": x_name,
            }
            for key, value in numeric.items():
                parsed_value = to_float_or_none(value)
                if parsed_value is not None:
                    row[key] = parsed_value
            rows.append(row)
    return rows


def parse_csv_rows(path: Path, max_lines: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_index, data in enumerate(reader, start=1):
            if max_lines is not None and max_lines > 0 and line_index > max_lines:
                break
            row: dict[str, object] = {
                "__tag": str(data.get("stream_name") or data.get("tag") or "csv"),
                "__line": line_index,
            }
            numeric_fields: dict[str, float] = {}
            for key, value in data.items():
                if key is None:
                    continue
                normalized = normalize_key(key)
                parsed_value = to_float_or_none(value)
                if parsed_value is None:
                    continue
                numeric_fields[normalized] = parsed_value

            x_value, x_name = choose_x(numeric_fields, None, len(rows))
            row["__x"] = x_value
            row["__x_name"] = x_name
            row.update(numeric_fields)
            rows.append(row)
    return rows


def load_rows(path: Path, max_lines: int | None = None) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return parse_jsonl(path, max_lines)
    if suffix == ".csv":
        return parse_csv_rows(path, max_lines)
    return parse_plain_log(path, max_lines)


def split_csv_argument(value: str | None) -> set[str]:
    if value is None or not value.strip():
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def field_allowed(field: str, selected_fields: set[str]) -> bool:
    if not selected_fields:
        return True
    lowered = field.lower()
    return lowered in {item.lower() for item in selected_fields}


def tag_allowed(tag: str, selected_tags: set[str]) -> bool:
    if not selected_tags:
        return True
    lowered = tag.lower()
    return lowered in {item.lower() for item in selected_tags}


def rolling_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed: list[float] = []
    running_sum = 0.0
    running_values: list[float] = []
    for value in values:
        running_values.append(value)
        running_sum += value
        if len(running_values) > window:
            running_sum -= running_values.pop(0)
        smoothed.append(running_sum / float(len(running_values)))
    return smoothed


def safe_filename(text: str) -> str:
    safe_chars = []
    for char in text:
        if char.isalnum() or char in "._-":
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "plot"


def field_looks_like_size_or_index(field: str) -> bool:
    lowered = field.lower()
    exact_size_fields = {
        "b",
        "d",
        "h",
        "m",
        "n",
        "r",
        "row",
        "col",
        "q_b",
        "k_b",
        "v_b",
        "o_b",
        "do_b",
        "dq_b",
        "total_q",
        "total_k",
        "total_v",
        "total_o",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "d_rounded",
    }
    if lowered in exact_size_fields:
        return True
    return lowered.startswith("seqlen_") or lowered.endswith("_rounded") or lowered.endswith("_bytes")


def collect_numeric_fields(
    rows: list[dict[str, object]],
    selected_fields: set[str],
    include_constant_fields: bool,
    include_size_fields: bool,
) -> list[str]:
    excluded = {"__x", "__line", "batch", "step", "global_step", "epoch", "level", "stride", "stream_idx", "step_count", "s"}
    ordered_fields: list[str] = []
    first_values: dict[str, float] = {}
    first_x_values: dict[str, float] = {}
    value_counts: dict[str, int] = {}
    varying_fields: set[str] = set()
    multi_x_fields: set[str] = set()
    explicit_field_selection = bool(selected_fields)
    for row in rows:
        x_value = to_float_or_none(row.get("__x"))
        for key, value in row.items():
            if key in excluded or key.startswith("__"):
                continue
            if not field_allowed(key, selected_fields):
                continue
            parsed_value = to_float_or_none(value)
            if parsed_value is None:
                continue
            if key not in first_values:
                first_values[key] = parsed_value
                if x_value is not None:
                    first_x_values[key] = x_value
                value_counts[key] = 1
                ordered_fields.append(key)
                continue
            value_counts[key] += 1
            if x_value is not None and key in first_x_values and not math.isclose(first_x_values[key], x_value, rel_tol=0.0, abs_tol=0.0):
                multi_x_fields.add(key)
            if not math.isclose(first_values[key], parsed_value, rel_tol=1e-12, abs_tol=1e-12):
                varying_fields.add(key)

    fields: list[str] = []
    for key in ordered_fields:
        if value_counts.get(key, 0) < 2 and not explicit_field_selection and not include_constant_fields:
            continue
        if key not in multi_x_fields and not explicit_field_selection and not include_constant_fields:
            continue
        if not explicit_field_selection and not include_size_fields and field_looks_like_size_or_index(key):
            continue
        if not explicit_field_selection and not include_constant_fields and key not in varying_fields:
            continue
        fields.append(key)
    return fields


def collect_series(rows: list[dict[str, object]], field: str) -> tuple[list[float], list[float], str]:
    points_by_x: dict[float, tuple[float, int]] = {}
    x_name = "x"
    for row in rows:
        value = to_float_or_none(row.get(field))
        x_value = to_float_or_none(row.get("__x"))
        if value is None or x_value is None:
            continue
        current_sum, current_count = points_by_x.get(x_value, (0.0, 0))
        points_by_x[x_value] = (current_sum + value, current_count + 1)
        x_name = str(row.get("__x_name", x_name))
    points = [(x_value, value_sum / float(value_count)) for x_value, (value_sum, value_count) in points_by_x.items()]
    points.sort(key=lambda item: item[0])
    return [item[0] for item in points], [item[1] for item in points], x_name


def plot_field_page(
    rows: list[dict[str, object]],
    fields: list[str],
    title: str,
    output_path: Path,
    smooth_window: int,
    show_raw: bool,
) -> None:
    import matplotlib.pyplot as plt

    plot_count = len(fields)
    cols = 2 if plot_count > 1 else 1
    rows_count = int(math.ceil(plot_count / float(cols)))
    fig, axes = plt.subplots(rows_count, cols, figsize=(8 * cols, 3.2 * rows_count), squeeze=False, constrained_layout=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    flat_axes = [axis for axis_row in axes for axis in axis_row]
    for axis, field in zip(flat_axes, fields):
        x_values, y_values, x_name = collect_series(rows, field)
        if not x_values:
            axis.set_visible(False)
            continue
        if smooth_window <= 1 or len(y_values) < 3:
            axis.plot(x_values, y_values, linewidth=1.2, alpha=0.85, label=field)
        if smooth_window > 1 and len(y_values) >= 3:
            smooth_values = rolling_average(y_values, smooth_window)
            if show_raw:
                axis.plot(x_values, y_values, linewidth=0.5, alpha=0.25, label="raw")
            axis.plot(x_values, smooth_values, linewidth=1.6, label=f"smooth-{smooth_window}")
        axis.set_title(field, fontsize=10)
        axis.set_xlabel(x_name)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    for axis in flat_axes[len(fields) :]:
        axis.set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_rows(
    rows: list[dict[str, object]],
    source_path: Path,
    output_dir: Path,
    selected_tags: set[str],
    selected_fields: set[str],
    max_fields_per_page: int,
    smooth_window: int,
    show_raw: bool,
    include_constant_fields: bool,
    include_size_fields: bool,
) -> list[Path]:
    rows_by_tag: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        tag = str(row.get("__tag", "untagged"))
        if tag_allowed(tag, selected_tags):
            rows_by_tag[tag].append(row)

    saved: list[Path] = []
    for tag in sorted(rows_by_tag):
        tag_rows = rows_by_tag[tag]
        fields = collect_numeric_fields(tag_rows, selected_fields, include_constant_fields, include_size_fields)
        fields = [field for field in fields if len(collect_series(tag_rows, field)[0]) > 0]
        if not fields:
            continue

        for page_start in range(0, len(fields), max_fields_per_page):
            page_fields = fields[page_start : page_start + max_fields_per_page]
            page_index = page_start // max_fields_per_page + 1
            suffix = f"_{page_index:02d}" if len(fields) > max_fields_per_page else ""
            filename = f"{safe_filename(source_path.stem)}_{safe_filename(tag)}{suffix}.png"
            output_path = output_dir / filename
            title = f"{source_path.name} — {tag}"
            if len(fields) > max_fields_per_page:
                title += f" (page {page_index})"
            plot_field_page(tag_rows, page_fields, title, output_path, smooth_window, show_raw)
            saved.append(output_path)

    return saved


def print_summary(rows: list[dict[str, object]], saved_paths: list[Path]) -> None:
    tags = sorted({str(row.get("__tag", "untagged")) for row in rows})
    print(f"Parsed numeric rows: {len(rows)}")
    print(f"Tags with numeric fields: {len(tags)}")
    if tags:
        print("Tags: " + ", ".join(tags[:30]) + (" ..." if len(tags) > 30 else ""))
    print(f"Saved plots: {len(saved_paths)}")
    for path in saved_paths[:30]:
        print(f"  {path}")
    if len(saved_paths) > 30:
        print(f"  ... {len(saved_paths) - 30} more")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot numeric fields from GRIM .log, .jsonl, or .csv files using Matplotlib.")
    parser.add_argument("path", nargs="?", help="Path to .log, .jsonl, or .csv. Defaults to GRIM-text latest_run.log when present.")
    parser.add_argument("--out-dir", help="Directory for PNG plots. Defaults to <log_stem>_field_plots next to the input file.")
    parser.add_argument("--tags", help="Comma-separated log tags to plot, e.g. LossStats,LogitSignal,Step")
    parser.add_argument("--fields", help="Comma-separated numeric fields to plot, e.g. loss,lr,loss_mean")
    parser.add_argument("--max-fields-per-page", type=int, default=12, help="Maximum subplots per saved PNG page.")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average smoothing window. 1 disables smoothing.")
    parser.add_argument("--raw", action="store_true", help="When smoothing is enabled, also show raw values faintly.")
    parser.add_argument("--max-lines", type=int, default=0, help="Only parse the first N lines/rows. 0 parses all.")
    parser.add_argument("--include-constant-fields", action="store_true", help="Also plot fields that do not vary across the selected x-axis.")
    parser.add_argument("--include-size-fields", action="store_true", help="Also plot obvious tensor size/index fields such as q_b, total_q, seqLen_q_rounded.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively after saving them.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.max_fields_per_page <= 0:
        parser.error("--max-fields-per-page must be positive")
    if args.smooth <= 0:
        parser.error("--smooth must be positive")

    maybe_use_headless_backend(args.show)

    source_path = Path(args.path).resolve() if args.path else default_log_path()
    if not source_path.exists():
        raise FileNotFoundError(f"Log path does not exist: {source_path}")

    output_dir = Path(args.out_dir).resolve() if args.out_dir else source_path.with_name(f"{source_path.stem}_field_plots")
    selected_tags = split_csv_argument(args.tags)
    selected_fields = split_csv_argument(args.fields)

    max_lines = args.max_lines if args.max_lines > 0 else None
    rows = load_rows(source_path, max_lines=max_lines)
    if not rows:
        print(f"No numeric fields found in {source_path}")
        return 1

    saved_paths = plot_rows(
        rows=rows,
        source_path=source_path,
        output_dir=output_dir,
        selected_tags=selected_tags,
        selected_fields=selected_fields,
        max_fields_per_page=args.max_fields_per_page,
        smooth_window=args.smooth,
        show_raw=args.raw,
        include_constant_fields=args.include_constant_fields,
        include_size_fields=args.include_size_fields,
    )
    print_summary(rows, saved_paths)

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise