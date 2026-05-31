#!/usr/bin/env python3
"""Analyze GRIM-text training logs using plain string search only.

This parser intentionally avoids regex. It uses only string operations such as
``in``, ``find()``, ``split()``, and ``partition()`` so it stays easy to audit
against the project's log format.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"
SECTION_MARKERS = (
	"LM_HEAD_GEMM_EQUATION",
	"LM_HEAD_GEMM_BACKWARD_EQUATION",
	"LOGIT_SCALE_EQUATION",
	"RHO_BUILDUP_EQUATION",
	"POSTCLIP_PARAM_GRAD_EMB_LM_EQUATION",
	"SPECIAL_TOKEN_EQUATION",
)
ANOMALY_RULES = {
	"fatal": ("[FATAL", "FATAL-SEH", "FATAL-VALIDATE", "EXCEPTION_ACCESS_VIOLATION"),
	"error": ("[ERROR]", "[SYS][ERROR]", "ERROR:", "Unhandled structured exception"),
	"warning": ("[WARNING]", "WARNING:"),
	"nonfinite": ("nan=", "inf=", " NaN", " nonfinite", "NONFINITE"),
}
TRAINING_BATCH_MARKERS = (
	"[LossStats]",
	"[LogitSignal]",
	"[DEBUG-LM-WEIGHTS]",
	"[LMHeadNorm]",
	"[SPECIAL_TOKEN_EQUATION]",
	"POST-CLIP-MEASURE",
	"POST-OPTIMIZER",
	"TOP-GRAD-GROUPS",
	"[FATAL-VALIDATE]",
)


def is_timestamp_text(text: str) -> bool:
	if len(text) < 19:
		return False
	if text[4:5] != "-" or text[7:8] != "-" or text[10:11] != " ":
		return False
	if text[13:14] != ":" or text[16:17] != ":":
		return False
	digits = text[:4] + text[5:7] + text[8:10] + text[11:13] + text[14:16] + text[17:19]
	return digits.isdigit()


def strip_timestamp_prefix(line: str) -> tuple[str | None, str]:
	stripped = line.rstrip("\n")
	if not stripped.startswith("["):
		return None, stripped

	end = stripped.find("]")
	if end <= 1:
		return None, stripped

	candidate = stripped[1:end]
	if not is_timestamp_text(candidate):
		return None, stripped

	return candidate, stripped[end + 1 :].lstrip()


def split_leading_brackets(text: str) -> tuple[list[str], str]:
	parts: list[str] = []
	rest = text.lstrip()
	while rest.startswith("["):
		end = rest.find("]")
		if end <= 0:
			break
		parts.append(rest[1:end])
		rest = rest[end + 1 :].lstrip()
	return parts, rest


def normalize_key(text: str) -> str:
	cleaned = text.strip().strip(",;:")
	if not cleaned:
		return ""
	cleaned = cleaned.replace("/", "_per_")
	cleaned = cleaned.replace("(", "_")
	cleaned = cleaned.replace(")", "_")
	cleaned = cleaned.replace("[", "_")
	cleaned = cleaned.replace("]", "_")
	cleaned = cleaned.replace("-", "_")
	cleaned = cleaned.replace(".", "_")
	cleaned = cleaned.replace(" ", "_")
	while "__" in cleaned:
		cleaned = cleaned.replace("__", "_")
	return cleaned.strip("_").lower()


def clean_value_text(text: str) -> str:
	value = text.strip().strip(",;")
	value = value.strip("\"'")
	while value and value[0] in "([{":
		value = value[1:].strip()
	while value and value[-1] in ")]},;":
		value = value[:-1].strip()
	if value.endswith("x") and len(value) > 1:
		probe = value[:-1]
		if probe[-1].isdigit():
			value = probe
	if value.endswith("%") and len(value) > 1:
		probe = value[:-1]
		if probe[-1].isdigit():
			value = probe
	return value


def parse_number(text: str) -> int | float | None:
	value = clean_value_text(text)
	if not value:
		return None
	if any(marker in value for marker in ("[", "]", ":", "0x", "0X", "\\", "tok")):
		return None

	lowered = value.lower()
	if lowered in {"yes", "no", "true", "false"}:
		return None

	try:
		if any(marker in value for marker in (".", "e", "E")):
			parsed = float(value)
			return parsed if math.isfinite(parsed) else None
		return int(value)
	except ValueError:
		return None


def parse_key_values(text: str) -> dict[str, Any]:
	fields: dict[str, Any] = {}
	for token in text.replace("\t", " ").split():
		key, separator, raw_value = token.partition("=")
		if not separator:
			continue
		normalized_key = normalize_key(key)
		if not normalized_key:
			continue
		numeric_value = parse_number(raw_value)
		if numeric_value is not None:
			fields[normalized_key] = numeric_value
			continue
		cleaned_value = clean_value_text(raw_value)
		if cleaned_value:
			fields[normalized_key] = cleaned_value
	return fields


def extract_step_from_tag(tag: str) -> int | None:
	tag = tag.strip()
	if not tag.startswith("Step "):
		return None
	value = parse_number(tag[5:].strip())
	return value if isinstance(value, int) else None


def find_first_number_after(marker: str, text: str) -> int | float | None:
	start = text.find(marker)
	if start < 0:
		return None
	cursor = start + len(marker)
	end = cursor
	while end < len(text) and not text[end].isspace():
		end += 1
	return parse_number(text[cursor:end])


def line_has_nonfinite_issue(line: str) -> bool:
	if "NONFINITE" in line or " nonfinite" in line:
		return True
	if " NaN" in line:
		return True

	for marker in ("nan=", "inf="):
		start = 0
		while True:
			index = line.find(marker, start)
			if index < 0:
				break
			value = find_first_number_after(marker, line[index:])
			if isinstance(value, (int, float)) and float(value) != 0.0:
				return True
			start = index + len(marker)
	return False


def line_reports_training_batch(line: str) -> bool:
	return any(marker in line for marker in TRAINING_BATCH_MARKERS)


def safe_mean(values: list[float | int]) -> float | None:
	if not values:
		return None
	return float(sum(values)) / float(len(values))


def safe_min(values: list[float | int]) -> float | int | None:
	return min(values) if values else None


def safe_max(values: list[float | int]) -> float | int | None:
	return max(values) if values else None


def make_excerpt(line_number: int, text: str, timestamp: str | None) -> dict[str, Any]:
	return {
		"line": line_number,
		"timestamp": timestamp,
		"text": text.strip(),
	}


def format_number(value: Any, digits: int = 6) -> str:
	if value is None:
		return "N/A"
	if isinstance(value, int):
		return str(value)
	if isinstance(value, float):
		magnitude = abs(value)
		if magnitude == 0.0:
			return "0.0"
		if magnitude >= 1000 or magnitude < 0.001:
			return f"{value:.6e}"
		return f"{value:.{digits}f}"
	return str(value)


def rolling_window_mean(values: list[float], width: int) -> float | None:
	if not values:
		return None
	window = values[:width] if len(values) <= width else values[-width:]
	return safe_mean(window)


def collect_top_counter(counter: Counter[str], limit: int = 12) -> list[dict[str, Any]]:
	return [{"name": name, "count": count} for name, count in counter.most_common(limit)]


def choose_default_log() -> Path:
	latest_run = DEFAULT_LOG_DIR / "latest_run.log"
	if latest_run.exists() and latest_run.stat().st_size > 0:
		return latest_run

	training_logs = sorted(DEFAULT_LOG_DIR.glob("training_*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
	if training_logs:
		return training_logs[0]

	raise FileNotFoundError(f"No training log found in {DEFAULT_LOG_DIR}")


def expand_inputs(paths: list[str], all_logs: bool) -> list[Path]:
	if not paths:
		if all_logs:
			candidates = sorted(DEFAULT_LOG_DIR.glob("training_*.log"))
			latest_run = DEFAULT_LOG_DIR / "latest_run.log"
			if latest_run.exists():
				candidates.append(latest_run)
			unique: list[Path] = []
			seen: set[Path] = set()
			for path in candidates:
				resolved = path.resolve()
				if resolved not in seen:
					seen.add(resolved)
					unique.append(resolved)
			return unique
		return [choose_default_log().resolve()]

	selected: list[Path] = []
	seen: set[Path] = set()
	for raw_path in paths:
		path = Path(raw_path).expanduser()
		if not path.is_absolute():
			path = (Path.cwd() / path).resolve()
		else:
			path = path.resolve()

		if not path.exists():
			raise FileNotFoundError(f"Path does not exist: {path}")

		candidates: list[Path]
		if path.is_dir():
			if all_logs:
				candidates = sorted(path.glob("training_*.log"))
				latest_run = path / "latest_run.log"
				if latest_run.exists():
					candidates.append(latest_run)
			else:
				latest_run = path / "latest_run.log"
				if latest_run.exists() and latest_run.stat().st_size > 0:
					candidates = [latest_run]
				else:
					logs = sorted(path.glob("training_*.log"), key=lambda candidate: candidate.stat().st_mtime, reverse=True)
					if not logs:
						raise FileNotFoundError(f"No training logs found in directory: {path}")
					candidates = [logs[0]]
		else:
			candidates = [path]

		for candidate in candidates:
			resolved = candidate.resolve()
			if resolved not in seen:
				seen.add(resolved)
				selected.append(resolved)

	return selected


def derive_output_token(log_path: Path) -> str:
	stem = log_path.stem
	if stem.startswith("training_"):
		return stem[len("training_") :]
	return stem


def derive_output_paths(log_path: Path, out_dir: Path | None) -> tuple[Path, Path]:
	token = derive_output_token(log_path)
	directory = out_dir if out_dir is not None else log_path.parent
	return directory / f"analysis_{token}.log", directory / f"analysis_{token}.json"


def analyze_log(log_path: Path, max_lines: int | None = None) -> dict[str, Any]:
	tag_counts: Counter[str] = Counter()
	section_counts: Counter[str] = Counter()
	anomaly_counts: Counter[str] = Counter()
	anomaly_samples: dict[str, list[dict[str, Any]]] = {name: [] for name in ANOMALY_RULES}
	loss_rows: list[dict[str, Any]] = []
	step_rows: list[dict[str, Any]] = []
	clip_rows: list[dict[str, Any]] = []
	post_optimizer_rows: list[dict[str, Any]] = []
	optimizer_component_rows: list[dict[str, Any]] = []
	logit_scale_rows: list[dict[str, Any]] = []
	lm_head_rows: list[dict[str, Any]] = []
	planned_batches: dict[str, Any] = {}
	fatal_context: list[dict[str, Any]] = []

	total_lines = 0
	first_timestamp: str | None = None
	last_timestamp: str | None = None
	session_id: str | None = None
	current_section: str | None = None
	current_batch: int | None = None
	current_step: int | None = None
	current_optimizer_step: int | None = None
	current_lm_head: dict[str, Any] | None = None
	current_logit_scale: dict[str, Any] | None = None

	with log_path.open("r", encoding="utf-8", errors="replace") as handle:
		for line_number, raw_line in enumerate(handle, start=1):
			if max_lines is not None and line_number > max_lines:
				break

			total_lines += 1
			line = raw_line.rstrip("\n")
			timestamp, remainder = strip_timestamp_prefix(line)

			if timestamp is not None:
				if first_timestamp is None:
					first_timestamp = timestamp
				last_timestamp = timestamp

			if "Session ID:" in line and session_id is None:
				after = line.partition("Session ID:")[2].strip()
				if after:
					session_id = after

			leading_tags, content = split_leading_brackets(remainder)
			primary_tag = None
			if leading_tags:
				primary_tag = leading_tags[-1].strip()
				if primary_tag:
					tag_counts[primary_tag] += 1
					step_value = extract_step_from_tag(primary_tag)
					if step_value is not None:
						current_step = step_value

			matched_section = None
			for marker in SECTION_MARKERS:
				if f"[{marker}]" in line:
					matched_section = marker
					section_counts[marker] += 1
					current_section = marker
					break

			if matched_section is None and line.startswith("[") and current_section is not None and content:
				if not any(f"[{marker}]" in line for marker in SECTION_MARKERS):
					current_section = None

			field_source = content if content else remainder
			parsed_fields = parse_key_values(field_source)
			if line_reports_training_batch(line) and isinstance(parsed_fields.get("batch"), int):
				current_batch = parsed_fields["batch"]
			if isinstance(parsed_fields.get("optimizer_step"), int):
				current_optimizer_step = parsed_fields["optimizer_step"]
			if isinstance(parsed_fields.get("step"), int):
				current_step = parsed_fields["step"]

			if "[LossStats]" in line:
				row = {
					"line": line_number,
					"timestamp": timestamp,
					"batch": parsed_fields.get("batch", current_batch),
					"loss_mean": parsed_fields.get("loss_mean"),
					"loss_sum": parsed_fields.get("loss_sum"),
					"valid_tokens": parsed_fields.get("valid_tokens"),
					"masked_tokens": parsed_fields.get("masked_tokens"),
					"total_tokens": parsed_fields.get("total_tokens"),
				}
				loss_rows.append(row)
				if isinstance(row.get("batch"), int):
					current_batch = row["batch"]

			if primary_tag and primary_tag.startswith("Step "):
				row = {
					"line": line_number,
					"timestamp": timestamp,
					"step": extract_step_from_tag(primary_tag),
					"loss": parsed_fields.get("loss"),
					"lr": parsed_fields.get("lr"),
				}
				step_rows.append(row)

			if "POST-CLIP-MEASURE" in line:
				row = {
					"line": line_number,
					"timestamp": timestamp,
					"batch": parsed_fields.get("batch", current_batch),
					"preclip_registered_global": parsed_fields.get("preclip_registered_global"),
					"postclip_registered_global": parsed_fields.get("postclip_registered_global"),
					"clipped": str(parsed_fields.get("clipped", "")).lower(),
					"emb_rms_pre": parsed_fields.get("emb_rms_pre"),
					"enc_rms_pre": parsed_fields.get("enc_rms_pre"),
					"sb_rms_pre": parsed_fields.get("sb_rms_pre"),
				}
				clip_rows.append(row)

			if "POST-OPTIMIZER" in line and "[GradTrace]" in line:
				row = {
					"line": line_number,
					"timestamp": timestamp,
					"batch": parsed_fields.get("batch", current_batch),
					"optimizer_step": parsed_fields.get("optimizer_step", current_optimizer_step),
					"iteration": parsed_fields.get("iteration"),
					"lr": parsed_fields.get("lr"),
					"rms": parsed_fields.get("rms"),
				}
				post_optimizer_rows.append(row)

			if "[OptimizerUpdateTrace]" in line:
				component_name = None
				if "COMPONENTS(" in line:
					start = line.find("COMPONENTS(") + len("COMPONENTS(")
					end = line.find(")", start)
					if end > start:
						component_name = line[start:end]
				elif "RATIOS(" in line:
					start = line.find("RATIOS(") + len("RATIOS(")
					end = line.find(")", start)
					if end > start:
						component_name = f"ratios_{line[start:end]}"

				row = {
					"line": line_number,
					"timestamp": timestamp,
					"component": component_name,
					"optimizer_step": parsed_fields.get("optimizer_step", current_optimizer_step),
					"iteration": parsed_fields.get("iteration"),
				}
				for key in ("emb", "lm", "attn", "ffn", "rmsnorm", "sb", "exec", "slot_selector"):
					if key in parsed_fields:
						row[key] = parsed_fields[key]
				optimizer_component_rows.append(row)

			if matched_section == "LM_HEAD_GEMM_EQUATION":
				current_lm_head = {
					"line": line_number,
					"timestamp": timestamp,
					"batch": current_batch,
					"step": current_step,
				}
				lm_head_rows.append(current_lm_head)
			elif current_section == "LM_HEAD_GEMM_EQUATION" and current_lm_head is not None:
				if "EXPECTED sample_logit_rms" in line:
					current_lm_head["expected_sample_logit_rms"] = find_first_number_after("= ", line)
				if "ACTUAL (logits prefix sample):" in line:
					actual_rms = find_first_number_after("rms=", line)
					ratio = find_first_number_after("ratio=", line)
					if actual_rms is not None:
						current_lm_head["actual_sample_logit_rms"] = actual_rms
					if ratio is not None:
						current_lm_head["ratio"] = ratio

			if matched_section == "LOGIT_SCALE_EQUATION":
				current_logit_scale = {
					"line": line_number,
					"timestamp": timestamp,
					"batch": current_batch,
					"step": current_step,
				}
				logit_scale_rows.append(current_logit_scale)
			elif current_section == "LOGIT_SCALE_EQUATION" and current_logit_scale is not None:
				if "LOGIT STATS:" in line:
					stats = parse_key_values(line)
					for key in ("std", "range", "avg_per_pos_range", "max_per_pos_range"):
						if key in stats:
							current_logit_scale[key] = stats[key]
				elif "EXPECTED logit_std" in line and current_logit_scale.get("expected_logit_std") is None:
					maybe_expected = find_first_number_after("= ", line)
					if maybe_expected is not None:
						current_logit_scale["expected_logit_std"] = maybe_expected
				elif "ACTUAL logit_std" in line:
					current_logit_scale["actual_logit_std"] = find_first_number_after("ACTUAL logit_std = ", line)
					current_logit_scale["ratio"] = find_first_number_after("ratio(actual/expected)=", line)

			if "Built " in line and "train payloads" in line:
				count = find_first_number_after("Built ", line)
				if count is not None:
					planned_batches["train_payloads"] = count
			if "Built " in line and "val payloads" in line:
				count = find_first_number_after("Built ", line)
				if count is not None:
					planned_batches["val_payloads"] = count
			if "Created " in line and "fixed batches" in line:
				count = find_first_number_after("Created ", line)
				if count is not None:
					planned_batches["fixed_batches"] = count
			if "Epoch " in line and "/" in line and line.startswith("["):
				epoch_text = line.partition("] ")[2].strip()
				planned_batches["last_epoch_line"] = epoch_text

			for anomaly_name, needles in ANOMALY_RULES.items():
				if anomaly_name == "nonfinite" and not line_has_nonfinite_issue(line):
					continue
				hit = False
				for needle in needles:
					if needle in line:
						hit = True
						break
				if not hit:
					continue
				anomaly_counts[anomaly_name] += 1
				samples = anomaly_samples[anomaly_name]
				if len(samples) < 8:
					samples.append(make_excerpt(line_number, line, timestamp))
				if anomaly_name == "fatal" and len(fatal_context) < 12:
					fatal_context.append(make_excerpt(line_number, line, timestamp))

	loss_values = [row["loss_mean"] for row in loss_rows if isinstance(row.get("loss_mean"), (int, float))]
	valid_tokens = [row["valid_tokens"] for row in loss_rows if isinstance(row.get("valid_tokens"), (int, float))]
	step_losses = [row["loss"] for row in step_rows if isinstance(row.get("loss"), (int, float))]
	step_lrs = [row["lr"] for row in step_rows if isinstance(row.get("lr"), (int, float))]
	clip_pre = [row["preclip_registered_global"] for row in clip_rows if isinstance(row.get("preclip_registered_global"), (int, float))]
	clip_post = [row["postclip_registered_global"] for row in clip_rows if isinstance(row.get("postclip_registered_global"), (int, float))]
	lm_ratios = [row["ratio"] for row in lm_head_rows if isinstance(row.get("ratio"), (int, float))]
	logit_ratios = [row["ratio"] for row in logit_scale_rows if isinstance(row.get("ratio"), (int, float))]

	best_loss_row = None
	if loss_rows:
		numeric_loss_rows = [row for row in loss_rows if isinstance(row.get("loss_mean"), (int, float))]
		if numeric_loss_rows:
			best_loss_row = min(numeric_loss_rows, key=lambda row: float(row["loss_mean"]))

	summary = {
		"log_path": str(log_path),
		"log_name": log_path.name,
		"session_id": session_id or derive_output_token(log_path),
		"generated_at": datetime.now().isoformat(timespec="seconds"),
		"line_count": total_lines,
		"first_timestamp": first_timestamp,
		"last_timestamp": last_timestamp,
		"tag_counts": dict(tag_counts.most_common()),
		"top_tags": collect_top_counter(tag_counts, limit=15),
		"section_counts": dict(section_counts.most_common()),
		"planned_batches": planned_batches,
		"loss": {
			"count": len(loss_rows),
			"first": loss_rows[0] if loss_rows else None,
			"last": loss_rows[-1] if loss_rows else None,
			"best": best_loss_row,
			"min_loss": safe_min(loss_values),
			"max_loss": safe_max(loss_values),
			"avg_first_5": safe_mean(loss_values[:5]),
			"avg_last_5": rolling_window_mean([float(value) for value in loss_values], 5),
			"valid_tokens_avg": safe_mean(valid_tokens),
			"valid_tokens_min": safe_min(valid_tokens),
			"valid_tokens_max": safe_max(valid_tokens),
		},
		"steps": {
			"count": len(step_rows),
			"last": step_rows[-1] if step_rows else None,
			"loss_avg_last_5": rolling_window_mean([float(value) for value in step_losses], 5),
			"lr_last": step_lrs[-1] if step_lrs else None,
		},
		"gradient_clip": {
			"count": len(clip_rows),
			"clipped_yes": sum(1 for row in clip_rows if row.get("clipped") == "yes"),
			"last": clip_rows[-1] if clip_rows else None,
			"preclip_avg": safe_mean(clip_pre),
			"postclip_avg": safe_mean(clip_post),
			"preclip_max": safe_max(clip_pre),
			"postclip_max": safe_max(clip_post),
		},
		"post_optimizer": {
			"count": len(post_optimizer_rows),
			"last": post_optimizer_rows[-1] if post_optimizer_rows else None,
		},
		"optimizer_update_trace": {
			"count": len(optimizer_component_rows),
			"last": optimizer_component_rows[-1] if optimizer_component_rows else None,
		},
		"lm_head_gemm": {
			"count": len(lm_head_rows),
			"last": lm_head_rows[-1] if lm_head_rows else None,
			"ratio_avg": safe_mean(lm_ratios),
			"ratio_min": safe_min(lm_ratios),
			"ratio_max": safe_max(lm_ratios),
		},
		"logit_scale": {
			"count": len(logit_scale_rows),
			"last": logit_scale_rows[-1] if logit_scale_rows else None,
			"ratio_avg": safe_mean(logit_ratios),
			"ratio_min": safe_min(logit_ratios),
			"ratio_max": safe_max(logit_ratios),
		},
		"anomalies": {
			"counts": dict(anomaly_counts),
			"samples": anomaly_samples,
			"fatal_context": fatal_context,
		},
	}
	return summary


def build_report(summary: dict[str, Any]) -> str:
	lines: list[str] = []
	loss = summary["loss"]
	steps = summary["steps"]
	clip = summary["gradient_clip"]
	post_optimizer = summary["post_optimizer"]
	optimizer_trace = summary["optimizer_update_trace"]
	lm_head = summary["lm_head_gemm"]
	logit_scale = summary["logit_scale"]
	anomalies = summary["anomalies"]

	lines.append("# GRIM-text Training Log Analysis")
	lines.append("")
	lines.append(f"- Source log: `{summary['log_path']}`")
	lines.append(f"- Session token: `{summary['session_id']}`")
	lines.append(f"- Generated: {summary['generated_at']}")
	lines.append(f"- Log lines scanned: {summary['line_count']}")
	lines.append(f"- First timestamp: {summary['first_timestamp'] or 'N/A'}")
	lines.append(f"- Last timestamp: {summary['last_timestamp'] or 'N/A'}")
	lines.append("")

	if summary["planned_batches"]:
		lines.append("## Planned workload")
		lines.append("")
		for key, value in summary["planned_batches"].items():
			label = key.replace("_", " ").title()
			lines.append(f"- {label}: {value}")
		lines.append("")

	lines.append("## Loss and step summary")
	lines.append("")
	lines.append(f"- LossStats rows: {loss['count']}")
	if loss["first"]:
		lines.append(
			f"- First loss: batch {loss['first'].get('batch', 'N/A')} -> {format_number(loss['first'].get('loss_mean'), 4)}"
		)
	if loss["last"]:
		lines.append(
			f"- Last loss: batch {loss['last'].get('batch', 'N/A')} -> {format_number(loss['last'].get('loss_mean'), 4)}"
		)
	if loss["best"]:
		lines.append(
			f"- Best loss: batch {loss['best'].get('batch', 'N/A')} -> {format_number(loss['best'].get('loss_mean'), 4)}"
		)
	lines.append(f"- Loss range: {format_number(loss['min_loss'], 4)} to {format_number(loss['max_loss'], 4)}")
	lines.append(f"- Avg first 5 losses: {format_number(loss['avg_first_5'], 4)}")
	lines.append(f"- Avg last 5 losses: {format_number(loss['avg_last_5'], 4)}")
	lines.append(
		f"- Valid tokens avg/min/max: {format_number(loss['valid_tokens_avg'], 2)} / {format_number(loss['valid_tokens_min'])} / {format_number(loss['valid_tokens_max'])}"
	)
	lines.append(f"- [Step N] rows: {steps['count']}")
	if steps["last"]:
		lines.append(
			f"- Last step line: step {steps['last'].get('step', 'N/A')} loss={format_number(steps['last'].get('loss'), 4)} lr={format_number(steps['last'].get('lr'))}"
		)
	lines.append(f"- Last-5 step loss average: {format_number(steps['loss_avg_last_5'], 4)}")
	lines.append("")

	lines.append("## Gradient and optimizer summary")
	lines.append("")
	lines.append(f"- POST-CLIP-MEASURE rows: {clip['count']}")
	lines.append(f"- Clipped=YES count: {clip['clipped_yes']}")
	lines.append(
		f"- Preclip avg/max: {format_number(clip['preclip_avg'])} / {format_number(clip['preclip_max'])}"
	)
	lines.append(
		f"- Postclip avg/max: {format_number(clip['postclip_avg'])} / {format_number(clip['postclip_max'])}"
	)
	if clip["last"]:
		lines.append(
			f"- Last clip row: batch {clip['last'].get('batch', 'N/A')} clipped={clip['last'].get('clipped', 'N/A')} pre={format_number(clip['last'].get('preclip_registered_global'))} post={format_number(clip['last'].get('postclip_registered_global'))}"
		)
	lines.append(f"- POST-OPTIMIZER rows: {post_optimizer['count']}")
	if post_optimizer["last"]:
		lines.append(
			f"- Last optimizer row: opt_step {post_optimizer['last'].get('optimizer_step', 'N/A')} iteration={post_optimizer['last'].get('iteration', 'N/A')} lr={format_number(post_optimizer['last'].get('lr'))} rms={format_number(post_optimizer['last'].get('rms'))}"
		)
	lines.append(f"- OptimizerUpdateTrace rows: {optimizer_trace['count']}")
	if optimizer_trace["last"]:
		lines.append(
			f"- Last OptimizerUpdateTrace component: {optimizer_trace['last'].get('component', 'N/A')} at opt_step {optimizer_trace['last'].get('optimizer_step', 'N/A')}"
		)
	lines.append("")

	lines.append("## Equation health checks")
	lines.append("")
	lines.append(f"- LM_HEAD_GEMM_EQUATION rows: {lm_head['count']}")
	lines.append(
		f"- LM_HEAD_GEMM ratio avg/min/max: {format_number(lm_head['ratio_avg'])} / {format_number(lm_head['ratio_min'])} / {format_number(lm_head['ratio_max'])}"
	)
	if lm_head["last"]:
		lines.append(
			f"- Last LM_HEAD_GEMM ratio: batch {lm_head['last'].get('batch', 'N/A')} step={lm_head['last'].get('step', 'N/A')} ratio={format_number(lm_head['last'].get('ratio'))}"
		)
	lines.append(f"- LOGIT_SCALE_EQUATION rows: {logit_scale['count']}")
	lines.append(
		f"- LOGIT_SCALE ratio avg/min/max: {format_number(logit_scale['ratio_avg'])} / {format_number(logit_scale['ratio_min'])} / {format_number(logit_scale['ratio_max'])}"
	)
	if logit_scale["last"]:
		lines.append(
			f"- Last LOGIT_SCALE ratio: batch {logit_scale['last'].get('batch', 'N/A')} ratio={format_number(logit_scale['last'].get('ratio'))} std={format_number(logit_scale['last'].get('actual_logit_std'))}"
		)
	lines.append("")

	lines.append("## Most common tags")
	lines.append("")
	for item in summary["top_tags"]:
		lines.append(f"- {item['name']}: {item['count']}")
	lines.append("")

	lines.append("## Anomalies")
	lines.append("")
	counts = anomalies["counts"]
	lines.append(f"- Fatal count: {counts.get('fatal', 0)}")
	lines.append(f"- Error count: {counts.get('error', 0)}")
	lines.append(f"- Warning count: {counts.get('warning', 0)}")
	lines.append(f"- Nonfinite markers: {counts.get('nonfinite', 0)}")
	lines.append("")

	for name in ("fatal", "error", "warning", "nonfinite"):
		samples = anomalies["samples"].get(name) or []
		if not samples:
			continue
		lines.append(f"### {name.title()} samples")
		lines.append("")
		for sample in samples:
			timestamp = sample.get("timestamp") or "no-ts"
			lines.append(f"- line {sample['line']} [{timestamp}] {sample['text']}")
		lines.append("")

	return "\n".join(lines).rstrip() + "\n"


def save_outputs(summary: dict[str, Any], report_path: Path, json_path: Path) -> None:
	report_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	report_text = build_report(summary)
	report_path.write_text(report_text, encoding="utf-8")
	json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Analyze GRIM-text training logs with plain string search and save text/JSON summaries."
	)
	parser.add_argument(
		"paths",
		nargs="*",
		help="Log files or directories. Defaults to the latest GRIM-text log when omitted.",
	)
	parser.add_argument(
		"--all-logs",
		action="store_true",
		help="Analyze every training_*.log in the selected directory or the default logs directory.",
	)
	parser.add_argument(
		"--out-dir",
		help="Directory for saved summaries. Defaults to the source log directory.",
	)
	parser.add_argument(
		"--max-lines",
		type=int,
		default=0,
		help="Only scan the first N lines from each log. 0 means scan the full file.",
	)
	parser.add_argument(
		"--stdout-only",
		action="store_true",
		help="Print the report but do not save files.",
	)
	return parser


def main() -> int:
	parser = build_arg_parser()
	args = parser.parse_args()

	if args.max_lines < 0:
		parser.error("--max-lines must be >= 0")

	out_dir = Path(args.out_dir).resolve() if args.out_dir else None
	selected_logs = expand_inputs(args.paths, all_logs=args.all_logs)
	if not selected_logs:
		raise FileNotFoundError("No logs selected for analysis")

	max_lines = args.max_lines if args.max_lines > 0 else None

	for index, log_path in enumerate(selected_logs, start=1):
		summary = analyze_log(log_path, max_lines=max_lines)
		report_text = build_report(summary)
		print(report_text)
		if not args.stdout_only:
			report_path, json_path = derive_output_paths(log_path, out_dir)
			save_outputs(summary, report_path, json_path)
			print(f"Saved report: {report_path}")
			print(f"Saved JSON:   {json_path}")
		if index != len(selected_logs):
			print("\n" + "=" * 88 + "\n")

	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except Exception as exc:
		print(f"ERROR: {exc}", file=sys.stderr)
		raise
