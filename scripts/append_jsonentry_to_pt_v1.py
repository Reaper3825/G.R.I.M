#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "resources" / "models" / "GRIM-text" / "training" / "data"

JSONENTRY_PATH = DATA_DIR / "jsonentry.txt"
CONCEPT_BLOCKS_PATH = DATA_DIR / "concept_blocks.jsonl"
CURRICULUM_REGISTRY_PATH = DATA_DIR / "curriculum_registry.json"
CURRICULUM_MANIFEST_PATH = DATA_DIR / "Pre-Trainingv1.json"
CURRICULUM_NAME = "Pre-Trainingv1"


def try_parse_json(text: str) -> object | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def read_required_entry_text() -> str:
    text = JSONENTRY_PATH.read_text(encoding="utf-8")
    if not (normalized := text.replace("\r\n", "\n").replace("\r", "\n").strip()):
        raise RuntimeError(f"{JSONENTRY_PATH} is empty; nothing to append")
    non_empty_lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    if len(non_empty_lines) > 1 and all(try_parse_json(line) is not None for line in non_empty_lines):
        raise RuntimeError(
            f"{JSONENTRY_PATH} contains multiple JSON documents; provide exactly one entry"
        )
    return normalized


def normalize_string_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError(f"{field_name} must be a JSON array of strings")
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise RuntimeError(f"{field_name}[{index}] must be a string")
        if stripped := item.strip():
            normalized.append(stripped)
    return normalized


def split_segments(segments: list[str]) -> tuple[str, list[str], str]:
    if not segments:
        raise RuntimeError("jsonentry content has no non-empty text segments")
    if len(segments) == 1:
        return "", [], segments[0]
    if len(segments) == 2:
        return segments[0], [], segments[1]
    return segments[0], segments[1:-1], segments[-1]


def split_plaintext_document(text: str) -> tuple[str, list[str], str]:
    return split_segments([line.strip() for line in text.split("\n") if line.strip()])


def build_canonical_plaintext(question: str, intermediates: list[str], answer: str) -> str:
    parts: list[str] = []
    if question:
        parts.append(question)
    parts.extend(intermediates)
    if answer:
        parts.append(answer)
    return "\n".join(parts)


def build_house_block(
    *,
    question: str,
    intermediates: list[str],
    answer: str,
    block_id: str | None = None,
    name: str | None = None,
    format_type: str = "chain_of_thought",
    timestamp: int = 0,
) -> dict:
    if not isinstance(question, str):
        raise RuntimeError("question must be a string")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("answer must be a non-empty string")

    normalized_question = question.strip()
    normalized_intermediates = [item.strip() for item in intermediates if item.strip()]
    normalized_answer = answer.strip()
    canonical_plaintext = build_canonical_plaintext(
        normalized_question,
        normalized_intermediates,
        normalized_answer,
    )
    content_hash = hashlib.sha256(canonical_plaintext.encode("utf-8")).hexdigest()[:12]
    resolved_id = block_id.strip() if isinstance(block_id, str) and block_id.strip() else f"cb_pt_{content_hash}"
    resolved_name = name.strip() if isinstance(name, str) and name.strip() else resolved_id
    step_index = list(range(len(normalized_intermediates)))

    return {
        "answer": normalized_answer,
        "explanation": normalized_intermediates,
        "format_type": format_type,
        "id": resolved_id,
        "intermediate_count": len(normalized_intermediates),
        "intermediates": normalized_intermediates,
        "name": resolved_name,
        "question": normalized_question,
        "step_index": step_index,
        "timestamp": timestamp,
    }


def build_block_from_concept_json(payload: dict) -> dict:
    if "answer" not in payload or not isinstance(payload["answer"], str):
        raise RuntimeError("concept-block JSON input must include a string 'answer'")

    intermediates = normalize_string_list(
        payload.get("intermediates", payload.get("explanation")),
        "intermediates",
    )
    question = payload.get("question", "")
    if not isinstance(question, str):
        raise RuntimeError("concept-block JSON input field 'question' must be a string")
    format_type = payload.get("format_type", "chain_of_thought")
    if not isinstance(format_type, str) or not format_type.strip():
        raise RuntimeError("concept-block JSON input field 'format_type' must be a non-empty string")

    timestamp = payload.get("timestamp", 0)
    if not isinstance(timestamp, int):
        raise RuntimeError("concept-block JSON input field 'timestamp' must be an integer")

    if not question.strip() and not intermediates and isinstance((nested_messages := try_parse_json(payload["answer"])), list):
        return build_block_from_message_json(
            nested_messages,
            block_id=payload.get("id"),
            name=payload.get("name"),
            format_type=format_type,
            timestamp=timestamp,
        )

    return build_house_block(
        question=question,
        intermediates=intermediates,
        answer=payload["answer"],
        block_id=payload.get("id"),
        name=payload.get("name"),
        format_type=format_type,
        timestamp=timestamp,
    )


def build_block_from_message_json(
    payload: list[object],
    *,
    block_id: str | None = None,
    name: str | None = None,
    format_type: str = "chain_of_thought",
    timestamp: int = 0,
) -> dict:
    contents: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RuntimeError(f"message JSON entry {index} must be an object")
        content = item.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"message JSON entry {index} is missing a string 'content'")
        if stripped := content.strip():
            contents.append(stripped)

    question, intermediates, answer = split_segments(contents)
    return build_house_block(
        question=question,
        intermediates=intermediates,
        answer=answer,
        block_id=block_id,
        name=name,
        format_type=format_type,
        timestamp=timestamp,
    )


def build_plaintext_block(text: str) -> dict:
    if (parsed := try_parse_json(text)) is not None:
        if isinstance(parsed, dict):
            if all(key not in parsed for key in ("question", "answer", "intermediates", "explanation")):
                raise RuntimeError(
                    "JSON object input must be a concept-block-style object with question/answer fields"
                )
            return build_block_from_concept_json(parsed)
        if isinstance(parsed, list):
            return build_block_from_message_json(parsed)
        raise RuntimeError(
            "JSON input must be either a concept-block object or an array of message objects"
        )

    question, intermediates, answer = split_plaintext_document(text)
    return build_house_block(
        question=question,
        intermediates=intermediates,
        answer=answer,
    )


def upsert_concept_block(block: dict) -> str:
    serialized_block = json.dumps(block, ensure_ascii=False, separators=(",", ":"))
    matched_existing = False

    fd, temp_path_str = tempfile.mkstemp(
        prefix="concept_blocks.",
        suffix=".jsonl.tmp",
        dir=str(DATA_DIR),
    )
    os.close(fd)
    temp_path = Path(temp_path_str)

    try:
        with CONCEPT_BLOCKS_PATH.open("r", encoding="utf-8") as src, temp_path.open("w", encoding="utf-8") as dst:
            for line_number, line in enumerate(src, start=1):
                stripped = line.strip()
                if not stripped:
                    dst.write("\n")
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON in {CONCEPT_BLOCKS_PATH} at line {line_number}: {exc}"
                    ) from exc

                if row.get("id") == block["id"]:
                    dst.write(serialized_block + "\n")
                    matched_existing = True
                    continue

                dst.write(line if line.endswith("\n") else line + "\n")

            if not matched_existing:
                dst.write(serialized_block + "\n")

        temp_path.replace(CONCEPT_BLOCKS_PATH)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return "updated" if matched_existing else "appended"


def load_registry() -> dict:
    with CURRICULUM_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    if not isinstance(registry, dict) or not isinstance(registry.get("curriculums"), list):
        raise RuntimeError(
            f"{CURRICULUM_REGISTRY_PATH} is missing a top-level 'curriculums' array"
        )
    return registry


def find_curriculum(registry: dict) -> dict:
    for curriculum in registry["curriculums"]:
        if curriculum.get("name") == CURRICULUM_NAME:
            return curriculum
    raise RuntimeError(
        f"Curriculum '{CURRICULUM_NAME}' was not found in {CURRICULUM_REGISTRY_PATH}"
    )


def ensure_plaintext_id_list(curriculum: dict) -> list[str]:
    if curriculum.get("format_as_concept", True):
        raise RuntimeError(
            f"Curriculum '{CURRICULUM_NAME}' must have format_as_concept=false for plaintext rows"
        )

    plaintext_ids = curriculum.get("plaintext_block_ids")
    if plaintext_ids is not None:
        if not isinstance(plaintext_ids, list):
            raise RuntimeError(
                f"Curriculum '{CURRICULUM_NAME}' has a non-list plaintext_block_ids field"
            )
        if "concept_block_ids" not in curriculum:
            curriculum["concept_block_ids"] = []
        return plaintext_ids

    concept_ids = curriculum.get("concept_block_ids")
    if concept_ids is None:
        curriculum["concept_block_ids"] = []
        curriculum["plaintext_block_ids"] = []
        return curriculum["plaintext_block_ids"]

    if not isinstance(concept_ids, list):
        raise RuntimeError(
            f"Curriculum '{CURRICULUM_NAME}' has a non-list concept_block_ids field"
        )

    curriculum["plaintext_block_ids"] = list(concept_ids)
    curriculum["concept_block_ids"] = []
    return curriculum["plaintext_block_ids"]


def write_registry(registry: dict) -> None:
    with CURRICULUM_REGISTRY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_manifest(plaintext_ids: list[str]) -> None:
    manifest = {
        "concept_block_ids": [],
        "plaintext_block_ids": plaintext_ids,
        "format_as_concept": False,
    }
    with CURRICULUM_MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False)
        handle.write("\n")


def clear_entry_file() -> None:
    JSONENTRY_PATH.write_text("", encoding="utf-8")


def main() -> None:
    entry_text = read_required_entry_text()
    block = build_plaintext_block(entry_text)

    concept_block_action = upsert_concept_block(block)

    registry = load_registry()
    curriculum = find_curriculum(registry)
    plaintext_ids = ensure_plaintext_id_list(curriculum)

    if block["id"] not in plaintext_ids:
        plaintext_ids.append(block["id"])
    curriculum["timestamp"] = int(time.time())

    write_registry(registry)
    write_manifest(plaintext_ids)
    clear_entry_file()

    print(f"{concept_block_action}: {block['id']}")
    print(f"updated curriculum: {CURRICULUM_NAME} ({len(plaintext_ids)} plaintext ids)")
    print(f"cleared: {JSONENTRY_PATH}")


if __name__ == "__main__":
    main()