"""Structural Goal validation shared by migration and the FlatBuffer compiler.

Predicate support, argument types and entity binding are environment checks.
"""


def object_fields(value, allowed):
    if not isinstance(value, dict):
        raise ValueError("Goal contract expects an object")
    extra = set(value) - set(allowed)
    if extra:
        raise ValueError(f"Unsupported Goal contract fields: {sorted(extra)}")


def validate_evidence(value):
    object_fields(value, {"predicate", "subject", "target"})
    for field in ("predicate", "subject"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(f"Evidence {field} must be a non-empty string")
    if "target" in value and not isinstance(value["target"], str):
        raise ValueError("Evidence target must be a string")


def validate_goal(goal):
    object_fields(goal, {"target_state", "success_criteria", "constraints"})
    if not isinstance(goal.get("target_state", ""), str):
        raise ValueError("target_state must be a string")
    for collection, text_key in (("success_criteria", "criterion"), ("constraints", "constraint")):
        entries = goal.get(collection, [])
        if not isinstance(entries, list):
            raise ValueError(f"{collection} must be an array")
        for entry in entries:
            object_fields(entry, {text_key, "evidence"})
            if not isinstance(entry.get(text_key), str):
                raise ValueError(f"{text_key} must be a string")
            if "evidence" in entry:
                validate_evidence(entry["evidence"])
