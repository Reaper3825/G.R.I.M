"""Add generic courses to a legacy DataHub registry without changing block membership.

The flat curriculum membership remains a compatibility projection for existing readers.
Run with a registry path; validates first and replaces the file atomically.
"""
import argparse
import copy
import json
from pathlib import Path


def migrate_registry(source):
    result = copy.deepcopy(source)
    courses = result.setdefault("courses", [])
    by_id = {}
    for course in courses:
        if not course["id"] or course["id"] in by_id:
            raise ValueError("Empty or duplicate course ID")
        by_id[course["id"]] = course
    for curriculum in result.get("curriculums", []):
        curriculum.setdefault("randomize_course_order", False)
        curriculum.setdefault("randomize_concept_block_order", False)
        if "course_ids" not in curriculum:
            course_id = "course_generic_" + curriculum["id"]
            if course_id in by_id:
                raise ValueError(f"Generic course ID collision: {course_id}")
            course = {"id": course_id, "name": "General - " + curriculum["name"],
                      "concept_block_ids": list(curriculum.get("concept_block_ids", []))}
            courses.append(course)
            by_id[course_id] = course
            curriculum["course_ids"] = [course_id]
        flattened = list(dict.fromkeys(
            block for course_id in curriculum["course_ids"]
            for block in by_id[course_id]["concept_block_ids"]))
        if flattened != curriculum.get("concept_block_ids", []):
            raise ValueError(f"Membership mismatch: {curriculum['id']}")
    result["schema_version"] = 2
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    args = parser.parse_args()
    source = json.loads(args.registry.read_text(encoding="utf-8"))
    result = migrate_registry(source)
    if result != source:
        temporary = args.registry.with_suffix(args.registry.suffix + ".courses.tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(args.registry)
    print(f"Validated {len(result['curriculums'])} curriculums and {len(result['courses'])} courses")


if __name__ == "__main__":
    main()
