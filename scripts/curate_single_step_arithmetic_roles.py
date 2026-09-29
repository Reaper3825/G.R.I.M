#!/usr/bin/env python3
"""Write a small, reviewed role-selection supplement; never merge or train it.

Each authored scenario has three questions, one for each possible unknown.
Expected numeric results and semantic roles live in the review manifest only.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "DataCollection" / "curated" / "single_step_arithmetic_roles_v1"

# Roles are ordered by their relationship, independently of sentence order.
# For decreases: initial - removed = remaining.
# For increases: initial + added = final.
# For groups: group count * quantity per group = total.
# For comparisons: smaller + difference = larger.
SCENARIOS = [
    ("tank", "train", "decrease", "liters", ("120", "36", "84"),
     ("starting volume", "volume used", "volume remaining"),
     ("A tank has some water. After 36 liters are used, 84 liters remain. How many liters were in the tank at first?",
      "A tank holds 120 liters. After using a few liters, 84 liters remain. How many liters were used?",
      "A tank holds 120 liters. After using 36 liters, how many liters remain?")),
    ("library", "train", "decrease", "books", ("95", "28", "67"),
     ("original number of books", "number of books borrowed", "number of books still on the shelf"),
     ("After readers borrowed 28 books, 67 books were still on the shelf. How many books were on the shelf before the borrowing?",
      "There were 95 books on a shelf. Readers borrowed some, leaving 67 books on the shelf. How many books did readers borrow?",
      "Readers borrowed 28 books from a shelf that originally held 95 books. How many books are still on the shelf?")),
    ("rope", "train", "decrease", "meters", ("18.5", "6.25", "12.25"),
     ("original rope length", "length cut off", "length left"),
     ("A piece 6.25 meters long was cut off a rope, leaving 12.25 meters. How long was the rope before it was cut?",
      "How many meters were cut off a rope if it was originally 18.5 meters long and 12.25 meters are left?",
      "A rope was originally 18.5 meters long. A piece 6.25 meters long was cut off. How many meters are left?")),
    ("tickets", "train", "decrease", "tickets", ("240", "175", "65"),
     ("initial number of tickets", "number of tickets sold", "number of unsold tickets"),
     ("There are 65 unsold tickets after 175 tickets were sold. How many tickets were available before any were sold?",
      "There are 65 unsold tickets out of the 240 initially available. How many tickets have been sold?",
      "Of the 240 tickets initially available, 175 have been sold. How many tickets are unsold?")),
    ("marbles", "train", "increase", "marbles", ("23", "19", "42"),
     ("original number of marbles", "number of marbles received", "final number of marbles"),
     ("Maya received 19 more marbles and now has 42. How many marbles did she have before receiving them?",
      "Maya had 23 marbles. After receiving some more, she has 42. How many marbles did she receive?",
      "Maya had 23 marbles and received 19 more. How many marbles does she have now?")),
    ("refill", "train", "increase", "liters", ("84", "36", "120"),
     ("volume before the refill", "volume added", "volume after the refill"),
     ("After 36 liters were added to a tank, it contained 120 liters. How many liters were in the tank before the refill?",
      "A tank contained 84 liters before a refill and 120 liters afterward. How many liters were added?",
      "A tank contained 84 liters. A refill added 36 liters. How many liters are in the tank afterward?")),
    ("boxes", "train", "groups", "books", ("7", "12", "84"),
     ("number of boxes", "number of books per box", "total number of books"),
     ("There are 84 books packed into boxes with 12 books in each box. How many boxes are there?",
      "There are 84 books packed equally into 7 boxes. How many books are in each box?",
      "There are 7 boxes with 12 books in each box. How many books are there in total?")),
    ("bottles", "train", "groups", "liters", ("6", "1.5", "9"),
     ("number of bottles", "volume per bottle", "total volume of juice"),
     ("Each bottle contains 1.5 liters of juice. The bottles contain 9 liters altogether. How many bottles are there?",
      "Six identical bottles contain 9 liters of juice altogether. How many liters does each bottle contain?",
      "Six bottles each contain 1.5 liters of juice. How many liters do the bottles contain altogether?")),
    ("collections", "train", "comparison", "books", ("28", "15", "43"),
     ("number of books Ben owns", "number of extra books Ada owns", "number of books Ada owns"),
     ("Ada owns 43 books, which is 15 more than Ben owns. How many books does Ben own?",
      "Ada owns 43 books and Ben owns 28. How many more books does Ada own than Ben?",
      "Ben owns 28 books. Ada owns 15 more books than Ben. How many books does Ada own?")),
    ("ribbons", "train", "comparison", "centimeters", ("32", "18", "50"),
     ("blue ribbon length", "difference in ribbon lengths", "red ribbon length"),
     ("The blue ribbon is 18 centimeters shorter than the red ribbon. The red ribbon is 50 centimeters long. How long is the blue ribbon?",
      "The blue ribbon is 32 centimeters long and the red ribbon is 50 centimeters long. How many centimeters shorter is the blue ribbon?",
      "The blue ribbon is 32 centimeters long, which is 18 centimeters shorter than the red ribbon. How long is the red ribbon?")),
    ("flour", "eval", "decrease", "kilograms", ("22.5", "8.75", "13.75"),
     ("original flour mass", "flour mass used", "flour mass left"),
     ("A bakery used 8.75 kilograms of flour and had 13.75 kilograms left. How many kilograms did it have before baking?",
      "A bakery had 22.5 kilograms of flour before baking and 13.75 kilograms afterward. How many kilograms went into the baking?",
      "A bakery used 8.75 kilograms from its 22.5 kilograms of flour. How many kilograms are left?")),
    ("chairs", "eval", "groups", "chairs", ("8", "9", "72"),
     ("number of rows", "number of chairs per row", "total number of chairs"),
     ("An auditorium has 72 chairs arranged with 9 chairs in every row. How many rows does it have?",
      "An auditorium has 72 chairs arranged in 8 equally sized rows. How many chairs are in one row?",
      "An auditorium has 8 rows with 9 chairs in each row. How many chairs does it have altogether?")),
]

# (left role, operator, right role) indexed by the requested role.
SOLUTIONS = {
    "decrease": ((1, "+", 2), (0, "-", 2), (0, "-", 1)),
    "increase": ((2, "-", 1), (2, "-", 0), (0, "+", 1)),
    "groups": ((2, "/", 1), (2, "/", 0), (0, "*", 1)),
    "comparison": ((2, "-", 1), (2, "-", 0), (0, "+", 1)),
}
PREFIXES = {
    "decrease": ("start", "change", "remainder"),
    "increase": ("start", "change", "total"),
    "groups": ("group", "quantity", "total"),
    "comparison": ("quantity", "change", "total"),
}


def make_entries():
    seen = set()
    for scenario, split, relationship, unit, values, roles, prompts in SCENARIOS:
        for target, prompt in enumerate(prompts):
            left, op, right = SOLUTIONS[relationship][target]
            a, b, expected = map(Decimal, (values[left], values[right], values[target]))
            actual = {"+": lambda: a + b, "-": lambda: a - b,
                      "*": lambda: a * b, "/": lambda: a / b}[op]()
            assert actual == expected, (scenario, target)
            assert prompt not in seen
            seen.add(prompt)
            names = [f"{prefix}_{unit}" for prefix in PREFIXES[relationship]]
            if relationship == "groups":
                names[0] = "group_count"
            refs = ["${" + name + "}" for name in names]
            method = {
                "+": f"Add the {roles[left]} and the {roles[right]}",
                "-": f"Subtract the {roles[right]} from the {roles[left]}",
                "*": f"Multiply the {roles[left]} by the {roles[right]}",
                "/": f"Divide the {roles[left]} by the {roles[right]}",
            }[op]
            block_id = f"ssat_roles_v1_{scenario}_{target}"
            # The inferred role appears in the supervised continuation. The goal
            # intentionally supplies no operation or role hint beyond the prompt.
            entry = {
                "id": block_id,
                "name": f"Arithmetic role selection: {scenario}, {roles[target]}",
                "prompt": prompt, "knowns": [], "unknowns": [],
                "determine": f"{method} to find the {roles[target]}.",
                "define": f"{refs[left]} -> {values[left]};\n{refs[right]} -> {values[right]};\n{refs[target]};",
                "execute": f"<TOOL>({refs[left]} {op} {refs[right]})</TOOL> -> {refs[target]}",
                "update": "",
                "answer": f"The {roles[target]} is {refs[target]}.",
                "goal": {
                    "target_state": "The quantity requested in the question is reported.",
                    "success_criteria": [{
                        "criterion": "The response answers the question using the stated quantities.",
                        "evidence": "",
                    }],
                    "constraints": ["Use exactly one single-step arithmetic tool call."],
                },
                "format_type": "derivation",
                "source_sequence_id": f"single_step_arithmetic_roles_v1:{scenario}",
                "timestamp": 0,
            }
            is_group_count = relationship == "groups" and target == 0
            result_unit = {"boxes": "boxes", "bottles": "bottles", "chairs": "rows"}[scenario] if is_group_count else unit
            if not is_group_count:
                entry["answer"] = f"The {roles[target]} is {refs[target]} {unit}."
            yield split, entry, {
                "id": block_id, "split": split, "scenario": scenario,
                "relationship": relationship, "requested_role": roles[target],
                "operator": op, "expected_value": str(expected),
                "expected_unit": result_unit,
            }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = list(make_entries())
    for split in ("train", "eval"):
        entries = [entry for label, entry, _ in records if label == split]
        (OUTPUT / f"{split}.jsonl").write_text(
            "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
            encoding="utf-8")
    (OUTPUT / "review_manifest.json").write_text(
        json.dumps([meta for _, _, meta in records], indent=2) + "\n", encoding="utf-8")
    print(f"Wrote 30 training entries and 6 evaluation entries to {OUTPUT}")


if __name__ == "__main__":
    main()
