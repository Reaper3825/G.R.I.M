"""Authored semantic relationships for the single-operation arithmetic course.

No operations are inferred from keywords: the requested role selects an inverse
of the story's relationship. All three roles of a case share its facts/wording.
"""
from dataclasses import dataclass
from decimal import Decimal
import random


@dataclass(frozen=True)
class Story:
    unit: str
    suffix: str
    item: str
    subject: str
    other: str
    removed: str
    added: str
    removal: str
    addition: str
    group: str

    @property
    def material(self):
        return self.item.split(" of ", 1)[-1]

    @property
    def measure(self):
        if self.suffix == "count":
            return f"number of {self.item}"
        return f"{self.material} " + {"length": "length", "weight": "mass", "volume": "volume"}[self.suffix]


STORIES = {
    "metric": (
        Story("millimeters", "length", "millimeters of wire", "the first spool", "the second spool", "are cut off", "are spliced on", "wire length cut off", "wire length spliced on", "spools"),
        Story("centimeters", "length", "centimeters of ribbon", "the red roll", "the blue roll", "are cut off", "are attached", "ribbon length cut off", "ribbon length attached", "rolls"),
        Story("meters", "length", "meters of cable", "the workshop", "the depot", "are used", "are delivered", "cable length used", "cable length delivered", "bundles"),
        Story("kilometers", "length", "kilometers of trail", "the northern trail network", "the southern trail network", "are removed from the network", "are added to the network", "trail length removed", "trail length added", "sections"),
        Story("milligrams", "weight", "milligrams of powder", "the first vial", "the second vial", "are extracted", "are added", "powder mass extracted", "powder mass added", "vials"),
        Story("grams", "weight", "grams of flour", "the bakery", "the cafe", "are used", "are delivered", "flour mass used", "flour mass delivered", "bags"),
        Story("kilograms", "weight", "kilograms of rice", "the first store", "the second store", "are sold", "are delivered", "rice mass sold", "rice mass delivered", "sacks"),
        Story("milliliters", "volume", "milliliters of juice", "the first jug", "the second jug", "are poured out", "are poured in", "juice volume poured out", "juice volume poured in", "bottles"),
        Story("liters", "volume", "liters of water", "the tank", "the reservoir", "are used", "are added", "water volume used", "water volume added", "containers"),
    ),
    "imperial": (
        Story("inches", "length", "inches of wood", "the first stack", "the second stack", "are used", "are added", "wood length used", "wood length added", "pieces"),
        Story("feet", "length", "feet of rope", "the red coil", "the blue coil", "are cut off", "are spliced on", "rope length cut off", "rope length spliced on", "coils"),
        Story("yards", "length", "yards of fabric", "the first shop", "the second shop", "are sold", "are delivered", "fabric length sold", "fabric length delivered", "rolls"),
        Story("miles", "length", "miles of road", "the northern road network", "the southern road network", "are removed from the network", "are added to the network", "road length removed", "road length added", "sections"),
        Story("ounces", "weight", "ounces of flour", "the bakery", "the cafe", "are used", "are delivered", "flour weight used", "flour weight delivered", "bags"),
        Story("pounds", "weight", "pounds of rice", "the first store", "the second store", "are sold", "are delivered", "rice weight sold", "rice weight delivered", "sacks"),
        Story("fluid ounces", "volume", "fluid ounces of juice", "the first jug", "the second jug", "are poured out", "are poured in", "juice volume poured out", "juice volume poured in", "bottles"),
        Story("gallons", "volume", "gallons of water", "the tank", "the reservoir", "are drained", "are added", "water volume drained", "water volume added", "containers"),
    ),
    "count": (
        Story("apples", "count", "apples", "Maya", "Ben", "are eaten", "are received", "number of apples eaten", "number of apples received", "baskets"),
        Story("places", "count", "places listed", "the first travel guide", "the second travel guide", "are removed from the guide", "are added to the guide", "number of places removed", "number of places added", "chapters"),
        Story("people", "count", "people", "the morning event", "the evening event", "leave the event", "join the event", "number of people who left", "number of people who joined", "teams"),
        Story("books", "count", "books", "Ada", "Noah", "are lent out", "are received", "number of books lent out", "number of books received", "boxes"),
        Story("tickets", "count", "tickets", "the first ticket office", "the second ticket office", "are sold", "are supplied", "number of tickets sold", "number of tickets supplied", "batches"),
        Story("chairs", "count", "chairs", "the east hall", "the west hall", "are taken away", "are brought in", "number of chairs taken away", "number of chairs brought in", "rows"),
        Story("boxes", "count", "boxes", "the warehouse", "the depot", "are shipped out", "are delivered", "number of boxes shipped out", "number of boxes delivered", "stacks"),
        Story("packages", "count", "packages", "the first depot", "the second depot", "are dispatched", "are delivered", "number of packages dispatched", "number of packages delivered", "batches"),
        Story("students", "count", "students", "the morning class", "the afternoon class", "leave the class", "join the class", "number of students who left", "number of students who joined", "teams"),
        Story("vehicles", "count", "vehicles", "the east parking area", "the west parking area", "leave the parking area", "enter the parking area", "number of vehicles that left", "number of vehicles that entered", "rows"),
        Story("marbles", "count", "marbles", "Lena", "Owen", "are given away", "are received", "number of marbles given away", "number of marbles received", "bags"),
        Story("tools", "count", "tools", "the workshop", "the garage", "are lent out", "are returned", "number of tools lent out", "number of tools returned", "kits"),
    ),
}

RELATIONSHIPS = ("decrease", "increase", "comparison", "groups")
TARGETS = {
    "decrease": ("start", "removed", "remaining"),
    "increase": ("start", "added", "final"),
    "comparison": ("smaller", "difference", "larger"),
    "groups": ("group_count", "per_group", "total"),
}
SOLUTIONS = {
    "decrease": ((1, "add", 2), (0, "sub", 2), (0, "sub", 1)),
    "increase": ((2, "sub", 1), (2, "sub", 0), (0, "add", 1)),
    "comparison": ((2, "sub", 1), (2, "sub", 0), (0, "add", 1)),
    "groups": ((2, "div", 1), (2, "div", 0), (0, "mul", 1)),
}
PREFIXES = {
    "decrease": ("start", "change", "remainder"),
    "increase": ("start", "change", "total"),
    "comparison": ("base", "difference", "total"),
    "groups": ("group", "quantity", "total"),
}


def decimal_text(value):
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def describe_case(story, relationship, values, wording):
    """Return three independent fact clauses, questions, and semantic role labels."""
    a, b, c = map(decimal_text, values)
    s, o, item = story.subject, story.other, story.item
    quantity = "number" if story.suffix == "count" else "amount"
    if relationship in ("decrease", "increase"):
        decreasing = relationship == "decrease"
        action = story.removed if decreasing else story.added
        change = story.removal if decreasing else story.addition
        # The action is also stated without a value when that is the unknown.
        # This keeps the event explicit for the initial/final known-fact pair.
        facts = (
            (f"Initially, {s} has {a} {item}.", f"Then {b} {item} {action}.", f"Afterward, {s} has {c} {item}."),
            (f"Before the change, {s} has {a} {item}.", f"During the change, {b} {item} {action}.", f"Once the change is complete, {s} has {c} {item}."),
            (f"The starting {quantity} for {s} is {a} {item}.", f"A total of {b} {item} {action}.", f"The final {quantity} for {s} is {c} {item}."),
            (f"At first, {s} has {a} {item}.", f"Next, {b} {item} {action}.", f"At the end, {s} has {c} {item}."),
        )[wording]
        questions = (
            f"How many {item} did {s} have at first?",
            f"What is the {change}, in {story.unit}?",
            f"How many {item} does {s} have afterward?",
        )
        if wording % 2:
            questions = (
                f"What was the original {story.measure} for {s}, in {story.unit}?",
                f"How many {item} {action.replace('are ', 'were ', 1)}?" if action.startswith("are ") else f"What is the {change}?",
                f"How many {item} remain?" if decreasing else f"What is the final {story.measure} for {s}, in {story.unit}?",
            )
        roles = (f"starting {story.measure}", change,
                 f"remaining {story.measure}" if decreasing else f"final {story.measure}")
        bridge = f"Some {item} {action}." if story.suffix == "count" else f"Some {story.material} {action.replace('are ', 'is ', 1)}."
    elif relationship == "comparison":
        facts = (
            (f"{s.capitalize()} has {a} {item}.", f"{o.capitalize()} has {b} more {item} than {s}.", f"{o.capitalize()} has {c} {item}."),
            (f"{s.capitalize()} has {a} {item}.", f"{s.capitalize()} has {b} fewer {item} than {o}.", f"{o.capitalize()} has {c} {item}."),
            (f"There are {a} {item} with {s}.", f"The {quantity} with {o} exceeds that with {s} by {b} {story.unit}.", f"There are {c} {item} with {o}."),
            (f"The {quantity} for {s} is {a} {item}.", f"The difference is {b} {story.unit}, with {o} having more.", f"The {quantity} for {o} is {c} {item}."),
        )[wording]
        questions = (f"How many {item} does {s} have?",
                     f"How many more {item} does {o} have than {s}?" if wording % 2 == 0 else f"How many fewer {item} does {s} have than {o}?",
                     f"How many {item} does {o} have?")
        roles = (f"{story.measure} with {s}", f"difference in {story.measure} between {s} and {o}", f"{story.measure} with {o}")
        bridge = ""
    else:
        group = story.group
        # 'per group' avoids incorrect singularization of boxes/batches/etc.
        facts = (
            (f"There are {a} equal {group}.", f"Each group contains {b} {item}.", f"Together they contain {c} {item}."),
            (f"There are {a} equal {group} in the distribution.", f"The amount per group is {b} {story.unit}.", f"The total is {c} {item}."),
            (f"There are {a} {group} of equal size.", f"Every group has {b} {item}.", f"The combined amount is {c} {item}."),
            (f"The distribution uses {a} equal {group}.", f"A single group receives {b} {item}.", f"There are {c} {item} in all."),
        )[wording]
        questions = (f"How many {group} are there?", f"How many {item} are in each group?", f"How many {item} are there altogether?")
        roles = (f"number of {group}", f"{story.measure} per group", f"total {story.measure}")
        bridge = f"The {group} all contain equal amounts."
    return facts, questions, roles, bridge


def make_case(relationship, occurrence, seed, category=None, values=None, story=None):
    """Deterministic case shared by its three possible unknown positions."""
    family = RELATIONSHIPS.index(relationship)
    category = category or tuple(STORIES)[(occurrence + family) % 3]
    story = story or STORIES[category][(occurrence // 3) % len(STORIES[category])]
    rng = random.Random(f"{seed}:{relationship}:{occurrence}")
    scale = Decimal(10 if category != "count" and rng.randrange(4) == 0 else 1)
    if values is None:
        # Two independent coprime periods exceed the number of authored cases.
        a = Decimal(10 + ((occurrence + seed) * 37) % 490) / scale
        b = Decimal(2 + ((occurrence + seed) * 53) % 97) / scale
        if relationship == "decrease":
            values = (a + b, b, a)
        elif relationship == "groups":
            groups = Decimal(2 + ((occurrence + seed) * 11) % 97)
            values = (groups, a, groups * a)
        else:
            values = (a, b, a + b)
    wording = rng.randrange(4)
    facts, questions, roles, bridge = describe_case(story, relationship, values, wording)
    return dict(relationship=relationship, category=category, story=story, values=values,
                facts=facts, questions=questions, roles=roles, bridge=bridge,
                reverse=rng.choice((False, True)), question_first=rng.randrange(3) == 0,
                wording=wording, determine_variant=rng.randrange(3))


def render_case(case, target):
    rel, story = case["relationship"], case["story"]
    left, operation, right = SOLUTIONS[rel][target]
    known_roles = [i for i in range(3) if i != target]
    if case["reverse"]:
        known_roles.reverse()
    facts = [case["facts"][i] for i in known_roles]
    if (rel in ("decrease", "increase") and target == 1) or (rel == "groups" and target == 0):
        facts.insert(0, case["bridge"])
    question = case["questions"][target]
    prompt = " ".join(([question] + facts) if case["question_first"] else (facts + [question]))
    roles = case["roles"]
    action = {
        "add": f"Add the {roles[left]} and the {roles[right]}",
        "sub": f"Subtract the {roles[right]} from the {roles[left]}",
        "mul": f"Multiply the {roles[left]} by the {roles[right]}",
        "div": f"Divide the {roles[left]} by the {roles[right]}",
    }[operation]
    determine = (
        f"{action} to find the {roles[target]}.",
        f"Find the {roles[target]} by {action[0].lower() + action[1:]}.",
        f"Calculate the {roles[target]}: {action[0].lower() + action[1:]}.",
    )[case["determine_variant"]]
    # The gerund must be grammatical in the 'by ...' variant.
    determine = determine.replace("by add ", "by adding ").replace("by subtract ", "by subtracting ").replace("by multiply ", "by multiplying ").replace("by divide ", "by dividing ")
    all_names = [f"{prefix}_{story.suffix}" for prefix in PREFIXES[rel]]
    if rel == "groups":
        all_names[0] = "group_count"
    unit = story.group if rel == "groups" and target == 0 else story.unit
    result = "${" + all_names[target] + "}"
    answer = f"The {roles[target]} is {result}" + ("." if rel == "groups" and target == 0 else f" {unit}.")
    return dict(prompt=prompt, determine=determine, answer=answer, unit=unit,
                names=tuple(all_names[i] for i in (left, right, target)),
                lhs=case["values"][left], rhs=case["values"][right], result=case["values"][target],
                operation=operation, relationship=rel, target_role=TARGETS[rel][target],
                category=case["category"], input_unit=story.unit, wording=case["wording"],
                question_first=case["question_first"], reverse_facts=case["reverse"])
