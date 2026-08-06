#!/usr/bin/env python3
"""Generate the GRIM-text Intent Action v1 curriculum.

Each row asks what the user wants done. The concise imperative action belongs
in ``answer``; ``goal.target_state`` is deliberately present and empty.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from vocab_playground import build_trie, load_vocab_bin, tokenize  # noqa: E402


DATA_DIR = ROOT_DIR / "resources" / "models" / "GRIM-text" / "training" / "data"
CURRICULUM_ID = "curr_intent_action_v1"
CURRICULUM_NAME = "Intent Action v1"
ENTRY_PREFIX = "iav1_"
DEFAULT_COUNT = 10_000
MAX_SEQUENCE_TOKENS = 1024
BOUNDARY_TOKEN_BUDGET = 2

CONTEXTS = (
    "Please handle this now:",
    "When convenient,",
    "For my current task,",
    "Before I leave,",
    "As part of my morning routine,",
    "For the rest of the day,",
    "In the workspace,",
    "On this device,",
    "For the household,",
    "Before my next meeting,",
    "While I am occupied,",
    "For this request,",
    "Could you help and",
    "I need you to",
    "Go ahead and",
    "At the next opportunity,",
    "For my evening routine,",
    "Before the workday begins,",
    "Once the current task is finished,",
    "Without changing anything else,",
    "For the setup I am using,",
    "As my next action,",
    "Before I start working,",
    "After the current activity,",
    "For this device session,",
    "As part of the current workflow,",
    "Before the next appointment,",
    "While the device is available,",
    "For the plan I am following,",
    "As soon as possible,",
    "For my home setup,",
    "Before I continue,",
    "During this session,",
    "For the task at hand,",
    "When the device is ready,",
    "As part of my daily routine,",
    "Before the next task,",
    "For my current workspace,",
    "When you are ready,",
    "For the active session,",
    "Before I step away,",
    "As the next step,",
    "For my personal setup,",
    "During the current activity,",
    "Before the day ends,",
    "For this particular task,",
    "Once you can,",
    "As part of the plan,",
)

ROOMS = ("kitchen", "bedroom", "living room", "office", "hallway", "garage", "studio", "guest room")
DEVICES = ("lights", "lamps", "ceiling lights", "desk lights", "wall lights", "accent lights")
DOORS = ("front door", "back door", "garage door", "side door", "patio door")
APPS = ("calendar", "mail", "browser", "notes", "music", "camera", "maps", "calculator")
FILES = ("project brief", "meeting notes", "budget draft", "design outline", "travel plan", "research summary")
FOLDERS = ("archive", "documents", "projects", "shared folder", "downloads", "work folder")
CONTACTS = ("Alex", "Morgan", "Taylor", "Jordan", "Casey", "Riley", "Sam", "Cameron")
CHANNELS = ("project channel", "family chat", "support thread", "design group", "team chat", "planning room")
MEDIA = ("morning playlist", "focus mix", "latest episode", "saved album", "news briefing", "audiobook")
PLACES = ("nearest pharmacy", "central station", "city library", "grocery store", "airport", "community center")
ITEMS = ("coffee", "notebooks", "soap", "printer paper", "tea", "batteries", "rice", "headphones")
DOCUMENTS = ("project brief", "travel itinerary", "meeting agenda", "expense report", "design proposal", "research notes")
LANGUAGES = ("Spanish", "French", "German", "Italian", "Japanese", "Portuguese")
NUMBERS = ("15", "20", "25", "30", "40", "50", "65", "75")
TIMES = ("7:00", "8:30", "10:15", "13:00", "16:45", "18:30", "21:00", "23:15")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass(frozen=True)
class Pattern:
    domain: str
    answer: str
    requests: tuple[str, ...]
    slots: tuple[tuple[str, tuple[str, ...]], ...] = ()


def pattern(
    domain: str,
    answer: str,
    requests: tuple[str, ...],
    **slots: tuple[str, ...],
) -> Pattern:
    return Pattern(domain, answer, requests, tuple(slots.items()))


PATTERNS = (
    pattern("smart_home", "Turn off {device}", ("turn off the {device} in the {room}.", "switch the {room} {device} off.", "make sure the {device} in the {room} are off."), device=DEVICES, room=ROOMS),
    pattern("smart_home", "Turn on {device}", ("turn on the {device} in the {room}.", "switch the {room} {device} on.", "activate the {device} in the {room}."), device=DEVICES, room=ROOMS),
    pattern("smart_home", "Dim {device}", ("dim the {device} in the {room} to {number} percent.", "set the {room} {device} brightness to {number} percent.", "lower the {device} in the {room} until they are at {number} percent."), device=DEVICES, room=ROOMS, number=NUMBERS),
    pattern("smart_home", "Brighten {device}", ("brighten the {device} in the {room} to {number} percent.", "raise the {room} {device} brightness to {number} percent.", "make the {device} in the {room} {number} percent bright."), device=DEVICES, room=ROOMS, number=NUMBERS),
    pattern("smart_home", "Lock {door}", ("lock the {door}.", "secure the {door} lock.", "make sure the {door} is locked."), door=DOORS),
    pattern("smart_home", "Unlock {door}", ("unlock the {door}.", "release the lock on the {door}.", "make sure the {door} is unlocked."), door=DOORS),
    pattern("smart_home", "Open {room} blinds", ("open the blinds in the {room}.", "raise the {room} window blinds.", "let daylight in through the {room} blinds."), room=ROOMS),
    pattern("smart_home", "Close {room} blinds", ("close the blinds in the {room}.", "lower the {room} window blinds.", "shut the blinds in the {room}."), room=ROOMS),
    pattern("smart_home", "Set {room} temperature", ("set the {room} temperature to {number} degrees.", "adjust the thermostat in the {room} to {number} degrees.", "make the {room} {number} degrees."), room=ROOMS, number=NUMBERS),
    pattern("smart_home", "Turn on {room} fan", ("turn on the fan in the {room}.", "start the {room} fan.", "switch the fan in the {room} on."), room=ROOMS),
    pattern("smart_home", "Turn off {room} fan", ("turn off the fan in the {room}.", "stop the {room} fan.", "switch the fan in the {room} off."), room=ROOMS),
    pattern("applications", "Open {app}", ("open the {app} app.", "launch {app}.", "bring up {app} on this device."), app=APPS),
    pattern("applications", "Close {app}", ("close the {app} app.", "quit {app}.", "shut down the open {app} application."), app=APPS),
    pattern("applications", "Switch to {app}", ("switch over to {app}.", "bring {app} to the foreground.", "show me the open {app} app."), app=APPS),
    pattern("applications", "Enable {app} notifications", ("enable notifications for {app}.", "allow {app} to send notifications.", "turn {app} alerts back on."), app=APPS),
    pattern("applications", "Disable {app} notifications", ("disable notifications for {app}.", "stop {app} from sending notifications.", "turn off alerts from {app}."), app=APPS),
    pattern("applications", "Clear {app} cache", ("clear the cache for {app}.", "remove cached data from {app}.", "clean out the {app} application cache."), app=APPS),
    pattern("files", "Create {file}", ("create a new file named {file}.", "make a document called {file}.", "start a new {file} file."), file=FILES),
    pattern("files", "Delete {file}", ("delete the file named {file}.", "remove {file} from my files.", "send the {file} document to the recycle bin."), file=FILES),
    pattern("files", "Rename {file}", ("rename the {file} document.", "change the filename for {file}.", "give the {file} file a new name."), file=FILES),
    pattern("files", "Move {file} to {folder}", ("move {file} into the {folder}.", "put the {file} document in the {folder}.", "relocate {file} to the {folder}."), file=FILES, folder=FOLDERS),
    pattern("files", "Copy {file} to {folder}", ("copy {file} into the {folder}.", "make a copy of {file} in the {folder}.", "duplicate {file} to the {folder}."), file=FILES, folder=FOLDERS),
    pattern("files", "Archive {folder}", ("archive the {folder}.", "compress the {folder} for storage.", "put the {folder} into an archive."), folder=FOLDERS),
    pattern("files", "Search files for {file}", ("search my files for {file}.", "find the document called {file}.", "look through storage for {file}."), file=FILES),
    pattern("files", "Open {file}", ("open the {file} document.", "show me the file named {file}.", "load {file} from my files."), file=FILES),
    pattern("communication", "Send message to {contact}", ("send {contact} a message saying I am on my way.", "message {contact} that the plan is confirmed.", "tell {contact} I will respond soon."), contact=CONTACTS),
    pattern("communication", "Reply to {contact}", ("reply to {contact} and say that works for me.", "respond to {contact} with a confirmation.", "answer {contact}'s latest message."), contact=CONTACTS),
    pattern("communication", "Forward message to {contact}", ("forward the latest message to {contact}.", "send {contact} a copy of this conversation.", "pass this message along to {contact}."), contact=CONTACTS),
    pattern("communication", "Call {contact}", ("call {contact}.", "start a voice call with {contact}.", "place a call to {contact}."), contact=CONTACTS),
    pattern("communication", "Email {contact}", ("email {contact} the project update.", "send the current summary to {contact} by email.", "write {contact} an email about the schedule."), contact=CONTACTS),
    pattern("communication", "Mute {channel}", ("mute the {channel}.", "silence notifications from the {channel}.", "stop alerts for the {channel}."), channel=CHANNELS),
    pattern("media", "Play {media}", ("play my {media}.", "start the {media}.", "put on the {media}."), media=MEDIA),
    pattern("media", "Pause media", ("pause what is currently playing.", "pause the audio for now.", "stop playback temporarily.")),
    pattern("media", "Resume media", ("resume what I was listening to.", "continue the paused playback.", "start playing the paused media again.")),
    pattern("media", "Skip media", ("skip the current track.", "move to the next item in the queue.", "advance past what is playing.")),
    pattern("media", "Replay media", ("replay the current track.", "start this item again from the beginning.", "play the current audio over again.")),
    pattern("media", "Set media volume", ("set the playback volume to {number} percent.", "change the media volume to {number} percent.", "make the audio volume {number} percent."), number=NUMBERS),
    pattern("media", "Save {media}", ("save the {media} to my library.", "add the {media} to my saved media.", "keep the {media} in my collection."), media=MEDIA),
    pattern("system", "Turn on WiFi", ("turn on WiFi.", "enable the wireless network connection.", "switch WiFi on for this device.")),
    pattern("system", "Turn off WiFi", ("turn off WiFi.", "disable the wireless network connection.", "switch WiFi off for this device.")),
    pattern("system", "Turn on Bluetooth", ("turn on Bluetooth.", "enable Bluetooth connectivity.", "switch Bluetooth on for this device.")),
    pattern("system", "Turn off Bluetooth", ("turn off Bluetooth.", "disable Bluetooth connectivity.", "switch Bluetooth off for this device.")),
    pattern("system", "Set screen brightness", ("set the screen brightness to {number} percent.", "change display brightness to {number} percent.", "make the screen {number} percent bright."), number=NUMBERS),
    pattern("system", "Take screenshot", ("take a screenshot.", "capture the current screen.", "save an image of what is on the display.")),
    pattern("system", "Restart device", ("restart this device.", "reboot the computer.", "perform a system restart.")),
    pattern("system", "Shut down device", ("shut down this device.", "turn off the computer.", "power the system down.")),
    pattern("scheduling", "Create meeting with {contact}", ("schedule a meeting with {contact} on {day} at {time}.", "create a {day} calendar event with {contact} for {time}.", "book time with {contact} at {time} on {day}."), contact=CONTACTS, day=DAYS, time=TIMES),
    pattern("scheduling", "Cancel meeting with {contact}", ("cancel my meeting with {contact} on {day}.", "remove the {day} appointment with {contact}.", "call off the calendar event with {contact} on {day}."), contact=CONTACTS, day=DAYS),
    pattern("scheduling", "Reschedule meeting with {contact}", ("move my meeting with {contact} to {day} at {time}.", "reschedule the appointment with {contact} for {time} on {day}.", "change the meeting with {contact} to {day} at {time}."), contact=CONTACTS, day=DAYS, time=TIMES),
    pattern("scheduling", "Create reminder", ("remind me on {day} at {time} to review the project.", "create a reminder for {time} on {day} about the report.", "set a {day} reminder at {time} to check my notes."), day=DAYS, time=TIMES),
    pattern("scheduling", "Start timer", ("start a timer for {number} minutes.", "set a {number} minute timer.", "begin counting down {number} minutes."), number=NUMBERS),
    pattern("scheduling", "Set alarm", ("set an alarm for {time} on {day}.", "wake me at {time} on {day}.", "create a {day} alarm for {time}."), time=TIMES, day=DAYS),
    pattern("search", "Search web for {file}", ("search the web for {file} examples.", "look online for information about {file}.", "find web results related to {file}."), file=FILES),
    pattern("search", "Navigate to {place}", ("give me directions to the {place}.", "navigate to the {place}.", "start a route to the {place}."), place=PLACES),
    pattern("search", "Find {place}", ("find the {place} nearby.", "show me where the {place} is.", "locate the {place} closest to me."), place=PLACES),
    pattern("search", "Check weather", ("check the weather for {day}.", "show me the forecast for {day}.", "look up what the weather will be like on {day}."), day=DAYS),
    pattern("commerce", "Add {item} to shopping list", ("add {item} to my shopping list.", "put {item} on the grocery list.", "remember that I need to buy {item}."), item=ITEMS),
    pattern("commerce", "Remove {item} from shopping list", ("remove {item} from my shopping list.", "take {item} off the grocery list.", "delete {item} from the things I need to buy."), item=ITEMS),
    pattern("commerce", "Order {item}", ("place an order for {number} packs of {item}.", "buy {number} packages of {item}.", "order {item} with a quantity of {number}."), item=ITEMS, number=NUMBERS),
    pattern("commerce", "Track order", ("track order {number} for me.", "check the delivery status of order {number}.", "find out where shipment {number} is."), number=NUMBERS),
    pattern("commerce", "Cancel order", ("cancel order {number}.", "stop the purchase associated with order {number}.", "request cancellation for order {number}."), number=NUMBERS),
    pattern("productivity", "Print {document}", ("print the {document}.", "send the {document} to the printer.", "make a paper copy of the {document}."), document=DOCUMENTS),
    pattern("productivity", "Scan {document}", ("scan the {document}.", "create a digital scan of the {document}.", "capture the {document} with the scanner."), document=DOCUMENTS),
    pattern("productivity", "Translate {document} to {language}", ("translate the {document} into {language}.", "convert the {document} text to {language}.", "make a {language} version of the {document}."), document=DOCUMENTS, language=LANGUAGES),
    pattern("productivity", "Summarize {document}", ("summarize the {document}.", "give me a concise summary of the {document}.", "condense the {document} into its main points."), document=DOCUMENTS),
    pattern("productivity", "Check spelling in {document}", ("check the spelling in the {document}.", "run a spelling review on the {document}.", "find spelling mistakes in the {document}."), document=DOCUMENTS),
    pattern("meetings", "Share screen", ("share my screen in the current meeting.", "start presenting my display.", "show the meeting participants my screen.")),
    pattern("meetings", "Start recording meeting", ("start recording the current meeting.", "begin a recording of this call.", "record the meeting from this point.")),
    pattern("meetings", "Stop recording meeting", ("stop recording the current meeting.", "end the recording of this call.", "finish and save the meeting recording.")),
    pattern("system", "Enable focus mode", ("enable focus mode.", "turn on do not disturb.", "silence distractions with focus mode.")),
    pattern("system", "Enable airplane mode", ("enable airplane mode.", "turn airplane mode on.", "switch this device to airplane mode.")),
)


def selected_values(item: Pattern, occurrence: int) -> dict[str, str]:
    quotient = occurrence // (len(item.requests) * len(CONTEXTS))
    selected: dict[str, str] = {}
    for key, values in item.slots:
        if not values:
            raise ValueError(f"pattern slot {key!r} has no values")
        selected[key] = values[quotient % len(values)]
        quotient //= len(values)
    if quotient:
        raise ValueError(f"pattern {item.answer!r} exhausted its unique combinations")
    return selected


def make_entry(index: int) -> dict[str, Any]:
    item_index = index % len(PATTERNS)
    occurrence = index // len(PATTERNS)
    item = PATTERNS[item_index]
    request_index = occurrence % len(item.requests)
    context_index = (occurrence // len(item.requests)) % len(CONTEXTS)
    values = selected_values(item, occurrence)
    question = f"{CONTEXTS[context_index]} {item.requests[request_index].format(**values)}"
    answer = item.answer.format(**values)
    block_id = f"{ENTRY_PREFIX}{index:05d}"
    return {
        "id": block_id,
        "name": f"Intent Action v1 {item.domain} {index:05d}",
        "question": question,
        "intermediates": [],
        "explanation": [],
        "answer": answer,
        "goal": {"target_state": ""},
        "intermediate_count": 0,
        "step_index": [],
        "format_type": "qa",
        "timestamp": 0,
    }


def generate_entries(count: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [make_entry(index) for index in range(count)]


def validate_entries(entries: list[dict[str, Any]], vocab_path: Path) -> dict[str, Any]:
    vocab = load_vocab_bin(vocab_path)
    trie = build_trie(vocab)
    ids: set[str] = set()
    questions: set[str] = set()
    domain_counts: dict[str, int] = {}
    max_tokens = 0
    max_id = ""

    for entry in entries:
        block_id = entry["id"]
        question = entry["question"]
        answer = entry["answer"]
        if block_id in ids:
            raise ValueError(f"duplicate generated id: {block_id}")
        if question in questions:
            raise ValueError(f"duplicate generated question: {question}")
        if not answer.strip():
            raise ValueError(f"{block_id}: answer is empty")
        if any(character.isdigit() for character in answer):
            raise ValueError(f"{block_id}: answer contains a number: {answer!r}")
        if entry.get("goal") != {"target_state": ""}:
            raise ValueError(f"{block_id}: goal must contain an empty target_state")
        if entry.get("execution"):
            raise ValueError(f"{block_id}: intent-action rows must not contain execution")

        canonical = f"{question}\nA: {answer}\n"
        token_count = len(tokenize(canonical, vocab, trie, detect_numbers=True).tokens)
        bounded_count = token_count + BOUNDARY_TOKEN_BUDGET
        if bounded_count > MAX_SEQUENCE_TOKENS:
            raise ValueError(
                f"{block_id}: canonical sequence uses {bounded_count} tokens, "
                f"exceeding {MAX_SEQUENCE_TOKENS}"
            )
        if bounded_count > max_tokens:
            max_tokens = bounded_count
            max_id = block_id

        ids.add(block_id)
        questions.add(question)
        domain = entry["name"].split()[3]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    return {
        "entries": len(entries),
        "domains": domain_counts,
        "maximum_sequence_tokens": max_tokens,
        "maximum_sequence_id": max_id,
    }


def append_missing_entries(path: Path, entries: list[dict[str, Any]], dry_run: bool) -> int:
    desired = {entry["id"]: entry for entry in entries}
    existing_ids: set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank JSONL row")
                row = json.loads(line)
                block_id = row.get("id")
                if not isinstance(block_id, str) or not block_id:
                    raise ValueError(f"{path}:{line_number}: missing string id")
                if block_id in desired:
                    if block_id in existing_ids:
                        raise ValueError(f"{path}:{line_number}: duplicate id {block_id}")
                    if row != desired[block_id]:
                        raise ValueError(f"{path}:{line_number}: existing {block_id} differs from generator")
                    existing_ids.add(block_id)

    missing = [entry for entry in entries if entry["id"] not in existing_ids]
    if dry_run or not missing:
        return len(missing)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, stage_name = tempfile.mkstemp(prefix="intent_action.", suffix=".jsonl", dir=path.parent)
    os.close(fd)
    stage_path = Path(stage_name)
    try:
        with stage_path.open("w", encoding="utf-8", newline="\n") as stage:
            for entry in missing:
                stage.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
            stage.flush()
            os.fsync(stage.fileno())
        with path.open("ab") as destination, stage_path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        stage_path.unlink(missing_ok=True)
    return len(missing)


def update_registry(path: Path, entries: list[dict[str, Any]], dry_run: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(f"curriculum registry not found: {path}")
    registry = json.loads(path.read_text(encoding="utf-8"))
    curricula = registry.get("curriculums")
    if not isinstance(curricula, list):
        raise ValueError("curriculum registry must contain a curriculums array")

    matches = [item for item in curricula if item.get("id") == CURRICULUM_ID]
    if len(matches) > 1:
        raise ValueError(f"curriculum registry contains duplicate id {CURRICULUM_ID}")
    if matches:
        curriculum = matches[0]
    else:
        curriculum = {"id": CURRICULUM_ID, "timestamp": int(time.time())}
        curricula.append(curriculum)
    curriculum.update({
        "name": CURRICULUM_NAME,
        "format_as_concept": True,
        "concept_block_ids": [entry["id"] for entry in entries],
    })

    if dry_run:
        return
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(registry, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--vocab", type=Path, default=DATA_DIR / "vocab.bin")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = generate_entries(args.count)
    report = validate_entries(entries, args.vocab)
    concept_path = args.data_dir / "concept_blocks.jsonl"
    registry_path = args.data_dir / "curriculum_registry.json"
    appended = append_missing_entries(concept_path, entries, args.dry_run)
    update_registry(registry_path, entries, args.dry_run)

    action = "would append" if args.dry_run else "appended"
    print(f"{CURRICULUM_NAME}: {report['entries']} validated blocks")
    print(f"  {action}: {appended}")
    print(
        f"  maximum sequence: {report['maximum_sequence_tokens']} tokens "
        f"including BOS/EOS ({report['maximum_sequence_id']})"
    )
    print(f"  domains: {report['domains']}")
    print(f"  concept blocks: {concept_path}")
    print(f"  registry: {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())