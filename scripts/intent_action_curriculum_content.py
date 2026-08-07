#!/usr/bin/env python3
"""Deterministic target-state prompts for the Intent Action v1 curriculum."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


NAMES = ("Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Harper", "Jordan", "Morgan", "Riley")
APPS = ("Calendar", "Mail", "Browser", "Notes", "Music", "Camera", "Maps", "Calculator", "Files", "Terminal")
FILES = ("project brief", "meeting notes", "budget draft", "design outline", "travel plan", "research summary", "inventory sheet", "reading list")
FOLDERS = ("Archive", "Documents", "Projects", "Shared", "Downloads", "Work")
ITEMS = ("notebooks", "batteries", "cables", "folders", "samples", "lamps", "cartons", "tickets")
PLACES = ("library", "station", "pharmacy", "museum", "market", "community center")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
TIMES = ("7:00", "8:30", "10:15", "13:00", "16:45", "18:30", "21:00")
REQUEST_FRAMES = (
    "", "Please help with this: ", "For a quick task, ", "Handle this request: ",
    "I need assistance: ", "For the current task, ", "Take care of this: ",
    "Could you help me: ", "My next request is to ", "Please proceed and ",
    "When ready, ", "As the next action, ", "For this request, ",
    "I would like you to ", "Help me complete this: ", "Please address this: ",
    "Work on the following: ", "The task is to ", "Go ahead and ", "I need you to ",
)
PROMPT_CONTEXTS = (
    "", " Keep the scope unchanged.", " Use the details provided.",
    " Do not omit the named item.", " Treat the stated values as authoritative.",
    " Preserve the requested order.", " Focus on the requested outcome.",
    " Use the specified destination.", " Respect the stated condition.",
    " Keep the named subject in scope.", " Follow the request as written.",
    " Use the available information.", " Apply this to the current task.",
    " Keep the response relevant.", " Address every requested action.",
    " Preserve the original meaning.", " Use the stated time and place.",
    " Do not broaden the task.", " Resolve the requested operation.",
    " Keep the result task-focused.",
)


@dataclass(frozen=True)
class GeneratedContent:
    family: str
    prompt: str
    answer: str


def pick(values: tuple[str, ...], occurrence: int, stride: int = 1) -> str:
    return values[(occurrence * stride) % len(values)]


def choose_answer(occurrence: int, answers: tuple[str, ...]) -> str:
    return answers[occurrence % len(answers)]


def factual_qa(occurrence: int, rng: random.Random) -> GeneratedContent:
    subjects = ("the capital of Japan", "the largest ocean", "the author of Pride and Prejudice", "the gas plants absorb", "the purpose of a CPU", "the freezing point of water")
    subject = pick(subjects, occurrence, 5)
    prompt = f"What is {subject}?"
    answer = choose_answer(occurrence, (
        "Provide the fact", "Answer the factual question", "Identify the requested fact",
        "Supply the correct factual answer", "State the requested piece of general knowledge",
        "Give the relevant fact for the question",
    ))
    return GeneratedContent("factual_qa", prompt, answer)


def explanation(occurrence: int, rng: random.Random) -> GeneratedContent:
    concept = pick(("photosynthesis", "caching", "inflation", "version control", "gravity", "encryption", "an ecosystem", "a database index"), occurrence, 3)
    prompt = f"Explain {concept} in plain language."
    answer = choose_answer(occurrence, (
        "Explain the concept", "Provide a plain-language explanation", "Describe the requested concept clearly",
        "Give an accessible explanation of the concept", "Explain the concept and its practical significance",
        "Provide a clear beginner-friendly description of the requested concept",
    ))
    return GeneratedContent("explanation", prompt, answer)


def comparison(occurrence: int, rng: random.Random) -> GeneratedContent:
    subject = pick(("RAM and storage", "weather and climate", "a list and a set", "authentication and authorization", "lossless and lossy compression", "a process and a thread"), occurrence, 5)
    prompt = f"Compare {subject}."
    answer = choose_answer(occurrence, (
        "Compare the options", "Explain the main differences", "Distinguish the requested concepts",
        "Provide a focused comparison of the two concepts", "Compare the subjects using their relevant characteristics",
        "Describe how the requested concepts differ and relate",
    ))
    return GeneratedContent("comparison", prompt, answer)


def extraction(occurrence: int, rng: random.Random) -> GeneratedContent:
    name = pick(NAMES, occurrence, 3)
    day = pick(DAYS, occurrence, 5)
    time = pick(TIMES, occurrence, 2)
    place = pick(PLACES, occurrence, 5)
    prompt = f"The appointment with {name} is at {time} on {day} in the {place}. Extract the person and time."
    answer = choose_answer(occurrence, (
        "Extract the details", "Return the requested fields", "Identify the person and time",
        "Extract the named person and appointment time", "Provide the requested structured appointment details",
        "Return the relevant person and time from the record",
    ))
    return GeneratedContent("extraction", prompt, answer)


def classification(occurrence: int, rng: random.Random) -> GeneratedContent:
    text = pick(("The battery lasts all day and charges quickly.", "The package arrived broken and late.", "The meeting starts at noon.", "I love how simple this interface is.", "The update deleted my settings.", "The report contains several sections."), occurrence, 5)
    prompt = f"Classify the sentiment as positive, negative, or neutral: {text}"
    answer = choose_answer(occurrence, (
        "Classify sentiment", "Assign the sentiment label", "Determine the text sentiment",
        "Classify the passage by emotional polarity", "Identify the appropriate sentiment category for the text",
        "Return the requested positive negative or neutral classification",
    ))
    return GeneratedContent("classification", prompt, answer)


def rewriting(occurrence: int, rng: random.Random) -> GeneratedContent:
    text = pick(("send me the report now", "we cannot ship because the test failed", "the meeting moved to friday", "your instructions are unclear", "need budget details soon"), occurrence, 3)
    tone = pick(("clear", "friendly", "formal", "concise"), occurrence, 3)
    prompt = f"Rewrite this in a {tone} tone: {text}"
    answer = choose_answer(occurrence, (
        "Rewrite the text", "Improve the wording", "Produce the requested rewrite",
        "Rewrite the message in the specified tone", "Preserve the meaning while improving the wording",
        "Create a polished rewrite that follows the requested tone",
    ))
    return GeneratedContent("rewriting", prompt, answer)


def summarization(occurrence: int, rng: random.Random) -> GeneratedContent:
    passage = pick(("The release passed tests, but deployment was postponed because staging was unavailable.", "The library closes early Friday for maintenance and reopens Saturday.", "Sales rose, stayed flat, and then declined slightly.", "The smaller design costs less to operate and is easier to repair."), occurrence, 3)
    prompt = f"Summarize in one sentence: {passage}"
    answer = choose_answer(occurrence, (
        "Summarize the passage", "Provide a concise summary", "State the central point",
        "Condense the passage into its key message", "Create a focused one-sentence summary of the passage",
        "Capture the main information in a concise self-contained summary",
    ))
    return GeneratedContent("summarization", prompt, answer)


def translation(occurrence: int, rng: random.Random) -> GeneratedContent:
    phrase, language = pick((("Good morning", "Spanish"), ("Thank you", "French"), ("Where is the station", "Spanish"), ("See you tomorrow", "German"), ("Please", "Italian"), ("Good night", "Portuguese")), occurrence, 5)
    prompt = f"Translate '{phrase}' into {language}."
    answer = choose_answer(occurrence, (
        "Translate the phrase", "Provide the translation", "Translate into the requested language",
        "Return the phrase in the specified language", "Produce an accurate translation of the supplied phrase",
        "Translate the provided wording while preserving its intended meaning",
    ))
    return GeneratedContent("translation", prompt, answer)


def logic(occurrence: int, rng: random.Random) -> GeneratedContent:
    name = pick(NAMES, occurrence, 3)
    prompt = pick((f"Either {name} or Morgan has the key. Morgan does not have it. Who has the key?", "All archived records are read-only. This record is archived. Is it read-only?", "No approved request is pending. This request is pending. Is it approved?", "If the alarm is armed, the indicator is red. The indicator is not red. Can the alarm be armed?"), occurrence, 3)
    answer = choose_answer(occurrence, (
        "Resolve the inference", "Answer the logic question", "Determine the logical conclusion",
        "Infer the conclusion from the stated premises", "Provide the conclusion supported by the logical conditions",
        "Resolve the question using only the supplied logical relationships",
    ))
    return GeneratedContent("logic", prompt, answer)


def arithmetic(occurrence: int, rng: random.Random) -> GeneratedContent:
    lhs = rng.randint(12, 900)
    rhs = rng.randint(2, 80)
    symbol = pick(("+", "-", "*", "/"), occurrence)
    prompt = f"Calculate {lhs} {symbol} {rhs}."
    answer = choose_answer(occurrence, (
        "Calculate the value", "Evaluate the expression", "Determine the arithmetic result",
        "Compute the value of the supplied expression", "Return the result of the requested arithmetic operation",
        "Evaluate the numeric expression and provide its resulting value",
    ))
    return GeneratedContent("arithmetic", prompt, answer)


def word_problem(occurrence: int, rng: random.Random) -> GeneratedContent:
    item = pick(ITEMS, occurrence, 3)
    start = rng.randint(30, 300)
    added = rng.randint(5, 100)
    removed = rng.randint(2, start + added - 1)
    prompt = f"A store had {start} {item}, received {added} more, then sold {removed}. How many remain?"
    answer = choose_answer(occurrence, (
        "Determine the remainder", "Find the remaining quantity", "Calculate the final inventory",
        "Determine how many items remain after the changes", "Compute the ending quantity from the inventory changes",
        "Find the final remaining amount after applying both inventory updates",
    ))
    return GeneratedContent("word_problem", prompt, answer)


def equation(occurrence: int, rng: random.Random) -> GeneratedContent:
    coefficient = rng.randint(2, 12)
    offset = rng.randint(1, 40)
    total = coefficient * rng.randint(2, 90) + offset
    prompt = f"Solve for x: {coefficient}x + {offset} = {total}."
    answer = choose_answer(occurrence, (
        "Solve the equation", "Find the unknown", "Determine the variable value",
        "Solve for the unknown variable", "Find the value that satisfies the equation",
        "Determine the unknown value from the supplied linear equation",
    ))
    return GeneratedContent("equation", prompt, answer)


def data_interpretation(occurrence: int, rng: random.Random) -> GeneratedContent:
    first, second, third = (rng.randint(20, 90) for _ in range(3))
    prompt = f"Monthly tickets were {first}, {second}, and {third}. Identify the highest month and describe the trend."
    answer = choose_answer(occurrence, (
        "Interpret the data", "Identify the leading value", "Analyze the reported trend",
        "Determine the highest period and summarize the trend", "Interpret the values and identify the strongest period",
        "Analyze the supplied data to report the peak and overall pattern",
    ))
    return GeneratedContent("data_interpretation", prompt, answer)


def planning(occurrence: int, rng: random.Random) -> GeneratedContent:
    task = pick(("a small team meeting", "a weekend trip", "a software release", "a study session", "a shared-folder cleanup", "a community event"), occurrence, 5)
    prompt = f"Create a practical plan for {task}."
    answer = choose_answer(occurrence, (
        "Create the plan", "Plan the task", "Develop a practical plan",
        "Create an organized plan for the requested task", "Produce a useful plan covering the task's major stages",
        "Develop a clear actionable plan for completing the requested activity",
    ))
    return GeneratedContent("planning", prompt, answer)


def task_decomposition(occurrence: int, rng: random.Random) -> GeneratedContent:
    task = pick(("moving a website to a new domain", "preparing a quarterly report", "onboarding a teammate", "cleaning up a repository", "organizing a workshop"), occurrence, 3)
    prompt = f"Break down the task of {task}."
    answer = choose_answer(occurrence, (
        "Decompose the task", "List the major steps", "Break the work into stages",
        "Create an ordered decomposition of the requested task", "Divide the task into clear manageable units of work",
        "Produce a structured sequence of actionable steps for the task",
    ))
    return GeneratedContent("task_decomposition", prompt, answer)


def troubleshooting(occurrence: int, rng: random.Random) -> GeneratedContent:
    issue = pick(("an application cannot connect to its server", "a script reports a missing module", "a laptop charges slowly", "a database query became slow", "a renamed file appears missing"), occurrence, 3)
    prompt = f"Troubleshoot this issue: {issue}."
    answer = choose_answer(occurrence, (
        "Diagnose the issue", "Provide troubleshooting guidance", "Identify the likely checks",
        "Create a focused diagnostic sequence for the problem", "Outline evidence-based steps to isolate the reported issue",
        "Provide a systematic troubleshooting approach for finding the root cause",
    ))
    return GeneratedContent("troubleshooting", prompt, answer)


def application_action(occurrence: int, rng: random.Random) -> GeneratedContent:
    app = pick(APPS, occurrence, 3)
    action = pick(("open", "close", "switch to", "restart"), occurrence)
    prompt = f"{action.capitalize()} {app}."
    answers = {
        "open": (f"Open {app}", f"Launch the {app} application", f"Make {app} the active application"),
        "close": (f"Close {app}", f"Exit the {app} application", f"End the active {app} application session"),
        "switch to": (f"Activate {app}", f"Switch focus to {app}", f"Bring the {app} application to the foreground"),
        "restart": (f"Restart {app}", f"Relaunch the {app} application", f"Close and reopen the {app} application"),
    }
    return GeneratedContent("application_action", prompt, answers[action][(occurrence // 4) % 3])


def file_action(occurrence: int, rng: random.Random) -> GeneratedContent:
    file_name = pick(FILES, occurrence, 3)
    folder = pick(FOLDERS, occurrence, 5)
    action = occurrence % 4
    prompts = (f"Move the {file_name} into {folder}.", f"Copy the {file_name} into {folder}.", f"Find and open the {file_name}.", f"Create {folder} and put the {file_name} there.")
    answers = (f"Move the {file_name}", f"Copy the {file_name} to {folder}", f"Locate and open the {file_name}", f"Create {folder} and relocate the {file_name} into it")
    return GeneratedContent("file_action", prompts[action], answers[action])


def multi_action(occurrence: int, rng: random.Random) -> GeneratedContent:
    app = pick(APPS, occurrence, 3)
    file_name = pick(FILES, occurrence, 5)
    name = pick(NAMES, occurrence, 7)
    variant = occurrence % 3
    prompts = (f"Open {app}, locate the {file_name}, and share it with {name}.", f"Close {app}, open Calendar, and schedule a review with {name}.", f"Find the {file_name}, copy it to {pick(FOLDERS, occurrence, 5)}, then open the copy.")
    answers = (f"Share the {file_name} with {name} through {app}", f"Close {app} and schedule a calendar review with {name}", f"Copy the {file_name} to the requested folder and open it")
    return GeneratedContent("multi_action", prompts[variant], answers[variant])


def clarification(occurrence: int, rng: random.Random) -> GeneratedContent:
    prompt, subject = pick((("Delete the report.", "report identity"), ("Schedule it for Friday.", "event and time"), ("Send that to Morgan.", "item to send"), ("Make it better.", "improvement target"), ("Open the file.", "file identity"), ("Order some supplies.", "supply details")), occurrence, 5)
    answer = choose_answer(occurrence, (
        "Request clarification", "Ask for the missing details", f"Clarify the {subject}",
        f"Request the missing {subject} before acting", f"Obtain enough information to resolve the {subject}",
        f"Ask the user to specify the missing {subject} required for the task",
    ))
    return GeneratedContent("clarification", prompt, answer)


def judgment(occurrence: int, rng: random.Random) -> GeneratedContent:
    prompt = pick(("A message asks for my password to verify an account. What should I do?", "I am about to overwrite the only copy of a document. What is safer?", "An unknown update asks for administrator access. Proceed?", "A recursive delete command has an ambiguous path. What should happen?", "A form requests unrelated sensitive data. Should I submit it?"), occurrence, 3)
    answer = choose_answer(occurrence, (
        "Recommend a safe action", "Assess the risk", "Choose the safer course",
        "Provide a safety-focused recommendation", "Determine the prudent response to the reported risk",
        "Recommend an appropriate action that avoids the identified safety risk",
    ))
    return GeneratedContent("judgment", prompt, answer)


def coding(occurrence: int, rng: random.Random) -> GeneratedContent:
    topic = pick(("a return statement", "a parameterized SQL query", "an off-by-one error", "a dictionary", "a unit test", "an informative error message"), occurrence, 5)
    prompt = f"Explain the purpose of {topic}."
    answer = choose_answer(occurrence, (
        "Explain the coding concept", "Describe its purpose", "Provide the technical explanation",
        "Explain the requested programming concept clearly", "Describe the concept and its practical role in software",
        "Provide a concise technical explanation of the requested programming concept",
    ))
    return GeneratedContent("coding", prompt, answer)


def instruction_following(occurrence: int, rng: random.Random) -> GeneratedContent:
    prompt = pick(("Return these words in alphabetical order: pear, apple, plum, banana.", "Convert this title to lowercase: A Practical Guide To Testing.", "Reverse these items: red, green, blue.", "Return the initials of North Atlantic Treaty Organization.", "Remove duplicate words while preserving order: one two one three two.", "Put plan, build, and verify on separate numbered lines."), occurrence, 5)
    answer = choose_answer(occurrence, (
        "Apply the instruction", "Transform the supplied text", "Return the requested format",
        "Perform the specified text transformation", "Produce the output in the explicitly requested form",
        "Follow the formatting instruction while preserving the supplied content",
    ))
    return GeneratedContent("instruction_following", prompt, answer)


def conversation(occurrence: int, rng: random.Random) -> GeneratedContent:
    prompt = pick(("I'm overwhelmed by my task list.", "I finished the draft but do not know if it is ready.", "Can you help me choose between two options?", "I made a mistake in the spreadsheet.", "What should I work on next?"), occurrence, 3)
    answer = choose_answer(occurrence, (
        "Offer useful guidance", "Help with the decision", "Respond with relevant support",
        "Provide focused assistance for the stated concern", "Guide the user toward an appropriate next step",
        "Offer a constructive response tailored to the user's immediate concern",
    ))
    return GeneratedContent("conversation", prompt, answer)


def creative(occurrence: int, rng: random.Random) -> GeneratedContent:
    subject = pick(("a community workshop", "a reading club", "a garden project", "a study group", "a repair cafe"), occurrence, 3)
    prompt = f"Suggest a concise name and invitation for {subject}."
    answer = choose_answer(occurrence, (
        "Create the wording", "Generate a name and invitation", "Provide the requested creative copy",
        "Create concise wording suited to the stated subject", "Generate an appropriate title and inviting description",
        "Produce original concise copy that fits the requested community activity",
    ))
    return GeneratedContent("creative", prompt, answer)


def procedure(occurrence: int, rng: random.Random) -> GeneratedContent:
    task = pick(("verify a downloaded file", "change an important configuration", "take reliable meeting notes", "evaluate an online source", "investigate an unfamiliar error"), occurrence, 3)
    prompt = f"Describe a reliable procedure to {task}."
    answer = choose_answer(occurrence, (
        "Describe the procedure", "Provide the process", "Outline a reliable method",
        "Give an ordered procedure for the requested task", "Describe a dependable process for completing the task",
        "Provide a clear sequence of actions for performing the task reliably",
    ))
    return GeneratedContent("procedure", prompt, answer)


BUILDERS: tuple[Callable[[int, random.Random], GeneratedContent], ...] = (
    factual_qa, explanation, comparison, extraction, classification, rewriting,
    summarization, translation, logic, arithmetic, word_problem, equation,
    data_interpretation, planning, task_decomposition, troubleshooting,
    application_action, file_action, multi_action, clarification, judgment,
    coding, instruction_following, conversation, creative, procedure,
)


def generate_content(index: int, seed: int) -> GeneratedContent:
    if index < 0:
        raise ValueError("index must be non-negative")
    builder_index = index % len(BUILDERS)
    occurrence = index // len(BUILDERS)
    rng = random.Random(seed + index * 1_000_003)
    content = BUILDERS[builder_index](occurrence, rng)
    frame = REQUEST_FRAMES[occurrence % len(REQUEST_FRAMES)]
    context = PROMPT_CONTEXTS[(occurrence // len(REQUEST_FRAMES)) % len(PROMPT_CONTEXTS)]
    return GeneratedContent(content.family, frame + content.prompt + context, content.answer)
