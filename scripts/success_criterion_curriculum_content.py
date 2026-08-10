#!/usr/bin/env python3
"""Generic success contracts for the Intent Action curriculum families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuccessContract:
    completion_clause: str
    observation: str
    state_based: bool = False


CONTRACTS: dict[str, SuccessContract] = {
    "factual_qa": SuccessContract(
        "the requested fact is provided", "a direct factual answer"),
    "explanation": SuccessContract(
        "the requested concept is explained", "an explanation addressing the requested concept"),
    "comparison": SuccessContract(
        "the requested subjects are compared", "a comparison addressing the requested subjects"),
    "extraction": SuccessContract(
        "the requested information is extracted", "the requested extracted information"),
    "classification": SuccessContract(
        "the supplied content is classified", "a classification of the supplied content"),
    "rewriting": SuccessContract(
        "the requested rewrite is produced", "a rewritten version of the supplied text"),
    "summarization": SuccessContract(
        "the requested summary is produced", "a summary of the supplied material"),
    "translation": SuccessContract(
        "the requested translation is produced", "a translation of the supplied wording"),
    "logic": SuccessContract(
        "the requested inference is resolved", "a conclusion addressing the logic question"),
    "arithmetic": SuccessContract(
        "the requested value is calculated", "a result for the supplied expression"),
    "word_problem": SuccessContract(
        "the requested quantity is determined", "a result addressing the quantitative problem"),
    "equation": SuccessContract(
        "the requested unknown is determined", "a solution to the supplied equation"),
    "data_interpretation": SuccessContract(
        "the supplied data is interpreted", "an interpretation of the supplied data"),
    "planning": SuccessContract(
        "the requested plan is produced", "a plan addressing the requested activity"),
    "task_decomposition": SuccessContract(
        "the requested task is decomposed", "a decomposition of the requested task"),
    "troubleshooting": SuccessContract(
        "the reported issue is addressed", "troubleshooting guidance for the reported issue"),
    "application_action": SuccessContract(
        "the requested application action is completed",
        "completion of the requested application action",
        state_based=True),
    "file_action": SuccessContract(
        "the requested file action is completed",
        "completion of the requested file action",
        state_based=True),
    "multi_action": SuccessContract(
        "the requested combined action is completed",
        "completion of the requested combined action",
        state_based=True),
    "clarification": SuccessContract(
        "the missing information is requested", "a request for the information needed to proceed"),
    "judgment": SuccessContract(
        "the reported situation receives an appropriate recommendation",
        "a recommendation addressing the reported situation"),
    "coding": SuccessContract(
        "the requested coding concept is explained", "an explanation of the requested coding concept"),
    "instruction_following": SuccessContract(
        "the requested transformation is performed", "an output reflecting the requested transformation"),
    "conversation": SuccessContract(
        "the user's stated concern is addressed", "a response addressing the stated concern"),
    "creative": SuccessContract(
        "the requested creative content is produced", "creative content addressing the request"),
    "procedure": SuccessContract(
        "the requested procedure is described", "a procedure addressing the requested task"),
}


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] + "."


def criterion_for(family: str, variant: int) -> str:
    contract = CONTRACTS[family]
    forms = (
        _sentence(contract.completion_clause),
        f"Successful completion means {contract.completion_clause}.",
        f"The outcome satisfies the identified intent when {contract.completion_clause}.",
    )
    return forms[variant % len(forms)]


def evidence_for(family: str, variant: int) -> str:
    contract = CONTRACTS[family]
    if contract.state_based:
        forms = (
            f"The resulting state shows {contract.observation}.",
            f"The observable state reflects {contract.observation}.",
            f"The recorded outcome confirms {contract.observation}.",
        )
    else:
        forms = (
            f"The response contains {contract.observation}.",
            f"The produced response presents {contract.observation}.",
            f"Reviewing the response shows {contract.observation}.",
        )
    return forms[variant % len(forms)]
