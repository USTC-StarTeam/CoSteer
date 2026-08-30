"""Dataset adapters for the eight evaluation datasets in the paper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


PREFERENCE_VALUES = ("concise", "creative", "uplifting", "verbose")


@dataclass(frozen=True)
class DatasetSpec:
    file_name: str
    category: str
    requires_preference: bool = False


DATASET_SPECS: dict[str, DatasetSpec] = {
    "cogenesis": DatasetSpec("cogenesis.jsonl", "personalized generation"),
    "longlamp_abstract": DatasetSpec(
        "longlamp_abstract.jsonl", "personalized generation"
    ),
    "longlamp_review": DatasetSpec(
        "longlamp_review.jsonl", "personalized generation"
    ),
    "longlamp_writing": DatasetSpec(
        "longlamp_writing.jsonl", "personalized generation"
    ),
    "helpsteer": DatasetSpec(
        "helpsteer.jsonl", "preference alignment", requires_preference=True
    ),
    "personal_preference_eval": DatasetSpec(
        "personal_preference_eval.jsonl",
        "preference alignment",
        requires_preference=True,
    ),
    "truthfulqa": DatasetSpec(
        "truthfulqa.jsonl", "preference alignment", requires_preference=True
    ),
    "ultrachat": DatasetSpec(
        "ultrachat.jsonl", "preference alignment", requires_preference=True
    ),
}


@dataclass(frozen=True)
class PreparedExample:
    example_id: str | int
    input_text: str
    personalized_input_text: str


def default_dataset_path(task: str) -> Path:
    return Path(__file__).resolve().parents[1] / "datasets" / DATASET_SPECS[task].file_name


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield item


def _longlamp_prompt(task: str, query: str, history: list[dict[str, Any]]) -> str:
    if task == "longlamp_abstract":
        heading = "The following are five titles with their abstracts."
        pairs = (("Title", "title"), ("Abstract", "abstract"))
    elif task == "longlamp_review":
        heading = "The following are product descriptions with their user reviews."
        pairs = (("Product", "description"), ("Review", "reviewText"))
    elif task == "longlamp_writing":
        heading = "The following are reddit post summaries with their content."
        pairs = (("Summary", "summary"), ("Content", "content"))
    else:
        raise ValueError(f"unsupported LongLaMP task: {task}")

    sections = [heading]
    for index, item in enumerate(history[:5], start=1):
        first_label, first_key = pairs[0]
        second_label, second_key = pairs[1]
        try:
            first_value = item[first_key]
            second_value = item[second_key]
        except KeyError as error:
            raise ValueError(f"missing LongLaMP history field: {error.args[0]}") from error
        sections.append(
            f"{first_label}[{index}]: {first_value}\n"
            f"{second_label}[{index}]: {second_value}\n"
        )
    sections.extend(("Now it's your turn\n", query))
    return "\n".join(sections)


def prepare_example(
    task: str,
    item: dict[str, Any],
    row_index: int,
    preference: str | None = None,
) -> PreparedExample:
    if task not in DATASET_SPECS:
        raise ValueError(f"unknown task: {task}")

    example_id = item.get("id", item.get("tid", item.get("index", row_index)))
    input_text = item.get("input", item.get("question"))
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError(f"{task} row {row_index} has no non-empty input")

    if task.startswith("longlamp_"):
        history = item.get("top_5")
        if not isinstance(history, list):
            raise ValueError(f"{task} row {row_index} requires a top_5 list")
        personalized_input = _longlamp_prompt(task, input_text, history)
    elif task == "cogenesis":
        personalized_input = item.get("personalized_input")
        if not isinstance(personalized_input, str) or not personalized_input.strip():
            raise ValueError(
                f"cogenesis row {row_index} requires personalized_input"
            )
    else:
        if preference not in PREFERENCE_VALUES:
            allowed = ", ".join(PREFERENCE_VALUES)
            raise ValueError(f"{task} requires --preference in: {allowed}")
        personalized_input = (
            input_text
            + "\nYour answer should be as "
            + preference
            + " as possible"
        )

    return PreparedExample(example_id, input_text, personalized_input)
