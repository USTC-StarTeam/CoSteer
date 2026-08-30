#!/usr/bin/env python3
"""Validate the public CoSteer source and eight released dataset inputs."""

from __future__ import annotations

import hashlib
import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"

DATASET_RULES = {
    "cogenesis.jsonl": (240, {"id", "input", "personalized_input", "output"}),
    "longlamp_abstract.jsonl": (200, {"id", "input", "output", "top_5"}),
    "longlamp_review.jsonl": (200, {"id", "input", "output", "top_5"}),
    "longlamp_writing.jsonl": (200, {"id", "input", "output", "top_5"}),
    "helpsteer.jsonl": (200, {"id", "input"}),
    "personal_preference_eval.jsonl": (200, {"id", "input"}),
    "truthfulqa.jsonl": (200, {"id", "input"}),
    "ultrachat.jsonl": (200, {"id", "input"}),
}

SECRET_PATTERNS = [
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"\bsk-ant-api\d+-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
]

HARD_PATH_PATTERNS = [
    re.compile(r"/(?:home|Users|zhdd|mnt|scratch)/[A-Za-z0-9_.~/-]+"),
    re.compile(r"[A-Za-z]:\\Users\\[^\s\"']+"),
]

FORBIDDEN_TRACKED_PARTS = {
    "outputs",
    "results",
    "rebuttal",
    "logs",
    "models",
    "checkpoints",
    "wandb",
    "runs",
}

FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    )
    return [Path(line) for line in output.splitlines() if line]


def check_python_syntax() -> None:
    paths = sorted(
        path
        for path in ROOT.rglob("*.py")
        if ".git" not in path.parts and ".venv" not in path.parts
    )
    for path in paths:
        py_compile.compile(str(path), doraise=True)
    print(f"OK: compiled {len(paths)} Python files")


def check_datasets() -> None:
    actual = {path.name for path in DATA_DIR.glob("*.jsonl")}
    expected = set(DATASET_RULES)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"dataset file mismatch; missing={missing}, extra={extra}")

    for file_name, (expected_rows, required) in DATASET_RULES.items():
        path = DATA_DIR / file_name
        rows = 0
        ids: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                rows += 1
                missing = required - item.keys()
                if missing:
                    raise ValueError(
                        f"{path}:{line_number} missing fields: {sorted(missing)}"
                    )
                if not isinstance(item["input"], str) or not item["input"].strip():
                    raise ValueError(f"{path}:{line_number} has an empty input")
                if "response" in item:
                    raise ValueError(
                        f"{path}:{line_number} contains a generated response field"
                    )
                if "top_5" in item and not isinstance(item["top_5"], list):
                    raise ValueError(f"{path}:{line_number} top_5 must be a list")
                item_id = str(item["id"])
                if item_id in ids:
                    raise ValueError(f"{path}:{line_number} duplicates id={item_id}")
                ids.add(item_id)
        if rows != expected_rows:
            raise ValueError(f"{path} has {rows} rows; expected {expected_rows}")
        print(f"OK: datasets/{file_name} has {rows} unique rows")


def check_dataset_checksums() -> None:
    checksum_path = DATA_DIR / "SHA256SUMS"
    expected = {}
    with checksum_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            digest, file_name = line.strip().split(maxsplit=1)
            expected[file_name] = digest
    if set(expected) != set(DATASET_RULES):
        raise RuntimeError("SHA256SUMS does not list exactly the eight datasets")
    for file_name, digest in expected.items():
        actual = hashlib.sha256((DATA_DIR / file_name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"checksum mismatch for datasets/{file_name}")
    print("OK: all eight dataset checksums match")


def check_no_forbidden_artifacts(paths: list[Path]) -> None:
    offenders = []
    for path in paths:
        if any(part in FORBIDDEN_TRACKED_PARTS for part in path.parts):
            offenders.append(str(path))
        elif path.suffix.lower() in FORBIDDEN_SUFFIXES:
            offenders.append(str(path))
    if offenders:
        raise RuntimeError(
            "Generated outputs, model artifacts, or private paths are tracked:\n"
            + "\n".join(offenders)
        )
    print("OK: no generated-output, checkpoint, or model artifacts are tracked")


def check_text_files(paths: list[Path]) -> None:
    text_suffixes = {
        ".py",
        ".sh",
        ".md",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".jsonl",
        "",
    }
    for relative_path in paths:
        path = ROOT / relative_path
        if path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise RuntimeError(f"potential credential found in {relative_path}")
        if path.suffix.lower() != ".jsonl":
            for pattern in HARD_PATH_PATTERNS:
                match = pattern.search(text)
                if match:
                    raise RuntimeError(
                        f"machine-specific path found in {relative_path}: {match.group(0)}"
                    )
    print("OK: no credentials or machine-specific absolute paths found")


def main() -> int:
    check_python_syntax()
    check_datasets()
    check_dataset_checksums()
    paths = tracked_files()
    check_no_forbidden_artifacts(paths)
    check_text_files(paths)
    print("Release validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
