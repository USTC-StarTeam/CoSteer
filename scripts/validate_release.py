#!/usr/bin/env python3
"""Lightweight release checks for the public CoSteer repository.

The script deliberately avoids loading model weights. It checks Python syntax,
the JSONL dataset schema, and obvious sensitive-file patterns before sharing the
repository with reviewers.
"""

from __future__ import annotations

import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "codes"
DATA_DIR = ROOT / "datasets"

SECRET_PATTERNS = [
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_]{24,}\b"),
    re.compile(r"\bsk-ant-api\d+-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(openai|anthropic|hf|huggingface).*api[_-]?key"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{20,}"),
]

FORBIDDEN_TRACKED_PARTS = {
    "outputs",
    "results",
    "rebuttal",
    "logs",
    "models",
    "checkpoints",
    "wandb",
}


def check_python_syntax() -> None:
    for path in sorted(CODE_DIR.glob("*.py")):
        py_compile.compile(str(path), doraise=True)
    print(f"OK: compiled {len(list(CODE_DIR.glob('*.py')))} Python files")


def check_datasets() -> None:
    required = {"input", "output"}
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                rows += 1
                item = json.loads(line)
                missing = required - item.keys()
                if missing:
                    raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
                if "top_5" in item and not isinstance(item["top_5"], list):
                    raise ValueError(f"{path}:{line_no} top_5 must be a list")
        print(f"OK: {path.relative_to(ROOT)} has {rows} rows")


def tracked_files() -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return [p.relative_to(ROOT) for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    return [Path(line) for line in output.splitlines() if line]


def check_no_forbidden_tracked_files(paths: list[Path]) -> None:
    offenders = []
    for path in paths:
        if any(part in FORBIDDEN_TRACKED_PARTS for part in path.parts):
            offenders.append(str(path))
    if offenders:
        raise RuntimeError("Forbidden generated/private paths are tracked:\n" + "\n".join(offenders))
    print("OK: no generated-output or model directories are tracked")


def check_no_obvious_secrets(paths: list[Path]) -> None:
    text_suffixes = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".jsonl", ""}
    for rel_path in paths:
        path = ROOT / rel_path
        if path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise RuntimeError(f"Potential secret pattern found in {rel_path}")
    print("OK: no obvious API key or token patterns found in tracked text files")


def main() -> int:
    check_python_syntax()
    check_datasets()
    paths = tracked_files()
    check_no_forbidden_tracked_files(paths)
    check_no_obvious_secrets(paths)
    print("Release validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
