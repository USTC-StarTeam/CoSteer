"""Command-line interface for the eight CoSteer evaluation datasets."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm

from .generation import generate_with_costeer, load_model_bundle
from .optimizer import CoSteerConfig, CoSteerOptimizer
from .tasks import (
    DATASET_SPECS,
    PREFERENCE_VALUES,
    default_dataset_path,
    iter_jsonl,
    prepare_example,
)


def build_parser(default_task: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run per-token CoSteer generation on a paper evaluation dataset."
    )
    parser.add_argument("--task", choices=sorted(DATASET_SPECS), default=default_task)
    parser.add_argument("--input-file", "--input_file", type=Path)
    parser.add_argument("--output-file", "--output_file", type=Path)
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--llm-model",
        "--llm_model_name",
        dest="llm_model",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--slm-model",
        "--slm_model_name",
        dest="slm_model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--preference", choices=PREFERENCE_VALUES)
    parser.add_argument("--iterations", "--T", dest="iterations", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--player-lambda", "--player_lambda", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=10.0)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=1024)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", "--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--llm-device-map", default="auto")
    parser.add_argument("--slm-device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _default_output(args: argparse.Namespace) -> Path:
    suffix = f"_{args.preference}" if args.preference else ""
    pair = f"{_safe_name(args.slm_model)}_to_{_safe_name(args.llm_model)}"
    return args.output_dir / f"{args.task}_{pair}{suffix}.jsonl"


def _processed_ids(path: Path) -> set[str]:
    processed: set[str] = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in item:
                processed.add(str(item["id"]))
    return processed


def main(default_task: str | None = None) -> None:
    args = build_parser(default_task).parse_args()
    if args.task is None:
        raise SystemExit("--task is required")
    spec = DATASET_SPECS[args.task]
    if spec.requires_preference and args.preference is None:
        raise SystemExit(f"--preference is required for {args.task}")
    if not spec.requires_preference and args.preference is not None:
        raise SystemExit(f"--preference does not apply to {args.task}")

    input_path = args.input_file or default_dataset_path(args.task)
    output_path = args.output_file or _default_output(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    models = load_model_bundle(
        args.llm_model,
        args.slm_model,
        llm_device_map=args.llm_device_map,
        slm_device_map=args.slm_device_map,
        trust_remote_code=args.trust_remote_code,
    )
    optimizer = CoSteerOptimizer(
        CoSteerConfig(
            iterations=args.iterations,
            alpha=args.alpha,
            beta=args.beta,
            player_lambda=args.player_lambda,
            eta=args.eta,
        )
    )

    rows = list(iter_jsonl(input_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    processed = _processed_ids(output_path)

    with output_path.open("a", encoding="utf-8") as output:
        for row_index, item in enumerate(tqdm(rows, desc=args.task)):
            example = prepare_example(
                args.task, item, row_index, preference=args.preference
            )
            if str(example.example_id) in processed:
                continue
            response = generate_with_costeer(
                example.input_text,
                example.personalized_input_text,
                models,
                optimizer,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            record = {
                "id": example.example_id,
                "task": args.task,
                "input": example.input_text,
                "response": response,
            }
            if args.preference:
                record["preference"] = args.preference
            json.dump(record, output, ensure_ascii=False)
            output.write("\n")
            output.flush()


if __name__ == "__main__":
    main()
