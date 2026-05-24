# CoSteer: Collaborative Decoding-Time Personalization via Local Delta Steering

This repository contains the reviewer-facing implementation for CoSteer, a
decoding-time framework where a remote large language model (LLM) is guided by a
smaller local language model (SLM) that can access private user context.

The release is intentionally lightweight. It includes the core generation code
and small JSONL evaluation splits, but does not include generated model outputs,
private API credentials, local model checkpoints, or cached experiment artifacts.

## Repository Layout

```text
.
├── codes/
│   ├── abstract_costeer.py       # CoSteer for same-tokenizer model pairs
│   ├── abstract_costeer_map.py   # Vocabulary-intersection alignment
│   ├── abstract_costeer_byte.py  # Byte-level alignment for tokenizer mismatch
│   └── abstract_adacosteer.py    # AdaCoSteer confidence-gated variant
├── datasets/
│   ├── abstract.jsonl
│   ├── review.jsonl
│   └── writing.jsonl
├── scripts/
│   └── validate_release.py       # Lightweight repository sanity check
└── requirements.txt
```

## Installation

Create a fresh Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts load Hugging Face models through `transformers`. If a model requires
authentication, please authenticate through the standard Hugging Face workflow
outside this repository. Do not place tokens, API keys, or private model paths in
tracked files.

## Data Format

Each JSONL row is one evaluation instance. The core fields are:

- `input`: the task prompt without retrieved personalization context.
- `output`: the reference response from the dataset.
- `top_5`: retrieved in-context examples used as local private context.

The three files under `datasets/` are the small reviewer-facing splits used by
the example scripts. They are safe to keep in git. Generated generations and
evaluation outputs should be written under `outputs/`, which is ignored.

## Running CoSteer

For a quick smoke run on one item:

```bash
python codes/abstract_costeer.py \
  --input_file datasets/abstract.jsonl \
  --output_dir outputs/costeer \
  --llm_model_name Qwen/Qwen2-7B-Instruct \
  --slm_model_name Qwen/Qwen2-1.5B-Instruct \
  --max_new_tokens 128 \
  --limit 1
```

`abstract_costeer.py` assumes the LLM and SLM share the same tokenizer and
vocabulary. For cross-tokenizer experiments, use one of the alignment variants:

```bash
python codes/abstract_costeer_map.py \
  --input_file datasets/abstract.jsonl \
  --output_dir outputs/costeer_map \
  --llm_model_name Qwen/Qwen2.5-7B-Instruct \
  --slm_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --max_new_tokens 128 \
  --limit 1

python codes/abstract_costeer_byte.py \
  --input_file datasets/abstract.jsonl \
  --output_dir outputs/costeer_byte \
  --llm_model_name Qwen/Qwen2.5-7B-Instruct \
  --slm_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --max_new_tokens 128 \
  --limit 1
```

To run AdaCoSteer:

```bash
python codes/abstract_adacosteer.py \
  --input_file datasets/abstract.jsonl \
  --output_dir outputs/adacosteer \
  --llm_model_name Qwen/Qwen2.5-7B-Instruct \
  --slm_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --greedy \
  --max_new_tokens 128 \
  --limit 1
```

The default hyperparameters follow the settings used in the paper experiments:
`T=20`, `alpha=2`, `beta=1`, `player_lambda=2`, and `eta=10`.

## Safety and Reproducibility Notes

- Generated outputs, evaluation results, logs, local checkpoints, caches, and API
  credentials are ignored by `.gitignore`.
- Use environment variables or external credential stores for private tokens.
- Keep private full-scale generations outside the repository, or under ignored
  directories such as `outputs/`, `results/`, or `logs/`.
- The public scripts focus on the abstract-generation setting. The review and
  writing datasets use the same JSONL pattern, but may require task-specific
  prompt formatting if you want to reproduce the full paper table.

Before sharing the repository, run:

```bash
python scripts/validate_release.py
```

This performs a lightweight syntax, dataset-schema, and sensitive-file check
without loading any model weights.
