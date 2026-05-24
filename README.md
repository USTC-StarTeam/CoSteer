# CoSteer: Collaborative Decoding-Time Personalization via Local Delta Steering

This repository contains the implementation of **CoSteer**, a decoding-time
personalization method for collaborative cloud-edge generation. CoSteer keeps
private user context on the local side: a small local language model (SLM)
observes the private context and produces a context-induced delta signal, while a
larger language model (LLM) generates the final response without directly seeing
that private context.

At each decoding step, CoSteer compares the SLM distribution with and without
the retrieved user context, then uses this local delta to steer the LLM's next
token distribution. The repository also includes two tokenizer-mismatch variants
for cross-architecture collaboration and an adaptive variant, AdaCoSteer, that
skips local fusion when the LLM is already sufficiently confident.

This public artifact is designed for paper review. It includes the core
generation code and small JSONL evaluation splits so reviewers can inspect the
method logic and run lightweight smoke tests. Generated model outputs,
evaluation tables, local checkpoints, and cached experiment artifacts are not
included.

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
authentication, please authenticate through the standard Hugging Face workflow.

## Data Format

Each JSONL row is one evaluation instance. The core fields are:

- `input`: the task prompt without retrieved personalization context.
- `output`: the reference response from the dataset.
- `top_5`: retrieved in-context examples used as local private context.

The three files under `datasets/` are small reviewer-facing splits used by the
example scripts. They follow the same JSONL pattern as the full experiments.

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

The default hyperparameters match the settings used in the paper experiments:
`T=20`, `alpha=2`, `beta=1`, `player_lambda=2`, and `eta=10`.

## Release Scope

- The repository contains source code and small JSONL splits for inspection and
  smoke testing.
- Generated responses and evaluation outputs are not included in the public
  artifact.
- The scripts write generated responses to `outputs/` by default. This keeps
  source files and released data unchanged during local runs.
- The example scripts focus on the abstract-generation setting. The review and
  writing splits are included for reference and use the same JSONL structure, but
  full-table reproduction may require task-specific prompt formatting.

For a lightweight sanity check that does not load model weights, run:

```bash
python scripts/validate_release.py
```

The check validates Python syntax, JSONL schemas, and the public repository
layout.
