# CoSteer: Collaborative Decoding-Time Personalization via Local Delta Steering

CoSteer is a tuning-free edge-cloud decoding framework for personalized text
generation. At every generation step, a local small language model (SLM)
computes the change in its next-token distribution caused by user context. The
cloud large language model (LLM) never receives that context directly; CoSteer
instead transfers the local distribution delta and incorporates it through an
FTRL policy update before selecting the next token.

This repository provides the per-token implementation used by the paper, task
adapters for all eight evaluation datasets, and the exact released evaluation
inputs. The canonical runner uses KV caching and supports Qwen2.5, Qwen3, and
other LLM/SLM pairs that share the same tokenizer and token IDs.

## Method at a Glance

For each decoding position, CoSteer performs three forward passes:

1. The cloud LLM predicts from the public query and generated prefix.
2. The local SLM predicts once without personal context and once with it.
3. The difference between the two local log distributions is fused into the
   cloud policy with the iterative update in `costeer/optimizer.py`.

Only the sampled token is appended to all three prefixes. The default
hyperparameters match the paper: `T=20`, `alpha=2`, `beta=1`,
`player_lambda=2`, and `eta=10`.

## Repository Layout

```text
.
├── costeer/
│   ├── optimizer.py             # FTRL policy update and local delta fusion
│   ├── generation.py            # Per-token generation with KV caching
│   ├── tasks.py                 # Prompt adapters for all eight datasets
│   └── cli.py                   # Shared command-line implementation
├── codes/
│   ├── abstract_costeer.py      # Backward-compatible abstract entry point
│   ├── abstract_costeer_map.py  # Vocabulary-intersection extension
│   ├── abstract_costeer_byte.py # Byte-level tokenizer alignment extension
│   └── abstract_adacosteer.py   # Confidence-gated AdaCoSteer extension
├── datasets/                    # Exact paper evaluation inputs
├── tests/                       # Optimizer and task-adapter unit tests
├── run_costeer.py               # Canonical eight-dataset entry point
└── scripts/validate_release.py  # Data, secret, path, and release checks
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model identifiers may be Hugging Face Hub names or local paths supplied at run
time. Authentication for gated models should use the standard Hugging Face
login or environment-based credential flow.

## Evaluation Data

The release contains the eight datasets described in the paper:

| Category | Dataset | Released file | Rows |
|---|---|---|---:|
| Personalized generation | CoGenesis | `cogenesis.jsonl` | 240 |
| Personalized generation | LongLaMP Abstract | `longlamp_abstract.jsonl` | 200 |
| Personalized generation | LongLaMP Review | `longlamp_review.jsonl` | 200 |
| Personalized generation | LongLaMP Writing | `longlamp_writing.jsonl` | 200 |
| Preference alignment | HelpSteer | `helpsteer.jsonl` | 200 |
| Preference alignment | Personal Preference Eval | `personal_preference_eval.jsonl` | 200 |
| Preference alignment | TruthfulQA | `truthfulqa.jsonl` | 200 |
| Preference alignment | UltraChat | `ultrachat.jsonl` | 200 |

The LongLaMP files include the top-five records retrieved with
`bge-reranker-v2-m3`. The four preference-alignment files are the 200-instance
samples used in the paper. Dataset provenance and field definitions are in
`datasets/README.md`.

## Running CoSteer

Run one LongLaMP abstract example:

```bash
python run_costeer.py \
  --task longlamp_abstract \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --slm-model Qwen/Qwen2.5-1.5B-Instruct \
  --max-new-tokens 128 \
  --limit 1
```

Run CoGenesis:

```bash
python run_costeer.py \
  --task cogenesis \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --slm-model Qwen/Qwen2.5-1.5B-Instruct
```

Run a preference-alignment dataset. Each attribute is generated independently,
as in the paper protocol:

```bash
python run_costeer.py \
  --task helpsteer \
  --preference concise \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --slm-model Qwen/Qwen2.5-1.5B-Instruct
```

Valid preference values are `concise`, `creative`, `uplifting`, and `verbose`.
The same command applies to `personal_preference_eval`, `truthfulqa`, and
`ultrachat`. Generation is greedy by default; `--do-sample`, `--temperature`,
and `--top-p` enable sampling. Existing output files are resumed by example ID.

The canonical runner requires identical token-to-ID mappings because sampled
tokens are shared across the cloud and local prefixes. The two scripts ending
in `_map.py` and `_byte.py` expose the tokenizer-mismatch extensions evaluated
in the paper.

## Release Validation

The repository does not contain generated responses, metric tables, logs,
checkpoints, model caches, or credentials. Runtime generations are written to
the ignored `outputs/` directory unless `--output-file` is provided.

Run all lightweight checks without loading model weights:

```bash
python -m unittest discover -s tests -v
python scripts/validate_release.py
```

The release validator checks all eight dataset schemas and row counts, compiles
the Python sources, and rejects tracked result artifacts, credentials, and
machine-specific absolute paths.
