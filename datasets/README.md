# Evaluation Datasets

This directory contains the exact input subsets used for the eight datasets in
the CoSteer paper. Reference answers stored in source datasets are labels, not
model generations produced by this repository.

| Dataset | Released file | Rows | Source |
|---|---|---:|---|
| CoGenesis | `cogenesis.jsonl` | 240 | [TsinghuaC3I/CoGenesis](https://github.com/TsinghuaC3I/CoGenesis) |
| LongLaMP Abstract | `longlamp_abstract.jsonl` | 200 | [LongLaMP](https://huggingface.co/datasets/LongLaMP/LongLaMP) |
| LongLaMP Review | `longlamp_review.jsonl` | 200 | [LongLaMP](https://huggingface.co/datasets/LongLaMP/LongLaMP) |
| LongLaMP Writing | `longlamp_writing.jsonl` | 200 | [LongLaMP](https://huggingface.co/datasets/LongLaMP/LongLaMP) |
| HelpSteer | `helpsteer.jsonl` | 200 | [nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) |
| Personal Preference Eval | `personal_preference_eval.jsonl` | 200 | [Linear Alignment](https://github.com/Wizardcoast/Linear_Alignment) |
| TruthfulQA | `truthfulqa.jsonl` | 200 | [sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA) |
| UltraChat | `ultrachat.jsonl` | 200 | [thunlp/UltraChat](https://github.com/thunlp/UltraChat) |

## Formats

All files use one JSON object per line and include a stable `id` plus the public
task `input`.

The three LongLaMP files additionally contain:

- `top_5`: the five user-history records retrieved for the local SLM branch.
- `output`: the benchmark reference text.
- Source metadata such as author or reviewer identifiers where provided by the
  benchmark.

`cogenesis.jsonl` merges the official paired `ctx_test_wo.json` and
`ctx_test.json` files:

- `input`: the prompt without user context.
- `personalized_input`: the paired prompt containing the benchmark context.
- `output`: the context-aware reference response.
- `output_without_context`: the paired context-free reference response.

The four preference-alignment files contain only `id` and `input`. The target
attribute is supplied at generation time with `--preference`; each attribute is
an independent generation.

`SHA256SUMS` records checksums for all eight released files.

The files retain the content and upstream terms of their respective source
datasets. Please cite the original datasets when using them.
