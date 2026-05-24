# Datasets

This directory contains the small reviewer-facing JSONL splits used by the
example scripts:

- `abstract.jsonl`
- `review.jsonl`
- `writing.jsonl`

Each line is a JSON object. The generation scripts primarily consume:

- `input`: task prompt without retrieved local context.
- `top_5`: retrieved examples used as local context for the SLM branch.
- `output`: reference response from the source dataset.

These dataset files are intentionally tracked. Generated responses, model
outputs, evaluation reports, and private full-scale result files should be kept
outside git or under ignored directories such as `outputs/` and `results/`.
