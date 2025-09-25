# CoSteer: COLLABORATIVE DECODING-TIME PER-SONALIZATION VIA LOCAL DELTA STEERING

This repository contains the implementation and datasets for our manuscipt. Our work introduces **CoSteer**, a framework for enabling a powerful, remote large language model (LLM) to be guided by a smaller, local language model (SLM) that has access to private user context, without compromising data privacy.

## Overview

This repository provides the necessary resources to reproduce the experiments and understand the core implementations of the CoSteer framework and its variants. The key contributions implemented here include:

* **CoSteer**: The original implementation of our collaborative framework.
* **Cross-Architecture Collaboration**: Two methods for enabling collaboration between models of different architectures and tokenizers.
* **Efficient CoSteer (AdaCoSteer)**: An adaptive variant of CoSteer designed to reduce communication cost for faster inference.

## Repository Structure
.
├── datasets/
│   ├── abstract.jsonl
│   ├── review.jsonl
│   └── writing.jsonl
└── codes/
├── abstract_costeer.py
├── abstract_costeer_map.py
├── abstract_costeer_byte.py
└── abstract_adacosteer.py

### Datasets

The `datasets/` directory contains the three datasets used in our experiments, as described in Section 4.1 of our paper. For each instance in the original datasets, we used the `bge-m3` model to retrieve the five most similar examples. These examples were then prepended to the input to serve as in-context examples.

### Code

The `codes/` directory contains the core implementations of our proposed methods. The current scripts are primarily test code focused on the `abstract` dataset.

* `costeer.py`: The original implementation of the **CoSteer** framework.
* `costeer_map.py`: The implementation of our cross-architecture collaboration strategy using **vocabulary mapping**, as detailed in Section 5.2.
* `costeer_byte.py`: The implementation of our more universal cross-architecture strategy using a **byte-level approach**, also detailed in Section 5.2.
* `adacosteer.py`: The implementation of **AdaCoSteer**, the efficient, adaptive variant of our framework described in Section 5.3.

**Note:** The current code is provided for reproducibility of our core results. Adapting the code for other datasets requires minor modifications to the prompt format and data loading logic. We plan to release a more user-friendly, refactored, and well-documented version of the code that supports all datasets and experiments after the peer-review process.