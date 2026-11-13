# Search-R1 Replication on Bamboogle

This project reproduces the inference-only evaluation of Qwen2.5-3B-Instruct on Bamboogle. It evaluates Direct Inference and the authors’ published Search-R1-Instruct checkpoints trained with PPO and GRPO on the same 125 questions.

## Results

| Method | Correct / Total | Replication EM | Paper EM | Difference | Run log |
| --- | ---: | ---: | ---: | ---: | --- |
| Direct Inference | 2 / 125 | **1.6%** | 2.4% | −0.8 pp | [Modal log](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-OJfUswXFNQRkrtC1CdsfK5?activeTab=logs) |
| Search-R1-Instruct (PPO) | 40 / 125 | **32.0%** | 26.4% | +5.6 pp | [Modal log](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-Q7fw4ZO7yCi56TiD0NwUgq?activeTab=logs) |
| Search-R1-Instruct (GRPO) | 31 / 125 | **24.8%** | 23.2% | +1.6 pp | [Modal log](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-4mL6zG9XwMSqbrt04Oovbt?activeTab=logs) |

The difference is the replication score minus the paper score in percentage points. The Direct Inference and PPO reference values come from Table 2. The GRPO reference value comes from Table 3. [Search-R1 paper, arXiv v5](https://arxiv.org/html/2503.09516v5#S4)

PPO answered 38 more questions correctly than Direct Inference, and GRPO answered 29 more. These comparisons describe the checkpoints and inference settings used in this replication.

## Experimental setup

- **Dataset.** All runs used the 125 questions in the original Bamboogle sheet. The loader kept both accepted answers for the one question with a second answer. [Original Bamboogle sheet](https://docs.google.com/spreadsheets/d/1jwcsA5kE4TObr9YHn9Gc-wQHYjTbLhDGx6tmIzMhl_U/edit?gid=0)
- **Metric.** Exact Match (EM) compares the prediction with each accepted answer after lowercasing, removing ASCII punctuation and English articles, and normalizing whitespace. Direct Inference scores the generated text. Search-R1 scores the text inside the generated `<answer>...</answer>` block.
- **Direct Inference.** The original `Qwen/Qwen2.5-3B-Instruct` received a short-answer prompt and generated up to 64 new tokens with greedy decoding. [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **Search-R1.** Both checkpoints used the official Search-R1 prompt and Qwen chat template, greedy decoding, up to 1,024 new tokens per generation turn, and at most four searches per question. Each search returned three passages from the official 2018 Wikipedia corpus. The inference loop inserted those passages in `<information>...</information>` and resumed generation. [Official inference example](https://github.com/PeterGriffinJin/Search-R1/blob/main/infer.py) · [Official retriever setup](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md#flat-indexing)
- **Runtime.** The full evaluations ran on Modal with one L4. For Search-R1, `intfloat/e5-base-v2` encoded queries on the GPU and FAISS searched the official Flat index on the CPU.

## Reproduce the runs

Place the original spreadsheet export in this directory as `bamboogle.csv`. Run the commands below from the project directory with the Modal CLI configured for your workspace.

Run Direct Inference over the full test set.

```bash
modal run direct.py
```

Prepare the official Wikipedia corpus and FAISS index once per Modal Volume, then verify that the retriever returns three passages.

```bash
modal run retriever_tools.py::prepare_index
modal run retriever_tools.py::check_top3
```

Each Search-R1 entry point uses its published checkpoint and defaults to 125 questions.

| Entry point | Published checkpoint |
| --- | --- |
| `official_ppo.py` | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.2` |
| `official_grpo.py` | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.2` |

```bash
modal run official_ppo.py
modal run official_grpo.py
```

Each entry point starts the retriever and evaluates the checkpoint in the same container. Each full run prints an EM score with a denominator of 125. The LoRA training and evaluation commands are in [README.md](README.md#run-the-experiments).

