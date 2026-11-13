# Search-R1 on Bamboogle

This project evaluates Qwen2.5-3B-Instruct on all 125 Bamboogle questions. It reproduces inference with the authors’ Search-R1-Instruct checkpoints and evaluates two adapters trained in this project with LoRA.

## Results

| Method | Correct / total | Bamboogle EM | Paper EM | Difference | Evaluation log |
| --- | ---: | ---: | ---: | ---: | --- |
| Direct Inference | 2 / 125 | 1.6% | 2.4% | −0.8 pp | [Modal](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-OJfUswXFNQRkrtC1CdsfK5?activeTab=logs) |
| Search-R1-Instruct, published PPO | 40 / 125 | **32.0%** | 26.4% | +5.6 pp | [Modal](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-Q7fw4ZO7yCi56TiD0NwUgq?activeTab=logs) |
| Search-R1-Instruct, published GRPO | 31 / 125 | 24.8% | 23.2% | +1.6 pp | [Modal](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-4mL6zG9XwMSqbrt04Oovbt?activeTab=logs) |
| Self-trained GRPO + LoRA | 23 / 125 | 18.4% | — | — | [Modal](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-jc0FI5CZMq5oEKaMZ2EzbR?activeTab=logs) |
| Self-trained PPO + LoRA | 21 / 125 | 16.8% | — | — | [Modal](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-oftXDyBoo98d1ZQV43yvov?activeTab=logs) |

The paper values are from Tables 2 and 3 of [Search-R1, arXiv v5](https://arxiv.org/html/2503.09516v5). The difference is the measured score minus the paper score in percentage points. The self-trained adapters used short runs with different training budgets, so the two-question gap between them does not establish an advantage for either algorithm.

## Evaluation and training setup

- **Evaluation.** Exact match (EM) compares each normalized prediction against all accepted answers. Direct Inference uses greedy decoding with up to 64 new tokens. Search-R1 scores the extracted `<answer>` and uses greedy decoding, up to 1,024 new tokens per turn, and up to four searches per question. Each search retrieves three passages from the official Wikipedia 2018 index with E5. All evaluations used a Modal L4. See [the inference replication notes](REPLICATION.md) for details.
- **PEFT.** Both self-trained runs started from `Qwen/Qwen2.5-3B-Instruct` with frozen base weights and rank-16 LoRA adapters, using examples streamed from the [NQ/HotpotQA training split](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train). GRPO sampled five rollouts per prompt and used an update budget of three. PPO used three batches of three rollouts with two epochs per batch, for six optimizer steps. The PPO run included five EM=0 rollouts and four EM=1 rollouts. Its value head was saved alongside the adapter and is only needed for training. [PPO training log](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-zrYfpTrMuFssGB2MiteF0i?activeTab=logs)

## Directory structure

```text
search-r1-replication/
├── README.md              # Results, project layout, and commands
├── REPLICATION.md         # Inference replication details
├── requirements.txt       # Local Python dependencies
├── bamboogle.csv          # Bamboogle evaluation questions and answers
├── direct.py              # Direct Inference entry point
├── official_ppo.py        # Published PPO checkpoint evaluation
├── official_grpo.py       # Published GRPO checkpoint evaluation
├── train_ppo_lora.py      # PPO + LoRA training entry point
├── train_grpo_lora.py     # GRPO + LoRA training entry point
├── eval_ppo_lora.py       # Saved PPO adapter evaluation
├── eval_grpo_lora.py      # Saved GRPO adapter evaluation
├── retriever_tools.py     # Wikipedia index preparation and retrieval check
├── modal_common.py        # Shared Modal image, Volume, and retriever lifecycle
├── direct_inference.py    # Direct answer generation
├── search_inference.py    # Search-R1 generation and evaluation
├── retriever.py           # Retrieval client
├── bamboogle.py           # Dataset loader
├── evaluate.py            # Evaluation loop
├── reward.py              # Answer normalization and exact match
├── train_ppo.py           # PPO training loop
├── train_grpo.py          # GRPO training loop
├── ppo.py                 # PPO losses and advantages
├── grpo.py                # GRPO update and losses
├── lora_policy.py         # LoRA configuration
├── rollout.py             # Search-R1 training rollouts
├── trajectory.py          # Token masks and policy log probabilities
└── ppo_local.py           # Local PPO update check
```

## Run the experiments

Run the commands from this directory with the Modal CLI configured. If needed, export the [Bamboogle sheet](https://docs.google.com/spreadsheets/d/1jwcsA5kE4TObr9YHn9Gc-wQHYjTbLhDGx6tmIzMhl_U/edit?gid=0) as `bamboogle.csv`. The Search-R1 runs use a Wikipedia index stored in the `search-r1-wiki18` Modal Volume. Prepare it once per Volume, then check retrieval.

```bash
modal run retriever_tools.py::prepare_index
modal run retriever_tools.py::check_top3
```

Run the baseline and the published checkpoint evaluations. Each command defaults to 125 questions.

```bash
modal run direct.py
modal run official_ppo.py
modal run official_grpo.py
```

Train new LoRA adapters. Each run saves its checkpoint to a new directory and prints the path.

```bash
modal run train_ppo_lora.py
modal run train_grpo_lora.py
```

Evaluate the saved adapters from the results table. Each command defaults to 125 questions. The saved checkpoints must be available in the mounted Modal Volume.

```bash
modal run eval_ppo_lora.py
modal run eval_grpo_lora.py
```

| Evaluation | Saved adapter directory in the container |
| --- | --- |
| PPO + LoRA | `/data/ppo-adapters/ppo-short-20261113T020435277459Z` |
| GRPO + LoRA | `/data/grpo-adapters/grpo-short-20261112T082650462520Z` |
