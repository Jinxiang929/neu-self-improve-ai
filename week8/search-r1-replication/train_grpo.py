from itertools import islice

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo import grpo_update
from lora_policy import attach_lora
from retriever import retrieve
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    torch.manual_seed(42)

    dataset = load_dataset(
        "PeterJinGo/nq_hotpotqa_train",
        split="train",
        streaming=True,
    )
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
    ).to("cuda")
    policy = attach_lora(base_model)
    policy.print_trainable_parameters()

    trainable = {
        name: parameter for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    assert trainable, "No trainable parameters found."
    assert all(".lora_A." in name or ".lora_B." in name for name in trainable)

    before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    optimizer = torch.optim.AdamW(
        trainable.values(), lr=1e-6, weight_decay=0.0,
    )

    max_updates = 3
    max_questions = 24
    updates = 0

    for example in islice(dataset, max_questions):
        print(f"Training example: {example['id']} ({example['data_source']})", flush=True)
        prompt = tokenizer.apply_chat_template(
            example["prompt"], add_generation_prompt=True, tokenize=False,
        )
        rewards, loss = grpo_update(
            policy, tokenizer, optimizer, prompt, example["golden_answers"], retrieve,
            group_size=5,
            max_searches=2,
            max_new_tokens=256,
            skip_flat_group=True,
        )
        print(f"EM rewards: {rewards}", flush=True)

        if loss is None:
            print("All rewards tied. Trying the next training question.", flush=True)
            continue

        updates += 1
        print(f"GRPO update {updates}/{max_updates}, loss: {loss:.6f}", flush=True)
        if updates == max_updates:
            break

    if updates == 0:
        raise RuntimeError(f"No effective GRPO updates in {max_questions} questions.")

    changed = sum(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in trainable.items()
    )
    assert changed > 0, "No adapter weights changed."
    assert all(
        parameter.grad is None
        for parameter in policy.parameters()
        if not parameter.requires_grad
    ), "A frozen parameter received a gradient."

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    adapter_dir = Path("/data/grpo-adapters") / f"grpo-short-{run_id}"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(adapter_dir, safe_serialization=True)

    print(f"Completed GRPO updates: {updates}/{max_updates}")
    print(f"Updated adapter tensors: {changed}")
    print(f"Saved LoRA adapter: {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()