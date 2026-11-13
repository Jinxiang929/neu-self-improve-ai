from itertools import islice

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_policy import attach_lora
from ppo import gae, ppo_policy_loss, ppo_value_loss
from retriever import retrieve
from reward import exact_match
from rollout import collect_search_rollout
from trajectory import build_policy_inputs, policy_token_logprobs
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    torch.manual_seed(42)
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    dataset = load_dataset(
        "PeterJinGo/nq_hotpotqa_train", split="train", streaming=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
    ).to("cuda")
    policy = attach_lora(base_model).eval()

    value_head = torch.nn.Linear(
        policy.config.hidden_size, 1, device=policy.device, dtype=torch.float32,
    )

    adapters = {
        name: param for name, param in policy.named_parameters()
        if param.requires_grad
    }
    assert adapters and all(
        ".lora_A." in name or ".lora_B." in name for name in adapters
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": adapters.values(), "lr": 1e-6},
            {"params": value_head.parameters(), "lr": 1e-5},
        ],
        weight_decay=0.0,
    )

    before_adapters = {
        name: param.detach().clone() for name, param in adapters.items()
    }
    before_head = value_head.weight.detach().clone()

    num_batches = 3
    rollouts_per_batch = 3
    ppo_epochs = 2
    examples = iter(dataset)
    em_counts = [0, 0]

    for batch_idx in range(1, num_batches + 1):
        batch = []
        batch_rewards = []

        for example in islice(examples, rollouts_per_batch):
            prompt = tokenizer.apply_chat_template(
                example["prompt"], add_generation_prompt=True, tokenize=False,
            )
            prompt_ids, segments, answer = collect_search_rollout(
                policy, tokenizer, prompt, retrieve,
                max_searches=2, max_new_tokens=256,
            )
            em = exact_match(answer, example["golden_answers"])
            print(f"{example['id']} ({example['data_source']}): EM={em}", flush=True)

            inputs = tuple(
                tensor.to(policy.device)
                for tensor in build_policy_inputs(prompt_ids, segments)
            )
            input_ids, attention_mask, policy_mask = inputs
            start = len(prompt_ids) - 1
            actor_mask = policy_mask[0, 1:][start:]
            generated_positions = actor_mask.nonzero(as_tuple=True)[0]
            if generated_positions.numel() == 0:
                raise RuntimeError(f"No generated tokens for {example['id']}.")

            with torch.no_grad():
                output = policy(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                response_states = output.hidden_states[-1][0, start:-1, :].float()
                del output
                old_values = value_head(response_states).squeeze(-1)
                old_logprobs = policy_token_logprobs(policy, *inputs)

                rewards = torch.zeros_like(old_values)
                rewards[generated_positions[-1]] = em
                advantages, returns = gae(rewards, old_values)

            batch.append((
                inputs, old_logprobs, response_states, old_values,
                advantages[actor_mask], returns,
            ))
            batch_rewards.append(em)
            em_counts[em] += 1

        if len(batch) != rollouts_per_batch:
            raise RuntimeError("Training data ended before the batch was full.")
        print(f"Batch {batch_idx}/{num_batches}: EM rewards={batch_rewards}", flush=True)

        for epoch in range(1, ppo_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            actor_total = critic_total = 0.0

            for inputs, old_logprobs, states, old_values, advantages, returns in batch:
                new_logprobs = policy_token_logprobs(policy, *inputs)
                new_values = value_head(states).squeeze(-1)

                actor_loss = ppo_policy_loss(new_logprobs, old_logprobs, advantages)
                critic_loss = ppo_value_loss(new_values, old_values, returns)
                loss = (actor_loss + critic_loss) / len(batch)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite PPO loss.")

                loss.backward()
                actor_total += actor_loss.item()
                critic_total += critic_loss.item()

            optimizer.step()
            print(
                f"Batch {batch_idx}, PPO epoch {epoch}/{ppo_epochs}: "
                f"actor={actor_total / len(batch):.6f}, "
                f"critic={critic_total / len(batch):.6f}",
                flush=True,
            )

    updated = sum(
        not torch.equal(param.detach(), before_adapters[name])
        for name, param in adapters.items()
    )
    assert updated > 0, "No LoRA weights changed."
    assert not torch.equal(value_head.weight.detach(), before_head), (
        "Value head weights did not change."
    )
    assert all(
        param.grad is None for param in policy.parameters() if not param.requires_grad
    ), "A frozen policy parameter received a gradient."

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = Path("/data/ppo-adapters") / f"ppo-short-{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(output_dir, safe_serialization=True)
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in value_head.state_dict().items()},
        output_dir / "value_head.pt",
    )

    print(f"Training rollouts: EM=0: {em_counts[0]}, EM=1: {em_counts[1]}")
    print(f"Optimizer steps: {num_batches * ppo_epochs}, updated LoRA tensors: {updated}")
    print(f"Saved PPO checkpoint: {output_dir}", flush=True)

if __name__ == "__main__":
    main()