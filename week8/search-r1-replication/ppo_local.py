import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from lora_policy import attach_lora
from ppo import gae, ppo_policy_loss, ppo_value_loss
from reward import exact_match
from trajectory import build_policy_inputs, policy_token_logprobs


def main() -> None:
    torch.manual_seed(42)

    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    # Keep dropout disabled when comparing old and new log probabilities.
    policy = attach_lora(Qwen2ForCausalLM(config)).eval()
    value_head = torch.nn.Linear(config.hidden_size, 1)
    torch.nn.init.zeros_(value_head.weight)
    torch.nn.init.zeros_(value_head.bias)

    policy_params = dict(policy.named_parameters())
    adapter_names = {
        name for name in policy_params
        if ".lora_A." in name or ".lora_B." in name
    }
    trainable_names = {
        name for name, param in policy_params.items() if param.requires_grad
    }
    assert adapter_names and trainable_names == adapter_names

    optimizer = torch.optim.AdamW(
        [param for param in policy_params.values() if param.requires_grad]
        + list(value_head.parameters()),
        lr=1e-3,
        weight_decay=0.0,
    )

    prompt_ids = [1, 10, 11]
    segments = [
        ([20, 21], True),   # Generated search
        ([30, 31], False),  # Retrieved information
        ([40, 2], True),    # Generated answer
    ]
    inputs = build_policy_inputs(prompt_ids, segments)
    input_ids, attention_mask, policy_mask = inputs
    response_start = len(prompt_ids) - 1
    actor_mask = policy_mask[0, 1:][response_start:]
    assert actor_mask.tolist() == [True, True, False, False, True, True]

    # Save the old policy scores and critic estimates before the update.
    with torch.no_grad():
        output = policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        response_states = output.hidden_states[-1][0, response_start:-1, :]
        old_values = value_head(response_states).squeeze(-1)
        old_logprobs = policy_token_logprobs(policy, *inputs)

        rewards = torch.zeros_like(old_values)
        rewards[-1] = exact_match("Paris", ["Paris"])
        advantages, returns = gae(rewards, old_values)

    before_policy = {
        name: param.detach().clone() for name, param in policy_params.items()
    }
    before_head = value_head.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    new_logprobs = policy_token_logprobs(policy, *inputs)
    new_values = value_head(response_states).squeeze(-1)
    torch.testing.assert_close(new_logprobs, old_logprobs)
    torch.testing.assert_close(new_values, old_values)

    actor_loss = ppo_policy_loss(
        new_logprobs, old_logprobs, advantages[actor_mask],
    )
    critic_loss = ppo_value_loss(new_values, old_values, returns)
    loss = actor_loss + critic_loss
    assert torch.isfinite(loss).item()
    loss.backward()

    base_names = policy_params.keys() - adapter_names
    assert all(policy_params[name].grad is None for name in base_names)
    optimizer.step()

    assert any(
        not torch.equal(policy_params[name], before_policy[name])
        for name in adapter_names
    ), "No LoRA adapter weights changed."
    assert not torch.equal(value_head.weight, before_head), (
        "Value head weights did not change."
    )
    assert all(
        torch.equal(policy_params[name], before_policy[name])
        for name in base_names
    ), "A base model weight changed."

    print(f"Actor loss: {actor_loss.item():.6f}")
    print(f"Critic loss: {critic_loss.item():.6f}")
    print("Local PPO + LoRA update passed: adapter and value head changed; base weights stayed frozen.")


if __name__ == "__main__":
    main()