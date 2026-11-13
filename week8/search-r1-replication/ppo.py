import torch


@torch.no_grad()
def gae(
    rewards: torch.Tensor,
    old_values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantages and returns for one terminated response."""
    if rewards.ndim != 1 or rewards.shape != old_values.shape or rewards.numel() == 0:
        raise ValueError("Expected matching, nonempty 1-D rewards and values.")

    advantages = torch.zeros_like(old_values)
    next_value = old_values.new_zeros(())
    next_advantage = old_values.new_zeros(())

    for t in range(rewards.numel() - 1, -1, -1):
        delta = rewards[t] + gamma * next_value - old_values[t]
        next_advantage = delta + gamma * lam * next_advantage
        advantages[t] = next_advantage
        next_value = old_values[t]

    return advantages, advantages + old_values


def ppo_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip: float = 0.2,
) -> torch.Tensor:
    """Compute actor loss on policy tokens only."""
    ratio = (new_logprobs - old_logprobs.detach()).exp()
    advantages = advantages.detach()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * advantages
    return -torch.minimum(unclipped, clipped).mean()


def ppo_value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip: float = 0.5,
) -> torch.Tensor:
    """Compute critic loss on response positions."""
    old_values = old_values.detach()
    returns = returns.detach()
    clipped = old_values + (new_values - old_values).clamp(-clip, clip)
    error = (new_values - returns).square()
    clipped_error = (clipped - returns).square()
    return 0.5 * torch.maximum(error, clipped_error).mean()


if __name__ == "__main__":
    from trajectory import build_policy_inputs

    # Same format as collect_search_rollout(): prompt, segments, answer.
    prompt_ids = [1, 10, 11]
    segments = [
        ([20, 21], True),   # Generated search
        ([30, 31], False),  # Retrieved information
        ([40, 2], True),    # Generated answer
    ]
    input_ids, _, policy_mask = build_policy_inputs(prompt_ids, segments)

    # State at position t predicts the action at position t + 1.
    response_start = len(prompt_ids) - 1
    actor_mask = policy_mask[0, 1:][response_start:]
    response_ids = input_ids[0, 1:][response_start:]
    assert response_ids[actor_mask].tolist() == [20, 21, 40, 2]

    # Fake critic outputs, one scalar per input position.
    state_values = torch.tensor(
        [[9.0, 9.0, 0.2, 1.2, 0.0, 0.0, 0.4, 1.4, 9.0]],
        requires_grad=True,
    )
    old_values = state_values[0, :-1][response_start:]
    rewards = torch.zeros_like(old_values)
    rewards[-1] = 1.0  # Simulated EM = 1 at the final generated token.

    advantages, returns = gae(rewards, old_values)
    torch.testing.assert_close(returns, torch.ones(6))
    torch.testing.assert_close(
        advantages[actor_mask], torch.tensor([0.8, -0.2, 0.6, -0.4])
    )

    # Also check the recurrence with discounting and lambda below 1.
    discounted, _ = gae(
        torch.tensor([0.0, 1.0]), torch.tensor([0.2, 0.4]),
        gamma=0.5, lam=0.5,
    )
    torch.testing.assert_close(discounted, torch.tensor([0.15, 0.6]))

    # In training, these vectors come from policy_token_logprobs().
    old_logprobs = torch.full((4,), -2.0, requires_grad=True)
    ratios = torch.tensor([1.5, 0.5, 0.5, 1.5])
    new_logprobs = (old_logprobs.detach() + ratios.log()).requires_grad_()
    actor_loss = ppo_policy_loss(new_logprobs, old_logprobs, advantages[actor_mask])
    torch.testing.assert_close(actor_loss, torch.tensor(-0.125))

    new_values = torch.tensor(
        [0.8, 1.1, 0.2, 0.3, 0.5, 1.3], requires_grad=True
    )
    critic_loss = ppo_value_loss(new_values, old_values, returns)
    torch.testing.assert_close(critic_loss, torch.tensor(1.57 / 12))

    actor_loss.backward()
    critic_loss.backward()
    assert old_logprobs.grad is None and state_values.grad is None
    assert new_logprobs.grad[:2].tolist() == [0.0, 0.0]
    assert new_logprobs.grad[2] < 0 and new_logprobs.grad[3] > 0
    assert new_values.grad[0] == 0
    assert torch.all(new_values.grad[~actor_mask] != 0)

    print(f"Actor loss: {actor_loss.item():.6f}")
    print(f"Critic loss: {critic_loss.item():.6f}")
    print("Local PPO advantage and loss check passed.")