import torch
from peft import PeftModel

from trajectory import policy_token_logprobs
from collections.abc import Callable

from transformers import PreTrainedTokenizerBase

from reward import exact_match
from rollout import collect_search_rollout
from trajectory import build_policy_inputs, policy_token_logprobs


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute one advantage per rollout for a single question."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("Expected at least two rewards for one question.")

    rewards = rewards.detach().float()
    centered_rewards = rewards - rewards.mean()
    return centered_rewards / (rewards.std(correction=1) + 1e-6)

def clipped_policy_loss(
    new_log_probs: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """Compute the clipped policy loss for one question's rollout group."""
    group_size = len(new_log_probs)
    if group_size == 0 or len(old_log_probs) != group_size:
        raise ValueError("Expected matching, non-empty rollout lists.")
    if advantages.shape != (group_size,):
        raise ValueError("Expected one advantage per rollout.")

    advantages = advantages.detach().to(new_log_probs[0].device)
    rollout_losses = []

    for current, previous, advantage in zip(new_log_probs, old_log_probs, advantages):
        if current.ndim != 1 or current.numel() == 0 or current.shape != previous.shape:
            raise ValueError("Each rollout needs matching, non-empty token vectors.")

        ratio = (current - previous.detach()).exp()
        unclipped = ratio * advantage
        clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
        rollout_losses.append(-torch.minimum(unclipped, clipped).mean())

    return torch.stack(rollout_losses).mean()

def reference_token_logprobs(
    policy: PeftModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    policy_mask: torch.Tensor,
) -> torch.Tensor:
    """Score the policy tokens with the frozen base model."""
    with torch.no_grad(), policy.disable_adapter():
        return policy_token_logprobs(policy, input_ids, attention_mask, policy_mask)


def token_kl(
    current_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the low-variance KL estimate for each policy token."""
    if current_log_probs.shape != reference_log_probs.shape:
        raise ValueError("Current and reference scores must cover the same tokens.")

    delta = reference_log_probs.detach().float() - current_log_probs.float()
    return (delta.expm1() - delta).clamp(min=-10.0, max=10.0)


def grpo_loss(
    new_log_probs: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    reference_log_probs: list[torch.Tensor],
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    beta: float = 0.001,
) -> torch.Tensor:
    """Combine the clipped policy loss and reference-policy KL."""
    if len(reference_log_probs) != len(new_log_probs):
        raise ValueError("Expected one reference score vector per rollout.")

    policy_loss = clipped_policy_loss(
        new_log_probs, old_log_probs, advantages, clip_epsilon=clip_epsilon,
    )
    rollout_kls = [
        token_kl(current, reference).mean()
        for current, reference in zip(new_log_probs, reference_log_probs)
    ]
    return policy_loss + beta * torch.stack(rollout_kls).mean()

def grpo_update(
    policy: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    prompt: str,
    answers: list[str],
    retrieve: Callable[[str], str],
    group_size: int = 5,
    max_searches: int = 4,
    max_new_tokens: int = 1024,
    skip_flat_group: bool = False,
) -> tuple[list[int], float | None]:
    """Collect one rollout group and make one GRPO optimizer update."""
    if group_size < 2:
        raise ValueError("GRPO requires at least two rollouts per question.")

    # Keep dropout disabled during sampling and log-probability scoring.
    policy.eval()
    rollouts = []
    rewards = []

    with torch.no_grad():
        for _ in range(group_size):
            prompt_ids, segments, answer = collect_search_rollout(
                policy, tokenizer, prompt, retrieve,
                max_searches=max_searches,
                max_new_tokens=max_new_tokens,
            )
            inputs = tuple(
                tensor.to(policy.device)
                for tensor in build_policy_inputs(prompt_ids, segments)
            )
            if not inputs[2][:, 1:].any().item():
                raise RuntimeError("Rollout has no policy tokens.")

            old_scores = policy_token_logprobs(policy, *inputs)
            reference_scores = reference_token_logprobs(policy, *inputs)
            rollouts.append((inputs, old_scores, reference_scores))
            rewards.append(exact_match(answer, answers))


    if skip_flat_group and len(set(rewards)) == 1:
        return rewards, None
    
    advantages = group_advantages(torch.tensor(rewards, device=policy.device))
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    for index, (inputs, old_scores, reference_scores) in enumerate(rollouts):
        current_scores = policy_token_logprobs(policy, *inputs)
        loss = grpo_loss(
            [current_scores],
            [old_scores],
            [reference_scores],
            advantages[index:index + 1],
        ) / group_size

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite GRPO loss for rollout {index}.")

        loss.backward()
        total_loss += loss.detach().item()

    optimizer.step()
    return rewards, total_loss