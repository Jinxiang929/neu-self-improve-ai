import torch
import torch.nn.functional as F


def build_policy_inputs(
    prompt_ids: list[int],
    segments: list[tuple[list[int], bool]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build inputs for one unpadded trajectory."""
    if not prompt_ids:
        raise ValueError("The trajectory must begin with a prompt.")

    token_ids = list(prompt_ids)
    policy_positions = [False] * len(prompt_ids)

    for segment_ids, generated_by_policy in segments:
        token_ids.extend(segment_ids)
        policy_positions.extend([generated_by_policy] * len(segment_ids))

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    policy_mask = torch.tensor([policy_positions], dtype=torch.bool)

    return input_ids, attention_mask, policy_mask


def policy_token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    policy_mask: torch.Tensor,
) -> torch.Tensor:
    """Return log probabilities for the policy tokens in one trajectory."""
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits

    target_ids = input_ids[:, 1:]
    target_mask = policy_mask[:, 1:]
    selected_logits = logits[:, :-1, :][target_mask].float()
    selected_ids = target_ids[target_mask]

    return -F.cross_entropy(selected_logits, selected_ids, reduction="none")