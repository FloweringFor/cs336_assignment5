import torch
from collections.abc import Callable


def compute_group_normalized_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    raw_rewards_list = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        score_dict = reward_fn(resp, gt)
        raw_rewards_list.append(score_dict["reward"])
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    grouped_rewards = raw_rewards.view(-1, group_size)    # (n_prompts, group_size)
    mean_r = grouped_rewards.mean(dim=-1, keepdim=True)   # (n_prompts, 1)
    if normalize_by_std:
        std_r = grouped_rewards.std(dim=-1, keepdim=True)   # (n_prompts, 1)
        grouped_advantages = (grouped_rewards - mean_r) / (std_r + advantage_eps)
    else:
        grouped_advantages = grouped_rewards - mean_r
    advantages = grouped_advantages.view(-1)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return - raw_rewards_or_advantages * policy_log_probs
