import torch
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from vllm import LLM

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm.model_executor import set_random_seed as vllm_set_random_seed


"""
model_id = "Qwen/Qwen2.5-Math-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
"""


def tokenize_prompt_and_output(
        prompt_strs: list[str],          # List of prompt strings
        output_strs: list[str],          # List of output strings,
        tokenizer: PreTrainedTokenizer,  # Tokenizer to use for tokenization
) -> dict[str, torch.Tensor]:
    all_input_ids = []
    all_response_masks = []

    for p_str, o_str in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(p_str)
        o_ids = tokenizer.encode(o_str)

        full_ids = p_ids + o_ids
        # 0 代表 prompt / padding 部分，1 代表 output 部分
        mask = [0] * len(p_ids) + [1] * len(o_ids)
        all_input_ids.append(torch.tensor(full_ids))
        all_response_masks.append(torch.tensor(mask))

    # 1. 对整个 batch 进行 padding
    # 使用 tokenizer.pad_sequence 或者手动 pad
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        all_input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    mask_padded = torch.nn.utils.rnn.pad_sequence(
        all_response_masks,
        batch_first=True,
        padding_value=0
    )

    return {
        "input_ids": input_ids_padded[:, :-1],
        "labels": input_ids_padded[:, 1:],
        "response_mask": mask_padded[:, 1:]
    }


@torch.no_grad()
def compute_entropy(
        logits: torch.Tensor  # Tensor of shape (batch_size, seq_len, vocab_size) containing unnormalized logits
) -> torch.Tensor:
    """
    这里计算的 熵（Entropy）是 衡量一个分布内部的混乱程度
    注意和交叉熵 （Cross Entropy）区分开来
    """
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    return - torch.sum(probs * (logits - log_z), dim=-1)


@torch.no_grad()
def get_response_log_probs(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    model.eval()
    logits = model(input_ids).logits
    log_probs_all = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result = {'log_probs': log_probs}
    if return_token_entropy:
        result['token_entropy'] = compute_entropy(logits)
    return result


def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None,
) -> torch.Tensor:
    mask = mask.to(tensor.dtype)
    masked_tensor = mask * tensor
    if dim is None:
        total_sum = torch.sum(masked_tensor)
    else:
        total_sum = torch.sum(masked_tensor, dim=dim)
    return total_sum / normalize_constant


def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    token_loss = - policy_log_probs
    total_loss = masked_normalize(token_loss, response_mask, normalize_constant, dim=-1)
    loss_for_backward = total_loss.mean() / gradient_accumulation_steps
    loss_for_backward.backward()
    meta_data = {
        "loss": loss_for_backward.detach()
    }
    return loss_for_backward.detach(), meta_data


@torch.no_grad()
def log_generations(
        model,
        tokenizer,
        prompt_strs: list[str],
        output_strs: list[str],
        generation_config: dict,
        device: str = "cuda"
) -> dict:
    model.eval()

    # 1. 准备输入 (仅针对 Prompt 部分进行编码，用于 generate)
    inputs = tokenizer(prompt_strs, return_tensor="pt", padding=True)
    inputs_ids = inputs.inputs_ids
    attention_mask = inputs.attention_mask

    # 2. 模型生成响应
    generate_outputs = model.generate(
        inputs_ids=inputs_ids,
        attention_mask=attention_mask,
        **generation_config,
        return_dict_in_generate=True,
        output_scores=True
    )

    gen_tokens = generate_outputs.sequences[:, inputs_ids.shape[-1]:]
    generated_responses = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    # 3. 计算 Token Entropy (平均标记熵)
    # scores 是一个元组，长度为生成步数，每个元素为 (batch_size, vocab_size)
    logits = torch.stack(generate_outputs.scores, dim=1)  # (batch_size, seq_len, vocab_size)
    entropies = compute_entropy(logits)   # (batch_size, seq_len)

    # 4. 统计指标
    results = []
    all_lengths = []
    correct_lengths = []
    incorrect_lengths = []

    for i in range(len(prompt_strs)):
        reward_info = r1_zero_reward_fn(generated_responses[i], output_strs[i])
        actual_gen_len = (gen_tokens[i] != tokenizer.pad_token_id).sum().item()
        avg_entropy = entropies[i, :actual_gen_len].mean().item() if actual_gen_len > 0 else 0.0

        all_lengths.append(actual_gen_len)
        is_correct = reward_info["reward"]
        if is_correct:
            correct_lengths.append(actual_gen_len)
        else:
            incorrect_lengths.append(actual_gen_len)

        results.append({
            "prompt": prompt_strs[i],
            "generation": generated_responses[i],
            "ground_truth": output_strs[i],
            "reward": is_correct,
            "reward_info": reward_info,
            "entropy": avg_entropy
        })

    # 5. 汇总指标
    metrics = {
        "avg_response_length": sum(all_lengths) / len(all_lengths),
        "avg_correct_length": sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0.0,
        "avg_incorrect_length": sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else 0.0,
        "avg_entropy": entropies.mean().item(),
        "samples": results
    }

    model.train()  # 恢复训练模式
    return metrics


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
        Start the inference process, here we use vLLM to hold a model on
        a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


