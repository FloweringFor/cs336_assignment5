import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import torch.functional as F

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
