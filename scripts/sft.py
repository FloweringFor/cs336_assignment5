import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
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

