import os
import torch
import wandb
import json
import random
import gc
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, PreTrainedModel
from vllm import LLM, SamplingParams
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from sft import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step, log_generations
from math_baseline import load_model


# --- 1. 配置参数 ---
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
RAW_DATA_PATH = "/root/autodl-tmp/cs336_assignment5/data/MATH/sft.jsonl"
FILTER_DATA_PATH = "/root/autodl-tmp/cs336_assignment5/data/MATH/sft_filter.jsonl"
VAL_DATA_PATH = "/root/autodl-tmp/cs336_assignment5/data/MATH/validation.jsonl"
VAL_RESULT_PATH = "/root/autodl-tmp/cs336_assignment5/data/MATH/result"
CHECKPOINTS_PATH = "/root/autodl-tmp/cs336_assignment5/checkpoints"
MAX_GRAD_NORM = 1.0
LEARNING_RATE = 3e-5
BATCH_SIZE = 32
GRAD_ACCUM = 8
EPOCHS = 10


# 任务 2 的过滤逻辑
def sft_filter():
    results = []
    with open(RAW_DATA_PATH, "r") as f:
        for line in f:
            data = json.loads(line)
            reward = r1_zero_reward_fn(data["response"], data["ground_truth"])
            if reward["reward"] > 0:
                results.append(data)

    with open(FILTER_DATA_PATH, "w") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_sft(dataset_size=None, is_filtered=False):
    # 初始化wandb
    run_name = f"sft_{dataset_size if dataset_size else 'full'}"
    if is_filtered: run_name += "_filtered"

    wandb.init(
        project="cs336-a5-sft",
        name=run_name,
        config={
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "dataset_size": dataset_size
        }
    )

    # 加载模型和Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 训练用right

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    # 显存优化的关键
    model.gradient_checkpointing_enable()

    # --- 2. 数据处理 ---
    with open(FILTER_DATA_PATH, "r") as f:
        all_data = [json.loads(line) for line in f]

    # 任务 1 的规模放缩
    if dataset_size is not None and dataset_size < len(all_data):
        all_data = all_data[:dataset_size]

    print(f"Training on {len(all_data)} samples...")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = (len(all_data) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
    num_warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    # 获取模型当前所在的设备 (cuda:0)
    device = next(model.parameters()).device

    model.train()
    global_steps = 0
    running_loss = 0
    for epoch in range(EPOCHS):
        random.shuffle(all_data)
        for i in range(0, len(all_data), BATCH_SIZE):
            batch = all_data[i:i + BATCH_SIZE]
            prompts = [b["prompt"] for b in batch]
            outputs = [b["response"] for b in batch]
            tokenizer_data = tokenize_prompt_and_output(
                prompt_strs=prompts,
                output_strs=outputs,
                tokenizer=tokenizer
            )

            input_ids = tokenizer_data["input_ids"].to(device)
            labels = tokenizer_data["labels"].to(device)
            response_mask = tokenizer_data["response_mask"].to(device)

            # 计算 loss
            log_probs = get_response_log_probs(model=model, input_ids=input_ids, labels=labels)["log_probs"]
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=GRAD_ACCUM,
            )
            running_loss += loss.item()

            if (global_steps + 1) % GRAD_ACCUM == 0:
                # --- 核心要求：梯度裁剪 ---
                clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 这里的 running_loss 已经是这一组累积步的平均值了
                wandb.log({"train/loss": running_loss, "step": (global_steps + 1) // GRAD_ACCUM})
                running_loss = 0
                """
                metric = log_generations(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_strs=prompts,
                    output_strs=outputs,
                    generation_config={
                        "max_new_tokens": 1024,  # 足够写完 MATH 题目的推导
                        "do_sample": True,  # 开启采样以获得多样性
                        "temperature": 0.7,  # 适中的随机性
                        "top_p": 0.9,  # 过滤长尾无效词
                        "repetition_penalty": 1.1,  # 轻微惩罚重复
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    }
                )
                wandb.log(metric)
                """

            global_steps += 1

        checkpoint_dir = f"{CHECKPOINTS_PATH}/{run_name}_epoch_{epoch + 1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    wandb.finish()


def evaluate(run_name):
    results = []
    results_path = f"{VAL_RESULT_PATH}/results.jsonl"
    for epoch in range(EPOCHS):
        checkpoint_dir = os.path.abspath(f"{CHECKPOINTS_PATH}/{run_name}_epoch_{epoch + 1}")
        accuracy, format_accuracy, answer_accuracy = \
            load_model(checkpoint_dir, VAL_DATA_PATH, f"{VAL_RESULT_PATH}/{run_name}_epoch_{epoch + 1}_val_result.json")
        results.append({
            "name": f"{run_name}_epoch_{epoch + 1}",
            "accuracy": accuracy,
            "format_accuracy": format_accuracy,
            "answer_accuracy": answer_accuracy,
        })
        gc.collect()
        torch.cuda.empty_cache()

    with open(results_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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


if __name__ == "__main__":
    # sft_filter()
    # run_sft()
    evaluate("sft_full")

