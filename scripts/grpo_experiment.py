import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import wandb
import json
import random

from sft import tokenize_prompt_and_output, get_response_log_probs
from math_baseline import format_prompt
from grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from cs336_alignment.drgrpo_grader import my_reward_fn


def sync_weights_vllm(train_policy, vllm_generator):
    """将 GPU 1 的最新权重同步到 GPU 0 的 vLLM 模型中"""
    # 1. 获取训练好的 state_dict
    state_dict = train_policy.state_dict()

    # 2. 找到 vLLM 内部的模型实例
    # 路径通常是: llm_engine -> model_executor -> driver_worker -> model_runner -> model
    vllm_model = vllm_generator.llm_engine.model_executor.driver_worker.model_runner.model

    with torch.no_grad():
        for name, param in vllm_model.named_parameters():
            if name in state_dict:
                # 跨显卡拷贝: GPU 1 -> GPU 0
                # .to(device_rollout) 会触发 PCIe 传输
                param.copy_(state_dict[name].to(device_rollout))


# --- 1. 设备与超参数配置 ---
device_rollout = "cuda:0"
device_train = "cuda:1"
n_grpo_steps = 3
learning_rate = 3e-5
advantages_eps = 1e-6
rollout_batch_size = 256
group_size = 16
seed = 42
top_p = 0.9
sampling_temperature = 0.8
sampling_min_tokens = 4
sampling_max_tokens = 1024
epochs_per_rollout_batch = 1
train_batch_size = 256
gradient_accumulation_steps = 32
gpu_memory_utilization = 0.8
cliprange = 0.2
loss_type = "grpo_clip"
use_std_normalization = True
save_iter = 50
model_id = "/root/autodl-tmp/cs336_assignment5/checkpoints/math_sft_best"
# model_id = "Qwen/Qwen2.5-Math-1.5B"
train_data_path = "/root/autodl-tmp/cs336_assignment5/data/MATH/train.jsonl"
valid_data_path = "/root/autodl-tmp/cs336_assignment5/data/MATH/validation.jsonl"
save_path = "/root/autodl-tmp/cs336_assignment5/checkpoints"
os.makedirs(save_path, exist_ok=True)

with open(train_data_path, "r") as f:
    all_data = [json.loads(line) for line in f]


# --- 2. 初始化双卡模型 ---
# GPU 0: vLLM 推理引擎
# 建议调低 gpu_memory_utilization，防止 vLLM 抢占过多导致同步时 OOM
generator = LLM(
    model=model_id,
    device=device_rollout,
    dtype="bfloat16",
    gpu_memory_utilization=gpu_memory_utilization
)

sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=top_p,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        seed=seed,
        stop=["</answer>"],
        include_stop_str_in_output=True
)

# GPU 1: PyTorch 训练策略
policy = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device_train,
    attn_implementation="flash_attention_2"
)

# 显存优化的关键
policy.gradient_checkpointing_enable()

# 加载模型和Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 训练用right

wandb.init(
    project="math-grpo-r1-zero",  # 项目名称
    name=f"grpo-rollout256-lr3e-5-gs8",  # 实验名称，建议包含核心超参数
    config={
        "learning_rate": learning_rate,
        "rollout_batch_size": rollout_batch_size,
        "group_size": group_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "cliprange": cliprange,
        "model_id": model_id,
    }
)

# 优化器
optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))

# 计算每一轮 Rollout 需要多少个不同的原始题目
n_prompts_per_rollout = rollout_batch_size // group_size
total_training_steps = n_grpo_steps * (len(all_data) // n_prompts_per_rollout) * epochs_per_rollout_batch
warmup_steps = int(0.03 * total_training_steps)  # 3% 的步数用来预热

scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps
)

global_step = 0

for step in range(n_grpo_steps):
    # 【阶段 A: Rollout - GPU 0】
    # 1. 抽样 Prompts
    random.shuffle(all_data)
    current_epoch_data = all_data[: (len(all_data) // n_prompts_per_rollout) * n_prompts_per_rollout]
    for i in range(0, len(current_epoch_data), n_prompts_per_rollout):
        batch = all_data[i:i+n_prompts_per_rollout]
        base_prompts = [format_prompt(b["problem"]) for b in batch]
        prompts = [p for p in base_prompts for _ in range(group_size)]
        base_truths = [b["answer"] for b in batch]
        truths = [t for t in base_truths for _ in range(group_size)]

        outputs = generator.generate(prompts, sampling_params=sampling_params)
        all_responses = [output.outputs[0].text for output in outputs]
        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn=my_reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=truths,
            group_size=group_size,
            advantage_eps=advantages_eps,
            normalize_by_std=use_std_normalization
        )

        """
        # 创建一个 WandB 表格
        table = wandb.Table(columns=["Step", "Prompt", "Response", "Reward"])
        # 随机取当前 batch 里的第一个
        table.add_data(step, prompts[0], all_responses[0], raw_rewards[0].item())
        wandb.log({"samples": table})
        """

        tokenizer_data = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=all_responses,
            tokenizer=tokenizer
        )

        input_ids = tokenizer_data["input_ids"].to(device_train)
        labels = tokenizer_data["labels"].to(device_train)
        response_mask = tokenizer_data["response_mask"].to(device_train)

        policy.eval()
        all_old_lp = []
        eval_micro_batch_size = train_batch_size // gradient_accumulation_steps

        with torch.no_grad():
            for m_idx in range(gradient_accumulation_steps):
                m_start = m_idx * eval_micro_batch_size
                m_end = m_start + eval_micro_batch_size

                m_input = input_ids[m_start:m_end]
                m_labels = labels[m_start:m_end]
                m_lp = get_response_log_probs(model=policy, input_ids=m_input, labels=m_labels)["log_probs"]
                all_old_lp.append(m_lp.detach())

        old_log_probs = torch.cat(all_old_lp, dim=0)

        policy.train()
        for epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()

            # 计算微批次大小: 256 / 128 = 2
            micro_batch_size = train_batch_size // gradient_accumulation_steps
            cumulative_kl = 0
            total_loss = 0
            for mb_idx in range(gradient_accumulation_steps):
                start = mb_idx * micro_batch_size
                end = start + micro_batch_size

                mb_input_ids = input_ids[start:end]
                mb_labels = labels[start:end]
                mb_mask = response_mask[start:end]
                mb_adv = advantages[start:end].unsqueeze(1).to(device_train)
                mb_raw_rewards = raw_rewards[start:end].to(device_train)
                mb_old_lp = old_log_probs[start:end]

                mb_current_lp = get_response_log_probs(
                    model=policy,
                    input_ids=mb_input_ids,
                    labels=mb_labels)["log_probs"]

                with torch.no_grad():
                    mb_kl = (mb_old_lp - mb_current_lp).mean().item()
                    cumulative_kl += mb_kl

                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=mb_current_lp,
                    response_mask=mb_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                    cliprange=cliprange,
                    constant_normalizer=sampling_max_tokens)
                total_loss += loss.item()

                # --- 新增 DEBUG 打印位置 ---
                # print(f"  mb_current_lp requires_grad: {mb_current_lp.requires_grad}")
                """
                if mb_idx == 0:  # 每一轮只打第一个微批次，不然刷屏
                    print(f"\n[DEBUG Step {global_step}]")
                    print(f"  Advantage Mean: {mb_adv.mean().item():.4f}")
                    print(f"  Mask Sum (Active Tokens): {mb_mask.sum().item()}")
                    print(f"  Old LogProb Mean: {mb_old_lp.mean().item():.4f}")
                    print(f"  Current LogProb Mean: {mb_current_lp.mean().item():.4f}")
                    print(f"  First 4 Advantages: {mb_adv[:4].flatten().tolist()}")  # 看具体数值！
                    print(f"  First 4 LogProbs: {mb_current_lp[0, :4].tolist()}")
                    print(f"  Loss requires_grad: {loss.requires_grad}")
                """
            # 梯度剪裁与参数更新
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            current_lr = scheduler.get_last_lr()[0]

            wandb.log({
                "step": step,
                "train/loss": total_loss / gradient_accumulation_steps,
                "train/grad_norm": grad_norm,
                "train/kl": cumulative_kl / gradient_accumulation_steps,
                "train/learning_rate": current_lr,
                "reward/raw_mean": raw_rewards.mean().item(),
                "reward/max": raw_rewards.max().item(),
                "reward/min": raw_rewards.min().item(),
                "reward/std": raw_rewards.std().item(),
            }, step=global_step)

            if global_step % save_iter == 0 or global_step == total_training_steps:
                checkpoint_dir = f"{save_path}/math_grpo_{global_step}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                policy.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                # 打印日志
                print(f"[System] Checkpoint saved. Current Reward Mean: {raw_rewards.mean().item():.4f}")

        # 训练完一个 Rollout Batch 后，必须把 GPU 1 的参数同步给 GPU 0 的 vLLM
        sync_weights_vllm(train_policy=policy, vllm_generator=generator)
        # 释放显存给下一次 Rollout
        torch.cuda.empty_cache()


