import json
import os
from collections import Counter

import torch
from vllm import SamplingParams, LLM
import torch.distributed as dist

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from math_baseline import format_prompt, read_train_val_json, evaluate_vllm


def generate_level_dataset(file_path, level, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{level}.jsonl"
    extract_count = 0
    with open(file_path, "r") as f, open(save_path, "w") as f_out:
        for line in f:
            item = json.loads(line)
            if item["level"] == level:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                extract_count += 1
    print(f"✅ 提取完成！{level} 共计 {extract_count} 条数据。")
    print(f"📂 文件已存至: {save_path}")


def generate_subject_dataset(file_path, subject, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{subject}.jsonl"
    extract_count = 0
    with open(file_path, "r") as f, open(save_path, "w") as f_out:
        for line in f:
            item = json.loads(line)
            if item["subject"] == subject:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                extract_count += 1
    print(f"✅ 提取完成！{subject} 共计 {extract_count} 条数据。")
    print(f"📂 文件已存至: {save_path}")


def generate_sft_dataset(model, prompts, answers, sampling_params, max_answer_count, file_path):
    results = model.generate(prompts=prompts, sampling_params=sampling_params)

    sft_sample = []
    success_count = 0
    for result, answer in zip(results, answers):
        question = result.prompt  # 拿到题目
        seen_responses = set()
        answer_count = 0
        for candidate in result.outputs:
            # 拿到模型生成的第 i 条推理过程
            reasoning_path = candidate.text
            if r1_zero_reward_fn(reasoning_path, answer)['reward'] > 0.0:
                if reasoning_path not in seen_responses and answer_count < max_answer_count:
                    sft_sample.append({
                        "prompt": format_prompt(question),
                        "response": reasoning_path,
                        "ground_truth": answer
                    })
                    seen_responses.add(reasoning_path)
                    answer_count += 1

        if len(seen_responses) > 0:
            success_count += 1
    with open(file_path, "a") as f:
        for entry in sft_sample:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"本轮迭代：{len(prompts)} 题中解决了 {success_count} 题，一共生成 {len(sft_sample)} 条有效数据")


def analysis_file(file_path):
    # file_path = "/root/autodl-tmp/cs336_assignment5/data/MATH/train.jsonl"
    level_record = Counter()  # 自动处理初始值 0 的逻辑
    subject_record = Counter()

    with open(file_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                # 统计各级别的数量
                lv = item.get("level", "Unknown")
                sub = item.get("subject", "Unknown")
                level_record[lv] += 1
                subject_record[sub] += 1
            except json.JSONDecodeError:
                continue  # 跳过损坏的行

    # 打印一份漂亮的报告
    print(f"{'难度级别':<10} | {'题目数量':<10}")
    print("-" * 25)
    for lv, count in sorted(level_record.items()):
        print(f"{str(lv):<10} | {count:<10}")

    print(f"{'科目类型':<20} | {'题目数量':<20}")
    print("-" * 25)
    for sub, count in sorted(subject_record.items()):
        print(f"{str(sub):<20} | {count:<20}")

    return dict(level_record), dict(subject_record)


def test_model(model, sampling_params, levels, subjects, file_dir):
    os.makedirs(file_dir, exist_ok=True)
    files = []
    for level in levels:
        file_path = f"{file_dir}/levels/{level}.jsonl"
        files.append(file_path)
    for subject in subjects:
        file_path = f"{file_dir}/subjects/{subject}.jsonl"
        files.append(file_path)
    for path in files:
        prompts, ground_truths = read_train_val_json(path)
        print("测试：", path)
        evaluate_vllm(
            vllm_model=model,
            reward_fn=r1_zero_reward_fn,
            prompts=prompts,
            ground_truths=ground_truths,
            eval_sampling_params=sampling_params,
        )


def generate_all_data(model, sampling_params, levels, subjects, file_dir, save_dir):
    os.makedirs(file_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    """
    save_path = f"{save_dir}/train_levels_sft.jsonl"
    if os.path.exists(save_path):
        os.remove(save_path)
    for level in levels:
        if level in ["Level 1", "Level 2"]:
            sampling_params.n = 8
            max_answer_count = 1
        elif level == "Level 3":
            sampling_params.n = 16
            max_answer_count = 2
        elif level == "Level 4":
            sampling_params.n = 32
            max_answer_count = 3
        else:
            sampling_params.n = 64
            max_answer_count = 4
        data_path = f"{file_dir}/levels/{level}.jsonl"
        prompts, ground_truths = read_train_val_json(data_path)
        print(f"🚀 开始采样 {level}: 题目数={len(prompts)}, n={sampling_params.n}, max_save={max_answer_count}")
        generate_sft_dataset(model=model, prompts=prompts, answers=ground_truths, sampling_params=sampling_params,
                             max_answer_count=max_answer_count, file_path=save_path)
        torch.cuda.empty_cache()
    """
    save_path = f"{save_dir}/train_subjects_sft.jsonl"
    if os.path.exists(save_path):
        os.remove(save_path)
    for subject in subjects:
        if subject in ["Algebra", "Prealgebra"]:
            sampling_params.n = 8
            max_answer_count = 1
        elif subject in ["Counting & Probability", "Number Theory"]:
            sampling_params.n = 16
            max_answer_count = 2
        elif subject in ["Geometry"]:
            sampling_params.n = 32
            max_answer_count = 3
        else:
            sampling_params.n = 64
            max_answer_count = 4
        data_path = f"{file_dir}/subjects/{subject}.jsonl"
        prompts, ground_truths = read_train_val_json(data_path)
        print(f"🚀 开始采样 {subject}: 题目数={len(prompts)}, n={sampling_params.n}, max_save={max_answer_count}")
        generate_sft_dataset(model=model, prompts=prompts, answers=ground_truths, sampling_params=sampling_params,
                             max_answer_count=max_answer_count, file_path=save_path)


if __name__ == "__main__":
    base_model_id = "/root/autodl-tmp/cs336_assignment5/checkpoints/math_sft_best"
    best_model_id = "/root/autodl-tmp/cs336_assignment5/checkpoints/ei_best_8"
    sampling_temperature = 0.8
    top_p = 0.9
    sampling_min_tokens = 4
    sampling_max_tokens = 1024
    G = 32
    seed = 42
    n_ei_step = 10

    model = LLM(model=best_model_id, device="auto", dtype="bfloat16", gpu_memory_utilization=0.85)

    sampling_params_test = SamplingParams(
        temperature=sampling_temperature,
        top_p=top_p,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        seed=seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    sampling_params_generate = SamplingParams(
        temperature=sampling_temperature,
        top_p=top_p,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=G,
        seed=seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    file_dir = "/root/autodl-tmp/cs336_assignment5/data/MATH/val"
    levels = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
    subjects = [
        "Algebra",
        "Counting & Probability",
        "Geometry",
        "Intermediate Algebra",
        "Number Theory",
        "Prealgebra",
        "Precalculus"]

    test_model(
        model=model,
        sampling_params=sampling_params_test,
        levels=levels,
        subjects=subjects,
        file_dir=file_dir
    )
    """
    save_dir = "/root/autodl-tmp/cs336_assignment5/data/MATH/train"

    generate_all_data(
        model=model,
        sampling_params=sampling_params_generate,
        levels=levels,
        subjects=subjects,
        file_dir=file_dir,
        save_dir=save_dir
    )
    """
    if dist.is_initialized():
        dist.destroy_process_group()
