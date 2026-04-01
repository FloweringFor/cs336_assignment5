import os
import json
from vllm import LLM, SamplingParams
from typing import Callable, List, Any

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


R1_ZERO_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. "
    "The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> "
    "and answer is enclosed within <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)


def format_prompt(question):
    return f"{R1_ZERO_SYSTEM_PROMPT}\n\nUser: {question}\nAssistant: <think>"


def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        ground_truths: List[str],
        eval_sampling_params: SamplingParams,
        results_path
) -> tuple[float | Any, float | Any, float | Any]:
    # 1、调用 vLLM 批量生成答案
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    for output, gold_answer in zip(outputs, ground_truths):
        generated_text = "<think>" + output.outputs[0].text
        # 2、调用 reward_fn 判定对错
        reward_dict = reward_fn(generated_text, gold_answer)
        is_correct = reward_dict.get("reward", 0.0)
        format_correct = reward_dict.get("format_reward", 0.0)
        answer_correct = reward_dict.get("answer_reward", 0.0)

        results.append({
            "prompt": output.prompt,
            "prediction": generated_text,
            "gold_answer": gold_answer,
            "correct": is_correct,
            "format_correct": format_correct,
            "answer_correct": answer_correct
        })

    # 3、计算得分并存盘
    accuracy = sum(r["correct"] for r in results) / len(results)
    format_accuracy = sum(r["format_correct"] for r in results) / len(results)
    answer_accuracy = sum(r["answer_correct"] for r in results) / len(results)
    print("-" * 30)
    print(f"Zero-shot Accuracy: {accuracy:.2%}")
    print(f"Zero-shot Format Accuracy: {format_accuracy:.2%}")
    print(f"Zero-shot Answer Accuracy: {answer_accuracy:.2%}")
    print("-" * 30)

    """
    # 4、序列化到磁盘 (JSONL 格式)
    # 确保存储目录存在
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)

    with open(results_path, "w", encoding="utf-8") as f:
        for entry in results:
            # ensure_ascii=False 保证数学公式里的特殊字符不乱码
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ 结果已成功保存至: {results_path}")
    """
    return accuracy, format_accuracy, answer_accuracy


def read_json(path):
    prompts = []
    ground_truths = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(format_prompt(data["problem"]))
            ground_truths.append(data["answer"])
    return prompts, ground_truths


def load_model(model_id, json_path, save_path):
    prompts, ground_truths = read_json(json_path)
    model = LLM(model=model_id, device="auto", dtype="bfloat16", gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    return evaluate_vllm(model, r1_zero_reward_fn, prompts, ground_truths, sampling_params, save_path)


if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    read_path = "../data/MATH/validation.jsonl"
    save_path = "../data/MATH/results_validation.jsonl"
    load_model(model_id, read_path, save_path)
