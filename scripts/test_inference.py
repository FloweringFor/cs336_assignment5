from vllm import LLM, SamplingParams

# 1. 初始化模型
# dtype="bfloat16" 是 3090 的标配，既省显存又保精度
model_id = "Qwen/Qwen2.5-Math-1.5B"
llm = LLM(model=model_id, dtype="bfloat16", device="cuda")

# 2. 设置推理参数
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

# 3. 准备测试数学题
prompts = ["Calculate the integral of x^2 from 0 to 3."]

# 4. 生成结果
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
