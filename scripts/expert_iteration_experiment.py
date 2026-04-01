from vllm import SamplingParams

sampling_temperature = 0.7
top_p = 0.9
sampling_min_tokens = 4
sampling_max_tokens = 1024
G = 32
seed = 42
sampling_params = SamplingParams(
    temperature=sampling_temperature,
    top_p=top_p,
    max_tokens=sampling_max_tokens,
    min_tokens=sampling_min_tokens,
    n=G,
    seed=seed,
)

