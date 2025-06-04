import ray
import os
import time
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams

# --------------------------
# 强制使用 V1 引擎
# --------------------------
os.environ["VLLM_USE_V1"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 单GPU
ray.init()

# --------------------------
# 显式创建 PlacementGroup
# --------------------------
pg = placement_group([{"CPU": 0, "GPU": 1}]*2)
ray.get(pg.ready())  # ✅ 等待 placement group 准备好

strategy = PlacementGroupSchedulingStrategy(
    placement_group=pg,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# --------------------------
# 创建 vLLM LLM actor
# --------------------------
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,  # 由 PG 控制
    scheduling_strategy=strategy,
)(LLM).remote(
    model="/root/.cache/modelscope/hub/models/facebook/opt-125m",
    tokenizer="/root/.cache/modelscope/hub/models/facebook/opt-125m",
    enforce_eager=True,  # 禁用 CUDA graph 以避免 async output 错误
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
    dtype="float16",  # 可选：节省内存
)

# --------------------------
# 推理测试
# --------------------------
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

# 可选延迟确保加载完成
time.sleep(5)

outputs = ray.get(llm.generate.remote(prompts, sampling_params))

print("-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)
