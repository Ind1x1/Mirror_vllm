import ray
from vllm import LLM, SamplingParams
import torch
import os
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import time

from vllm.utils import get_ip, get_open_port

class LLMRayActor(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['VLLM_USE_V1'] = '0'  
ray.init()
pg_inference = placement_group([{"GPU": 1, "CPU": 0}])
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(LLMRayActor).remote(
    model="/root/.cache/modelscope/hub/models/facebook/opt-125m",
    enforce_eager=True,
    worker_extension_cls="test_ray_utils.WorkerExtension",
    tensor_parallel_size=1,
    distributed_executor_backend="ray",
)


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)
time.sleep(20)
outputs = ray.get(llm.generate.remote(prompts, sampling_params))

print("-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)

result_ref = llm.collective_rpc.remote("create_ops", args=("adam"))
print(ray.get(result_ref))

result_ref = llm.collective_rpc.remote("print_params", args=("adam"))
print(ray.get(result_ref))

result_ref = llm.collective_rpc.remote("step", args=("adam"))
print(ray.get(result_ref))

result_ref = llm.collective_rpc.remote("print_params", args=("adam"))
print(ray.get(result_ref))

sampling_params = SamplingParams(temperature=0)
time.sleep(20)
outputs = ray.get(llm.generate.remote(prompts, sampling_params))

print("-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)