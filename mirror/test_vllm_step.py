import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['VLLM_USE_V1'] = '0'  
from vllm import LLM
import torch

if __name__ == "__main__":
    llm = LLM(model="/root/.cache/modelscope/hub/models/facebook/opt-125m")

    def print_first_ten_params(model):
        print("\n前10个模型参数:")
        for i, (name, param) in enumerate(model.named_parameters()):
            if i < 10:
                print(f"{name}: {param.data[:10]}")
            else:
                break
                
    llm.apply_model(print_first_ten_params)

    from torch.optim import Adam

    def attach_optimizer(model):
        # 只在第一次时创建并绑定
        if not hasattr(model, "optimizer"):
            model.optimizer = Adam(model.parameters(), lr=1)
            print("optimizer attached to model")
        else:
            print("optimizer already exists")
        return model.optimizer

    llm.apply_model(attach_optimizer)

    optimizer = llm.apply_model(attach_optimizer)

    def apply_grads_and_step(model, scale=100.0):
        grads = {}
        for name, param in model.named_parameters():
            grad = torch.randn_like(param.data) * scale
            grads[name] = grad
        # 赋值梯度并优化
        for name, param in model.named_parameters():
            if name in grads:
                param.grad = grads[name]
        model.optimizer.step()
        model.optimizer.zero_grad()
        
    llm.apply_model(lambda m: apply_grads_and_step(m, scale=100.0))

    llm.apply_model(print_first_ten_params)

