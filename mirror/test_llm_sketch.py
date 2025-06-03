import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['VLLM_USE_V1'] = '0'  
from vllm import LLM
import torch

if __name__ == "__main__":
    # 加载模型（会自动下载到本地缓存）
    llm = LLM(model="/root/.cache/modelscope/hub/models/facebook/opt-125m")

    # 打印vLLM模型结构
    print("vLLM model structure:")
    # 打印模型结构  、、
    # 方法4: 打印模型参数信息  
    def print_model_params(model):  
        print(f"\nModel parameters:")  
        total_params = sum(p.numel() for p in model.parameters())  
        print(f"Total parameters: {total_params:,}")  
    
    # 方法3: 打印模型的详细层级结构  
    def print_model_structure(model):  
        print("\nDetailed model structure:")  
        for name, module in model.named_modules():  
            print(f"{name}: {module.__class__.__name__}")  
    
    llm.apply_model(print_model_structure)
    llm.apply_model(print_model_params)
    # llm.apply_model(lambda model: print(type(model)))

    def print_state_dict(model):
        print("\nvLLM state_dict:")
        total_params = 0
        for name, param in model.state_dict().items():
            print(f"{name}: {param.shape}")
            total_params += param.numel()
        print(f"\n总参数量: {total_params:,}")

    llm.apply_model(print_state_dict)

    def create_random_grads(model):
        grads = {}
        for name, param in model.named_parameters():
            # 创建与参数同shape、同dtype、同device的随机梯度
            grad = torch.randn_like(param.data)
            grads[name] = grad
            print(f"{name}: grad shape={grad.shape}, dtype={grad.dtype}, device={grad.device}")
        return grads
    
    grads = llm.apply_model(create_random_grads)