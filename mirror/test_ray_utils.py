# SPDX-License-Identifier: Apache-2.0
import torch


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream()
        )

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated
    
    def create_ops(self, *args, **kwargs):
        from deepspeed.ops.adam import FusedAdam
        import torch 

        model_params = self.model_runner.model.parameters()
        optimizer = FusedAdam(
            model_params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
        self.optimizer = optimizer
        print("success create ops")
        return optimizer
    
    def step(self, *args, **kwargs):
        scale = 100
        grads = {}
        model = self.model_runner.model
        # param = list(model.parameters())[0]
        #print("before:", param.view(-1)[:5].detach().cpu().numpy())

        for name, param in model.named_parameters():
            grad = torch.randn_like(param.data) * scale
            grads[name] = grad
        for name, param in model.named_parameters():
            if name in grads:
                param.grad = grads[name]
                # 打印梯度均值
                # print(f"{name} grad mean: {param.grad.float().mean().item()}")

        self.optimizer.step()
        self.optimizer.zero_grad()

        # param = list(model.parameters())[0]
        # print("after:", param.view(-1)[:5].detach().cpu().numpy())
        return True
        
    # def step(self, *args, **kwargs):
    #     scale = 10
    #     grads = {}
    #     model = self.model_runner.model
    #     for name, param in model.named_parameters():
    #         grad = torch.randn_like(param.data) * scale
    #         grads[name] = grad
    #     # 赋值梯度并优化
    #     for name, param in model.named_parameters():
    #         if name in grads:
    #             param.grad = grads[name]

    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
    #     return True

    def print_params(self, *args, **kwargs):
        model = self.model_runner.model
        print("check params")
        print("-"*50)
        for i, (name, param) in enumerate(model.named_parameters()):
            if i < 10:
                print(f"{name}: {param.data[:10]}")
            else:
                break
        print("-"*50)

class ColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def report_device_id(self) -> str:
        from vllm.platforms import current_platform

        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

    def update_weights_from_ipc_handles(self, ipc_handles):
        handles = ipc_handles[self.device_uuid]
        device_id = self.device.index
        weights = []
        for name, handle in handles.items():
            func, args = handle
            list_args = list(args)
            # the key is to change device id to the current device id
            # in case two processes have different CUDA_VISIBLE_DEVICES
            list_args[6] = device_id
            tensor = func(*list_args)
            weights.append((name, tensor))
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated
