"""
Average metric code taken from: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L347-L406
MFU code adapted from: https://github.com/karpathy/nanoGPT/blob/master/train.py
"""
import os
from enum import Enum

import torch
import torch.distributed as dist

from utils import Registry
from training.constants import LossConstants


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMetric(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE, process_group=None):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.process_group = process_group
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            local_rank = os.environ["LOCAL_RANK"]
            device = torch.device(f"cuda:{local_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False, group=self.process_group)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def get_avg(self, reset=False):
        self.all_reduce()
        avg = self.avg
        if reset:
            self.reset()
        return avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ModelFlopsUtilization(AverageMetric):

    GPU_TYPE_TO_MAX_FLOPS_PER_SEC = {
        # for H100 see page 20 https://resources.nvidia.com/en-us-tensor-core?_gl=1*ighl1e*_gcl_au*OTczOTM3NTMzLjE3MTcxOTYwNjk.
        # H100 GPU with bfloat16, this is the SXM version
        "H100_SXM_bf16": 989.4e12,
        # H100 GPU with bfloat16, this is the PICe version
        "H100_PCIe_bf16": 756e12,
        # V100 GPU with FP 32
        "V100_fp32": 14e12,
        # A100 GPU with FP 32
        "A100_fp32": 19.5e12,
        # A100 GPU with tensor float 32 peak flops is 156 TFLOPS
        "A100_tf32": 156e12,
        # A100 GPU bfloat16 peak flops is 312 TFLOPS
        "A100_bf16": 312e12,
        # 0 result for CPU
        "cpu": float("-inf"),
    }

    def __init__(
        self,
        n_params,
        seq_len,
        n_layers,
        n_heads,
        d_head,
        gpu_type,
        n_gpus_per_replica=1,
        process_group=None,
        validate_gpu=True,
    ) -> None:

        super().__init__("ModelFlopsUtilization", process_group=process_group)
        if validate_gpu:
            self._validate_gpu_type(gpu_type)
        self.seq_len = seq_len
        self.max_gpu_flops_per_sec = self.GPU_TYPE_TO_MAX_FLOPS_PER_SEC[gpu_type] * n_gpus_per_replica
        self.model_flops_per_token = 6 * n_params + 12 * n_layers * n_heads * d_head * seq_len

    @classmethod
    def from_config(cls, config, process_group=None):
        top_k_experts = 1
        n_mlp_layers = 2
        is_moe = hasattr(config.model, "moe")

        if is_moe:
            top_k_experts = config.model.moe.top_k

        if is_moe and config.model.moe.mlp_type == "glu":
            n_mlp_layers = 3
        elif hasattr(config.model, "mlp") and config.model.mlp[-3:] == "glu":
            n_mlp_layers = 3

        kwargs = {
            "seq_len": config.model.max_len,
            "n_layers": config.model.n_layers,
            "n_heads": config.model.n_heads,
            "d_head": config.model.d_model // config.model.n_heads,
            "gpu_type": config.compute.gpu_type,
            "process_group": process_group,
            "n_params": cls.n_params_in_model(
                n_layers=config.model.n_layers,
                n_mlp_layers=n_mlp_layers,
                d_model=config.model.d_model,
                d_hidden=config.model.d_hidden,
                n_heads=config.model.n_heads,
                n_kv_heads=config.model.n_kv_heads,
                vocab_size=config.model.vocab_size,
                top_k_experts=top_k_experts,
            ),
        }

        if hasattr(config.model, "num_pipeline_stages"):
            kwargs["n_gpus_per_replica"] = config.model.num_pipeline_stages

        return cls(**kwargs)

    @classmethod
    def n_params_in_model(cls, n_layers, n_mlp_layers, d_model, d_hidden, n_heads, n_kv_heads, vocab_size, top_k_experts):
        n_params = 0

        # input embedding params
        n_params += d_model * vocab_size

        # params per layer
        d_head = d_model // n_heads
        n_norm_params = 2 * d_model
        n_attn_params = 2 * d_model * d_model + 2 * d_model * n_kv_heads * d_head
        n_mlp_params = n_mlp_layers * d_model * d_hidden * top_k_experts
        n_layer_params = 2 * n_norm_params + n_attn_params + n_mlp_params
        n_params += n_layers * n_layer_params

        # output layer params
        n_params += d_model * vocab_size
        return n_params

    def get_avg(self, reset=False):
        # get the avg. throughput then compare against max_throughput
        avg_throughput = super().get_avg(reset)
        flops_per_sec = avg_throughput * self.model_flops_per_token
        avg_mfu = flops_per_sec / self.max_gpu_flops_per_sec
        return avg_mfu, avg_throughput

    def update(self, batch_size, n_iters, iter_secs):
        n_tokens = batch_size * self.seq_len * n_iters
        throughput = n_tokens / iter_secs
        super().update(throughput, iter_secs)

    def _validate_gpu_type(self, gpu_type):
        if gpu_type not in self.GPU_TYPE_TO_MAX_FLOPS_PER_SEC:
            raise ValueError(f"GPU type {gpu_type} unsupported. Must be in {self.GPU_TYPE_TO_MAX_FLOPS_PER_SEC.keys()}")
        elif gpu_type != "cpu":
            gpu_name = gpu_type.split("_")[0].lower()
            device_name = torch.cuda.get_device_name(0).lower()
            if gpu_name not in device_name:
                raise ValueError(f"GPU type {gpu_type} does not match the device 0 name {device_name}")


@Registry.register("accuracy")
class Accuracy(AverageMetric):

    def __init__(self, name, process_group=None):
        super().__init__(name, process_group)

    def update(self, logits_dict, target_dict, ignore_index=LossConstants.IGNORE_INDEX):
        """
        TODO: need to test this now that we re-factored to make sure we didn't mess it up
        """
        logits = logits_dict["input"]
        n = logits.shape[0]
        labels = target_dict["target"]
        predictions = logits.argmax(dim=-1)
        mask = labels != ignore_index
        correct = torch.sum(predictions[mask] == labels[mask]).item()
        total = mask.sum().item()
        accuracy = correct / total
        super().update(accuracy, n)

