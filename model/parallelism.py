from enum import Enum
from functools import partial
from contextlib import ExitStack

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import ShardingStrategy, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper

from utils import Registry, Precision
from model.initialization import prepare_model_for_fsdp_with_meta_device, validate_no_params_on_meta_device

try:
    from model.te_model import prepare_te_modules_for_fsdp
except ModuleNotFoundError as e:
    print(f"Failed to import Transformer Engine: {e}")


class ParallelismTypes(Enum):

    FULLY_SHARDED_DATA_PARALLEL = "fsdp"
    DISTRIBUTED_DATA_PARALLEL = "ddp"
    PIPELINE_DATA_PARALLEL = "pdp"

    @classmethod
    def is_parallelism_type(cls, parallelism_type):
        contains_value = any(parallelism_type == item.value for item in cls)
        contains_elem = any(parallelism_type == item for item in cls)
        return contains_value or contains_elem

    def __eq__(self, other):
        if isinstance(other, Enum):
            return super().__eq__(other)
        else:
            return self.value == other


@Registry.register("ddp_model", "from_config")
class DDPModel(DDP):

    @classmethod
    def from_config(cls, config):
        model_cls = Registry.get(config.model.id)
        model = model_cls(config.model)

        model.to(config.local_rank)

        if config.model.do_compile:
            model = torch.compile(model)

        return cls(model, device_ids=[config.local_rank])


@Registry.register("fsdp_model", "from_config")
class FSDPModel(FSDP):
    """
    Wrapper around FSDP that allows for loading from a config
    """

    @classmethod
    def from_config(cls, config):
        model_cls = Registry.get(config.model.id)
        with ExitStack() as stack:
            if config.model.use_meta_device:
                stack.enter_context(torch.device("meta"))
            model = model_cls(config.model)

        transformer_layer_cls = model.get_transformer_layer_cls()
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={transformer_layer_cls})
        mixed_precision = Precision.PRECISION_NAME_TO_FSDP_MIXED_PRECISION[config.model.precision]

        model_init_kwargs = {}
        is_fms_model = config.model.id == "fms_llama"
        if config.model.use_meta_device and not is_fms_model:
            assert config.model.initializer is None
            model = prepare_model_for_fsdp_with_meta_device(model)
        elif config.model.use_meta_device and is_fms_model:
            assert config.model.initializer == "fms_init"
            initializer_cls = Registry.get(config.model.initializer)
            init_kwargs = initializer_cls.get_kwargs_from_model_config(config.model)
            model_init_kwargs["param_init_fn"] = partial(initializer_cls.init_params, did_skip_init=True, **init_kwargs)

        if config.model.precision == "pure_bf16":
            model = model.to(Precision.PRECISION_NAME_TO_DTYPE["bf16"])

        model = cls(
           model,
           auto_wrap_policy=auto_wrap_policy,
           mixed_precision=mixed_precision,
           sharding_strategy=ShardingStrategy.HYBRID_SHARD,
           use_orig_params=True,
           device_id=config.local_rank,
           limit_all_gathers=True,
           **model_init_kwargs,
        )

        validate_no_params_on_meta_device(model)

        if config.model.id == "te_llama":
            print("calling prepare for FSDP on TE model")
            prepare_te_modules_for_fsdp(model)

        if config.model.id == "fms_llama":
            print("calling compute freqs cis")
            # we need this post-fsdp call to avoid graph break with torch.compile, until we figure out a better solution.
            model.rot_emb.compute_freqs_cis(
                torch.device("cuda", torch.cuda.current_device()),
                config.model.max_len,
            )
        elif config.model.qk_encoding != "identify" and hasattr(model, "qk_encoding"):
            model.qk_encoding.init_params(torch.device("cuda", torch.cuda.current_device()))

        if config.model.do_offload_activations:
            model = offload_wrapper(model)

        return model

