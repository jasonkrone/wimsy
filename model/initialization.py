from typing import Dict
import math
from itertools import chain
from contextlib import ExitStack

import torch
import torch.nn as nn

try:
    from fms.modules.attention import MultiHeadAttention
    from fms.modules.embedding import WordEmbedding
    from fms.modules.feedforward import GatedLinearUnit
    from fms.modules.layernorm import LayerNormParameterized
except ModuleNotFoundError as e:
    print(f"Failed to import FMS: {e}")

try:
    from model.te_model import TransformerLayerWithPOS
except ModuleNotFoundError as e:
    print(f"Failed to import Transformer Engine: {e}")

from utils import Registry, Config
from model.pos_encoding import PositionalEncoding
from model.normalization import LayerNorm, RMSNorm
from model.model import LMHead


@Registry.register("default_init")
class Initializer(object):

    @classmethod
    def get_kwargs(cls, **kwargs) -> Dict:
        return {"stdev": 0.02}

    @classmethod
    def get_kwargs_from_model_config(cls, config: Config) -> Dict:
        return cls.get_kwargs()

    @classmethod
    def init_params(cls, module: nn.Module, did_skip_init: bool, **kwargs) -> None:
        if did_skip_init:
            # TODO: to empty is recursive so it'll reset all sub-modules
            module.to_empty(device=torch.cuda.current_device())

        with ExitStack() as stack:
            if did_skip_init:
                stack.enter_context(torch.no_grad())
            cls._init_params(module, **kwargs)

    @classmethod
    def _init_params(cls, module: nn.Module, stdev: float) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=stdev, a=-3.0*stdev, b=3.0*stdev)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=stdev, a=-3.0*stdev, b=3.0*stdev)
        elif isinstance(module, LayerNorm) or isinstance(module, RMSNorm):
            module.init_params()
        elif isinstance(module, PositionalEncoding):
            module.init_params()


class Wang2021Init(Initializer):
    """
    MLP output layer init method used in Wang et al. 2021 https://github.com/kingoflolz/mesh-transformer-jax
    """

    @classmethod
    def get_kwargs(cls, n_layers: int, d_model: int) -> Dict:
        return {"mlp_out_stdev": 2.0 / (n_layers * math.sqrt(d_model))}

    @classmethod
    def get_kwargs_from_model_config(cls, config: Config) -> Dict:
        return cls.get_kwargs(config.n_layers, config.d_model)


@Registry.register("small_init")
class SmallInit(Initializer):
    """
    Small init scheme from Nguyen and Salazar (2019) https://arxiv.org/abs/1910.05895
    We apply the small init to both attn and mlp linear layers, whereas the paper
    only applies them to the attn layers.
    """

    @classmethod
    def get_kwargs(cls, d_model: int) -> Dict:
        return {"stdev": math.sqrt(2.0 / d_model * 5)}

    @classmethod
    def get_kwargs_from_model_config(cls, config: Config) -> Dict:
        return cls.get_kwargs(config.d_model)


@Registry.register("gpt_neox_init")
class GPTNeoXInit(Initializer):
    """
    follow the GPT NeoX initialization approach https://arxiv.org/pdf/2204.06745.pdf
    Use small init scheme from Nguyen and Salazar (2019) for all layers except last MLP layer
    Use Wang 2021 init approach for last MLP layer in each transformer block
    """

    @classmethod
    def get_kwargs(cls, n_layers: int, d_model: int, d_hidden: int) -> Dict:
        kwargs = {
            "d_model": d_model,
            "d_hidden": d_hidden,
            "default_stdev": SmallInit.get_kwargs(d_model)["stdev"],
            "mlp_out_stdev": Wang2021Init.get_kwargs(n_layers, d_model)["mlp_out_stdev"],
        }
        return kwargs

    @classmethod
    def get_kwargs_from_model_config(cls, config: Config) -> Dict:
        return cls.get_kwargs(n_layers=config.n_layers, d_model=config.d_model, d_hidden=config.d_hidden)

    @classmethod
    def _init_params(cls, module: nn.Module, default_stdev: float, mlp_out_stdev: float, d_model: int, d_hidden: int) -> None:
        if cls._is_mlp_out_layer(module=module, d_model=d_model, d_hidden=d_hidden):
            nn.init.normal_(module.weight, mean=0.0, std=mlp_out_stdev)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            super()._init_params(module, stdev=default_stdev)

    @classmethod
    def _is_mlp_out_layer(cls, module: nn.Module, d_model: int, d_hidden: int) -> bool:
        return isinstance(module, nn.Linear) and module.in_features == d_hidden and module.out_features == d_model


@Registry.register("te_init")
class TEInit(Initializer):

    @classmethod
    def init_params(cls, module: nn.Module, did_skip_init: bool, **kwargs) -> None:
        if did_skip_init and module in cls._modules_to_init():
            module.to_empty(device=torch.cuda.current_device())

        with ExitStack() as stack:
            if did_skip_init:
                stack.enter_context(torch.no_grad())
            cls._init_params(module, **kwargs)

    @classmethod
    def _modules_to_init(cls):
        return [TransformerLayerWithPOS, nn.Embedding, LMHead]

    @classmethod
    def _init_params(cls, module: nn.Module, stdev: float) -> None:
        if isinstance(module, TransformerLayerWithPOS):
            # MultiheadAttention
            cls._init_attn(module.self_attention, stdev)
            # LayerNormMLP
            cls._init_layernorm_mlp(module.layernorm_mlp, stdev)
            # output layernorm
            if hasattr(module, "layernorm"):
                # RMSLayerNorm
                nn.init.ones_(module.layernorm.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=stdev, a=-2.0*stdev, b=2.0*stdev)
        elif isinstance(module, LMHead):
            super()._init_params(module.W_out, stdev)
            super()._init_params(module.norm, stdev)

    @classmethod
    def _init_attn(cls, module, stdev):
        # QKV
        if hasattr(module, "qkv"):
            nn.init.trunc_normal_(module.qkv.weight, mean=0.0, std=stdev, a=-2.0*stdev, b=2.0*stdev)
            if module.qkv.bias.numel() != 0:
                nn.init.zeros_(module.qkv.bias)
        elif hasattr(module, "layernorm_qkv"):
            nn.init.ones_(module.layernorm_qkv.layer_norm_weight)
            nn.init.trunc_normal_(module.layernorm_qkv.weight, mean=0.0, std=stdev, a=-2.0*stdev, b=2.0*stdev)
            if module.layernorm_qkv.bias.numel() != 0:
                nn.init.zeros_(module.layernorm_qkv.bias)
        # proj
        nn.init.trunc_normal_(module.proj.weight, mean=0.0, std=stdev, a=-2.0*stdev, b=2.0*stdev)
        if module.proj.bias.numel() != 0:
            nn.init.zeros_(module.proj.bias)

    @classmethod
    def _init_layernorm_mlp(cls, module, stdev):
        nn.init.ones_(module.layer_norm_weight)
        # Linear 1
        nn.init.trunc_normal_(module.fc1_weight, mean=0.0, std=stdev, a=-2.0*stdev, b=2.0*stdev)
        if module.fc1_bias.numel() != 0:
            nn.init.zeros_(module.fc1_bias)
        # Linear 2
        nn.init.trunc_normal_(module.fc2_weight, mean=0.0, std=stdev, a=-2.0*stdev, b=2.0*stdev)
        if module.fc2_bias.numel() != 0:
            nn.init.zeros_(module.fc2_bias)


@Registry.register("fms_init")
class FMSInit(Initializer):

    @classmethod
    def get_kwargs(cls, **kwargs) -> Dict:
        return {}

    @classmethod
    def get_kwargs_from_model_config(cls, config: Config) -> Dict:
        return cls.get_kwargs()

    @classmethod
    def init_params(cls, module: nn.Module, did_skip_init) -> None:
        if (
            isinstance(module, MultiHeadAttention)
            or isinstance(module, WordEmbedding)
            or isinstance(module, GatedLinearUnit)
            or isinstance(module, LayerNormParameterized)
        ):
            if did_skip_init:
                module.to_empty(device=torch.cuda.current_device())
            with torch.no_grad():
                module.reset_parameters()


#######################################################################################################################
#                            The following methods were taken from torchtune                                          #
# https://github.com/pytorch/torchtune/blob/ef6e196d8e47e9bc584bc9f7ce836f646443381f/torchtune/utils/_distributed.py  #
#######################################################################################################################


def _dummy_reset_params(x: nn.Module) -> None:
    """
    Dummy method for patching no-op reset_parameters() when using
    FSDP with meta device.
    """
    return


def prepare_model_for_fsdp_with_meta_device(model: nn.Module) -> nn.Module:
    """
    Dynamically define reset_parameters on every submodule of the model. For LoRA models,
    ensure that the FSDP contract of reset_parameters only modifying a module's directly-owned
    parameters is satisfied. More details here: https://github.com/pytorch/pytorch/issues/104187.

    Args:
        model (nn.Module): model class to prepare for usage with FSDP and meta device.

    Returns:
        nn.Module: Model with reset_parameters defined on every submodule.
        In the case of a LoRA model, we override the default reset_parameters of nn.Linear.

    Raises:
        RuntimeError: if model contains submodule with non-callable attribute reset_parameters
    """
    for k, v in model.named_modules():
        # If the module does not have reset_parameters defined, we define
        # a no-op reset_parameters method to satisfy FSDP's contract.
        reset_params = getattr(v, "reset_parameters", None)

        if reset_params is not None and not callable(reset_params):
            raise RuntimeError(
                f"Cannot override existing reset_parameters variable for FSDP init in {k}"
            )

        if reset_params is None:
            v.reset_parameters = _dummy_reset_params.__get__(v)

    return model


def validate_no_params_on_meta_device(model: torch.nn.Module) -> None:
    """
    Utility to validate that model has no params or buffers on meta device.
    If a meta param or buffer is found, an error indicating the param name will
    be raised.

    Args:
        model (nn.Module): model to check for meta params

    Raises:
        RuntimeError: If meta params or buffers exist in model
    """
    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")

