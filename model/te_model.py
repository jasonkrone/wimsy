from typing import Optional, Callable, Tuple, Union, List
import os
import warnings
import collections
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
from torch.distributed.fsdp._traversal_utils import _get_fsdp_states_with_modules

import transformer_engine.pytorch as te
import transformer_engine_extensions as tex
from transformer_engine.pytorch.transformer import DropPath, TransformerLayer
from transformer_engine.pytorch.module import (
    LayerNormLinear,
    Linear,
    LayerNormMLP,
    LayerNorm,
    RMSNorm,
)

from transformer_engine.pytorch.attention import (
    InferenceParams,
    RotaryPositionEmbedding,
    check_set_window_size,
    DotProductAttention,
    _SplitAlongDim,
    apply_rotary_pos_emb,
)
from transformer_engine.pytorch.export import is_in_onnx_export_mode

from transformer_engine.pytorch.constants import (
    AttnTypes,
    LayerTypes,
    AttnMaskTypes,
    AttnBiasTypes,
    dist_group_type,
)

from transformer_engine.pytorch.utils import (
    divide,
    get_default_init_method,
    cast_if_needed,
)

from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
)

from transformer_engine.pytorch.float8_tensor import Float8Tensor

from transformer_engine.pytorch.jit import (
    set_jit_fusion_options,
    warmup_jit_bias_dropout_add_all_dtypes,
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)


"""

https://github.com/NVIDIA/TransformerEngine/blob/main/examples/pytorch/fsdp/fsdp.py

TODO: you've got to prepare the model for 
FSDP b/c some buffers etc. need to get split
"""

from model.pos_encoding import RotaryEncoding
from model.model import InputEmbedding, LMHead, SequentialWithArgs
from model.normalization import RMSNorm as JPKRMSNorm
from utils import Registry, Precision


class MultiheadAttentionWithRope(torch.nn.Module):
    r"""
    Multi-head Attention (MHA), including Query,
    Key, Value and Output projection.

    .. note::

        Argument :attr:`attention_mask` in the `forward` call is only used when
        :attr:`attn_mask_type` includes '"padding"' or `"arbitrary"`.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    kv_channels: int, default = `None`
                number of key-value channels. defaults to
                :attr:`hidden_size` / :attr:`num_attention_heads` if `None`.
    attention_dropout: float, default = 0.1
                      dropout probability for the dropout op during multi-head attention.
    layernorm_epsilon : float, default = 1e-5
                       a value added to the denominator of layer normalization
                       for numerical stability.
    init_method : Callable, default = `None`
                 used for initializing weights of QKV and FC1 weights in the following way:
                 `init_method(weight)`. When set to `None`, defaults to
                 `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing weights of PROJ and FC2 in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    layer_number: int, default = `None`
                 layer number of the current `TransformerLayer` when multiple such modules are
                 concatenated to form a transformer block.
    attn_mask_type: {'no_mask', 'padding', 'causal', 'padding_causal' 'arbitrary'},
                   default = `causal`
                   type of attention mask passed into softmax operation. Overridden by
                   :attr:`attn_mask_type` in the `forward` method. The forward
                   arg is useful for dynamically changing mask types, e.g. a different
                   mask for training and inference. The init arg is useful for cases
                   involving compilation/tracing, e.g. ONNX export.
    window_size: Optional[Tuple[int, int]], default = `None`
                sliding window size for local attention, where query at position i attends to keys
                in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
                + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
                window and causal mask specifically. Similar to :attr:`attn_mask_type`, it can
                be overridden by :attr:`window_size` in `forward` as well.
    num_gqa_groups : int, default = `None`
                         number of GQA groups in the transformer layer.
                         Grouped Query Attention is described in
                         `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                         This only affects the keys and values, not the querys.
                         GQA-1 is equivalent to Multi-Query Attention
                         (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                         is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module is
                             taken post layernorm.
    input_layernorm: bool, default = `False`
                     if set to `True`, layer normalization to the input is applied.
    attention_type: { 'self', 'cross' }, default = 'self'
                   type of attention applied.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    qkv_weight_interleaved : bool, default = `True`
                            if set to `False`, the QKV weight is interpreted as a concatenation of
                            query, key, and value weights along the `0th` dimension. The default
                            interpretation is that the individual `q`, `k`, and `v` weights for each
                            attention head are interleaved. This parameter is set to `False` when
                            using :attr:`fuse_qkv_params=False`.
    bias : bool, default = `True`
          if set to `False`, the transformer layer will not learn any additive biases.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    qkv_format: str, default = `sbhd`
            dimension format for `query_layer`, `key_layer` and `value_layer`,
            {`sbhd`, `bshd`}. `s` stands for the sequence length, `b` batch size,
            `h` the number of heads and `d` head size. `sbhd` and `bshd` formats
            are used for when sequences in a batch are of equal length or padded to
            equal length. Please note that these formats do not reflect how
            tensors `query_layer`, `key_layer`, `value_layer` are laid out in memory.
            For that, please use `_get_qkv_layout` to gain the layout information.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, QKV and FC1 layers are used as Column Parallel
                      whereas PROJ and FC2 is used as Row Parallel as described
                      `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    fuse_qkv_params: bool, default = 'False'
                    if set to `True`, `TransformerLayer` module exposes a single fused
                    parameter for query-key-value. This enables optimizations such as QKV
                    fusion without concatentations/splits and also enables the argument
                    `fuse_wgrad_accumulation`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_channels: Optional[int] = None,
        attention_dropout: float = 0.1,
        layernorm_epsilon: float = 1e-5,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        layer_number: Optional[int] = None,
        attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        num_gqa_groups: Optional[int] = None,
        fuse_wgrad_accumulation: bool = False,
        get_rng_state_tracker: Optional[Callable] = None,
        sequence_parallel: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_bias: bool = False,
        return_layernorm_output: bool = False,
        input_layernorm: bool = False,
        attention_type: str = "self",
        set_parallel_mode: bool = False,
        fuse_qkv_params: bool = False,
        zero_centered_gamma: bool = False,
        qkv_weight_interleaved: bool = True,
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_split_rs: bool = False,
        ub_split_ag: bool = False,
        ub_atomic_gemm_rs: bool = False,
        ub_atomic_gemm_ag: bool = False,
        bias: bool = True,
        normalization: str = "LayerNorm",
        device: Union[torch.device, str] = "cuda",
        qkv_format: str = "sbhd",
        qk_encoding: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.qk_encoding = qk_encoding
        self.qkv_format = qkv_format
        self.attn_mask_type = attn_mask_type
        self.window_size = window_size
        self.window_size = check_set_window_size(attn_mask_type, self.window_size)
        self.layer_number = layer_number
        self.input_layernorm = input_layernorm
        self.attention_type = attention_type
        self.get_rng_state_tracker = get_rng_state_tracker
        self.tp_group = tp_group
        self.return_layernorm_output = return_layernorm_output
        self.params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.num_attention_heads = num_attention_heads
        self.return_bias = return_bias

        kv_channels = kv_channels if kv_channels else (hidden_size // num_attention_heads)

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        if not fuse_qkv_params:
            qkv_weight_interleaved = False
        self.qkv_weight_interleaved = qkv_weight_interleaved

        assert attention_type in AttnTypes, f"attention_type {attention_type} not supported"
        if layer_number is not None:
            assert layer_number > 0, "layer_number must be a positive integer"

        tp_size = tp_size if tp_group is None else get_distributed_world_size(tp_group)
        self.tp_size = tp_size
        self.sequence_parallel = (tp_size > 1) and sequence_parallel

        self.hidden_size_per_attention_head = kv_channels
        self.num_attention_heads_per_partition = divide(num_attention_heads, tp_size)
        self.num_gqa_groups = (
            num_attention_heads if num_gqa_groups is None else num_gqa_groups
        )
        assert (num_attention_heads % self.num_gqa_groups == 0
                ), "The number of attention heads must be divisible by the number of GQA groups!"
        assert (self.num_gqa_groups % tp_size == 0
                ), "The number of GQA groups must be divisible by tensor parallel size!"
        self.num_gqa_groups_per_partition = int(self.num_gqa_groups // tp_size)
        self.hidden_size_kv = int(hidden_size * self.num_gqa_groups // num_attention_heads)

        common_gemm_kwargs = {
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "tp_group": tp_group,
            "tp_size": tp_size,
            "get_rng_state_tracker": get_rng_state_tracker,
            "sequence_parallel": sequence_parallel,
            "params_dtype": self.params_dtype,
            "device": device,
        }

        qkv_parallel_mode = "column" if set_parallel_mode else None

        if self.attention_type == "self":
            parameters_split = None
            if not fuse_qkv_params:
                parameters_split = collections.OrderedDict([
                    ("query", hidden_size),
                    ("key", self.hidden_size_kv),
                    ("value", self.hidden_size_kv),
                ])
            if self.input_layernorm:
                self.layernorm_qkv = LayerNormLinear(
                    hidden_size,
                    hidden_size + 2 * self.hidden_size_kv,
                    eps=layernorm_epsilon,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    return_layernorm_output=return_layernorm_output,
                    parameters_split=parameters_split,
                    zero_centered_gamma=zero_centered_gamma,
                    ub_bulk_wgrad=ub_bulk_wgrad,
                    ub_bulk_dgrad=ub_bulk_dgrad,
                    ub_split_ag=ub_split_ag,
                    normalization=normalization,
                    ub_atomic_gemm_ag=ub_atomic_gemm_ag,
                    ub_name="qkv",
                    **common_gemm_kwargs,
                )
            else:
                self.qkv = Linear(
                    hidden_size,
                    hidden_size + 2 * self.hidden_size_kv,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    parameters_split=parameters_split,
                    **common_gemm_kwargs,
                )
        elif self.attention_type == "cross":
            if self.input_layernorm:
                self.layernorm_query = LayerNormLinear(
                    hidden_size,
                    hidden_size,
                    eps=layernorm_epsilon,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    parameters_split=("query",) if not fuse_qkv_params else None,
                    return_layernorm_output=return_layernorm_output,
                    zero_centered_gamma=zero_centered_gamma,
                    ub_bulk_wgrad=ub_bulk_wgrad,
                    ub_bulk_dgrad=ub_bulk_dgrad,
                    ub_split_ag=ub_split_ag,
                    normalization=normalization,
                    ub_atomic_gemm_ag=ub_atomic_gemm_ag,
                    ub_name="qkv",
                    **common_gemm_kwargs,
                )
            else:
                self.query_layer = Linear(
                    hidden_size,
                    hidden_size,
                    init_method=init_method,
                    bias=bias,
                    return_bias=False,
                    parallel_mode=qkv_parallel_mode,
                    **common_gemm_kwargs,
                )
            self.key_value = Linear(
                hidden_size,
                2 * self.hidden_size_kv,
                init_method=init_method,
                bias=bias,
                return_bias=False,
                parallel_mode=qkv_parallel_mode,
                parameters_split=("key", "value") if not fuse_qkv_params else None,
                **common_gemm_kwargs,
            )

        # Attention.
        self.core_attention = DotProductAttention(
            num_attention_heads,
            kv_channels,
            num_gqa_groups=self.num_gqa_groups,
            attention_dropout=attention_dropout,
            qkv_format=self.qkv_format,
            tp_size=tp_size,
            get_rng_state_tracker=get_rng_state_tracker,
            sequence_parallel=sequence_parallel,
            tp_group=tp_group,
            layer_number=self.layer_number,
            attention_type=self.attention_type,
        )

        # Linear
        self.proj = Linear(
            hidden_size,
            hidden_size,
            init_method=output_layer_init_method,
            bias=bias,
            return_bias=return_bias,
            parallel_mode="row" if set_parallel_mode else None,
            ub_split_rs=ub_split_rs,
            ub_split_ag=ub_split_ag,
            ub_atomic_gemm_rs=ub_atomic_gemm_rs,
            ub_atomic_gemm_ag=ub_atomic_gemm_ag,
            ub_name="proj",
            **common_gemm_kwargs,
        )




    def _allocate_memory(
        self, inference_max_sequence_len: int, batch_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_gqa_groups_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """
        Set the tensor parallel group for the given
        module before executing the forward pass.

        Parameters
        ----------
        tp_group : ProcessGroup, default = `None`
                  tensor parallel process group.
        """
        self.tp_group = tp_group

    def set_context_parallel_group(
        self,
        cp_group: Union[dist_group_type, None],
        cp_global_ranks: List[int],
        cp_stream: torch.cuda.Stream,
    ) -> None:
        """
        Set the context parallel attributes for the given
        module before executing the forward pass.

        Parameters
        ----------
        cp_group : ProcessGroup
                  context parallel process group.
        cp_global_ranks : List[int]
                         list of global ranks in the context group.
        cp_stream : torch.cuda.Stream
                   cuda stream for context parallel execution.
        """
        # Deep iterate but skip self to avoid infinite recursion.
        for index, child in enumerate(self.modules()):
            if index == 0:
                continue
            if hasattr(child, "set_context_parallel_group"):
                child.set_context_parallel_group(cp_group, cp_global_ranks, cp_stream)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        encoder_output: Optional[torch.Tensor] = None,
        attn_mask_type: Optional[str] = None,
        window_size: Optional[Tuple[int, int]] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: bool = False,
        inference_params: Optional[InferenceParams] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """
        Forward propagation for MultiheadAttention layer.

        .. note::

            Argument :attr:`attention_mask` is only used when :attr:`attn_mask_type`
            includes `"padding"` or `"arbitrary"`.

        Parameters
        ----------
        hidden_states : torch.Tensor
             Input tensor.
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
             default = `None`. Boolean tensor(s) used to mask out attention softmax input.
             It should be 'None' for 'causal' and 'no_mask' types. For 'padding' masks, it should be
             a single tensor of [batch_size, 1, 1, seqlen_q] for self-attention, and a tuple of
             two tensors in shapes [batch_size, 1, 1, seqlen_q] and [batch_size, 1, 1, seqlen_kv]
             for cross-attention. For the 'arbitrary' mask type, it should be in a shape that is
             broadcastable to [batch_size, num_heads, max_seqlen_q, max_seqlen_kv].
        attn_mask_type: {'no_mask', 'padding', 'causal', 'padding_causal', 'arbitrary'},
                       default = `None`
                       type of attention mask passed into softmax operation.
        window_size: Optional[Tuple[int, int]], default = `None`
                    sliding window size for local attention.
        encoder_output : Optional[torch.Tensor], default = `None`
             Output of the encoder block to be fed into the decoder block if using
             `layer_type="decoder"`.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        checkpoint_core_attention: bool, default = `False`
                                  If true, forward activations for core attention are recomputed
                                  during the backward pass in order to save memory that would
                                  otherwise be occupied to store the forward activations until
                                  backprop.
        rotary_pos_emb: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], default = `None`
                       Embeddings for query and key tensors for applying rotary position
                       embedding. By default no input embedding is applied.
        core_attention_bias_type: str, default = `no_bias`
                    Bias type, {`no_bias`, `pre_scale_bias`, 'post_scale_bias`, `alibi`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
                    Bias tensor for Q * K.T, shape [1, num_head, max_seqlen_q, max_seqlen_kv].
                    It should be 'None' for 'no_bias' and 'alibi' bias types.
        alibi_slopes: Optional[torch.Tensor], default = `None`
                     ALiBi slopes in FP32 and shape [nheads] or [batch_size, nheads].
                     It adds a bias of (-alibi_slope * (i + seqlen_k - seqlen_q - j))
                     to the attention score of query i and key j.
        fast_zero_fill: bool, default = `True`
                    Whether to set output tensors to 0 or not before use.
        """
        # hidden_states: [sq, b, h]

        if attn_mask_type is not None:
            window_size = check_set_window_size(attn_mask_type, window_size)
        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        if window_size is None:
            window_size = self.window_size

        if "padding" in attn_mask_type and attention_mask is not None:
            for i,_ in enumerate(attention_mask):
                assert (
                    attention_mask[i].dtype == torch.bool
                ), "Attention mask must be in boolean type!"

        assert (core_attention_bias_type in AttnBiasTypes
                ), f"core_attention_bias_type {core_attention_bias_type} is not supported!"

        # =================================================
        # Pre-allocate memory for key-values for inference
        # =================================================

        if inference_params and self.layer_number is not None:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_length
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size, hidden_states.dtype
                )
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size, hidden_states.dtype
                )
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory,
                    inference_value_memory,
                )
            else:
                (
                    inference_key_memory,
                    inference_value_memory,
                ) = inference_params.key_value_memory_dict[self.layer_number]

        # ======================
        # Query, Key, and Value
        # ======================

        if self.attention_type == "self":
            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn]
            if self.input_layernorm:
                layernorm_qkv_outputs = self.layernorm_qkv(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                )
                if self.return_layernorm_output:
                    mixed_x_layer, layernorm_output = layernorm_qkv_outputs
                else:
                    mixed_x_layer = layernorm_qkv_outputs
            else:
                mixed_x_layer = self.qkv(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                )

            num_queries_per_key_value = (self.num_attention_heads_per_partition //
                                         self.num_gqa_groups_per_partition)
            if self.qkv_weight_interleaved:
                # [sq, b, ng * (np/ng + 2) * hn] --> [sq, b, ng, (np/ng + 2), hn]
                new_tensor_shape = mixed_x_layer.size()[:-1] + (
                    self.num_gqa_groups_per_partition,
                    (num_queries_per_key_value + 2),
                    self.hidden_size_per_attention_head,
                )
                # split along second last dimension
                split_dim = -2
            else:
                # [sq, b, ng * (np/ng + 2) * hn] --> [sq, b, (np/ng + 2), ng, hn]
                new_tensor_shape = mixed_x_layer.size()[:-1] + (
                    (num_queries_per_key_value + 2),
                    self.num_gqa_groups_per_partition,
                    self.hidden_size_per_attention_head
                )
                # split along third last dimension
                split_dim = -3

            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # qkv_weight_interleaved:
            #  [sq, b, ng, (np/ng + 2), hn]
            #  --> [sq, b, ng, np/ng, hn], [sq, b, ng, 1, hn], [sq, b, ng, 1, hn]
            # not qkv_weight_interleaved:
            #  [sq, b, (np/ng + 2), ng, hn]
            #  --> [sq, b, np/ng, np, hn], [sq, b, 1, ng, hn], [sq, b, 1, ng, hn]
            if not is_in_onnx_export_mode():
                query_layer, key_layer, value_layer = _SplitAlongDim.apply(
                    mixed_x_layer, split_dim, (num_queries_per_key_value, 1, 1)
                )
            else:
                query_layer, key_layer, value_layer = torch.split(
                    mixed_x_layer, (num_queries_per_key_value, 1, 1), dim = split_dim,
                 )

            # query: -> [sq, b, np, hn]
            # key, value: -> [sq, b, ng, hn]
            query_layer, key_layer, value_layer = (x.reshape(x.size(0), x.size(1), -1,
                                                             self.hidden_size_per_attention_head)
                                                   for x in (query_layer, key_layer, value_layer))

        elif self.attention_type == "cross":
            # Attention heads [sk, b, h] --> [sk, b, (ng * 2 * hn)]
            mixed_kv_layer = self.key_value(
                encoder_output,
                is_first_microbatch=is_first_microbatch,
            )

            if self.qkv_weight_interleaved:
                # [sq, b, (ng * 2 * hn)] --> [sq, b, ng, 2 * hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    self.num_gqa_groups_per_partition,
                    2 * self.hidden_size_per_attention_head,
                )
                # split along last dimension
                split_dim = -1
            else:
                # [sq, b, (ng * 2 * hn)] --> [sq, b, 2 * ng, hn]
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    2 * self.num_gqa_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
                # split along second last dimension
                split_dim = -2

            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # mixed_kv_layer --> 2 [sk, b, ng, hn]
            if not is_in_onnx_export_mode():
                key_layer, value_layer = _SplitAlongDim.apply(
                    mixed_kv_layer, split_dim, mixed_kv_layer.shape[split_dim] // 2,
                )
            else:
                key_layer, value_layer = torch.split(
                    mixed_kv_layer, mixed_kv_layer.shape[split_dim] // 2, dim = split_dim,
                )

            # Attention head [sq, b, h] --> [sq, b, hp]
            if self.input_layernorm:
                layernorm_query_outputs = self.layernorm_query(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                )
                if self.return_layernorm_output:
                    query_layer, layernorm_output = layernorm_query_outputs
                else:
                    query_layer = layernorm_query_outputs
            else:
                query_layer = self.query_layer(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                )

            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        # ======================================================
        # Apply relative positional encoding (rotary embedding)
        # ======================================================

        use_mine = False
        rotary_pos_emb = self.qk_encoding.rotation

        if rotary_pos_emb is not None or use_mine:
           # duplicate the pos_emb for self attention
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = ((rotary_pos_emb,) * 2)
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb

            # adjust key and value for inference
            if inference_params is not None and rotary_pos_emb is not None:
                if self.qkv_format == "sbhd":
                    sequence_length = key_layer.size(0)
                elif self.qkv_format == "bshd":
                    sequence_length = key_layer.size(1)

                sequence_start = inference_params.sequence_len_offset
                sequence_end = sequence_start + sequence_length

                q_pos_emb = q_pos_emb[sequence_start:sequence_end, ...]
                k_pos_emb = k_pos_emb[sequence_start:sequence_end, ...]

            #print("K shape before:", key_layer.shape)
            #print("Q shape before:", query_layer.shape)
            if use_mine:
                # in: [T, N, H, E] => [N, H, T, E]
                # out: [N, H, T, E] => [T, N, H, E]
                #print("--------mine---------")
                to_input_shape_fn = lambda in_qk: in_qk.transpose(0, 1).transpose(1, 2)
                to_output_shape_fn = lambda out_qk: out_qk.transpose(1, 2).transpose(0, 1)
                query_layer = to_output_shape_fn(self.qk_encoding(to_input_shape_fn(query_layer)))
                key_layer = to_output_shape_fn(self.qk_encoding(to_input_shape_fn(key_layer)))
            else:
                #print("--------theirs---------")
                # TODO: you wanna apply your rope here but it may have been created in a diff class
                query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb, self.qkv_format, fused=True)
                key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb, self.qkv_format, fused=True)
            #print("K shape after:", key_layer.shape)
            #print("Q shape after:", query_layer.shape)

        # ===========================
        # Core attention computation
        # ===========================

        context_layer = self.core_attention(
            query_layer,
            key_layer,
            value_layer,
            qkv_format=self.qkv_format,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            window_size=window_size,
            checkpoint_core_attention=checkpoint_core_attention,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias=core_attention_bias,
            alibi_slopes=alibi_slopes,
            fast_zero_fill=fast_zero_fill,
            inference_params=inference_params,
        )

        # ===================
        # Output. [sq, b, h]
        # ===================

        projection_output = self.proj(
            context_layer, is_first_microbatch=is_first_microbatch
        )

        if self.return_bias:
            attention_output, attention_bias = projection_output
        else:
            attention_output, attention_bias = projection_output, None

        outputs = (attention_output,)
        if self.return_bias:
            outputs += (attention_bias,)
        if self.input_layernorm and self.return_layernorm_output:
            outputs += (layernorm_output,)
        return outputs if len(outputs) > 1 else outputs[0]


class TransformerLayerWithRope(torch.nn.Module):
    r"""
    TransformerLayer is made up of an attention block and a feedforward network (MLP).
    This standard layer is based on the paper "Attention Is All You Need".

    .. note::

        Argument :attr:`attention_mask` in the `forward` call is only used when
        :attr:`self_attn_mask_type` includes `"padding"` or `"arbitrary"`.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    ffn_hidden_size : int
                     intermediate size to which input samples are projected.
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    num_gqa_groups : int, default = `None`
                         number of GQA groups in the transformer layer.
                         Grouped Query Attention is described in
                         `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                         This only affects the keys and values, not the querys.
                         GQA-1 is equivalent to Multi-Query Attention
                         (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                         is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    layernorm_epsilon : float, default = 1e-5
                       a value added to the denominator of layer normalization
                       for numerical stability.
    hidden_dropout: float, default = 0.1
                   dropout probability for the dropout op after FC2 layer.
    attention_dropout: float, default = 0.1
                      dropout probability for the dropout op during multi-head attention.
    init_method : Callable, default = `None`
                 used for initializing weights of QKV and FC1 weights in the following way:
                 `init_method(weight)`. When set to `None`, defaults to
                 `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing weights of PROJ and FC2 in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    apply_residual_connection_post_layernorm : bool, default = `False`
                                              if set to `True`, residual connections are taken
                                              from the output of layer norm (default is taken
                                              from input of layer norm)
    layer_number: int, default = `None`
                 layer number of the current `TransformerLayer` when multiple such modules are
                 concatenated to form a transformer block.
    output_layernorm: bool, default = `False`
                     if set to `True`, layer normalization is applied on the output side,
                     after the final dropout-add. default behavior is to apply layer
                     normalization on the input side, before the QKV transformation.
    parallel_attention_mlp: bool, default = `False`
                           if set to `True`, self-attention and feedforward network are computed
                           based on the same input (in parallel) instead of sequentially.
                           Both blocks have an independent normalization.
                           This architecture is used in `Falcon` models.
    layer_type: {'encoder', 'decoder'}, default = `encoder`
               if set to `decoder`, an additional cross-attn block is added after self-attn.
               This can be used for structures like `T5` Transformer in conjunction with the
               `encoder` option.
    kv_channels: int, default = `None`
                number of key-value channels. defaults to
                :attr:`hidden_size` / :attr:`num_attention_heads` if `None`.
    self_attn_mask_type: {'no_mask', 'padding', 'causal', 'padding_causal', 'arbitrary'},
                        default = `causal`
                        type of attention mask passed into softmax operation. Overridden by
                        :attr:`self_attn_mask_type` in the `forward` method. The forward
                        arg is useful for dynamically changing mask types, e.g. a different
                        mask for training and inference. The init arg is useful for cases
                        involving compilation/tracing, e.g. ONNX export.
    window_size: Optional[Tuple[int, int]], default = `None`
                sliding window size for local attention, where query at position i attends to keys
                in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
                + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
                window and causal mask specifically. Similar to :attr:`self_attn_mask_type`, it can
                be overridden by :attr:`window_size` in `forward` as well.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    qkv_weight_interleaved : bool, default = `True`
                            if set to `False`, the QKV weight is interpreted as a concatenation of
                            query, key, and value weights along the `0th` dimension. The default
                            interpretation is that the individual `q`, `k`, and `v` weights for each
                            attention head are interleaved. This parameter is set to `False` when
                            using :attr:`fuse_qkv_params=False`.
    bias : bool, default = `True`
          if set to `False`, the transformer layer will not learn any additive biases.
    activation : str, default = 'gelu'
          Type of activation used in MLP block.
          Options are: 'gelu', 'relu', 'reglu', 'geglu', 'swiglu' and 'qgelu'.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    attn_input_format: {'sbhd', 'bshd'}, default = 'sbhd'
                         This controls whether the dimensions of the
                         intermediate hidden states is 'batch first' ('bshd') or
                         'sequence first' ('sbhd'). `s` stands for the sequence
                         length, `b` batch size, `h` the number of heads, `d`
                         head size. Note that these formats are very closely
                         related to the `qkv_format` in the `MultiHeadAttention`
                         and `DotProductAttention` modules.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, QKV and FC1 layers are used as Column Parallel
                      whereas PROJ and FC2 is used as Row Parallel as described
                      `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    seq_length: int
               sequence length of input samples. Needed for JIT Warmup, a technique where jit
               fused functions are warmed up before training to ensure same kernels are used for
               forward propogation and activation recompute phase.
    micro_batch_size: int
                     batch size per training step. Needed for JIT Warmup, a technique where jit
                     fused functions are warmed up before training to ensure same kernels are
                     used for forward propogation and activation recompute phase.
    drop_path_rate: float, default = 0.0
                   when > 0.0, applies stochastic depth per sample in
                   the main path of the residual block.
    fuse_qkv_params: bool, default = 'False'
                    if set to `True`, `TransformerLayer` module exposes a single fused
                    parameter for query-key-value. This enables optimizations such as QKV
                    fusion without concatentations/splits and also enables the argument
                    `fuse_wgrad_accumulation`.
    """

    def __init__(
            self,
            hidden_size: int,
            ffn_hidden_size: int,
            num_attention_heads: int,
            num_gqa_groups: Optional[int] = None,
            layernorm_epsilon: float = 1e-5,
            hidden_dropout: float = 0.1,
            attention_dropout: float = 0.1,
            init_method: Optional[Callable] = None,
            output_layer_init_method: Optional[Callable] = None,
            layer_number: Optional[int] = None,
            kv_channels: Optional[int] = None,
            self_attn_mask_type: str = "causal",
            window_size: Optional[Tuple[int, int]] = None,
            tp_group: Optional[dist_group_type] = None,
            tp_size: int = 1,
            params_dtype: Optional[torch.dtype] = None,
            get_rng_state_tracker: Optional[Callable] = None,
            fuse_wgrad_accumulation: bool = False,
            seq_length: Optional[int] = None,
            micro_batch_size: Optional[int] = None,
            sequence_parallel: bool = False,
            apply_residual_connection_post_layernorm: bool = False,
            output_layernorm: bool = False,
            parallel_attention_mlp: bool = False,
            layer_type: str = "encoder",
            drop_path_rate: float = 0.0,
            set_parallel_mode: bool = False,
            fuse_qkv_params: bool = False,
            zero_centered_gamma: bool = False,
            qkv_weight_interleaved: bool = True,
            ub_tp_comm_overlap: bool = False,
            ub_bulk_wgrad: bool = True,
            ub_bulk_dgrad: bool = True,
            ub_split_ag: bool = True,
            ub_split_rs: bool = True,
            ub_atomic_gemm_ag: bool = False,
            ub_atomic_gemm_rs: bool = False,
            bias: bool = True,
            activation: str = 'gelu',
            normalization: str = "LayerNorm",
            device: Union[torch.device, str] = "cuda",
            attn_input_format: str = "sbhd",
            qk_encoding: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        if ub_tp_comm_overlap:
            assert (
                tex.userbuf_comm_available()
            ), "Userbuffer communication backend not available."

        self.self_attn_mask_type = self_attn_mask_type
        self.window_size = window_size
        self.window_size = check_set_window_size(self_attn_mask_type, self.window_size)
        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        ub_bulk_wgrad = ub_tp_comm_overlap and ub_bulk_wgrad
        ub_bulk_dgrad = ub_tp_comm_overlap and ub_bulk_dgrad
        ub_split_ag = ub_tp_comm_overlap and ub_split_ag
        ub_split_rs = ub_tp_comm_overlap and ub_split_rs
        ub_atomic_gemm_rs = ub_tp_comm_overlap and ub_atomic_gemm_rs
        assert (
            not (ub_split_rs and ub_atomic_gemm_rs)
        ), "Only one type of RS overlap ub_split_rs/ub_atomic_gemm_rs should be enabled."
        ub_atomic_gemm_ag = ub_tp_comm_overlap and ub_atomic_gemm_ag
        assert (
            not (ub_split_ag and ub_atomic_gemm_ag)
        ), "Only one type of AG overlap ub_split_ag/ub_atomic_gemm_ag should be enabled."

        if ub_atomic_gemm_rs or ub_atomic_gemm_ag:
            warnings.warn(
                "Atomic gemm uses a beta API from cublas and is not tested for all use cases."
            )

        bias_dropout_fusion = bool(int(os.getenv("NVTE_BIAS_DROPOUT_FUSION", "1")))
        self.layer_number = layer_number
        self.output_layernorm = output_layernorm
        self.layer_type = layer_type
        self.apply_residual_connection_post_layernorm = (
            apply_residual_connection_post_layernorm
        )

        if parallel_attention_mlp:
            assert self.layer_type == "encoder", "parallel_attention requires layer_type='encoder'"
            assert (
                not self.apply_residual_connection_post_layernorm
            ), "parallel_attention and apply_residual_connection_post_layernorm " \
               "not supported simultaneously."
            assert (
                not self.output_layernorm
            ), "parallel_attention and output_layernorm not supported simultaneously"

        self.parallel_attention_mlp = parallel_attention_mlp

        assert layer_type in LayerTypes, f"layer_type {layer_type} not supported"

        if not fuse_qkv_params:
            assert (
                not fuse_wgrad_accumulation
            ), "Gradient accumulation fusion requires single QKV parameter."

        if not fuse_qkv_params:
            qkv_weight_interleaved = False

        self.kv_channels = (
            kv_channels if kv_channels else (hidden_size // num_attention_heads)
        )

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        self.tp_size = tp_size if tp_group is None else get_distributed_world_size(tp_group)
        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel
        self.seq_length = seq_length

        self.get_rng_state_tracker = get_rng_state_tracker

        self.attn_input_format = attn_input_format

        attention_args = (
            hidden_size,
            num_attention_heads,
            self.kv_channels,
            attention_dropout,
            layernorm_epsilon,
            init_method,
            output_layer_init_method,
        )
        common_attention_kwargs = {
            "layer_number": layer_number,
            "tp_group": tp_group,
            "tp_size": self.tp_size,
            "num_gqa_groups": num_gqa_groups,
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "get_rng_state_tracker": get_rng_state_tracker,
            "sequence_parallel": self.sequence_parallel,
            "params_dtype": params_dtype,
            "return_layernorm_output": apply_residual_connection_post_layernorm,
            "set_parallel_mode": set_parallel_mode,
            "fuse_qkv_params": fuse_qkv_params,
            "zero_centered_gamma": zero_centered_gamma,
            "qkv_weight_interleaved": qkv_weight_interleaved,
            "ub_bulk_wgrad": ub_bulk_wgrad,
            "ub_bulk_dgrad": ub_bulk_dgrad,
            "ub_split_ag": ub_split_ag,
            "ub_split_rs": ub_split_rs,
            "ub_atomic_gemm_rs": ub_atomic_gemm_rs,
            "ub_atomic_gemm_ag": ub_atomic_gemm_ag,
            "qkv_format": self.attn_input_format,
        }

        self.qk_encoding = qk_encoding

        self.self_attention = MultiheadAttentionWithRope(
            *attention_args,
            **common_attention_kwargs,
            input_layernorm=not output_layernorm,
            attention_type="self",
            bias=bias,
            return_bias=not self.parallel_attention_mlp,
            normalization=normalization,
            device=device,
            qk_encoding=self.qk_encoding,
        )

        if layer_type == "decoder":
            self.inter_attention = MultiheadAttentionWithRope(
                *attention_args,
                **common_attention_kwargs,
                attn_mask_type="padding",
                input_layernorm=True,
                attention_type="cross",
                bias=bias,
                return_bias=True,
                normalization=normalization,
                device=device,
                qk_encoding=self.qk_encoding,
            )

        # LayerNorm -> activation(Linear + Bias) -> Linear
        # parallel_mode not supported for LayerNormMLP,
        # FC1 is CPL and FC2 is RPL
        # In the case of GLU activation, FC1 handles both
        # Linear layers before the activation

        self.layernorm_mlp = LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            eps=layernorm_epsilon,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            tp_group=tp_group,
            tp_size=self.tp_size,
            get_rng_state_tracker=get_rng_state_tracker,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            bias=bias,
            return_bias=not self.parallel_attention_mlp,
            sequence_parallel=self.sequence_parallel,
            params_dtype=params_dtype,
            return_layernorm_output=apply_residual_connection_post_layernorm,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            set_parallel_mode=set_parallel_mode,
            zero_centered_gamma=zero_centered_gamma,
            ub_bulk_wgrad=ub_bulk_wgrad,
            ub_bulk_dgrad=ub_bulk_dgrad,
            ub_split_rs=ub_split_rs,
            ub_split_ag=ub_split_ag,
            ub_atomic_gemm_rs=ub_atomic_gemm_rs,
            ub_atomic_gemm_ag=ub_atomic_gemm_ag,
            activation=activation,
            normalization=normalization,
            device=device,
        )

        self.hidden_dropout = hidden_dropout
        self.bias_dropout_fusion = bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = (
            nullcontext if use_nvfuser else torch.enable_grad
        )

        if self.bias_dropout_fusion:
            set_jit_fusion_options()
            if seq_length and micro_batch_size:
                if self.sequence_parallel:
                    seq_length = seq_length // self.tp_size
                warmup_jit_bias_dropout_add_all_dtypes(
                    hidden_size, seq_length, micro_batch_size
                )

        norm_module = {
            "LayerNorm": LayerNorm,
            "RMSNorm": RMSNorm,
        }
        if self.output_layernorm:
            self.layernorm = norm_module[normalization](
                hidden_size,
                eps=layernorm_epsilon,
                sequence_parallel=self.sequence_parallel,
                params_dtype=params_dtype,
                zero_centered_gamma=zero_centered_gamma,
                device=device,
            )

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """
        Set the tensor parallel group for the given
        module before executing the forward pass.

        Parameters
        ----------
        tp_group : ProcessGroup, default = `None`
                  tensor parallel process group.
        """
        # Deep iterate but skip self to avoid infinite recursion.
        for index, child in enumerate(self.modules()):
            if index == 0:
                continue
            if hasattr(child, "set_tensor_parallel_group"):
                child.set_tensor_parallel_group(tp_group)

    def set_context_parallel_group(
        self,
        cp_group: Union[dist_group_type, None],
        cp_global_ranks: List[int],
        cp_stream: torch.cuda.Stream,
    ) -> None:
        """
        Set the context parallel attributes for the given
        module before executing the forward pass.

        Parameters
        ----------
        cp_group : ProcessGroup
                  context parallel process group.
        cp_global_ranks : List[int]
                         list of global ranks in the context group.
        cp_stream : torch.cuda.Stream
                   cuda stream for context parallel execution.
        """
        # Deep iterate but skip self to avoid infinite recursion.
        for index, child in enumerate(self.modules()):
            if index == 0:
                continue
            if hasattr(child, "set_context_parallel_group"):
                child.set_context_parallel_group(cp_group, cp_global_ranks, cp_stream)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        self_attn_mask_type: Optional[str] = None,
        window_size: Optional[Tuple[int, int]] = None,
        encoder_output: Optional[torch.Tensor] = None,
        enc_dec_attn_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: bool = False,
        inference_params: Optional[InferenceParams] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        fast_zero_fill: bool = True,
    ) -> torch.Tensor:
        """
        Transformer Layer: attention block and a feedforward network (MLP)

        .. note::

            Argument :attr:`attention_mask` is only used when :attr:`self_attn_mask_type`
            includes `"padding"` or `"arbitrary"`.

        Parameters
        ----------
        hidden_states : torch.Tensor
             Input tensor.
        attention_mask : Optional[torch.Tensor], default = `None`
                        Boolean tensor used to mask out self-attention softmax input.
                        It should be in [batch_size, 1, 1, seqlen_q] for 'padding' mask,
                        and broadcastable to [batch_size, num_heads, max_seqlen_q, max_seqlen_kv]
                        for 'arbitrary'. It should be 'None' for 'causal' and 'no_mask'.
        self_attn_mask_type: {'no_mask', 'causal', 'padding', 'padding_causal', 'arbitrary'},
                            default = `causal`
                            Type of attention mask passed into softmax operation.
        window_size: Optional[Tuple[int, int]], default = `None`
                    sliding window size for local attention.
        encoder_output : Optional[torch.Tensor], default = `None`
             Output of the encoder block to be fed into the decoder block if using
             `layer_type="decoder"`.
        enc_dec_attn_mask : Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
             default = `None`. Boolean tensors used to mask out inter-attention softmax input if
             using `layer_type="decoder"`. It should be a tuple of two masks in
             [batch_size, 1, 1, seqlen_q] and [batch_size, 1, 1, seqlen_kv] for 'padding' mask.
             It should be broadcastable to [batch_size, num_heads, max_seqlen_q, max_seqlen_kv]
             for 'arbitrary' mask. It should be 'None' for 'causal' and 'no_mask'.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        checkpoint_core_attention: bool, default = `False`
                                  If true, forward activations for core attention are recomputed
                                  during the backward pass in order to save memory that would
                                  otherwise be occupied to store the forward activations until
                                  backprop.
        rotary_pos_emb: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], default = `None`
                       Embeddings for query and key tensors for applying rotary position
                       embedding. By default no input embedding is applied.
        core_attention_bias_type: str, default = `no_bias`
                    Bias type, {`no_bias`, `pre_scale_bias`, `post_scale_bias`, `alibi`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
                    Bias tensor for Q * K.T
        alibi_slopes: Optional[torch.Tensor], default = `None`
                     ALiBi slopes in FP32 and shape [nheads] or [batch_size, nheads].
                     It adds a bias of (-alibi_slope * (i + seqlen_k - seqlen_q - j))
                     to the attention score of query i and key j.
        fast_zero_fill: bool, default = `True`
                    Whether to set output tensors to 0 or not before use.
        inference_params: InferenceParams, default = None
                         Inference parameters that are passed to the main model in order
                         to efficienly calculate and store the context during inference.
        """
        if self_attn_mask_type is not None:
            window_size = check_set_window_size(self_attn_mask_type, window_size)
        if self_attn_mask_type is None:
            self_attn_mask_type = self.self_attn_mask_type
        if window_size is None:
            window_size = self.window_size

        assert (
                self_attn_mask_type in AttnMaskTypes
        ), f"self_attn_mask_type {self_attn_mask_type} not supported"

        hidden_states = hidden_states.contiguous()

        if self.sequence_parallel and self.seq_length is not None:
            assert (
                    hidden_states.shape[0] == self.seq_length // self.tp_size
            ), "Sequence dimension must be split across TP group when using sequence parallel."

        if (("padding" in self_attn_mask_type
             or self_attn_mask_type == "arbitrary")
                and attention_mask is not None):
            assert (
                    attention_mask.dtype == torch.bool
            ), "Attention mask must be a boolean tensor"

        # For AMP
        if torch.is_autocast_enabled():
            hidden_states = cast_if_needed(
                hidden_states, torch.get_autocast_gpu_dtype()
            )

        # Self attention.
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            attn_mask_type=self_attn_mask_type,
            window_size=window_size,
            inference_params=inference_params,
            is_first_microbatch=is_first_microbatch,
            checkpoint_core_attention=checkpoint_core_attention,
            rotary_pos_emb=rotary_pos_emb,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias=core_attention_bias,
            alibi_slopes=alibi_slopes,
            fast_zero_fill=fast_zero_fill,
        )

        if self.apply_residual_connection_post_layernorm and not self.output_layernorm:
            attention_output, attention_bias, residual = self_attention_outputs
            hidden_states = self._bias_dropout_add(
                attention_output, attention_bias, residual, self.drop_path
            )
        elif not self.parallel_attention_mlp:
            attention_output, attention_bias = self_attention_outputs
            hidden_states = self._bias_dropout_add(
                attention_output, attention_bias, hidden_states, self.drop_path
            )

        # Cross attention.
        if self.layer_type == "decoder":
            inter_attention_outputs = self.inter_attention(
                hidden_states,
                attention_mask=enc_dec_attn_mask,
                window_size=window_size,
                encoder_output=encoder_output,
                is_first_microbatch=is_first_microbatch,
                checkpoint_core_attention=checkpoint_core_attention,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias,
                alibi_slopes=alibi_slopes,
                fast_zero_fill=fast_zero_fill,
            )
            if self.apply_residual_connection_post_layernorm:
                attention_output, attention_bias, residual = inter_attention_outputs
            else:
                attention_output, attention_bias = inter_attention_outputs
                residual = hidden_states

            hidden_states = self._bias_dropout_add(attention_output, attention_bias, residual)

        # MLP.
        mlp_outputs = self.layernorm_mlp(
            hidden_states, is_first_microbatch=is_first_microbatch
        )
        if self.apply_residual_connection_post_layernorm:
            mlp_output, mlp_bias, residual = mlp_outputs
            output = self._bias_dropout_add(mlp_output, mlp_bias, residual, self.drop_path)
        elif self.parallel_attention_mlp:
            output = self._bias_dropout_add(
                self_attention_outputs, mlp_outputs, hidden_states, self.drop_path
            )
        else:
            mlp_output, mlp_bias = mlp_outputs
            output = self._bias_dropout_add(mlp_output, mlp_bias, hidden_states, self.drop_path)

        # For BERT like architectures.
        if self.output_layernorm:
            output = self.layernorm(output)

        # output: [s, b, h]
        return output

    def _bias_dropout_add(self, hidden_state, bias, residual, drop_path=None):
        if drop_path is None and bias.numel() != 0:
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    hidden_state, bias, residual, self.hidden_dropout
                )
        else:
            if bias.numel() != 0:
                hidden_state = hidden_state + bias
            out = torch.nn.functional.dropout(
                hidden_state, p=self.hidden_dropout, training=self.training
            )
            if drop_path is not None:
                out = drop_path(out)
            output = residual + out

        return output


@Registry.register("te_llama_mod")
class TELLama(nn.Module):

    def __init__(self, config):
        super().__init__()

        # create input embedding
        input_embed = InputEmbedding(
            config.pos_encoding,
            config.vocab_size,
            config.d_model,
            config.p_dropout,
            config.max_len,
            precision=config.precision,
            layer_id=0,
            cache=None,
        )
        layer_list = [input_embed]

        args = self._get_layer_args_from_config(config)
        transformer_layer_cls = self.get_transformer_layer_cls()

        # create transformer layers
        for i in range(config.n_layers - 1):
            args["layer_number"] = i + 1
            layer_list.append(transformer_layer_cls(**args))
        # add an output norm for the last layer so we only use the TE norm type
        args["layer_number"] = config.n_layers
        args["output_layernorm"] = True
        layer_list.append(transformer_layer_cls(**args))

        # create the head
        # set norm to identity b/c norm is handled in the final transformer layer
        identity_cls = Registry.get("identity")
        lm_head = LMHead(vocab_size=config.vocab_size, d_model=config.d_model, norm_cls=identity_cls)
        layer_list.append(lm_head)

        self.layers = SequentialWithArgs(*layer_list)

    def _get_layer_args_from_config(self, config):
        device = "meta" if config.use_meta_device else "cuda"
        self.qk_encoding = RotaryEncoding(
            d_head=config.d_model // config.n_heads,
            max_len=config.max_len,
            base=500_000,
            device=device,
        )
        return {
            "device": device,
            "params_dtype": Precision.PRECISION_NAME_TO_DTYPE[config.precision],
            "hidden_size": config.d_model,
            "ffn_hidden_size": config.d_hidden,
            "num_attention_heads": config.n_heads,
            "self_attn_mask_type": "causal", #TODO: can change to "padding_causal" or "arbitrary" need to check it out
            "normalization": "RMSNorm",
            "bias": False, # won't learn bias terms
            "activation": "swiglu",
            "attn_input_format": "bshd", # whether the dims of the hidden states are batch first or seqlen first
            "fuse_wgrad_accumulation": False, # this is default
            #"params_dtype": , dtype for the model params; guessing we don't need but maybe
            "seq_length": config.max_len,
            "fuse_qkv_params": True,
            "qk_encoding": self.qk_encoding,
        }

    @staticmethod
    def get_transformer_layer_cls():
        return TransformerLayerWithRope

    def forward(self, input_ids):
        """
        Returns the logits for the given input_ids
        """
        return self.layers(input_ids)


class TransformerLayerWithPOS(TransformerLayer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rope = RotaryPositionEmbedding(kwargs["hidden_size"] // kwargs["num_attention_heads"])
        self.rope_freqs = rope(max_seq_len=kwargs["seq_length"]).cuda()

    def forward(self, hidden_states):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `LlamaDecoderLayer`.
        """
        return super().forward(hidden_states, rotary_pos_emb=self.rope_freqs)


@Registry.register("te_llama_og")
class TELLamaOg(TELLama):

    def _get_layer_args_from_config(self, config):
        device = "meta" if config.use_meta_device else "cuda"
        return {
            "device": device,
            "params_dtype": torch.float32, #Precision.PRECISION_NAME_TO_DTYPE[config.precision],
            "hidden_size": config.d_model,
            "ffn_hidden_size": config.d_hidden,
            "num_attention_heads": config.n_heads,
            "self_attn_mask_type": "causal", #TODO: can change to "padding_causal" or "arbitrary" need to check it out
            "normalization": "RMSNorm",
            "bias": False, # won't learn bias terms
            "activation": "swiglu",
            "attn_input_format": "bshd", # whether the dims of the hidden states are batch first or seqlen first
            "fuse_wgrad_accumulation": False, # this is default
            "seq_length": config.max_len,
            "fuse_qkv_params": True,
        }

    @staticmethod
    def get_transformer_layer_cls():
        return TransformerLayerWithPOS


"""
Below here copied from https://github.com/NVIDIA/TransformerEngine/blob/0edf30b87159e82048b5f248e4b379aebb8f364a/transformer_engine/pytorch/distributed.py
since latest version of nvidia pytorch container does not yet have these changes
"""

def _is_te_module(module):
    """
    Check if given module is a Transformer Engine module that requires the TE checkpoint
    implementation for activation recompute.
    """
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    from transformer_engine.pytorch.attention import UnfusedDotProductAttention, MultiheadAttention

    te_classes_list = [
        LayerNorm,
        RMSNorm,
        TransformerEngineBaseModule,
        UnfusedDotProductAttention,
        DotProductAttention,
        MultiheadAttention,
        TransformerLayer,
    ]
    is_te_module = False
    for te_class in te_classes_list:
        if isinstance(module, te_class):
            is_te_module = True
            break
    return is_te_module


def prepare_te_modules_for_fsdp(fsdp_root: nn.Module) -> None:
    """
    Inject FSDP process gorup references into FSDP-wrapped TE modules in an FSDP-wrapped root
    module in order to scatter/gather the Fp8 weight copies at the same time FSDP scatters/gathers
    its `FlatParameters`.

    Parameters
    ----------
    fsdp_root: torch.nn.Module
               FSDP-wrapped root module that may contain FSDP-wrapped TE modules.
    """
    assert isinstance(fsdp_root, FSDP), "Root module must be FSDP-wrapped."
    # TODO: jpk added
    if hasattr(fsdp_root, "primary_weights_in_fp8"):
        assert not fsdp_root.primary_weights_in_fp8, (
            "TE modules with primary weights in FP8 cannot be FSDP-wrapped. "
            "Please initialize your model without the te.fp8_model_init(...) context."
        )

    # If the root module is a TE module, inject FSDP information into it
    if _is_te_module(fsdp_root.module):
        root_state = _get_module_fsdp_state(fsdp_root)
        assert root_state is not None, "Root module does not have a valid _FSDPState."
        setattr(fsdp_root.module, "fsdp_group", root_state.process_group)

    # Iterate through all FSDP-wrapped submodules and inject FSDP information into TE modules
    fsdp_states, fsdp_modules = _get_fsdp_states_with_modules(fsdp_root)
    for state, fsdp_module in zip(fsdp_states, fsdp_modules):
        if hasattr(fsdp_module.module, "primary_weights_in_fp8"):
            print("==========hazzzzzzzzzzzz=============")
            assert not fsdp_module.module.primary_weights_in_fp8, (
                "TE modules with primary weights in FP8 cannot be FSDP-wrapped. "
                "Please initialize your model without the te.fp8_model_init(...) context."
            )
        if _is_te_module(fsdp_module.module):
            setattr(fsdp_module.module, "fsdp_group", state.process_group)
