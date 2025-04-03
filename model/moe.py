import argparse

from utils import Config, Registry

# TODO: maybe we also need to import mpu ?
from megablocks.layers.arguments import Arguments as MoEArguments
from megablocks.layers.dmoe import dMoE

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)


@Registry.register("dmoe", "from_config")
class DroplessMoE(dMoE):
    """
    Wrapper around the DMoE class make it compatible with our data dims and config

    TODO: need to allow this to be selected as the MLP class
    """

    @classmethod
    def from_config(cls, config):
        moe_args = MoEArguments(
            hidden_size=config.d_model,
            ffn_hidden_size=config.d_hidden,
            # include a bias term that is added to the output of the expert
            bias=config.moe.has_bias,
            return_bias=False,
            moe_num_experts=config.moe.n_experts,
            moe_top_k=config.moe.top_k,
            moe_normalize_expert_weights=config.moe.normalize_expert_weights,
            # not used for dropless MoE
            moe_loss_weight=None,
            moe_jitter_eps=config.moe.router_jitter_eps,
            # not used for dropless MoE
            moe_lbl_in_fp32=None,
            # Enables TP + EP
            # not sure this is supported for dMoE
            moe_expert_model_parallelism=config.moe.use_expert_and_tensor_parallelism,
            # TODO: see if this is required, should check megatron
            expert_parallel_group=None,
            # says weight parallelism but i think it's really expert parallelism
            moe_weight_parallelism=config.moe.use_expert_parallelism,
            weight_parallel_group=None,
            memory_optimized_mlp=config.moe.use_memory_optimized_mlp,
            # TODO: do they support fp32?
            bf16=config.precision == "bf16",
            fp16=config.precision == "fp16",
            device=None, # TODO: IDK exactly the impact of this
        )
        return cls(moe_args)

    def forward(self, x):
        """
        Wrapper around DMoE that permutes x to (T, N, D) shape before calling dMoE forward.
        We have x in (N, T, D) shape.
        """
        x = x.transpose(0, 1)
        if x.shape[0] > 1:
            x = x.contiguous()
        out = super().forward(x)
        out = out.transpose(0, 1)
        return out



