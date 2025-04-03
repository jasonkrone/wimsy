from fms.models.llama import LLaMA, LLaMABlock, LLaMAConfig

from utils import Registry


@Registry.register("fms_llama", "from_config")
class FMSLlamaDecoder(LLaMA):

    @classmethod
    def from_config(cls, config):
        model_config = LLaMAConfig(
            src_vocab_size=config.vocab_size,
            emb_dim=config.d_model,
            nheads=config.n_heads,
            nlayers=config.n_layers,
            max_expected_seq_len=config.max_len,
        )
        model = cls(model_config)
        if not config.use_meta_device:
            model.reset_parameters()
        return model

    @classmethod
    def get_transformer_layer_cls(cls):
        return LLaMABlock

