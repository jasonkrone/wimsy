import os
import math
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import huggingface_hub
from transformers import AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
try: 
    from hf_olmo import OLMoForCausalLM
    from olmo.model import OLMoSequentialBlock
except:
    print("olmo package not available")

from utils import Registry, Precision, logger
from training.trainers.trainer import Trainer


@Registry.register("hf_trainer")
class HFTrainer(Trainer):

    @classmethod
    def get_logits(cls, model, input_dict):
        logits = model(**input_dict).logits
        logits = logits.view(-1, logits.size(-1))
        logits_dict = {"input": logits}
        return logits_dict
   
    @classmethod
    def get_model(cls, config):
        assert config.model.id == "hf"
        assert config.model.parallelism == "fsdp_model"
        huggingface_hub.login(token=os.environ["HUGGINFGACE_WRITE_TOKEN"])

        model, model_name = cls.get_causal_lm(config)

        if "pythia" in model_name:
            transformer_layer_cls = GPTNeoXLayer
            if not hasattr(config.model, "max_len"):
                config.model.max_len = model.config.max_position_embeddings
            config.model.n_layers = model.config.num_hidden_layers
            config.model.n_heads = model.config.num_attention_heads
            config.model.n_kv_heads = model.config.num_attention_heads
            config.model.d_model = model.config.hidden_size
            config.model.d_hidden = model.config.intermediate_size
            config.model.vocab_size = model.config.vocab_size
        elif "olmo" in model_name:
            transformer_layer_cls = OLMoSequentialBlock
            if not hasattr(config.model, "max_len"):
                config.model.max_len = model.config.max_sequence_length
            config.model.n_layers = model.config.n_layers
            config.model.n_heads = model.config.n_heads
            config.model.n_kv_heads = model.config.n_heads
            config.model.d_model = model.config.d_model
            config.model.d_hidden = model.config.mlp_ratio * model.config.d_model
            config.model.vocab_size = model.config.vocab_size
        elif "Llama-3.1" in model_name:
            transformer_layer_cls = LlamaDecoderLayer
            if not hasattr(config.model, "max_len"):
                config.model.max_len = model.config.rope_scaling["original_max_position_embeddings"]
            config.model.n_layers = model.config.num_hidden_layers
            config.model.n_heads = model.config.num_attention_heads
            config.model.n_kv_heads = model.config.num_key_value_heads
            config.model.d_model = model.config.hidden_size
            config.model.d_hidden = model.config.intermediate_size
        else:
            raise ValueError(f"model {model_name} not supported")

        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={transformer_layer_cls})
        mixed_precision = Precision.PRECISION_NAME_TO_FSDP_MIXED_PRECISION[config.model.precision]

        model = FSDP(
            model, 
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            device_id=config.local_rank,
            limit_all_gathers=True,
        )
        return model

    @classmethod
    def save_checkpoint_for_eval(cls, config, model, checkpointer, ckpt_path, ckpt_dir, repo_id = None):
        if repo_id is None:
            repo_id = f"{config.ckpt.hf_username}/{config.wandb.run_name}"

        checkpointer.load_checkpoint(config, model, strip_prefix=config.ckpt.strip_prefix, ckpt_path=ckpt_path)
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config):
            model_state = model.state_dict()

        hf_ckpt_dir = os.path.join(ckpt_dir, "hf_ckpt")
        if config.is_master:
            # create another instance b/c model is FSDP wrapped which corrupts the weight dims
            temp_model, _ = cls.get_causal_lm(config)
            temp_model.load_state_dict(model_state)
            logger.info(f"pushing model to hub with repo_id {repo_id}")
            temp_model.save_pretrained(hf_ckpt_dir, push_to_hub=True, repo_id=repo_id)

        dist.barrier()
        return repo_id 

    @classmethod
    def get_causal_lm(cls, config):
        """
        Gets the config specified huggingface model. Also handles setting precision 
        given in string format to data type and vocab resizing if requested. 
        """
        model_kwargs, _ = cls.split_args_str(config.model.args)

        if "torch_dtype" in model_kwargs:
            if model_kwargs["torch_dtype"] == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                raise ValueError(f"dtype: {model_kwargs['torch_dtype']} not supported")

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        if hasattr(config.model, "n_extend_vocab"):
            pad_to_multiple_of = config.model.pad_vocab_size_to_multiple_of # 8
            #1 for n_extend_vocab
            with_pad_vocab_size = math.ceil((model.config.vocab_size + config.model.n_extend_vocab) / pad_to_multiple_of) * pad_to_multiple_of
            config.model.vocab_size = with_pad_vocab_size
            model.resize_token_embeddings(with_pad_vocab_size)
        
        model_name = model_kwargs["pretrained_model_name_or_path"]
        return model, model_name

    @classmethod
    def split_args_str(
        cls,
        args_str,
        model_keys=["pretrained", "revision", "trust_remote_code", "torch_dtype", "attn_implementation"],
        tokenizer_keys=["tokenizer", "add_bos_token", "trust_remote_code"],
    ):
        """
        Util so we can use the same args str format as lm-eval-harness
        """
        model_args = {}
        tokenizer_args = {}
        for pair in args_str.split(","):
            k, v = pair.split("=")
            if k in model_keys:
                if k == "pretrained":
                    k = "pretrained_model_name_or_path"
                model_args[k] = v
            if k in tokenizer_keys:
                if k == "tokenizer":
                    k = "pretrained_model_name_or_path"
                tokenizer_args[k] = v

        assert "pretrained_model_name_or_path" in model_args
        return model_args, tokenizer_args
