import argparse
from tqdm import tqdm

import torch
from transformers import StoppingCriteriaList, MaxLengthCriteria
from accelerate import load_checkpoint_and_dispatch

from tokenizer import *
from model.model import Decoder
from utils import Config, Precision, Registry

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config file in yaml format")

torch.manual_seed(0)


def main(config):
    outputs = []
    prompts = ["once upon a time there was a", "the best way to find a good restaurant is to search"]

    config.is_device_cuda = config.compute.device == "cuda"

    model_cls = Registry.get(config.model.id)
    model = model_cls(config.model)
    print("not loading any checkpoint")
    #transformer_block_cls_name = model.get_transformer_layer_cls().__name__
    #model = load_checkpoint_and_dispatch(
    #    model, config.decoding.ckpt_path, device_map="auto", no_split_module_classes=[transformer_block_cls_name],
    #)

    print("using cache: ", config.model.use_cache)

    model.eval()

    tokenizer_cls = Registry.get(config.decoding.tokenizer.id)
    tokenizer = tokenizer_cls(**config.decoding.tokenizer.args)

    decoding_cls = Registry.get(config.decoding.decoding_algorithm)
    decoding = decoding_cls()

    print("pad id:", tokenizer.pad_token_id, "pad tok:", tokenizer.pad_token)

    stop_criteria = StoppingCriteriaList([MaxLengthCriteria(config.decoding.out_len)])
    prompt_ids = [tokenizer.encode(prompt_str) for prompt_str in prompts]

    output_ids, logits, out_lengths = decoding(
        prompt_list=prompt_ids,
        model=model,
        stop_criteria=stop_criteria,
        temperature=config.decoding.temperature,
        pad_token_id=tokenizer.pad_token_id,
        max_length=config.decoding.out_len,
        device=config.compute.device,
    )

    output_str = tokenizer.batch_decode(output_ids)
    outputs.append(output_str)

    print(outputs)


if __name__ == "__main__":
    args = parser.parse_args()
    generate_config = Config.from_yaml(args.config)
    main(generate_config)
