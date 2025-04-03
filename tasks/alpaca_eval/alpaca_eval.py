"""
Code largely copied from https://github.com/tatsu-lab/alpaca_eval

Minor modification to the alpaca huggingface_local_completions and evaluate_from_model 
functions to support the lm-evaluation-harness model interface. 
"""
from typing import Optional, Sequence
from functools import partial
import argparse
import sys
import os

import ray
import wandb
import huggingface_hub
import datasets

import torch
import pandas as pd
from tqdm import tqdm


sys.path.append("./lm-evaluation-harness")
from lm_eval.models.huggingface import HFLM

sys.path.append("./alpaca_eval/src")
from alpaca_eval import constants, utils, evaluate

from utils import Config
from evaluation.eval_wrapper import get_model_wrapper


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)



@ray.remote(num_gpus=1)
def eval_harness_model_completions(
    prompts: Sequence[str],
    model_config,
    completion_kwargs,
    batch_size: int = 1,
    remove_ending: Optional[str] = None,
) -> dict[str, list]:
    """Decode locally using huggingface transformers pipeline.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model: model that satisfies the lm-evaluation-harness api 

    generate_kwargs: kwargs to pass to model._model_generate

    batch_size : int, optional
        Batch size to use for decoding. This currently does not work well with to_bettertransformer.

    remove_ending : str, optional
        The ending string to be removed from completions. Typically eos_token.
    """
    n_examples = len(prompts)
    assert n_examples > 0

    max_new_tokens = completion_kwargs.pop("max_new_tokens")
    system_prompt = completion_kwargs.pop("system_prompt", None)

    device = f"cuda:{torch.cuda.current_device()}"

    if model_config.model.id == "hf":
        model = HFLM.create_from_arg_string(model_config.model.args, {"batch_size": batch_size, "device": device})
    else:
        model, _ = get_model_wrapper(model_config)

    if model.tokenizer.padding_side != "left":
        model.tokenizer.padding_side = "left"

    # there's an issue with llama2 if the pad_token_id is set to unk_token_id
    if model.tokenizer.pad_token_id == model.tokenizer.unk_token_id:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model._model.generation_config.pad_token_id = model.tokenizer.pad_token_id

    conversation = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": "{instruction}"})

    prompt_template = model.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    prompts, _ = utils.make_prompts(prompts, template=prompt_template)

    if batch_size > 1:
        # sort the prompts by length so that we don't necessarily pad them by too much
        # save also index to reorder the completions
        original_order, prompts = zip(*sorted(enumerate(prompts), key=lambda x: len(x[1])))
        prompts = list(prompts)

    ## compute and log the time for completions
    completions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="generating outputs"):
        batch_prompts = prompts[i:i+batch_size]
        
        assert model.tokenizer.padding_side == "left"
        inputs_dict = model.tokenizer.batch_encode_plus(batch_prompts, padding='longest', return_tensors="pt")
        input_ids = inputs_dict["input_ids"]
        attn_mask = inputs_dict["attention_mask"]

        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        max_length = max_new_tokens + input_ids.shape[1]

        # output ids contain both input ids and continuation
        output_ids = model._model_generate(
            context=input_ids,
            max_length=max_length,
            stop=[model.tok_decode(model.eot_token_id, skip_special_tokens=False)],
            attention_mask=attn_mask,
            **completion_kwargs,
        )

        for output in output_ids.tolist():
            continuation_ids = output[input_ids.shape[1]:]
            generated_text = model.tokenizer.decode(continuation_ids, skip_special_tokens=True)
            if remove_ending is not None and generated_text.endswith(remove_ending):
                generated_text = generated_text[:-len(remove_ending)]
            completions.append(generated_text)

    if batch_size > 1:
        # reorder the completions to match the original order
        completions, _ = zip(*sorted(list(zip(completions, original_order)), key=lambda x: x[1]))
        completions = list(completions)

    return completions 



def get_completions(
    model_config,
    model_id, 
    df: pd.DataFrame, 
    batch_size: int, 
    is_strip_output=True, 
    completion_kwargs={}, 
    max_instances=None, 
    num_gpus=1
):
    columns_to_keep = ["dataset", "instruction", "output", "generator"]
    columns_to_keep = [c for c in columns_to_keep if c in df.columns]
    curr_outputs = df[columns_to_keep].copy()
    if max_instances is not None:
        curr_outputs = curr_outputs[:max_instances]
    
    # Create remote tasks
    remote_tasks = []
    chunksize = len(curr_outputs) // num_gpus
    for df_chunk in utils.dataframe_chunk_generator(curr_outputs, chunksize=chunksize, tqdm_desc="Chunking for multi-gpu"):
        remote_tasks.append(eval_harness_model_completions.remote(
            prompts=df_chunk,
            model_config=model_config,
            completion_kwargs=completion_kwargs,
            batch_size=batch_size,
            remove_ending=None  # Adjust if needed
        ))

    # Get results
    completions = []
    for result in ray.get(remote_tasks):
        completions.extend(result)

    if is_strip_output:
        completions = [c.strip() for c in completions]

    curr_outputs["output"] = completions
    curr_outputs["generator"] = model_id

    return curr_outputs


def get_eval_dataset(**kwargs):
    return datasets.load_dataset(**kwargs)["eval"]


def evaluate_from_model(
    model_config, 
    model_id,
    output_path: str,
    batch_size,
    eval_dataset_kwargs,
    annotators_config = constants.DEFAULT_ANNOTATOR_CONFIG,
    max_instances: int = None,
    num_gpus: int = 1,
    completion_kwargs={},
):

    evaluation_dataset = partial(get_eval_dataset, **eval_dataset_kwargs)
       
    df_dataset = utils.load_or_convert_to_dataframe(evaluation_dataset)

    model_outputs = get_completions(
        model_config=model_config, 
        model_id=model_id, 
        df=df_dataset, 
        max_instances=max_instances, 
        batch_size=batch_size, 
        num_gpus=num_gpus,
        completion_kwargs=completion_kwargs,
    )

    # use reference outputs from evaluation_dataset
    assert "output" in df_dataset.columns
    reference_outputs = df_dataset.copy()
    # write model and reference outputs
    model_outputs.to_json(os.path.join(output_path, "model_outputs.json"), orient="records", indent=2)
    reference_outputs.to_json(os.path.join(output_path, "reference_outputs.json"), orient="records", indent=2)
    # using a default reference outputs => uses the right leaderboard
    reference_outputs = evaluation_dataset

    return evaluate(
        model_outputs=model_outputs,
        reference_outputs=reference_outputs,
        annotators_config=annotators_config,
        output_path=output_path,
        max_instances=max_instances,
        is_return_instead_of_print=True,
        is_overwrite_leaderboard=True,
    )


def main(config):

    huggingface_hub.login(token=os.environ["HUGGINGFACE_TOKEN"])
    wandb.init(project=config.wandb.project_name, name=config.wandb.run_name, config=config.as_dict())

    num_gpus = config.compute.world_size  # Assuming this is set in your config
    ray.init(num_gpus=num_gpus)

    max_instances = config.test_eval.limit

    df_leaderboard, _ = evaluate_from_model(
        config,
        config.model.name,
        config.test_eval.output_path,
        max_instances=max_instances,
        num_gpus=num_gpus,
        batch_size=config.test_eval.batch_size,
        completion_kwargs=config.test_eval.completion_kwargs,
        eval_dataset_kwargs=config.test_eval.eval_dataset_kwargs,
    )

    ray.shutdown()

    metrics_dict = dict(df_leaderboard.loc[config.model.name])
    metrics_dict = {f"alpaca_eval/{k}": v for k, v in metrics_dict.items()}

    wandb.log(metrics_dict)
    wandb.finish()

    print("metrics:", metrics_dict)

   
if __name__ == "__main__":
    args = parser.parse_args()
    alpaca_config = Config.from_yaml(args.config)

    output_root = alpaca_config.test_eval.output_path
    for baseline_name, baseline_config in alpaca_config.baselines.items():
        alpaca_config.wandb.run_name = baseline_name
        alpaca_config.model.id = baseline_config["id"]
        alpaca_config.model.args = baseline_config["args"]
        alpaca_config.model.name = baseline_name
        alpaca_config.test_eval.completion_kwargs = baseline_config["completion_kwargs"]
        print("baseline name:", baseline_name)
        alpaca_config.test_eval.output_path = os.path.join(output_root, baseline_name)
        os.makedirs(alpaca_config.test_eval.output_path, exist_ok=True)
        main(alpaca_config)


