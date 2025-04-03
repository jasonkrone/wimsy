"""
Wraps the pre-trained model in an interface that's compatible with the LM evaluation harness.

Based on https://github.com/EleutherAI/gpt-neox/blob/f48d3a6ba11c29b38c713b976ad28d95f8f5d142/eval_tasks/eval_adapter.py
"""
import os
import sys
import time
import json
import shutil
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from datetime import timedelta

import torch
from accelerate import Accelerator, InitProcessGroupKwargs

import numpy as np
import huggingface_hub

sys.path.append("./lm-evaluation-harness")
from lm_eval.evaluator import simple_evaluate
from lm_eval.loggers import EvaluationTracker, WandbLogger
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.utils import (
    make_table,
    eval_logger,
    sanitize_list,
    handle_non_serializable,
)

from tokenizer import *
from training import Trainer, World
from utils import Config, Registry, copy_source_to_dest
from evaluation.model_wrapper import EvalHarnessModelWrapper

eval_logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to evaluation config file in yaml format")


class NoAppendEvaluationTracker(EvaluationTracker):
    """
    Modifies the save_results_samples to use file write vs. append
    because append is not available on some OS.
    """
    def save_results_samples(self, task_name: str, samples: dict) -> None:
        eval_logger.info(f"Saving per-sample results for: {task_name}")

        path = Path(self.output_path if self.output_path else Path.cwd())
        path = path.joinpath(self.general_config_tracker.model_name_sanitized)
        path.mkdir(parents=True, exist_ok=True)

        file_results_samples = path.joinpath(f"samples_{task_name}_{self.date_id}.jsonl")

        sample_list = []
        for sample in samples:
            arguments = {}
            for i, arg in enumerate(sample["arguments"]):
                arguments[f"gen_args_{i}"] = {}
                for j, tmp in enumerate(arg):
                    arguments[f"gen_args_{i}"][f"arg_{j}"] = tmp

            sample["resps"] = sanitize_list(sample["resps"])
            sample["filtered_resps"] = sanitize_list(sample["filtered_resps"])
            sample["arguments"] = arguments

            sample_dump = json.dumps(sample, default=handle_non_serializable, ensure_ascii=False)
            sample_list.append(sample_dump+"\n")

        with open(file_results_samples, "w") as f:
            f.writelines(sample_list)


def postprocess_results(task_name, results):
    if results is None:
        return results

    # add combined metric for ifeval
    # this is easier to do in post-processing b/c eval harness doesn't easily support using other metrics to create a new metric
    if task_name == "ifeval":
        results["results"]["ifeval"]["combined_acc,none"] = np.mean([
            results["results"]["ifeval"]["prompt_level_strict_acc,none"],
            results["results"]["ifeval"]["inst_level_strict_acc,none"],
            results["results"]["ifeval"]["prompt_level_loose_acc,none"],
            results["results"]["ifeval"]["inst_level_loose_acc,none"],
        ])

    return results


def aggregate_results(results_list):
    agg_results = defaultdict(dict)
    for result in results_list:
        for k, v in result.items():
            if isinstance(v, dict):
                agg_results[k].update(v)
            else:
                agg_results[k] = v
    return agg_results


def save_results(results_list, evaluation_tracker, wandb_logger, config):
    results = aggregate_results(results_list)
    # save outputs and scores
    if results is not None:
        samples = results.pop("samples", None)
        # write results to wandb
        try:
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            if config.test_eval.log_samples:
                wandb_logger.log_eval_samples(samples)
        except Exception as e:
            eval_logger.info(f"Logging to Weights and Biases failed due to {e}")
        # write results to file
        evaluation_tracker.save_results_aggregated(results=results, samples=samples)
        # write samples to file
        if config.test_eval.log_samples:
            for task_name in results["configs"]:
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )
        # print results
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))



def get_model_wrapper(config):
    ckpt_path = config.test_eval.ckpt_path
    log_if_master(eval_logger, f"Evaluating checkpoint: {ckpt_path}", config.is_master)

    # add ckpt path to wandb run name
    config.wandb.run_name = f"{config.wandb.run_name}-ckpt-{ckpt_path}"
    # name or path required by eval harness
    config.model._name_or_path = f"run-{config.wandb.run_name}-ckpt-{ckpt_path}"

    model_cls = Registry.get(config.model.id)
    model = model_cls(config.model)

    # accelerator setup taken from lm-evaluation-harness huggingface.py
    accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
    accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
    assert "npu" not in accelerator.device.type

    # copy the checkpoint to the local instance
    # required to prevent eval from hanging from ckpt is on mounted s3 bucket
    local_ckpt_path = ckpt_path
    if config.is_master and config.test_eval.get("copy_ckpt_to_local", True):
        local_ckpt_path = ckpt_path.replace(config.ckpt.local_output_dir, "/tmp")
        copy_source_to_dest(ckpt_path, local_ckpt_path)
    accelerator.wait_for_everyone()

    device = f"cuda:{config.local_rank}"
    state_dict = torch.load(local_ckpt_path, map_location=device)
    model.load_state_dict(state_dict["model"])

    # remove temp checkpoint
    if config.is_master and config.test_eval.get("copy_ckpt_to_local", True):
        os.remove(local_ckpt_path)

    model = model.to(device)
    if config.model.do_compile:
        model = torch.compile(model)

    # create tokenizer
    tokenizer_cls = Registry.get(config.test_eval.tokenizer.id)
    tokenizer = tokenizer_cls(**config.test_eval.tokenizer.args)

    model = EvalHarnessModelWrapper(
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        batch_size=None,
        device=config.compute.device,
    )

    # make results dir
    ckpt_str = f"{os.path.basename(ckpt_path).split('.')[0]}"
    output_dir = getattr(config.test_eval, "output_path", None)
    if output_dir is None:
        # create dir at the same level as ./checkpoints
        # checkpoint file could be under ./checkpoints or ./checkpoints/num
        if os.path.dirname(ckpt_path) == "checkpoints":
            output_dir = Path(ckpt_path).parents[1]
        else:
            output_dir = Path(ckpt_path).parents[2]
        config.test_eval.output_path = os.path.join(output_dir, "results", ckpt_str)
        log_if_master(eval_logger, f"Created results dir at: {config.test_eval.output_path}", config.is_master)

    return model, f"path={ckpt_path}"


def log_if_master(logger, message, is_master, log_fn="info"):
    if is_master:
        getattr(logger, log_fn)(message)


def clear_hf_cache(cache_dir):
    path_list = [os.path.join(cache_dir, item) for item in os.listdir(cache_dir)]
    for path in path_list:
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except Exception as e:
                eval_logger.warning(f"Exception while clearing hf cache: {e}")


def main(config, shutdown_distributed_processes_at_end=True):
    """
    Based on the lm evaluation cli_evaluate function
    """
    huggingface_hub.login(token=os.environ["HUGGINGFACE_TOKEN"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # allow code eval for humaneval dataset
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    World.set_world_state(config)

    if config.test_eval.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if config.model.id == "hf":
        model = "hf"
        model_args = config.model.args
        log_if_master(eval_logger, f"evaluating Hugging Face baseline model: {config.model.args}", config.is_master)
    else:
        model, model_args = get_model_wrapper(config)
        log_if_master(eval_logger, f"evaluating my model: {model_args}", config.is_master)

    if config.is_master:
        wandb_logger = WandbLogger(project=config.wandb.project_name, name=config.wandb.run_name, config=config.as_dict())

    assert config.test_eval.output_path is not None
    log_if_master(eval_logger, f"output path: {config.test_eval.output_path}", config.is_master)
    evaluation_tracker = NoAppendEvaluationTracker(output_path=config.test_eval.output_path)

    results_list = []
    task_manager = TaskManager(include_path=config.test_eval.include_path)
    for task, task_config in config.test_eval.tasks.items():
        num_fewshot = task_config["shot"]
        batch_size = task_config["batch_size"]
        gen_kwargs = task_config.get("gen_kwargs", None)
        predict_only = task_config.get("predict_only", False)
        apply_chat_template = task_config.get("apply_chat_template", False)
        system_instruction = task_config.get("system_instruction", None)

        log_if_master(eval_logger, f"running eval on n_fewshot: {num_fewshot} tasks: {task}", config.is_master)

        # get the tasks
        task_dict = get_task_dict([task], task_manager)
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, tuple):
                _, task_obj = task_obj
            # update generation_kwargs if provided in user specified config
            if task_obj is not None and gen_kwargs is not None:
                task_obj.set_config(key="generation_kwargs", value=gen_kwargs, update=True)

        # set batch size for this task if wrapped model
        if isinstance(model, EvalHarnessModelWrapper):
            model.batch_size = batch_size

        start_time = time.time()
        # only returns results for the rank 0 process
        results = simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=task_dict,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=config.compute.device,
            limit=config.test_eval.limit,
            log_samples=config.test_eval.log_samples,
            evaluation_tracker=evaluation_tracker,
            task_manager=task_manager,
            random_seed=config.seed,
            numpy_random_seed=config.seed,
            torch_random_seed=config.seed,
            fewshot_random_seed=config.seed,
            apply_chat_template=apply_chat_template,
            predict_only=predict_only,
        )
        results = postprocess_results(task, results)
        results_list.append(results)

        end_time = time.time()
        duration = (end_time - start_time) / 60.0
        if config.is_master:
            wandb_logger.run.log({f"{task}/eval_mins": duration})
            # log results for this dataset just in case something fails before we save results for everything
            wandb_logger.post_init(results)
            _, wandb_results = wandb_logger._sanitize_results_dict()
            wandb_logger.run.log(wandb_results)

        # reset auto batch size after task completes
        if isinstance(model, EvalHarnessModelWrapper):
            if model.auto_batch_size is not None and config.is_master:
                wandb_logger.run.log({f"{task}_auto_batch_size": model.auto_batch_size})
            model.auto_batch_size = None

    if config.is_master:
        save_results(results_list, evaluation_tracker, wandb_logger, config)
        wandb_logger.run.finish()

    if torch.distributed.is_initialized() and shutdown_distributed_processes_at_end:
        torch.distributed.barrier()
        Trainer.shutdown_distributed_processes(config)


if __name__ == "__main__":
    args = parser.parse_args()
    eval_config = Config.from_yaml(args.config)
    # eval list of baselines
    if hasattr(eval_config, "baselines"):
        output_root = eval_config.test_eval.output_path
        for baseline_name, baseline_config in eval_config.baselines.items():
            eval_config.model.id = baseline_config["id"]
            eval_config.model.args = baseline_config["args"]
            eval_config.wandb.run_name = baseline_name
            eval_config.test_eval.output_path = os.path.join(output_root, baseline_name)
            main(eval_config, shutdown_distributed_processes_at_end=False)
            if eval_config.test_eval.get("clear_cache", False) and eval_config.is_master:
                clear_hf_cache(os.environ["TRANSFORMERS_CACHE"])
            torch.distributed.barrier()
    # eval list of checkpoints
    elif hasattr(eval_config.test_eval, "ckpt_path") and isinstance(eval_config.test_eval.ckpt_path, list):
        ckpt_list = eval_config.test_eval.ckpt_path
        run_name = eval_config.wandb.run_name
        for ckpt_path in ckpt_list:
            # we re-set the run name before eval because the ckpt path gets added to it by the script
            eval_config.wandb.run_name = run_name
            eval_config.test_eval.ckpt_path = ckpt_path
            main(eval_config, shutdown_distributed_processes_at_end=False)
    else:
        main(eval_config)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        Trainer.shutdown_distributed_processes(eval_config)
