import argparse
import os
import sys
from functools import partial

import wandb
import huggingface_hub

from lm_eval.models.huggingface import HFLM

from tasks.mt_bench.gen_mt_answers import write_mt_bench_answer_file
from tasks.mt_bench.gen_mt_judgement import write_mt_bench_judgements_file
from tasks.mt_bench.show_mt_result import compute_mt_bench_metrics
from evaluation.eval_wrapper import get_model_wrapper
from utils import Config

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to evaluation config file in yaml format")


def get_model(config, device):
    if config.model.id == "hf":
        # we put the model on CPU because it's moved to each device later when ray is used
        model = HFLM.create_from_arg_string(config.model.args, {"batch_size": 1, "device": device})
    else:
        model, _ = get_model_wrapper(config)
    return model


def main(config):
    # TODO: might not need this
    huggingface_hub.login(token=os.environ["HUGGINGFACE_TOKEN"])
    wandb.init(project=config.wandb.project_name, name=config.wandb.run_name, config=config.as_dict())

    # get the mt bench answers
    answers_path = os.path.join(config.test_eval.output_path, f"{config.model.name}_answers.jsonl")
    write_mt_bench_answer_file(
        get_model_fn=partial(get_model, config),
        model_id=config.model.name,
        device=config.compute.device,
        questions_path=config.mt_bench.questions_path,
        questions_begin=config.mt_bench.questions_begin,
        questions_end=config.mt_bench.questions_end,
        answers_path=answers_path,
        max_new_tokens=config.mt_bench.max_new_tokens,
        num_choices=config.mt_bench.num_choices,
        limit=config.test_eval.limit,
        world_size=config.compute.world_size,
    )

    # use automated eval via model as judge
    judgements_path = os.path.join(config.test_eval.output_path, f"{config.model.name}_judgements.jsonl")
    write_mt_bench_judgements_file(
        models=[config.model.name],
        output_file=judgements_path,
        question_file=config.mt_bench.questions_path,
        model_answers_file=answers_path,
        reference_answers_file=config.mt_bench.reference_answers_path,
        judge_file=config.mt_bench.judge_prompts_path,
        judge_model=config.mt_bench.judge_model,
        baseline_model=config.mt_bench.baseline_model,
        mode=config.mt_bench.mode,
        num_workers=config.mt_bench.num_workers,
        limit=config.test_eval.limit,
    )

    # compute metrics
    metrics_dict = compute_mt_bench_metrics(
        input_file=judgements_path,
        models=[config.model.name],
        baseline_model=config.mt_bench.baseline_model,
        mode=config.mt_bench.mode,
    )
    wandb.log(metrics_dict)
    wandb.finish()

    print("metrics:", metrics_dict)


if __name__ == "__main__":
    args = parser.parse_args()
    mt_config = Config.from_yaml(args.config)

    output_root = mt_config.test_eval.output_path
    for baseline_name, baseline_config in mt_config.baselines.items():
        mt_config.wandb.run_name = baseline_name
        mt_config.model.id = baseline_config["id"]
        mt_config.model.args = baseline_config["args"]
        mt_config.model.name = baseline_name
        mt_config.test_eval.output_path = os.path.join(output_root, baseline_name)
        main(mt_config)


