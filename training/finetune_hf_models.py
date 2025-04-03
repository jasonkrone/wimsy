import argparse

from utils import Config, logger
from training.train import train_model
from training.trainers.hf_trainer import HFTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to evaluation config file in yaml format")


def get_tokenizer_args(model_args):
    tokenizer_args = None
    model_args_dict, tokenizer_args_dict = HFTrainer.split_args_str(model_args)
    if len(tokenizer_args_dict):
        tokenizer_args = tokenizer_args_dict
    else:
        tokenizer_args = {"pretrained_model_name_or_path": model_args_dict["pretrained_model_name_or_path"]}
    return tokenizer_args


def finetune_and_eval(finetune_config):
    # for now we assume we're fine-tuning multiple models
    assert hasattr(finetune_config, "base_models")
    num_base_models = len(finetune_config.base_models)
    for i, (model_name, model_config) in enumerate(finetune_config.base_models.items()):
        finetune_config.compute.init_process_group_at_start = i == 0
        finetune_config.compute.shutdown_distributed_processes_at_end = i == num_base_models - 1
        model_args = model_config["args"]
        finetune_config.model.id = model_config["id"]
        finetune_config.model.args = model_args
        finetune_config.wandb.run_name = f"{model_name}-mc-finetune-hpo-lr-with-mmlu"
        tokenizer_args = get_tokenizer_args(model_args)
        finetune_config.data.train_dataset.args.tokenizer.args = tokenizer_args
        finetune_config.data.dev_dataset.args.tokenizer.args = tokenizer_args
        hf_repo_id, _ = train_model(finetune_config)
        logger.info(f"saved fine-tuned version of {model_name} to {hf_repo_id}")


if __name__ == "__main__":
    args = parser.parse_args()
    finetune_config = Config.from_yaml(args.config)
    finetune_and_eval(finetune_config)
