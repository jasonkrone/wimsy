import argparse

import torch

from utils import Config, Registry, logger
from training import Trainer, HFTrainer
from training.train import train_model

parser = argparse.ArgumentParser()
parser.add_argument("--configs", type=str, nargs="+", help="training configs ordered by phase")


class PhasedConfigUpdater(object):
    """
    Updates the current phase config with info from prior phase config & checkpoint 
    """
    @classmethod
    def __call__(cls, last_phase_ckpt, last_phase_config, cur_phase_config):
        # set config to resume from prior phase checkpoint
        cur_phase_config.ckpt.resume_ckpt = last_phase_ckpt
        return cur_phase_config


@Registry.register("to_linear_annealing_lr_schedule")
class LearningRatePhasedConfigUpdater(PhasedConfigUpdater):
    """
    Sets the max learning rate for the current phases linear annealing lr schedule 
    to the lr saved in the prior phases ckpt
    """
    @classmethod
    def __call__(cls, last_phase_ckpt, last_phase_config, cur_phase_config):
        cur_phase_config = super().__call__(last_phase_ckpt, last_phase_config, cur_phase_config)
        checkpointer_cls = Registry.get(last_phase_config.ckpt.checkpointer)
        metadata_path = checkpointer_cls.get_metadata_path(last_phase_ckpt)
        metadata_dict = torch.load(metadata_path)
        cur_phase_config.lr_schedule.args.lr_max = metadata_dict["learning_rate"]
        cur_phase_config.optimizer.learning_rate = metadata_dict["learning_rate"]
        return cur_phase_config


def train_model_in_phases(configs):

    prev_ckpt = None
    prev_config = None
    for i, cur_config in enumerate(configs):
        logger.info(f"training phase {i+1} of {len(configs)}")
        # update current config with info from prev config
        if hasattr(cur_config, "training_phases"):
            config_updater_cls = Registry.get(cur_config.training_phases.config_updater.id)
            config_updater = config_updater_cls()
            cur_config = config_updater(prev_ckpt, prev_config, cur_config)

        prev_ckpt, _ = train_model(cur_config)
        prev_config = cur_config

    return prev_ckpt


if __name__ == "__main__":
    args = parser.parse_args()
    configs = [Config.from_yaml(config_path) for config_path in args.configs]
    train_model_in_phases(configs)


