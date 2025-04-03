"""
Runs hyper-parameter optimization. Wraps functions in train.py for use with Ray Tune.
Written for Ray Tune version 2.6.3. Unfortunately, Ray Tune 2.6.3 causes skypilot logging to break.
The fix is to install Ray in a conda env vs. using the base env (see ./configs/sky/hpo/hpo.yaml).

HPO notes:

types of params
(1) scientific - trying to opt over
(2) nuisance - need to opt over in order to fairly compare #1
(3) fixed - we keep these fixed
optim is largely dep on batch size so be very weary of changing batch size after it's set
how long to training when training is compute bounded
reasonable to increase per-trial training limits over time
e.g., could start w/ 30min, then go to 1hr, and then 2hr
1-3 rounds is typically most practical
they mention 10% and 20% of production run time as examples

adam params to tune dep on num trials
< 10 trials: LR only
10-25 trials, LR & beta_1 <<
25+ trials, LR & beta_1 & epsilon
if much more than 25 trials, tune beta_2

We trained that model I last had for 115,000 steps and it did pretty well
30k steps looks right << seems like we could increase the batch size by a hair

Quasi-random search is there suggested algo
After that they'd do pseudo random search but that's not their fave

best clip threshold is just above the "typical" grad norm
seems like we'd wanna log the grad norm and then use that to pick
pick grad clip at 90% percentile of grad norms
track what % of the grads are clipped (> 50% would be a lot)
"""
import os
import sys
import argparse

import torch

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.train import RunConfig
from ray.train.torch import TorchTrainer, TorchConfig
from ray.air import session
from ray.air.config import ScalingConfig
from ray.air.integrations.wandb import setup_wandb
from optuna.samplers import QMCSampler
from ray.tune.search import ConcurrencyLimiter

from utils import Config, Registry

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to training config file in yaml format")


class HPOConfigUpdater(object):

    @classmethod
    def __call__(cls, train_config, hpo_config):
        for key, value in hpo_config.items():
            setattr(train_config, key, value)
        return train_config


class LRScheduleUpdater(HPOConfigUpdater):

    @classmethod
    def __call__(cls, train_config, hpo_config):
        train_config = super().__call__(train_config, hpo_config)
        train_config.lr_schedule.args.lr_max = hpo_config["optimizer.learning_rate"]
        return train_config


def report_dev_loss(dev_loss, num_iters):
    session.report({"dev_loss": dev_loss, "iteration": num_iters})


def train_loop(config, train_config):
    """
    config provided by ray and train_config provided by us
    """
    # local inmport here required otherwise non-master processes don't have trainer registered
    from training import Trainer, HFTrainer

    config_updater = LRScheduleUpdater()
    train_config = config_updater(train_config, config)

    wandb_run = setup_wandb(train_config.as_dict(), project=train_config.wandb.project_name, rank_zero_only=False)
    train_config.wandb.run_name = wandb_run.name
    
    trainer_cls = Registry.get(train_config.trainer.id)
    trainer = trainer_cls()
    trainer.train(train_config, wandb=wandb_run, hpo_fn=report_dev_loss)


def get_ray_trainer(train_config):
    """
    Uses TorchTrainer to handle initialization of distributed processes.
    When using this, do not init / destroy distributed processes in your train loop.
    """
    scaling_config = ScalingConfig(
        num_workers=torch.cuda.device_count(),
        use_gpu=train_config.compute.device == "cuda",
    )
    torch_config = TorchConfig(backend=train_config.compute.backend)

    train_loop_fn = lambda x: train_loop(x, train_config)

    storage_path = os.path.join(train_config.ckpt.local_output_dir, "raytune_hpo")
    run_config = RunConfig(name=train_config.wandb.run_name, storage_path=storage_path)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_fn,
        scaling_config=scaling_config,
        torch_config=torch_config,
        run_config=run_config,
    )
    return trainer


def main(config):

    # this assumes we use ray port 6379
    ip_address = os.environ["SKYPILOT_NODE_IPS"]
    ip_address = f"{ip_address}:6379" 
    env_dict = {
        "env_vars": {
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"], 
            "PYTHONPATH": os.environ["PYTHONPATH"],
            "TOKENIZERS_PARALLELISM": os.environ["TOKENIZERS_PARALLELISM"],
        }
    }
    ray.init(ip_address, runtime_env=env_dict)

    param_space = {
        # wrap with training loop config b/c it's expected by trainer
        "train_loop_config": {
            "optimizer.learning_rate": tune.loguniform(lower=1e-6, upper=5e-3),
        }
    }

    # it's good for this search type when the num trials is a power of 2 (e.g., we could do 16?)
    # TODO: try to figure out what the QMC sampler is doing
    sampler = QMCSampler(seed=config.seed)
    search = OptunaSearch(sampler=sampler, metric="dev_loss", mode="min")
    search = ConcurrencyLimiter(search, max_concurrent=1)
    tune_config = tune.TuneConfig(
        search_alg=search,
        metric="dev_loss",
        mode="min",
        num_samples=config.hpo.num_hpo_trials,
    )

    trainer = get_ray_trainer(config)
    tuner = tune.Tuner(trainer, tune_config=tune_config, param_space=param_space)

    results = tuner.fit()
    print(results.get_best_result(metric="dev_loss", mode="min"))


if __name__ == "__main__":
    args = parser.parse_args()
    hpo_config = Config.from_yaml(args.config)
    main(hpo_config)