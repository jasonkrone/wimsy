import torch

from utils import Registry, logger
from training.trainers.trainer import Trainer
from evaluation import AverageMetric


Registry.register("dpo_trainer")
class DPOTrainer(Trainer):

    TRAIN_REWARD_METRIC_PREFIX = "rewards"
    DEV_REWARD_METRIC_PREFIX = "dev_rewards"

    @classmethod
    def prepare_batch(cls, batch, config):
        input_ids_accept = batch.pop("input_ids_accept")
        input_ids_reject = batch.pop("input_ids_reject") 
        # concat input ids on batch axis
        assert input_ids_accept.shape[0] == input_ids_reject.shape[0]
        batch["input_ids"] = torch.cat([input_ids_accept, input_ids_reject], dim=0)
        target_dict = {
            "reference_logits_accept": batch.pop("reference_logits_accept").to(config.compute.targets_device),
            "reference_logits_reject": batch.pop("reference_logits_reject").to(config.compute.targets_device),
            "targets_accept": batch.pop("targets_accept").to(config.compute.targets_device),
            "targets_reject": batch.pop("targets_reject").to(config.compute.targets_device),
        }
        return batch, target_dict
    
    @classmethod
    def get_logits(cls, model, input_dict):
        logits = model(**input_dict)
        batch_size = logits.shape[0] // 2
        logits_dict = {
            "policy_logits_accept": logits[:batch_size],
            "policy_logits_reject": logits[batch_size:],
        }
        return logits_dict

    @classmethod
    def update_loss_dict(cls, loss_dict, step_dict, step_num):
        loss_dict = super().update_loss_dict(loss_dict, step_dict, step_num)
        if step_num > 0:
            loss_dict["reward_accept"] = torch.cat([loss_dict["reward_accept"], step_dict["reward_accept"]], dim=0)
            loss_dict["reward_reject"] = torch.cat([loss_dict["reward_reject"], step_dict["reward_reject"]], dim=0)
        return loss_dict

    @classmethod
    def get_train_metrics(cls, config):
        """
        we're leaving out logps/chosen and logps/rejected
        """
        train_metrics = super().get_train_metrics(config)
        train_metrics.update(cls.get_reward_metrics(cls.TRAIN_REWARD_METRIC_PREFIX))
        return train_metrics

    @classmethod
    def get_dev_metrics(cls, config):
        dev_metrics = super().get_dev_metrics(config)
        dev_metrics.update(cls.get_reward_metrics(cls.DEV_REWARD_METRIC_PREFIX))
        return dev_metrics

    @classmethod
    def get_reward_metrics(cls, prefix):
        return {
            f"{prefix}/accept": AverageMetric(f"{prefix}/accept"),
            f"{prefix}/rejected": AverageMetric(f"{prefix}/rejected"),
            f"{prefix}/average": AverageMetric(f"{prefix}/average"),
            f"{prefix}/accuracy": AverageMetric(f"{prefix}/accuracy"),
            f"{prefix}/margin": AverageMetric(f"{prefix}/margin"),
        }

    @classmethod
    def update_train_metrics(cls, train_metrics, loss_dict, grad_norm):
        super().update_train_metrics(train_metrics, loss_dict, grad_norm)
        train_reward_metrics = cls.compute_reward_metrics_from_loss_dict(loss_dict, cls.TRAIN_REWARD_METRIC_PREFIX)
        for metric, score in train_reward_metrics.items():
            train_metrics[metric].update(score.item())

    @classmethod
    def update_dev_metrics(cls, dev_metrics, loss_dict, logits_dict, target_dict):
        super().update_dev_metrics(dev_metrics, loss_dict, logits_dict, target_dict)
        dev_reward_metrics = cls.compute_reward_metrics_from_loss_dict(loss_dict, cls.DEV_REWARD_METRIC_PREFIX)
        for metric, score in dev_reward_metrics.items():
            dev_metrics[metric].update(score.item())

    @classmethod
    def compute_reward_metrics_from_loss_dict(cls, loss_dict, prefix):
        reward_accept = loss_dict["reward_accept"]
        reward_reject = loss_dict["reward_reject"]
        avg_reward_accept = reward_accept.mean()
        avg_reward_reject = reward_reject.mean()
        return {
            f"{prefix}/accept": avg_reward_accept,
            f"{prefix}/rejected": avg_reward_reject,
            f"{prefix}/average": (avg_reward_accept + avg_reward_reject) / 2.0,
            f"{prefix}/accuracy": (reward_accept > reward_reject).float().mean(),
            f"{prefix}/margin": (reward_accept - reward_reject).mean(),
        }

    @classmethod
    def log_train_metrics(cls, config, wandb, train_metrics, lr, i, t_delta, n_iters):
        super().log_train_metrics(config, wandb, train_metrics, lr, i, t_delta, n_iters)
        cls.log_reward_metrics(config, wandb, train_metrics, i, cls.TRAIN_REWARD_METRIC_PREFIX)

    @classmethod
    def log_dev_metrics(cls, config, wandb, dev_metrics, hpo_fn, i):
        avg_dev_loss = super().log_train_metrics(config, wandb, dev_metrics, hpo_fn, i)
        cls.log_reward_metrics(config, wandb, dev_metrics, i, cls.DEV_REWARD_METRIC_PREFIX)
        return avg_dev_loss

    @classmethod
    def log_reward_metrics(cls, config, wandb, metrics, i, prefix):
        reward_metrics_to_log = {
            name: metric.get_avg(reset=True) for name, metric in metrics.items() 
            if name.startswith(prefix)
        }
        if config.is_master:
            wandb.log(reward_metrics_to_log, step=i)
            logger.info(f"iter: {i}, rewards: {reward_metrics_to_log}")
