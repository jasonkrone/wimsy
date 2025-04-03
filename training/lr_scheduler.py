import math

import torch
from torch.optim.lr_scheduler import LRScheduler

from utils import Registry


@Registry.register("cosine_with_linear_warmup_lr_schedule")
class CosineAnnealingWithLinearWarmupLR(LRScheduler):

    def __init__(self, optimizer, warmup_iters, max_iters, lr_max, cos_lr_min=None, warmup_lr_min=0, last_epoch=-1, verbose=False):
        """
        Warms the LR from warmup_lr_min to lr_max over t_warmup steps, then anneals lr_max down to 
        cos_lr_min over t_max - t_warmup steps.

        If cos_lr_min is None, sets cos_lr_min = lr_max / 10 as rec'd by the Chinchilla paper
        """
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.anneal_steps = max_iters - warmup_iters
        self.lr_max = lr_max
        self.cos_lr_min = cos_lr_min
        self.warmup_lr_min = warmup_lr_min
        if self.cos_lr_min is None:
            self.cos_lr_min = lr_max / 10.0
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # last epoch is like the init counter value for step num
        lr_list = None
        t = self.last_epoch
        if t < self.warmup_iters:
            lr_list = self._get_lr_linear_warmup(t)
        elif self.warmup_iters <= t < self.max_iters:
            lr_list = self._get_lr_cos_anneal(t)
        else:
            return [self.cos_lr_min for _ in self.optimizer.param_groups]
        return lr_list

    def _get_lr_linear_warmup(self, t):
        c = t / float(self.warmup_iters)
        lr = self.warmup_lr_min + (self.lr_max - self.warmup_lr_min) * c
        return [lr for _ in self.optimizer.param_groups]

    def _get_lr_cos_anneal(self, t):
        t_anneal = t - self.warmup_iters
        lr = self.cos_lr_min + 0.5*(self.lr_max - self.cos_lr_min)*(1 + math.cos(math.pi * t_anneal / self.anneal_steps))
        return [lr for _ in self.optimizer.param_groups]


@Registry.register("linear_annealing_lr_schedule")
class LinearAnnealingLR(LRScheduler):

    def __init__(self, optimizer, max_iters, lr_max, lr_min, warmup_iters=0, last_epoch=-1, verbose=False):
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.slope = (lr_max - lr_min) / (max_iters - warmup_iters)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        iters_past_warmup = self.last_epoch - self.warmup_iters
        if iters_past_warmup >= 0 and self.last_epoch < self.max_iters:
            lr = self.lr_max - iters_past_warmup * self.slope
        elif self.last_epoch >= self.max_iters:
            lr = self.lr_min
        else:
            lr = (self.last_epoch / self.warmup_iters) * self.lr_max
        return [lr for _ in self.optimizer.param_groups]


def get_learning_rate_scheduler(config, optimizer, ckpt_iter=None):
    """
    Returns the learning rate scheduler specified in the config.
    If ckpt is not None, sets the scheduler's current iter from the ckpt.
    """
    lr_scheduler_kwargs = config.lr_schedule.args.as_dict()

    if ckpt_iter is not None:
        lr_scheduler_kwargs["last_epoch"] = ckpt_iter
        # there's currently a bug where the distributed checkpointer doesn't save
        # initial_lr which is expected by the learning rate scheduler, so we set it
        # https://github.com/pytorch/pytorch/issues/140900
        if config.ckpt.checkpointer == "fsdp_checkpointer":
            for group in optimizer.param_groups:
                initial_lr = group["lr"]
                if isinstance(initial_lr, torch.Tensor):
                    initial_lr = initial_lr.clone()
                group.setdefault("initial_lr", initial_lr)
    
    # if max_iter_units == percentage, we set it as a percentage of config.iters.max_iters
    if lr_scheduler_kwargs.pop("max_iter_units", None) == "percentage":
        schedule_max_iters = lr_scheduler_kwargs["max_iters"] * config.iters.max_iters
        lr_scheduler_kwargs["max_iters"] = schedule_max_iters
    if lr_scheduler_kwargs.pop("warmup_iter_units", None) == "percentage":
        schedule_max_iters = lr_scheduler_kwargs["warmup_iters"] * config.iters.max_iters
        lr_scheduler_kwargs["warmup_iters"] = schedule_max_iters

    scheduler_cls = Registry.get(config.lr_schedule.id)
    scheduler = scheduler_cls(
        optimizer=optimizer,
        **lr_scheduler_kwargs,
    )
    return scheduler

