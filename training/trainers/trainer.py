import os
import math
import time
import json
import random
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm
import wandb as weights_and_biases

import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from model import FSDPModel, LayerNorm, RMSNorm
from data_loading import get_data_loader
from utils import EarlyStopper, logger, Registry
from tokenizer import *
from training import get_learning_rate_scheduler, World, FSDPCheckpointer
from evaluation import AverageMetric, ModelFlopsUtilization


torch._dynamo.config.suppress_errors = True


@Registry.register("trainer")
class Trainer(object):

    PROFILER_SUBDIR_NAME = "profile_traces"

    @classmethod
    def train(cls, config, wandb=None, hpo_fn=None) -> None:
        """
        Trains the model specified in the given config. If hpo_fn is provided, we use it to log the dev loss.
        Uses wandb to log metrics. If wandb is not provided, we initialize a new wandb run.
        """
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

        config.world_size = 1
        config.is_master = True

        #assert config.ckpt.ckpt_interval % config.dev_eval.eval_interval == 0

        if config.model.precision == "fp16":
            raise ValueError("fp16 not currently supported. Implementation with amp autocast & grad scaler required.")

        if "cuda" in config.compute.device:
            World.set_world_state(config)
            cls.setup_distributed_processes(config)

        model = cls.get_model(config)
        optimizer = cls.get_optimizer(config, model)

        wandb = cls.setup_wandb(config, wandb)
        logger.info(f"model params: {sum(p.numel() for p in model.parameters())}")

        sampler = config.data.get("sampler")
        train_loader = get_data_loader(config.data.train_dataset, config.data.data_loader, sampler)

        dev_loader = None
        if hasattr(config.data, "dev_dataset"):
            dev_loader = get_data_loader(config.data.dev_dataset, config.data.data_loader, sampler)

        # we store loss by ckpt so we can load best model by dev loss later
        best_ckpt = None
        avg_dev_loss = None
        ckpt_to_dev_loss = {}

        checkpointer, ckpt_iter = cls.load_checkpoint(config, model, optimizer, train_loader)
        ckpt_dir = checkpointer.get_checkpoint_dir(config)

        # compile the model, if needed after loading ckpt
        if config.model.do_compile:
            model = torch.compile(model)

        cls.set_max_tokens_or_iters(config)
        lr_scheduler = get_learning_rate_scheduler(config, optimizer, ckpt_iter)

        stop_early = False
        early_stopper = EarlyStopper(config.regularization.early_stopping_patience)

        train_metrics_dict = cls.get_train_metrics(config)
        dev_metrics_dict = cls.get_dev_metrics(config)

        profiler = cls.get_profiler(config)

        i = 0 if ckpt_iter is None else ckpt_iter + 1
        pbar = tqdm(total=config.iters.max_iters, initial=i, desc="Training")
        t_log = time.time()
        i_log = -1 if ckpt_iter is None else ckpt_iter
        t_eval = time.time() # TODO: t eval is temporary

        # set gradient accumlation steps to 1 if not set
        if not hasattr(config.loss, "grad_accum_steps"):
            config.loss.grad_accum_steps = 1

        assert config.iters.batch_size % config.loss.grad_accum_steps == 0
        per_step_batch_size = config.iters.batch_size // config.loss.grad_accum_steps
        per_step_loss_weight = config.loss.get("per_grad_accum_step_loss_coef", 1.0 / config.loss.grad_accum_steps)

        # run dev eval before train to prevent torch.compile issue
        #cls.evaluate(model, dev_loader, config, 1, dev_metrics_dict)

        for i in range(i, config.iters.max_iters):
            loss_dict = {}
            batch = next(train_loader)
            optimizer.zero_grad(set_to_none=True)

            # grad accum approach for FSDP taken from 
            # https://github.com/mlfoundations/open_lm/blob/9bb92ef1689333534b7057942a20d18a46d1fa52/open_lm/train.py
            for j in range(config.loss.grad_accum_steps):
                # avoid syncing grads until last accum step if using FSDP
                grad_sync_context = nullcontext
                if isinstance(model, FSDPModel) and j != config.loss.grad_accum_steps - 1:
                    grad_sync_context = model.no_sync

                mini_batch = {
                    k: v[j*per_step_batch_size:(j+1)*per_step_batch_size]
                    for k, v in batch.items()
                }

                step_dict = cls.accumulate_gradient(config, model, grad_sync_context, mini_batch, per_step_loss_weight)
                loss_dict = cls.update_loss_dict(loss_dict, step_dict, j)

            loss_dict["grad_norm"] = cls.clip_grad_norm(model, config.loss.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            if profiler:
                profiler.step()

            if torch.all(loss_dict["loss"].isnan()):
                raise ValueError(f"loss is NaN at step: {i}")

            cls.update_train_metrics(train_metrics_dict, loss_dict)

            # log training metrics
            if cls.is_log_interval(config, i):
                t_delta = time.time() - t_log
                n_iters = i - i_log
                cls.log_train_metrics(config, wandb, train_metrics_dict, lr_scheduler.get_last_lr()[0], i, t_delta, n_iters)
                pbar.update(n_iters)
                t_log = time.time()
                i_log = i

            # compute dev metrics
            if dev_loader and cls.is_eval_interval(config, i) or cls.is_eval_time(config, t_eval):
                cls.evaluate(model, dev_loader, config, config.dev_eval.max_eval_iters, dev_metrics_dict)
                avg_dev_loss = cls.log_dev_metrics(config, wandb, dev_metrics_dict, hpo_fn, i)
                stop_early = early_stopper(avg_dev_loss)
                # TODO: this makes eval only run once @ n_secs vs. @ every n_secs
                t_eval = None

            if stop_early:
                pbar.update()
                logger.info(f"Stopping early at iter {i}")
                break

            if cls.is_checkpoint_interval(config, i):
                ckpt_path = checkpointer.save_checkpoint(
                    config, ckpt_dir, model, optimizer, lr_scheduler.get_last_lr()[0], train_loader, i
                )
                if avg_dev_loss is not None:
                    ckpt_to_dev_loss[ckpt_path] = avg_dev_loss
                else:
                    # if no dev eval, we assume the most recent ckpt is the best model
                    best_ckpt = ckpt_path

        if len(ckpt_to_dev_loss) > 0:
            cls.save_dict_as_json(config, ckpt_to_dev_loss, os.path.join(ckpt_dir, "ckpt_to_dev_loss.json"))
            best_ckpt, best_dev_loss = min(ckpt_to_dev_loss.items(), key=lambda t: t[1])
            logger.info(f"best ckpt: {best_ckpt} had dev loss: {best_dev_loss}")

        if config.ckpt.get("save_best_model_at_end_for_eval") and best_ckpt is not None:
            eval_ckpt = cls.save_checkpoint_for_eval(config, model, checkpointer, best_ckpt, ckpt_dir)
            logger.info(f"saved best ckpt: {best_ckpt} to {eval_ckpt} for eval")
            best_ckpt = eval_ckpt

        # optional (e.g., in case want to shutdown after final phase of training)
        if config.compute.get("shutdown_distributed_processes_at_end", True):
            cls.shutdown_distributed_processes(config)

        if not hasattr(config, "hpo"):
            weights_and_biases.finish()

        return best_ckpt, ckpt_to_dev_loss

    @classmethod
    def update_loss_dict(cls, loss_dict, step_dict, step_num):
        if step_num == 0:
            loss_dict = step_dict
        else:
            loss_dict["loss"] += step_dict["loss"]
        return loss_dict

    @classmethod
    def accumulate_gradient(cls, config, model, context, batch, loss_coef):
        with context():
            input_dict, target_dict = cls.prepare_batch(batch, config)
            logits_dict = cls.get_logits(model, input_dict)
            # instantiating loss each iter runs faster than creating it once
            loss_fn = cls.get_loss_fn(config)
            loss_dict = loss_fn(**logits_dict, **target_dict)
            loss_dict["loss"] = loss_dict["loss"] * loss_coef
            loss_dict["loss"].backward()
        return loss_dict

    @classmethod
    def get_train_metrics(cls, config):
        return {
            "loss": AverageMetric("train_loss"),
            "mfu": ModelFlopsUtilization.from_config(config),
            "grad_norm": AverageMetric("grad_norm"),
        }

    @classmethod
    def get_dev_metrics(cls, config):
        dev_metrics = {"loss": AverageMetric("dev_loss")}
        for key, metric_dict in config.dev_eval.get("dev_metrics", {}).items():
            metric_cls = Registry.get(metric_dict["id"])
            dev_metrics[key] = metric_cls(**metric_dict["args"])
        return dev_metrics

    @classmethod
    def update_train_metrics(cls, train_metrics, loss_dict):
        train_metrics["loss"].update(loss_dict["loss"].item())
        train_metrics["grad_norm"].update(loss_dict["grad_norm"])

    @classmethod
    def update_dev_metrics(cls, dev_metrics, loss_dict, logits_dict, target_dict):
        for name, metric in dev_metrics.items():
            if name == "loss":
                dev_metrics["loss"].update(loss_dict["loss"].item())
            else:
                metric.update(logits_dict=logits_dict, target_dict=target_dict)

    @classmethod
    def log_train_metrics(cls, config, wandb, train_metrics, lr, i, t_delta, n_iters):
        mfu = train_metrics["mfu"]
        train_loss = train_metrics["loss"]
        grad_norm = train_metrics["grad_norm"]

        mfu.update(batch_size=config.iters.batch_size, n_iters=n_iters, iter_secs=t_delta)
        avg_mfu, avg_throughput = mfu.get_avg(reset=config.dev_eval.reset_mfu)
        avg_train_loss = train_loss.get_avg(reset=True)
        avg_grad_norm = grad_norm.get_avg(reset=True)

        if config.is_master:
            n_tokens_seen = (i + 1) * config.iters.batch_size * config.model.max_len * config.num_replicas
            metrics_dict = {
                "loss": avg_train_loss,
                "current throughput (token per gpu per sec)": avg_throughput,
                "learning rate": lr,
                "token seen": n_tokens_seen,
                "mfu": avg_mfu,
                "gradient norm": avg_grad_norm,
            }
            wandb.log(metrics_dict, step=i)
            logger.info(f"iter: {i}, loss: {avg_train_loss}, grad_norm: {avg_grad_norm}")

    @classmethod
    def log_dev_metrics(cls, config, wandb, dev_metrics, hpo_fn, i):
        scores_dict = {name: metric.get_avg(reset=True) for name, metric in dev_metrics.items()}
        avg_dev_loss = scores_dict.pop("loss")
        if config.is_master:
            scores_dict.update({"dev_loss": avg_dev_loss, "iter": i})
            logger.info(f"{scores_dict}")
            wandb.log(scores_dict)
        if hpo_fn is not None:
            hpo_fn(avg_dev_loss, i)
        return avg_dev_loss

    @classmethod
    def save_dict_as_json(cls, config, a_dict, save_path):
        if config.is_master:
            with open(save_path, "w") as f:
                json.dump(a_dict, f)

    @classmethod
    def get_loss_fn(cls, config):
        loss_cls = Registry.get(config.loss.loss_fn.id)
        loss_fn = loss_cls(**config.loss.loss_fn.args)
        return loss_fn

    @classmethod
    def get_logits(cls, model, input_dict):
        logits = model(**input_dict)
        logits = logits.view(-1, logits.size(-1))
        logits_dict = {"input": logits}
        return logits_dict

    @classmethod
    def clip_grad_norm(cls, model, max_grad_norm):
        if hasattr(model, "clip_grad_norm_"):
            # use FSDP clip grad b/c shared and non-sharded params need to be handled separately
            norm = model.clip_grad_norm_(max_grad_norm).item()
        else:
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
        return norm

    @classmethod
    def load_checkpoint(cls, config, model, optimizer, loader):
        ckpt_iter = None
        checkpointer_cls = Registry.get(config.ckpt.checkpointer)
        checkpointer = checkpointer_cls()

        if hasattr(config.ckpt, "resume_ckpt"):
            load_kwargs = {
                "config": config,
                "model": model,
                "strip_prefix": config.ckpt.get("strip_prefix"),
            }
            if config.ckpt.get("load_dataset_loader", True):
                load_kwargs["loader"] = loader
            if config.ckpt.get("load_optimizer", True):
                load_kwargs["optimizer"] = optimizer
            ckpt_iter = checkpointer.load_checkpoint(**load_kwargs)
 
        # resume training from specific iter, impacts lr_schedule
        if hasattr(config.ckpt, "resume_from_iter"):
            ckpt_iter = config.ckpt.resume_from_iter
            logger.info(f"resuming training config specified ckpt iter: {ckpt_iter}")

        return checkpointer, ckpt_iter

    @classmethod
    def save_checkpoint_for_eval(cls, config, model, checkpointer, ckpt_path, ckpt_dir):
        assert isinstance(checkpointer, FSDPCheckpointer)
        save_path = os.path.join(ckpt_dir, f"{os.path.basename(ckpt_path)}.pt")
        if config.is_master:
            dcp_to_torch_save(ckpt_path, save_path)
        torch.distributed.barrier()
        return save_path

    @classmethod
    def prepare_batch(cls, batch, config):
        batch["input_ids"] = batch["input_ids"].to(config.compute.inputs_device)
        if "mask" in batch:
            batch["mask"] = batch["mask"].to(config.compute.targets_device)
        labels = batch.pop("labels").to(config.compute.targets_device)
        labels = labels.long().view(-1)
        target_dict = {"target": labels}
        return batch, target_dict

    @classmethod
    def is_log_interval(cls, config, i):
        return (i % config.dev_eval.log_interval == 0) or i == config.iters.max_iters - 1

    @classmethod
    def is_eval_interval(cls, config, i):
        is_interval = False
        if hasattr(config.dev_eval, "eval_interval"):
            is_interval = (i % config.dev_eval.eval_interval == 0 and i != 0) or i == config.iters.max_iters - 1
        return is_interval

    @classmethod
    def is_eval_time(cls, config, t_eval):
        """
        t_start: time-stamp from start of training or prior eval
        """
        is_time = False
        if t_eval is not None and hasattr(config.dev_eval, "eval_every_n_secs"):
            n_secs_since_eval = time.time() - t_eval
            if n_secs_since_eval >= config.dev_eval.eval_every_n_secs:
                is_time = True
        return is_time

    @classmethod
    def is_checkpoint_interval(cls, config, i):
        return (i % config.ckpt.ckpt_interval == 0 and i != 0) or i == config.iters.max_iters - 1

    @classmethod
    @torch.no_grad()
    def evaluate(cls, model, loader, config, max_iters, dev_metrics) -> AverageMetric:
        """
        Returns the average loss on the dataset. If decoding & tokenizer are given, we return
        a table of generated text and gold text examples.
        """
        model.eval()
        for i, batch in tqdm(enumerate(loader), desc="Dev set evaluation"):
            input_dict, target_dict = cls.prepare_batch(batch, config)
            logits_dict = cls.get_logits(model, input_dict)
            loss_fn = cls.get_loss_fn(config)
            loss_dict = loss_fn(**logits_dict, **target_dict)
            cls.update_dev_metrics(dev_metrics, loss_dict, logits_dict, target_dict)

            if i == max_iters - 1:
                break

        model.train()

    @classmethod
    def get_model(cls, config):
        model_cls_id = config.model.id if not hasattr(config.model, "parallelism") else config.model.parallelism
        model_cls = Registry.get(model_cls_id)
        model_kwargs = {"config": config}
        model = model_cls(**model_kwargs)
        return model

    @classmethod
    def get_optimizer(cls, config, model, optimizer_cls=torch.optim.AdamW) -> torch.optim.Optimizer:
        # only apply weight decay to >= 2D layers (i.e. not the biases)
        # importance of this choice noted here: https://arxiv.org/pdf/2102.06171.pdf
        is_fsdp = isinstance(model, FSDP)
        if not is_fsdp or (is_fsdp and config.model.do_compile):
            params = [
                {"params": param_list, "weight_decay": decay} 
                for decay, param_list in cls.group_params_by_decay(config, model).items()
            ]
        elif is_fsdp and not config.model.do_compile:
            # TODO: keep this as temp
            params = model.parameters()
        else:
            raise ValueError("unsupported configuration for optimizer")

        kwargs = {
            "params": params,
            "lr": config.optimizer.learning_rate,
            "betas": (config.optimizer.beta_1, config.optimizer.beta_2),
            "eps": config.optimizer.epsilon,
            "fused": config.compute.device == "cuda",
        }
        optimizer = optimizer_cls(**kwargs)
        return optimizer

    @classmethod
    def group_params_by_decay(cls, config, model):
        """
        Adapted from get_param_groups in the OLMo code base
        https://github.com/allenai/OLMo/blob/878cc8ca3f2820d162f373b22ee6263f8252a28d/olmo/optim.py#L375
        """
        decay = set()
        no_decay = set()
        all_params = {}
        decay_to_params = {}

        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                all_params[fpn] = p

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, nn.Linear):
                    decay.add(fpn)
                elif isinstance(m, RMSNorm) or isinstance(m, LayerNorm):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, nn.Embedding):
                    if config.regularization.get("do_decay_embeddings", False):
                        decay.add(fpn)
                    else:
                        no_decay.add(fpn)

        # Validate that we've considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert (len(all_params.keys() - union_params) == 0), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

        if len(decay) > 0:
            decay_to_params[config.regularization.weight_decay] = [all_params[pn] for pn in sorted(list(decay))]
        if len(no_decay) > 0:
            decay_to_params[0.0] = [all_params[pn] for pn in sorted(list(no_decay))]

        return decay_to_params

    @classmethod
    def setup_wandb(cls, config, wandb):
        if config.is_master and wandb is None:
            wandb = weights_and_biases
            wandb.init(project=config.wandb.project_name, name=config.wandb.run_name, config=config.as_dict())
        return wandb

    @classmethod
    def get_profiler(cls, config):
        profiler = None
        if config.is_master and hasattr(config, "profiler") and config.profiler.use_profiler:
            trace_dir = os.path.join(config.ckpt.local_output_dir, cls.PROFILER_SUBDIR_NAME, config.wandb.run_name)
            os.makedirs(trace_dir, exist_ok=True)
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(**config.profiler.schedule_args),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
                profile_memory=True,
                with_stack=False,
                record_shapes=True,
            )
        return profiler

    @classmethod
    def set_max_tokens_or_iters(cls, config):
        """
        Infers max tokens or max iters if either is not set and updates their value in the config
        """
        if not hasattr(config.iters, "max_tokens"):
            config.iters.max_tokens = None
        if not hasattr(config.iters, "max_iters"):
            config.iters.max_iters = None

        tokens_per_iter = config.iters.batch_size * config.num_replicas * config.model.max_len

        if config.iters.max_iters is None:
            # number of tokens that we're training on per iter, accounts for distributed training via num_replicas
            config.iters.max_iters = math.ceil(config.iters.max_tokens / tokens_per_iter)
        elif config.iters.max_tokens is None:
            config.iters.max_tokens = config.iters.max_iters * tokens_per_iter
        elif config.iters.max_tokens is not None and config.iters.max_iters is not None:
            assert config.iters.max_iters == math.ceil(config.iters.max_tokens / tokens_per_iter)

        logger.info(f"Training for max_iters: {config.iters.max_iters} == max_tokens: {config.iters.max_tokens}")

    @classmethod
    def shutdown_distributed_processes(cls, config):
        if config.compute.is_distributed:
            destroy_process_group()

    @classmethod
    def setup_distributed_processes(cls, config):
        config.compute.device = f"cuda:{config.local_rank}"
        config.compute.inputs_device = config.local_rank
        config.compute.targets_device = config.local_rank
        torch.cuda.set_device(config.compute.device)

        config.num_replicas = config.world_size

        if config.compute.get("init_process_group_at_start", True):
            init_process_group(backend=config.compute.backend, rank=config.rank, world_size=config.world_size, device_id=torch.device(config.compute.device))
 