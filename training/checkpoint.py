import os
import pickle
from pathlib import Path
from collections import OrderedDict

import torch
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.state_dict import set_state_dict, get_state_dict, get_model_state_dict, set_model_state_dict
from torch.distributed.fsdp import ShardingStrategy

from data_loading import InfiniteDataLoader
from utils import S3UploadDownload, Registry, logger, copy_source_to_dest


class MountedFileSystemWriter(DCP.FileSystemWriter):

    METADATA_FILE_EXTENSION = "metadata"

    def finish(self, metadata, results) -> None:
        storage_md = dict()
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md
        with (self.path / f".{self.METADATA_FILE_EXTENSION}").open("wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            if self.sync_files:
                os.fsync(metadata_file.fileno())


class Checkpointer(object):

    CKPT_SUBDIR_NAME = "checkpoints"
    CKPT_FILE_EXTENSION = "pt"
    RESUME_CKPT_ON_S3 = "s3"

    @classmethod
    def get_checkpoint_dir(cls, config, ckpt_config_key="local_output_dir"):
        ckpt_base_dir = config.ckpt.get(ckpt_config_key)
        ckpt_dir = os.path.join(ckpt_base_dir, config.wandb.run_name, cls.CKPT_SUBDIR_NAME)
        if config.is_master:
            os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    @classmethod
    def get_latest_checkpoint_path(cls, ckpt_dir):
        ckpt_path = None
        if S3UploadDownload.is_s3_path(ckpt_dir):
            glob_path = os.path.join(ckpt_dir, f"*.{cls.CKPT_FILE_EXTENSION}")
            ckpt_list = S3UploadDownload().get_s3_files_matching_pattern(glob_path)
        else:
            ckpt_list = list(Path(ckpt_dir).rglob("*.pt"))

        if len(ckpt_list) > 0:
            ckpt_path = max(ckpt_list, key=cls.get_step_from_ckpt_path)

        return ckpt_path

    @classmethod
    def get_step_from_ckpt_path(cls, path):
        return int(os.path.splitext(os.path.basename(path))[0])

    @classmethod
    def save_checkpoint(cls, config, ckpt_dir, model, optimizer, learning_rate, loader, i) -> None:
        ckpt_path = os.path.join(ckpt_dir, f"{i}.{cls.CKPT_FILE_EXTENSION}")
        if config.is_master:
            ckpt_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": i,
                "config": config.as_dict(),
                "learning_rate": learning_rate,
            }
            if hasattr(loader, "state_dict"):
                ckpt_dict["loader"] = loader.state_dict()
            torch.save(ckpt_dict, ckpt_path)
        return ckpt_path

    @classmethod
    def load_checkpoint(cls, config, model, optimizer=None, loader=None, strip_prefix=None, ckpt_path=None):
        """
        Loads the given checkpoint into the model, optimizer, and loader. If strip_prefix is provided, removes
        the prefix from the model state dict before loading.

        Returns the iteration number of the checkpoint.
        """
        ckpt_iter = None
        if ckpt_path is None:
            ckpt_path = cls.get_checkpoint_to_resume(config)

        if ckpt_path is not None:
            logger.info(f"loading checkpoint: {ckpt_path}")
            # b/c only master replica is ckpt'd we set map_location so all models use the right GPUs (vs the master GPUs)
            device_map = {f"cuda:{i}": f"cuda:{config.compute.inputs_device + i}" for i in range(config.n_gpus_per_replica)}
            logger.info(f"loaded ckpt({config.rank}): {ckpt_path} with device map: {device_map}")
            ckpt = torch.load(ckpt_path, map_location=device_map)
            ckpt_iter = ckpt["iter"]

            if strip_prefix is not None:
                ckpt["model"] = cls.strip_prefix_from_state_dict(ckpt["model"], strip_prefix)

            model.load_state_dict(ckpt["model"])
            if optimizer is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if loader is not None and hasattr(loader, "load_state_dict"):
                loader.load_state_dict(ckpt["loader"])

        return ckpt_iter

    @classmethod
    def strip_prefix_from_state_dict(cls, state_dict, strip_prefix):
        out_state_dict = OrderedDict()
        prefix_len = len(strip_prefix)
        for key, value in state_dict.items():
            if key[:prefix_len] == strip_prefix:
                key = key[prefix_len:]
            out_state_dict[key] = value
        return out_state_dict

    @classmethod
    def get_checkpoint_to_resume(cls, config):
        """
        TODO: we should change our file system so that we don't need to download from S3
        """
        ckpt_path = None
        has_ckpt_to_resume = hasattr(config.ckpt, "resume_ckpt")
        # when resume = True we automatically resume from the lastest checkpoint
        if has_ckpt_to_resume and config.ckpt.resume_ckpt == True:
            ckpt_dir = cls.get_checkpoint_dir(config)
            ckpt_path = cls.get_latest_checkpoint_path(ckpt_dir)
        # when remote_output_dir is given, resume latest checkpoint stored in s3 bucket
        elif hasattr(config.ckpt, "remote_output_dir") and config.ckpt.resume_ckpt == True:
            ckpt_dir = cls.get_checkpoint_dir(config)
            ckpt_path = ckpt_dir.replace(config.ckpt.local_output_dir, config.ckpt.remote_output_dir)
        elif has_ckpt_to_resume and os.path.exists(config.ckpt.resume_ckpt):
            ckpt_path = config.ckpt.resume_ckpt

        # download checkpoint to resume if it's on S3
        if ckpt_path is not None and S3UploadDownload.is_s3_path(ckpt_path):
            s3_path = cls.get_latest_checkpoint_path(ckpt_path)
            ckpt_path = None
            if s3_path is not None:
                ckpt_dir = cls.get_checkpoint_dir(config, ckpt_config_key="copy_dir")
                ckpt_path = os.path.join(ckpt_dir, os.path.basename(s3_path))
                if not os.path.exists(ckpt_path) and config.local_rank == 0:
                    os.makedirs(ckpt_dir, exist_ok=True)
                    logger.info(f"downloading s3 checkpoint to resume from {s3_path} to {ckpt_path}")
                    if s3_path[-3:] == f".{cls.CKPT_FILE_EXTENSION}":
                        S3UploadDownload().download_file(s3_path, ckpt_path)
                    else:
                        S3UploadDownload().download_dir(s3_path, ckpt_path)
                S3UploadDownload.wait_for_download_to_complete(ckpt_path)
        return ckpt_path



@Registry.register("fsdp_checkpointer")
class FSDPCheckpointer(Checkpointer):

    @classmethod
    def get_latest_checkpoint_path(cls, ckpt_dir):
        """
        Correctness of this function requires that the .pt files are stored w/in the FSDP checkpoint dir
        """
        ckpt_path = super().get_latest_checkpoint_path(ckpt_dir)
        if ckpt_path is not None:
            ckpt_path = os.path.dirname(ckpt_path)
        return ckpt_path

    @classmethod
    def _get_state_dict(cls, model, optimizer=None):
        if optimizer is None:
            model_state_dict = get_model_state_dict(model)
            state_dict = {"model": model_state_dict}
        else:
            model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
            state_dict = {
                "model": model_state_dict,
                "optim": optimizer_state_dict
            }
        return state_dict

    @classmethod
    def _set_state_dict(cls, state_dict, model, optimizer=None):
        if optimizer is None:
            set_model_state_dict(model, model_state_dict=state_dict["model"])
        else:
            set_state_dict(
                model,
                optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"]
            )

    @classmethod
    def save_checkpoint(cls, config, ckpt_dir, model, optimizer, learning_rate, loader, i) -> None:
        """
        Checkpointing approach copied from
        https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/examples/fsdp_checkpoint_example.py
        """
        assert model.sharding_strategy == ShardingStrategy.HYBRID_SHARD

        ckpt_path = os.path.join(ckpt_dir, f"{i}")
        os.makedirs(ckpt_path, exist_ok=True)
        cls.save_metadata(config, learning_rate, loader, i, ckpt_path)

        state_dict = cls._get_state_dict(model, optimizer)

        if config.rank == config.local_rank:
            DCP.save(state_dict, storage_writer=MountedFileSystemWriter(ckpt_path), process_group=model.process_group)

        return ckpt_path

    @classmethod
    def get_metadata_path(cls, ckpt_dir):
        return os.path.join(ckpt_dir, f"meta.{cls.CKPT_FILE_EXTENSION}")

    @classmethod
    def get_step_from_ckpt_path(cls, path):
        # note that here path is gonna get that path of the metadata.pt file
        return int(os.path.basename(os.path.dirname(path)))

    @classmethod
    def save_metadata(cls, config, learning_rate, loader, i, ckpt_path):
        if config.is_master:
            metadata = {"iter": i, "config": config.as_dict(), "learning_rate": learning_rate}
            if hasattr(loader, "state_dict"):
                metadata["loader"] = loader.state_dict()
            metadata_path = cls.get_metadata_path(ckpt_path)
            torch.save(metadata, metadata_path)

    @classmethod
    def load_checkpoint(cls, config, model, optimizer=None, loader=None, strip_prefix=None, ckpt_path=None):
        """
        Loads the given checkpoint into the model, optimizer, and loader.

        Returns the iteration number of the checkpoint.
        """
        ckpt_iter = None
        if ckpt_path is None:
            ckpt_path = cls.get_checkpoint_to_resume(config)

        # copy to local otherwise ckpt hangs sometimes if it's on s3 mount
        if ckpt_path is not None and config.ckpt.get("copy_ckpt_to_local", True) and config.local_rank == 0:
            local_ckpt_path = ckpt_path.replace(config.ckpt.local_output_dir, "/tmp")
            copy_source_to_dest(ckpt_path, local_ckpt_path)
            ckpt_path = local_ckpt_path

        torch.distributed.barrier()

        if ckpt_path is not None:
            logger.info(f"loading checkpoint: {ckpt_path}")
            state_dict = cls._get_state_dict(model, optimizer)
            DCP.load(state_dict=state_dict, storage_reader=DCP.FileSystemReader(ckpt_path))
            cls._set_state_dict(state_dict, model, optimizer)
            ckpt_iter = cls.load_metadata(ckpt_path, loader=loader)
        return ckpt_iter

    @classmethod
    def load_metadata(cls, ckpt_path, loader=None):
        metadata_path = cls.get_metadata_path(ckpt_path)
        metadata = torch.load(metadata_path, map_location="cpu")
        # infinite dataloader has a wrapper around the actual dataloader
        loader_obj = loader if not isinstance(loader, InfiniteDataLoader) else loader.data_loader
        if loader is not None and hasattr(loader_obj, "load_state_dict"):
            loader.load_state_dict(metadata["loader"])
        return metadata["iter"]