import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import streaming
import numpy as np

from transformers import DataCollatorForSeq2Seq

from data_loading import MMAPDataset
from utils import Registry, logger

# register data classes that can be specified in the config
Registry.register("distributed_sampler")(DistributedSampler)
Registry.register("random_sampler")(RandomSampler)
Registry.register("data_loader")(DataLoader)


class InfiniteDataLoader(object):
    """
    Wraps a DataLoader to make it infinite, always reloading the data when it runs out.
    """

    def __init__(self, data_loader) -> None:
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)

    def state_dict(self):
        state_dict = None
        if hasattr(self.data_loader, "state_dict"):
            state_dict = self.data_loader.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        if hasattr(self.data_loader, "load_state_dict"):
            logger.info(f"Loading state dict for DataLoader")
            self.data_loader.load_state_dict(state_dict)
        else:
            error_message = f"DataLoader of class: {self.data_loader.__class__} doesn't have load_state_dict method"
            raise NotImplementedError(error_message)

    def __iter__(self):
        return self

    def __next__(self):
        out = None
        try:
            out = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            out = next(self.data_iter)
        return out


@Registry.register("default_collator")
class DefaultCollator(object):

    def __call__(self, batch):
        out = {}
        masks = []
        labels = []
        input_ids = []
        for a_dict in batch:
            input_ids.append(a_dict["input_ids"])
            labels.append(a_dict["labels"])
            if "mask" in a_dict:
                masks.append(a_dict["mask"])
        out["input_ids"] = torch.as_tensor(np.stack(input_ids).astype(np.int64))
        out["labels"] = torch.as_tensor(np.stack(labels).astype(np.int64))
        if len(masks):
            out["mask"] = torch.as_tensor(np.stack(masks).astype(np.float32))
        return out


@Registry.register("seq2seq_collator", "from_kwargs")
class Seq2SeqCollator(DataCollatorForSeq2Seq):

    @classmethod
    def from_kwargs(cls, tokenizer, padding):
        tokenizer_cls = Registry.get(tokenizer.id)
        return cls(tokenizer=tokenizer_cls(**tokenizer.args), padding=padding)


def get_data_loader(dataset_cfg, loader_cfg, sampler_cfg=None, ckpt=None):
    sampler = None
    # avoid pickling error that occurs if we try to use MMAPDataset from registry
    if dataset_cfg["id"] == "mmap_dataset":
        dataset_cls = MMAPDataset
    else:
        dataset_cls = Registry.get(dataset_cfg["id"])

    dataset = dataset_cls(**dataset_cfg["args"])

    if sampler_cfg is not None:
        sampler_args = sampler_cfg.get("args", {})
        sampler_cls = Registry.get(sampler_cfg["id"])
        sampler = sampler_cls(dataset, **sampler_args)

    loader_cls = Registry.get(loader_cfg["id"])

    if loader_cfg["args"].get("collator"):
        collator_cfg = loader_cfg["args"].pop("collator")
        collator_cls = Registry.get(collator_cfg["id"])
        collator = collator_cls(**collator_cfg["args"])
    else:
        collator = DefaultCollator()

    loader = loader_cls(dataset, sampler=sampler, collate_fn=collator, **loader_cfg["args"])
    # auto-reloads data when it runs out after 1 epoch
    loader = InfiniteDataLoader(loader)
    return loader

