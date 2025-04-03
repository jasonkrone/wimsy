from typing import List
from functools import partial

from datasets import load_dataset
from torch.utils.data import Dataset

from utils import Registry, logger
from training.constants import LossConstants


@Registry.register("hf_mc")
class HuggingFaceMultiChoiceDataset(Dataset):

    def __init__(
        self, 
        tokenizer, 
        dataset_path: str, 
        split: str, 
        choices: List[str],
        task: str = None,
        ignore_index: int = LossConstants.IGNORE_INDEX,
        max_length: int = None,
    ) -> None:

        self.ignore_index = ignore_index
        # set padding for tokenizer
        tokenizer_cls = Registry.get(tokenizer.id)
        self.tokenizer = tokenizer_cls(**tokenizer.args)
        if self.tokenizer.pad_token:
            pass
        elif self.tokenizer.unk_token:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        elif self.tokenizer.eos_token:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            raise ValueError("tokenizer has no pad_token or eos_token")

        # check that all choices are single tokens
        for choice in choices:
            assert len(self.tokenizer.encode(choice)) == 1

        logger.info(f"loading dataset from {dataset_path} with split {split}")
        self.hf_dataset = load_dataset(dataset_path, split=split)

        # filter dataset to only contain task
        self.task = task
        if self.task is not None:
            self.hf_dataset = self.hf_dataset.filter(lambda x: x["task"] == self.task)
        
        # tokenize the dataset
        logger.info("tokenizing dataset")
        self.hf_dataset = self.hf_dataset.map(self._get_token_ids_and_len)

        # pad each example to max length
        if max_length is None:
            max_length = max(self.hf_dataset["length"])
        logger.info(f"padding dataset to max length {max_length}")
        pad_fn = partial(self._add_padding, max_length=max_length)
        self.hf_dataset = self.hf_dataset.map(pad_fn)
 
        # Drop unnecessary columns from the dataset
        columns_to_drop = ["input", "target", "length", "task"]
        self.hf_dataset = self.hf_dataset.remove_columns(columns_to_drop)
        self.size = len(self.hf_dataset)

    def _add_padding(self, example, max_length):
        n_pad = max_length - example["length"]
        pad_tokens = [self.tokenizer.pad_token_id]*n_pad
        pad_labels = [self.ignore_index]*n_pad
        example["input_ids"] = example["input_ids"] + pad_tokens
        example["labels"] = example["labels"] + pad_labels
        return example

    def _get_token_ids_and_len(self, example):
        tokens = self.tokenizer.encode(example["input"]+example["target"])
        labels = [self.ignore_index] * len(tokens)
        # two tokens back is the token prior to the target, since the target is one token long
        labels[-2] = tokens[-1]
        return {"input_ids": tokens, "length": len(tokens), "labels": labels}

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        return row
