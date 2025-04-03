from typing import Union, List, Dict
import os
import sys
import shutil
from collections import Counter

import numpy as np
import torch

sys.path.append("./streaming")
from streaming import Stream, StreamingDataset, StreamingDataLoader

from utils import Registry

# register streaming loader to be specified in config
Registry.register("streaming_data_loader")(StreamingDataLoader)


@Registry.register("streaming_text_dataset", "from_args")
class StreamingTextDataset(StreamingDataset):

    NUM_PREP_SHARD_RETRIES = 20 

    def __init__(self, load_mask=False, **kwargs):
        self.load_mask = load_mask
        super().__init__(**kwargs)

    @classmethod
    def from_args(cls, **kwargs):
        streams = [Stream(**stream) for stream in kwargs.pop("streams")]
        return cls(streams=streams, **kwargs)

    def get_item(self, sample_id: int):
        """
        Retry which defaults to 7 in the original code as of 12/30/24.
        We wrap the parent function to increase retry.
        """
        return super().get_item(sample_id, retry=self.NUM_PREP_SHARD_RETRIES)

    def __getitem__(self, idx: int) -> Union[Dict[str, List[int]], torch.Tensor]:
        sample = super().__getitem__(idx)
        out = {
            "input_ids": sample["inputs"],
            "labels": sample["targets"]
        }
        if self.load_mask:
            out["mask"] = self._mask_for_doc_ids(sample["doc_ids"], out["input_ids"].size)
        return out

    @staticmethod
    def _mask_for_doc_ids(ids, context_len, pad_value=float("-inf"), nonpad_value=0, return_none_if_causal=False):
        """
        Returns a causal mask that respects document boundaries with mask == pad_value
        where attention should be masked and nonpad_value elsewhere.

        ids:
            list of document ids. ids are required to be unique for each document
            and there must be one id per token in the input sequence.
        """
        assert len(ids) == context_len

        if len(set(ids)) == 1 and return_none_if_causal:
            return None

        offset = 0
        id_counts = Counter(ids)
        mask = np.ones((context_len, context_len)) * pad_value

        while offset < context_len:
            doc_id = ids[offset]
            count = id_counts[doc_id]
            ones = np.ones((count, count))
            lower = np.tril(ones * nonpad_value)
            upper = np.triu(ones * pad_value, k=1)
            causal_mask = lower + upper
            mask[offset:offset + count, offset:offset + count] = causal_mask
            offset += count

        return mask

