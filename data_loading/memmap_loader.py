from typing import List
import glob

import numpy as np
from torch.utils.data import Dataset

from utils import Registry, logger


@Registry.register("mmap_dataset")
class MMAPDataset(Dataset):

    NAME_TO_DTYPE = {"uint32": np.uint32, "uint16": np.uint16}

    def __init__(self, glob_paths: List[str], context_len: int, dtype: str) -> None:
        offset = 0
        self.sizes = []
        self.shards = []
        # offsets[i] = number of examples that proceed shard i
        self.offsets = []
        # shard_idxs[i] = the shard that example i is stored in
        self.shard_idxs = []
        self.context_len = context_len
        shard_paths = sorted([shard_path for glob_path in glob_paths for shard_path in glob.glob(glob_path)])
        logger.info(f"Loading memmap shards: {shard_paths}")
        for i, path in enumerate(shard_paths):
            data = np.memmap(path, dtype=self.NAME_TO_DTYPE[dtype], mode="r")
            # subtract 1 to ensure we have a extra token for targets
            num_examples_in_shard = (len(data) - 1) // self.context_len
            self.offsets.append(offset)
            self.sizes.append(num_examples_in_shard)
            self.shards.append(data)
            self.shard_idxs += [i] * num_examples_in_shard
            offset += num_examples_in_shard
        self.size = sum(self.sizes)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        shard_idx = self.shard_idxs[idx]
        elem_idx = idx - self.offsets[shard_idx]
        start_idx = elem_idx * self.context_len
        end_idx = start_idx + self.context_len
        inputs = self.shards[shard_idx][start_idx:end_idx]
        targets = self.shards[shard_idx][start_idx+1:end_idx+1]
        assert len(inputs) == self.context_len and len(targets) == self.context_len
        return {"input_ids": inputs, "labels": targets}
