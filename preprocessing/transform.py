import os
import re
import json
import math
import glob
import pickle
import string
import shutil
from copy import deepcopy
from contextlib import ExitStack
from abc import abstractmethod, ABC
from collections import defaultdict
from multiprocessing import Pool

import msgspec
import smart_open
import numpy as np
import pandas as pd
from jsonpath_ng import parse
from jsonpath_ng.ext import parse as parse_extended

from dolma.core.parallel import BaseParallelProcessor, METADATA_SUFFIX, AllPathsTuple
from dolma.core.paths import mkdir_p, split_path, join_path
from dolma.core.errors import DolmaError, DolmaRetryableFailure

from preprocessing.constants import SkypilotConstants
from utils import Registry, get_value_at_jsonpath, logger, split_path_with_compound_extension, ParallelTQDM


def parent(path: str) -> str:
    """
    Get the parent directory of a path; if the parent is the root, return the root.
    Taken from https://github.com/allenai/dolma/blob/5b8109d718f1e69a87094623bca109aee1c33378/python/dolma/core/paths.py#L285
    """
    prot, parts = split_path(path)
    if len(parts) == 1:
        return path
    return join_path(prot, *parts[:-1])


class JSONTransform(ABC):

    @abstractmethod
    def __call__(self, json_dict):
        pass


@Registry.register("remove_json_keys")
class RemoveKeys(JSONTransform):

    def __init__(self, keys_to_remove):
        self.keys_to_remove = keys_to_remove

    def __call__(self, json_dict):
        return {k: v for k, v in json_dict.items() if k not in self.keys_to_remove}


@Registry.register("filter_json_keys")
class FilterKeys(JSONTransform):

    def __init__(self, keys_to_keep):
        self.keys_to_keep = keys_to_keep

    def __call__(self, json_dict):
        return {k: v for k, v in json_dict.items() if k in self.keys_to_keep}


class ParallelPreprocessingStep(BaseParallelProcessor):
    """
    We modify the default _process_single_and_save_status method to pass metadata to the process_single method
    """
    def __init__(self, **kwargs):
        self.suffix_depth = kwargs.pop("suffix_depth")
        super().__init__(**kwargs)

    def __call__(self, **kwargs):
        all_paths = self._get_all_paths()
        if len(all_paths.src) == 0:
            return
        super().__call__(**kwargs)

    @classmethod
    def increment_progressbar(cls, queue, /, files=0, documents=0):
        return super().increment_progressbar(queue, files=files, documents=documents)

    def _get_all_paths(self):
        all_paths = AllPathsTuple.empty()
        prefixes = [self.src_prefixes, self.dst_prefixes, self.meta_prefixes]
        if all([len(prefix) == 1 for prefix in prefixes]):
            for path in glob.glob(self.src_prefixes[0]):
                suffix = "/".join(path.split("/")[-self.suffix_depth:])
                all_paths.dst.append(os.path.join(self.dst_prefixes[0], suffix))
                all_paths.meta.append(os.path.join(self.meta_prefixes[0], suffix + METADATA_SUFFIX))
                all_paths.src.append(path)
        else:
            # we don't support the distributed case
            assert not self.is_distributed
            all_paths = super()._get_all_paths()
        return all_paths

    @classmethod
    def _process_single_and_save_status(cls, source_path, destination_path, metadata_path, queue, serialized_kwargs):
        mkdir_p(parent(destination_path))
        mkdir_p(parent(metadata_path))
        kwargs = pickle.loads(serialized_kwargs)
        retries_on_error = kwargs.get("retries_on_error", 0) + 1
        while True:
            try:
                cls.process_single(
                    source_path=source_path, destination_path=destination_path, metadata_path=metadata_path,
                    queue=queue, **kwargs,
                )
                break
            except DolmaRetryableFailure as exception:
                retries_on_error -= 1
                if retries_on_error == 0:
                    raise DolmaError from exception

    @classmethod
    def write_metadata(cls, metadata_path, metadata_dict):
        with smart_open.open(metadata_path, "wt") as f:
            f.write(json.dumps(metadata_dict))

    @classmethod
    def read_metadata(cls, metadata_path):
        with smart_open.open(metadata_path, "rt") as f:
            return json.loads(f.read())


@Registry.register("map_jsonl", "from_args")
class MapJSONL(ParallelPreprocessingStep):

    def __init__(self, transforms, **kwargs):
        self.transforms = transforms
        super().__init__(**kwargs)

    @classmethod
    def from_args(cls, json_transforms, **kwargs):
        transforms = []
        for transform in json_transforms:
            transform_cls = Registry.get(transform["id"])
            transforms.append(transform_cls(**transform["args"]))
        return cls(transforms, **kwargs)

    def __call__(self, **kwargs):
        super().__call__(transforms=self.transforms, **kwargs)

    @classmethod
    def process_single(cls, source_path, destination_path, metadata_path, queue, transforms, cleanup, **kwargs):
        num_documents = 0
        pbar_interval = 100
        decoder = msgspec.json.Decoder()

        with ExitStack() as stack:
            source_file = stack.enter_context(smart_open.open(source_path, "rt"))
            destination_file = stack.enter_context(smart_open.open(destination_path, "wt"))

            for line in source_file:
                num_documents += 1
                data = decoder.decode(line)
                for transform in transforms:
                    data = transform(data)
                json_str = json.dumps(data) + "\n"
                destination_file.write(json_str)
                if num_documents % pbar_interval == 0:
                    cls.increment_progressbar(queue, documents=pbar_interval)

        if cleanup:
            os.remove(source_path)

        cls.increment_progressbar(queue, files=1, documents=num_documents % pbar_interval)
        cls.write_metadata(metadata_path, {"num_documents": num_documents})


@Registry.register("document_splitter")
class DocumentSplitter(ParallelPreprocessingStep):

    @classmethod
    def process_single(cls, source_path, destination_path, metadata_path, queue, property_to_jsonpath, min_chars_per_chunk, cleanup, **kwargs):
        i = 0
        metadata_list = []
        pbar_interval = 1000
        property_to_jsonpath = {
            property: parse(jsonpath) for property, jsonpath in property_to_jsonpath.items()
        }

        with smart_open.open(source_path, "rt") as f_source, smart_open.open(destination_path, "wt") as f_dest:

            while True:
                line = f_source.readline()
                if line:
                    json_dict = json.loads(line)
                    text = get_value_at_jsonpath(json_dict, property_to_jsonpath["text"])
                    chunk_list = cls.split_document(text, min_chars=min_chars_per_chunk)
                    doc_id = get_value_at_jsonpath(json_dict, property_to_jsonpath["id"])
                    metadata_list.append({
                        "id": doc_id,
                        "num_source_chars": len(text),
                        "chunk_lens": [len(s) for s in chunk_list],
                    })
                    num_chunks = len(chunk_list)
                    assert num_chunks > 0

                    for j, chunk in enumerate(chunk_list):
                        chunk_dict = deepcopy(json_dict)
                        # set the text for the row to the chunk
                        property_to_jsonpath["text"].update(chunk_dict, chunk)
                        # update the id 
                        chunk_id = f"chunk-{j+1}-of-{num_chunks}-{doc_id}"
                        property_to_jsonpath["id"].update(chunk_dict, chunk_id)
                        json.dump(chunk_dict, f_dest, separators=(",", ":"))
                        f_dest.write("\n")
                else:
                    break

                i += 1
                if i % pbar_interval == 0:
                    cls.increment_progressbar(queue, documents=pbar_interval)

        if cleanup:
            os.remove(source_path)

        cls.write_metadata(metadata_path, {"source": source_path, "split_stats": metadata_list})
        cls.increment_progressbar(queue, files=1, documents=i % pbar_interval)

    @classmethod
    def split_document(cls, text, min_chars, delimiter="\n\n"):
        """
        Written by o1-preview. Splits the text by the delimiter into chunks of at least min_chars.
        """
        chunks = []
        current_chunk = ''
        current_count = 0
        paragraphs = text.split(delimiter)

        # Function to count non-whitespace, non-punctuation characters
        def count_chars(s):
            return len([c for c in s if c not in string.whitespace + string.punctuation])

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue  # Skip empty paragraphs
            
            # Include the delimiter in the chunk to preserve the original text
            paragraph_with_sep = paragraph + delimiter
            segment_count = count_chars(paragraph)
            current_chunk += paragraph_with_sep
            current_count += segment_count

            if current_count >= min_chars:
                chunks.append(current_chunk.rstrip())
                current_chunk = ''
                current_count = 0

        # Add any remaining text as the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.rstrip())

        return chunks


@Registry.register("grouper")
class Grouper(ParallelPreprocessingStep):

    @classmethod
    def process_single(cls, group_to_jsonpath, source_path, destination_path, metadata_path, queue, cleanup, **kwargs):
        i = 0
        pbar_interval = 1000
        group_to_count = defaultdict(int)
        group_to_jsonpath = {group: parse_extended(jsonpath) for group, jsonpath in group_to_jsonpath.items()}

        group_to_dest = {}
        dirname = os.path.dirname(destination_path)
        basename = os.path.basename(destination_path)

        for group in group_to_jsonpath:
            group_dir = f"{dirname}-group-{group}"
            group_dest = os.path.join(group_dir, basename)
            os.makedirs(group_dir, exist_ok=True)
            group_to_dest[group] = group_dest

        with ExitStack() as stack:
            group_to_file = {}
            f_source = stack.enter_context(smart_open.open(source_path, "rt"))
            while True:
                line = f_source.readline()
                if line:
                    json_dict = json.loads(line)
                    for group, jsonpath in group_to_jsonpath.items():
                        is_member = get_value_at_jsonpath([json_dict], jsonpath)
                        if is_member:
                            if group not in group_to_file:
                                group_to_file[group] = stack.enter_context(smart_open.open(group_to_dest[group], "wt"))    
                            f_dest = group_to_file[group]
                            json.dump(json_dict, f_dest, separators=(",", ":"))
                            f_dest.write("\n")
                            group_to_count[group] += 1
                            break
                else:
                    break

                i += 1
                if i % pbar_interval == 0:
                    cls.increment_progressbar(queue, documents=pbar_interval)
        if cleanup:
            os.remove(source_path)

        cls.write_metadata(metadata_path, {"group_to_count": group_to_count})
        cls.increment_progressbar(queue, files=1, documents=i % pbar_interval)



@Registry.register("sampler")
class Sampler(ParallelPreprocessingStep):

    @classmethod
    def process_single(
        cls, 
        source_path, 
        destination_path, 
        metadata_path, 
        queue, 
        source_regex_to_value_to_samples, 
        jsonpath, 
        is_dryrun, 
        **kwargs
    ):
        i = 0
        pbar_interval = 1000
        temp_path = os.path.join(os.path.dirname(source_path), f"temp_{os.path.basename(source_path)}")

        jsonpath_expr = parse(jsonpath)
        value_to_samples = [
            value_to_samples for source_regex, value_to_samples in source_regex_to_value_to_samples.items() 
            if re.match(source_regex, source_path)
        ]
        if len(value_to_samples) == 0:
            logger.info(f"No samples specified for file {source_path}. Leaving unchanged.")
            cls.increment_progressbar(queue, files=1, documents=i % pbar_interval)
            return

        value_to_samples = value_to_samples[0]
        total_to_sample = sum(value_to_samples.values())

        with smart_open.open(source_path, "rt") as f,  smart_open.open(temp_path, "wt") as f_temp:
            while True:
                line = f.readline()
                if line:
                    json_dict = json.loads(line)
                    value = get_value_at_jsonpath(json_dict, jsonpath_expr)
                    # add row to samples
                    if value in value_to_samples and value_to_samples[value] > 0:
                        f_temp.write(line)
                        value_to_samples[value] -= 1
                        total_to_sample -= 1
                    if total_to_sample == 0:
                        break
                else:
                    break

                i += 1
                if i % pbar_interval == 0:
                    cls.increment_progressbar(queue, documents=pbar_interval)
        
        if not is_dryrun:
            os.replace(temp_path, source_path)

        cls.write_metadata(metadata_path, {"source": source_path, "success": total_to_sample == 0})
        cls.increment_progressbar(queue, files=1, documents=i % pbar_interval)


@Registry.register("split_maker")
class SplitMaker(ParallelPreprocessingStep):

    def __init__(self, **kwargs):
        random_seed = kwargs.pop("random_seed", 0)
        np.random.seed(random_seed)
        super().__init__(**kwargs)

    def __call__(self, **process_single_kwargs):
        """
        Modify the default __call__ method to divide the split size by the number of source paths
        """
        split_to_num_docs = {}
        all_paths = self._get_all_paths()
        if len(all_paths.src) == 0:
            return
        # determine the size of each split in terms of documents
        context_len = process_single_kwargs.pop("context_len")
        num_tokens_per_doc = process_single_kwargs.pop("avg_num_tokens_per_doc")
        num_examples_per_doc = num_tokens_per_doc / context_len
        for split, num_examples in process_single_kwargs.pop("split_to_num_examples").items():
            num_docs_per_call = None
            if num_examples is not None:
                num_docs = num_examples / num_examples_per_doc
                num_docs_per_call = math.ceil(num_docs / len(all_paths.src))
            split_to_num_docs[split] = num_docs_per_call
        process_single_kwargs["split_to_size"] = split_to_num_docs
        super().__call__(**process_single_kwargs)

    @classmethod
    def process_single(cls, source_path, destination_path, metadata_path, queue, split_to_size, cleanup, **kwargs):
        i = 0
        pbar_interval = 100
        # get assignment of documents in file to splits
        metadata = cls.read_metadata(metadata_path)
        num_documents = metadata["num_documents"]
        remainder_size_splits = [split for split, size in split_to_size.items() if size is None]
        assert len(remainder_size_splits) <= 1
        remainder_split = None if len(remainder_size_splits) == 0 else remainder_size_splits[0]

        # confirm we have enough docs for the sample we're trying to take
        # change split sizes if there aren't enough
        num_requested = sum([size for size in split_to_size.values() if size is not None])
        if num_documents < num_requested:
            logger.warning(f"docs in file: {num_documents} < docs requested: {num_requested}")
            split_to_size = cls._update_split_sizes(split_to_size, num_documents)

        split_ids = cls._assign_docs_to_splits(split_to_size, num_documents, remainder_split)

        # get split file paths
        split_paths = {}
        destination_dir = parent(destination_path)
        basename = os.path.basename(destination_path)
        for split in split_to_size:
            split_dir = os.path.join(destination_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            split_paths[split] = os.path.join(destination_dir, split, f"{split}_{basename}")

        decoder = msgspec.json.Decoder()
        # write documents to split files
        with ExitStack() as stack:
            source_file = stack.enter_context(smart_open.open(source_path, "rt"))
            split_files = {
                split: stack.enter_context(smart_open.open(path, "wt"))
                for split, path in split_paths.items()
            }
            for line in source_file:
                data = decoder.decode(line)
                split_id = split_ids[i]
                if split_id is not None:
                    # we don't need to add a newline b/c the prior step added it
                    json_str = json.dumps(data) + "\n"
                    split_files[split_id].write(json_str)
                i += 1
                if i % pbar_interval == 0:
                    cls.increment_progressbar(queue, documents=pbar_interval)

        if cleanup:
            os.remove(source_path)

        cls.increment_progressbar(queue, files=1, documents=i % pbar_interval)
        # save metadata
        split_to_size[remainder_split] = num_documents - sum([size for size in split_to_size.values() if size])
        metadata["split_to_size"] = split_to_size
        cls.write_metadata(metadata_path, metadata)

    @classmethod
    def _update_split_sizes(cls, split_to_size, num_documents):
        has_remainder = True
        updated_split_to_size = {}
        num_requested = sum([size for size in split_to_size.values() if size is not None])
        delta = num_requested - num_documents
        num_sized_splits = len([split for split, size in split_to_size.items() if size is not None])
        per_split_delta = delta // num_sized_splits
        remainder = delta - per_split_delta * num_sized_splits
        for split, size in split_to_size.items():
            # pull remainder from the first sized split
            if has_remainder and size is not None:
                updated_split_to_size[split] = size - per_split_delta - remainder
                has_remainder = False
            elif size is not None:
                updated_split_to_size[split] = size - per_split_delta
            elif size is None:
                updated_split_to_size[split] = size
        return updated_split_to_size

    @classmethod
    def _assign_docs_to_splits(cls, split_to_size, num_documents, remainder_split):
        # assign documents to each split
        start_idx = 0
        total_size = sum([size for size in split_to_size.values() if size is not None])
        split_ids = np.array([remainder_split] * num_documents)
        rand_idxs = np.random.choice(num_documents, size=total_size, replace=False)

        for split, size in split_to_size.items():
            if size is not None:
                split_ids[rand_idxs[start_idx:start_idx + size]] = split
                start_idx += size
        return split_ids


@Registry.register("merger")
class Merger(object):

    def __init__(self, num_processes, max_docs_per_file, metadata_prefix, buffer_size, add_rank_to_filename):
        self.num_processes = num_processes
        self.max_docs_per_file = max_docs_per_file
        self.metadata_prefix = metadata_prefix
        self.buffer_size = buffer_size
        self.filename_prefix = ""
        if add_rank_to_filename:
            self.filename_prefix = f"rank_{SkypilotConstants.get_rank(default=0)}_"

    def __call__(self, sources_and_destinations, cleanup):
        df_list = []
        args_list = []
        tqdm_parallel = ParallelTQDM()
        num_source_files = sum([len(glob.glob(a_dict["source_prefix"])) for a_dict in sources_and_destinations])
        pbar_queue = tqdm_parallel.start_pbar(num_source_files, "Merging sources")

        for merge_dict in sources_and_destinations:
            source_prefix = merge_dict["source_prefix"]
            destination_dir = merge_dict["destination"]
            args_list.append((
                source_prefix, 
                destination_dir, 
                self.max_docs_per_file, 
                cleanup, 
                pbar_queue, 
                self.buffer_size, 
                self.filename_prefix
            ))

        with Pool(processes=self.num_processes) as pool:
            for out in pool.starmap(self.process_single, args_list):
                df_list += out

        tqdm_parallel.stop_pbar()

        os.makedirs(self.metadata_prefix, exist_ok=True)
        metadata_path = os.path.join(self.metadata_prefix, "mixture_stats.jsonl")
        df = pd.DataFrame(df_list)
        df.to_json(metadata_path, lines=True, orient="records")

 
    @classmethod
    def process_single(cls, source_glob, destination_dir, max_docs_per_file, cleanup, pbar_queue, buffer_size, filename_prefix):
        buffer = []
        out_list = []
        output_index = 0
        docs_written = 0
        files_written = 0

        input_paths = glob.glob(source_glob)
        pbar_interval = max(len(input_paths) // 50, 1)
        os.makedirs(destination_dir, exist_ok=True)

        _, extension = split_path_with_compound_extension(source_glob)
        output_path = os.path.join(destination_dir, f"{filename_prefix}idx_{output_index}{extension}")

        output_file = smart_open.open(output_path, "wb")
        for path in input_paths:
            with smart_open.open(path, "rb") as input_file:
                for line in input_file:
                    buffer.append(line)
                    docs_written += 1

                    if len(buffer) >= buffer_size:
                        output_file.writelines(buffer)
                        buffer = []

                    if docs_written >= max_docs_per_file:
                        if buffer:
                            output_file.writelines(buffer)
                            buffer = []

                        output_file.close()
                        output_index += 1
                        out_list.append({"path": output_path, "num_docs": docs_written})
                        output_path = os.path.join(destination_dir, f"{filename_prefix}idx_{output_index}{extension}")
                        output_file = smart_open.open(output_path, "wb")
                        docs_written = 0

            files_written += 1
            if files_written % pbar_interval == 0:
                pbar_queue.put(pbar_interval)

            if cleanup:
                os.remove(path)

        if len(buffer):
            output_file.writelines(buffer)
        output_file.close()

        pbar_queue.put(files_written % pbar_interval)
        
        if docs_written != max_docs_per_file:
            out_list.append({"path": output_path, "num_docs": docs_written})

        return out_list
