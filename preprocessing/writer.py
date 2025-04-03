import os
import sys
import json
import gzip
import math
import shutil
from glob import glob
from functools import partial
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Pool

import smart_open
import numpy as np
from tqdm import tqdm
from jsonpath_ng import parse
from datasets import load_dataset
from streaming import MDSWriter
import huggingface_hub

from preprocessing.constants import NgramConstants
from preprocessing.decontaminate import ElasticSearchPreprocessingStep
from utils import Registry, logger, S3UploadDownload, get_value_at_jsonpath, group_docs

try:
    sys.path.append("./lm-evaluation-harness")
    import lm_eval.tasks as eval_tasks
except:
    logger.info("EvalHarnessDocumentWriter not available due to missing lm evaluation harness dependency")


def encode(example_dict, tokenizer, property_to_jsonpath, property_to_default):
    # add EOS token to the end of the document
    input_text = get_value_at_jsonpath(example_dict, property_to_jsonpath["text"]) + tokenizer.eos_token
    out_dict = tokenizer(input_text)
    out_dict["id"] = get_value_at_jsonpath(example_dict, property_to_jsonpath.get("id"), property_to_default.get("id"))
    out_dict["num_tokens"] = len(out_dict["input_ids"])
    out_dict["source"] = get_value_at_jsonpath(example_dict, property_to_jsonpath.get("source"), property_to_default.get("source"))
    return out_dict


class Writer(object):

    def __init__(
        self,
        source_prefix,
        destination_prefix,
        shard_from_source_regex,
        context_len,
        tokenizer,
        property_to_jsonpath,
        hf_cache_dir,
        num_processes,
        num_tokens_per_chunk,
        is_distributed,
        property_to_default = {},
    ):
        self.hf_cache_dir = hf_cache_dir
        self.num_processes = num_processes
        self.num_tokens_per_chunk = num_tokens_per_chunk
        self.is_distributed = is_distributed
        self.source_prefix = source_prefix
        self.destination_prefix = self.shorten_path_if_needed(destination_prefix)
        self.shard_from_source_regex = shard_from_source_regex
        self.context_len = context_len
        self.tokenizer = tokenizer
        self.property_to_jsonpath = {
            property: parse(jsonpath) for property, jsonpath in property_to_jsonpath.items()
        }
        self.property_to_default = property_to_default
        assert "text" in property_to_jsonpath
        assert "id" in property_to_jsonpath

    @classmethod
    def shorten_path_if_needed(cls, path, max_len=63):
        s3 = S3UploadDownload()
        did_change = False
        if s3.is_s3_path(path):
            bucket, key = s3._get_bucket_and_key_from_s3_path(path)
            if len(bucket) > max_len:
                bucket = bucket[:max_len]
                did_change = True
            path = os.path.join(f"s3://{bucket}", key)

        if did_change: 
            logger.warning(f"changed path to {path} to ensure bucket was no more than 63 chars")
        return path

    @classmethod
    def from_args(cls, tokenizer, **kwargs):
        tokenizer = Registry.get(tokenizer["id"])(**tokenizer["args"])
        return cls(tokenizer=tokenizer, **kwargs)

    def __call__(self, **kwargs):
        s3 = S3UploadDownload()
        if s3.is_s3_path(self.destination_prefix):
            bucket, _ = s3._get_bucket_and_key_from_s3_path(self.destination_prefix)
            if not s3.bucket_exists(bucket):
                s3.client.create_bucket(Bucket=bucket)

        shard_to_metadata = {}
        shard_to_sources = group_docs(self.source_prefix, self.shard_from_source_regex)
        for shard, source_list in shard_to_sources.items():
            destination = os.path.join(self.destination_prefix, shard)
            shard_to_metadata[shard] = self._write(source_list, destination, **kwargs)

        if self.is_distributed or len(shard_to_metadata) == 0:
            # merge must be done separately after all shards are written
            return

        # save metadata
        shard_to_metadata["num_tokens"] = sum([v["num_tokens"] for v in shard_to_metadata.values()])
        shard_to_metadata["context_len"] = self.context_len
        shard_to_metadata["created_on"] = str(datetime.now())
        metadata_path = os.path.join(self.destination_prefix, "metadata.json")
        with smart_open.open(metadata_path, "wt") as f:
            f.write(json.dumps(shard_to_metadata))


    def _write(self, source_paths, destination, cleanup=False):
        dataset = load_dataset("json", data_files=source_paths, num_proc=self.num_processes, cache_dir=self.hf_cache_dir)

        # remove examples with empty text field
        dataset = dataset.filter(
            lambda x: len(get_value_at_jsonpath(x, self.property_to_jsonpath["text"])) > 0,
            num_proc=self.num_processes,
            desc="removing examples with empty text field",
        )

        kwargs_dict = {
            "tokenizer": self.tokenizer, 
            "property_to_jsonpath": self.property_to_jsonpath, 
            "property_to_default": self.property_to_default,
        }

        tokenized_dataset = dataset.map(
            encode,
            fn_kwargs=kwargs_dict,
            remove_columns=[str(self.property_to_jsonpath["text"])],
            num_proc=self.num_processes,
            desc="tokenizing dataset"
        )

        index_paths = []
        assert len(tokenized_dataset.keys()) == 1
        split = list(tokenized_dataset.keys())[0]
        data = tokenized_dataset[split]
        num_tokens = sum(data["num_tokens"])
        # we drop the remainder of tokens that don't go into context len
        num_tokens = num_tokens - (num_tokens % self.context_len)
        num_chunks = max(1, num_tokens / self.num_tokens_per_chunk)
        num_processes = int(min(self.num_processes, num_chunks))

        # write the data to MDS format in parallel
        with Pool(processes=num_processes) as pool:
            map_fn = partial(
                self._write_shard, num_shards=num_processes, data=data, destination_prefix=destination
            )
            for path in pool.imap(map_fn, range(num_processes)):
                index_paths.append(path)

        if cleanup:
            shutil.rmtree(self.hf_cache_dir)

        metadata = {"num_tokens": num_tokens, "destination": destination, "index_paths": index_paths}
        return metadata


@Registry.register("mds_writer", "from_args")
class ParallelMDSWriter(Writer):

    INDEX_FILE_NAME = "index.json"

    def __init__(
        self,
        source_prefix,
        destination_prefix,
        shard_from_source_regex,
        context_len,
        tokenizer,
        property_to_jsonpath,
        property_to_default,
        hf_cache_dir,
        num_processes,
        compression,
        num_tokens_per_chunk,
        mds_size_limit,
        is_distributed=False,
    ):
        super().__init__(
            source_prefix=source_prefix,
            destination_prefix=destination_prefix,
            shard_from_source_regex=shard_from_source_regex,
            context_len=context_len,
            tokenizer=tokenizer,
            property_to_jsonpath=property_to_jsonpath,
            property_to_default=property_to_default,
            hf_cache_dir=hf_cache_dir,
            num_processes=num_processes,
            num_tokens_per_chunk=num_tokens_per_chunk,
            is_distributed=is_distributed,
        )
        self.compression = compression
        self.mds_size_limit = mds_size_limit
        self.columns = {
            "inputs": "ndarray",
            "targets": "ndarray",
            "doc_ids": "ndarray",
            "source_id": "str",
            "uid": "str"
        }

    def _write_shard(self, shard_idx, num_shards, data, destination_prefix):
        shard = data.shard(num_shards, index=shard_idx, contiguous=True)
        destination = f"{destination_prefix}-part-{shard_idx+1}-of-{num_shards}"
        index_path = os.path.join(destination, self.INDEX_FILE_NAME)
        num_tokens = sum(shard["num_tokens"])
        num_chunks = math.ceil(num_tokens / self.num_tokens_per_chunk)
        with MDSWriter(out=destination, columns=self.columns, compression=self.compression, size_limit=self.mds_size_limit) as writer:
            for i in range(num_chunks):
                start = 0
                end = start + self.context_len
                chunk = shard.shard(num_chunks, index=i, contiguous=True).with_format("numpy")
                input_ids = np.concatenate(chunk["input_ids"])
                # ensure document ids are unique within the chunk. Doc ids used for creating masks at loading time.
                doc_ids = np.concatenate([[j]*n for j, n in enumerate(chunk["num_tokens"])])
                num_examples = (len(input_ids) - 1) // self.context_len
                source_id = Counter(chunk["source"]).most_common(1)[0][0]
                for _ in range(num_examples):
                    x = input_ids[start:end]
                    y = input_ids[start+1:end+1]
                    doc_id = doc_ids[start:end]
                    uid = f"shard-{shard_idx}-of-{num_shards}-chunk-{i}-of-{num_chunks}-idx-{start}-to-{end}"
                    sample = {"inputs": x, "targets": y, "doc_ids": doc_id, "source_id": source_id, "uid": uid}
                    writer.write(sample)
                    start += self.context_len
                    end += self.context_len
        return index_path


@Registry.register("memmap_writer", "from_args")
class MemmapWriter(Writer):

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        self.split = split

    def _write_shard(self, shard_idx, num_shards, data, destination_prefix):
        shard = data.shard(num_shards, index=shard_idx, contiguous=True)

        local_destination = destination_prefix
        destination_is_on_s3 = S3UploadDownload.is_s3_path(destination_prefix)
        if destination_is_on_s3:
            local_destination = os.path.join(self.hf_cache_dir, destination_prefix[5:])
            logger.info(f"writing shard {shard_idx + 1} to {local_destination}")
            os.makedirs(local_destination, exist_ok=True)

        filename_prefix = os.path.join(local_destination, f"{self.split}-part-{shard_idx + 1}-of-{num_shards}")
        bin_path = f"{filename_prefix}.bin"
        idx_path = f"{filename_prefix}.idx"

        num_tokens = sum(shard["num_tokens"])
        num_chunks = int(math.ceil(num_tokens / self.num_tokens_per_chunk))

        # create memmap arrays bin (to store data) and idx (to store num toks in each doc & num docs in shard)
        shard_array = np.memmap(bin_path, dtype=np.uint32, mode="w+", shape=(num_tokens,))
        idx_array = np.memmap(idx_path, dtype=np.uint32, mode="w+", shape=(len(shard) + 1,))

        bin_start = 0
        idx_start = 0
        # write the shard to disk in chunks
        for i in tqdm(range(num_chunks), desc=f"writing chunks of shard {shard_idx + 1}"):
            chunk = shard.shard(num_shards=num_chunks, index=i, contiguous=True).with_format("numpy")
            chunk_size = len(chunk)
            # add examples in chunk and num tokens in each example to memmap
            data_chunk = np.concatenate(chunk["input_ids"])
            shard_array[bin_start:bin_start + len(data_chunk)] = data_chunk
            idx_array[idx_start:idx_start + chunk_size] = chunk["num_tokens"]
            bin_start += len(data_chunk)
            idx_start += chunk_size

        # write num docs in shard
        idx_array[-1] = len(shard)
        shard_array.flush()
        idx_array.flush()

        if destination_is_on_s3:
            s3 = S3UploadDownload()
            bucket_name, prefix = destination_prefix.replace("s3://", "").split("/", 1)
            key_prefix = os.path.join(prefix, f"{self.split}-part-{shard_idx + 1}-of-{num_shards}")
            bin_key = f"{key_prefix}.bin"
            idx_key = f"{key_prefix}.idx"
            s3.client.upload_file(bin_path, bucket_name, bin_key)
            s3.client.upload_file(bin_path, bucket_name, idx_key)


@Registry.register("ngram_writer")
class NgramWriter(ElasticSearchPreprocessingStep):

    def __init__(self, metadata_dir, source_glob, default_ngram_size, source_to_ngram_size, index_properties, **kwargs):
        super().__init__(**kwargs)
        self.metadata_dir = metadata_dir
        self.source_paths = glob(source_glob)
        self.default_ngram_size = default_ngram_size
        self.source_to_ngram_size = source_to_ngram_size

        os.makedirs(self.metadata_dir, exist_ok=True)
        # elastic search tokenizer requires an index, so we make an index if it doesn't exist
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, mappings={"properties": index_properties})

    def __call__(self):
        ngram_set = set()
        df_list = []
        decontam_df_list = []
        ngram_to_locations = defaultdict(list)
        pbar = tqdm(total=len(self.source_paths))
        for path in self.source_paths:
            with smart_open.open(path) as f_source:
                while True:
                    line = f_source.readline()
                    if line:                    
                        json_dict = json.loads(line)
                        text = json_dict["text"]
                        source = json_dict["source"]
                        split = json_dict["split"]
                        index = json_dict["index"]
                        uid = json_dict["id"]

                        ngram_size = self.default_ngram_size
                        if source in self.source_to_ngram_size:
                            ngram_size = self.source_to_ngram_size[source]

                        tokens = [t for t in self.es_tokenizer(self.es, text, self.index_name)]
                        ngrams = self.ngrams_for_tokens(tokens, ngram_size, ngram_size)
                        unique_ngrams = [gram for gram in ngrams if gram not in ngram_set]
                        ngram_set.update(unique_ngrams)

                        for gram in unique_ngrams:
                            ngram_to_locations[gram].append({"source": source, "split": split, "index": index, "num_ngrams": len(ngrams)})
                            decontam_df_list.append({"ngram_for_decontam": gram})

                        row = {
                            "source":source,
                            "split": split,
                            "index": index,
                            "id": uid,
                            "text": text,
                            "tokens": tokens,
                            "ngram_size": ngram_size,
                            "num_tokens": len(tokens),
                            "ngrams": ngrams,
                            "ngrams_for_decontam": unique_ngrams,
                        }
                        df_list.append(row)
                    else:
                        break
            pbar.update(1)

        ngrams_path = os.path.join(self.metadata_dir, f"instances-and-ngrams--ngram-size-{self.default_ngram_size}.jsonl")
        self._safe_save_as_jsonl(df_list, ngrams_path, f"wrote ngrams to: {ngrams_path}")

        decontam_ngrams_path = os.path.join(self.metadata_dir, f"decontam-ngram-queries--ngram-size-{self.default_ngram_size}.jsonl")
        self._safe_save_as_jsonl(decontam_df_list, decontam_ngrams_path, f"wrote ngrams for decontam queries to: {decontam_ngrams_path}")

        locations_path = os.path.join(self.metadata_dir, f"ngram-source-locations--ngram-size-{self.default_ngram_size}.json")
        with smart_open.open(locations_path, "wt") as f:
            json.dump(ngram_to_locations, f)
        logger.info(f"wrote ngram locations to: {locations_path}")

                
    @classmethod
    def ngrams_for_tokens(cls, tokens, ngram_size, min_ngram_size):
        ngrams = [
            NgramConstants.TOKEN_DELIMITER.join(ngram) 
            for ngram in cls.ngram_generator(tokens, ngram_size, min_ngram_size)
        ]
        return ngrams

    @classmethod
    def es_tokenizer(cls, es, text, index):
        body = {"analyzer": "standard", "text": text}
        response = es.indices.analyze(index=index, body=body)
        for token in response["tokens"]:
            yield token["token"]

    @classmethod
    def ngram_generator(cls, sequence, n, min_n):
        """
        Written by o1-preview
        """
        sequence = iter(sequence)
        history = []
        for _ in range(n):
            try:
                history.append(next(sequence))
            except StopIteration:
                break  # Not enough items to fill history up to n

        if len(history) >= n:
            # We have enough items to yield n-grams of size n
            yield tuple(history)
            for item in sequence:
                history.append(item)
                del history[0]
                yield tuple(history)
        elif len(history) >= min_n:
            # We have at least min_n items, yield this shorter n-gram
            yield tuple(history)
        # If history length is less than min_n, do not yield anything


@Registry.register("eval_harness_document_writer")
class EvalHarnessDocumentWriter(object):
    """
    Writes evaluation datasets present in the lm-evaluation-harness package in jsonl format for use in decontamination
    """

    def __init__(self, input_datasets, destination_prefix, include_path):
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        huggingface_hub.login(token=os.environ["HUGGINGFACE_TOKEN"])
        self.include_path = include_path
        self.input_datasets = input_datasets
        self.destination_prefix = destination_prefix
        os.makedirs(self.destination_prefix, exist_ok=True)

    def __call__(self, **kwargs):
        num_docs = 0
        task_manager = eval_tasks.TaskManager("INFO", include_path=os.path.expanduser(self.include_path))
        task_names = task_manager.match_tasks(self.input_datasets)
        task_dict = eval_tasks.get_task_dict(task_names, task_manager)

        for task_name, task in tqdm(task_dict.items(), desc="writing eval harness docs"):
            task_jsons = []

            if isinstance(task, tuple):
                _, task = task
                if task is None:
                    continue

            logger.info(f"writing docs for task {task}")
            if task.has_test_docs():
                task_jsons += self._docs_to_jsons(task, task.test_docs(), task.config.test_split, task_name)
            if task.has_validation_docs():
                task_jsons += self._docs_to_jsons(task, task.validation_docs(), task.config.validation_split, task_name)
            if task.has_training_docs():
                task_jsons += self._docs_to_jsons(task, task.training_docs(), task.config.training_split, task_name)

            num_docs += len(task_jsons)
            json_path = os.path.join(self.destination_prefix, f"{task_name}.jsonl.gz")
            self._write_jsonl_gzip(json_path, task_jsons)

        logger.info(f"Total number of eval harness documents: {num_docs}")

    @staticmethod
    def _docs_to_jsons(task, docs, split, task_name):
        json_list = []
        for i, doc in enumerate(docs):
            text = task.doc_to_decontamination_query(doc)
            
            if text is None:
                continue

            id = f"{task_name}-{split}-{i}"
            json_list.append({
                "created": "", "added": "", "id": id, "metadata": {},
                "index": i, "split": split, "source": task_name, 
                "text": text, "version": "v0",
            })
        return json_list

    @staticmethod
    def _write_jsonl_gzip(path, json_list):
        with gzip.open(path, "wt") as f:
            for dict in json_list:
                json.dump(dict, f)
                f.write("\n")

