"""
Contains steps to be run after pre-processing is complete.
Namely, merging Mosaic Data Shard index files and saving dataset statistics.
"""
import os
import re
import json
import pdb
from pathlib import Path
from collections import Counter

import pandas as pd
import smart_open
import numpy as np
from tqdm import tqdm
from streaming.base.util import merge_index

from preprocessing.constants import SkypilotConstants
from utils import Registry, S3UploadDownload, logger


def find_s3_file_recursive(s3, s3_dir, file_name):
    """
    Returns every path to a file with the given name in the s3 bucket.
    """
    matching_file_paths = []

    s3_dir = s3_dir.replace("s3://", "")
    bucket_name, prefix = s3_dir.split("/", 1)
    if prefix[-1:] != "/":
        prefix += "/"

    paginator = s3.client.get_paginator('list_objects_v2')
    response_iter = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    for response in response_iter:
        if "Contents" in response:
            for item in response["Contents"]:
                path = os.path.join(f"s3://{bucket_name}", item["Key"])
                if os.path.basename(path) == file_name:
                    matching_file_paths.append(path)
        if "CommonPrefixes" in response:
            for subdir in response["CommonPrefixes"]:
                subdir_path = os.path.join(f"s3://{bucket_name}", subdir["Prefix"])
                matching_file_paths += find_s3_file_recursive(s3, subdir_path, file_name)

    return matching_file_paths


@Registry.register("index_merger")
class IndexMerger(object):

    def __init__(self, s3_dirs, split_subdirs, index_file_name="index.json"):
        self.s3_dirs = [os.path.join(a_dir, split_name) for a_dir in s3_dirs for split_name in split_subdirs] 
        self.index_file_name = index_file_name
        self.s3 = S3UploadDownload()

    def __call__(self):
        """
        For each s3 root_dir in s3_dirs, merges all index files under that root_dir
        and saves the merged index to root_dir/index.json. 
        Deletes root_dir/index.json is one already exists. 
        """
        for dir_path in self.s3_dirs:
            index_paths = find_s3_file_recursive(self.s3, dir_path, self.index_file_name)
            root_index_path = os.path.join(dir_path, self.index_file_name)
            if root_index_path in index_paths:
                index_paths.remove(root_index_path)
                logger.info(f"removing root: {root_index_path}")
                self.s3.delete_files([root_index_path])
            logger.info(f"merging for root: {root_index_path} \n\n")
            merge_index(index_paths, out=dir_path, keep_local=False)


@Registry.register("dataset_stats_reporter")
class DatasetStatsReporter(object):
    
    def __init__(
        self,
        dataset_dirs,
        context_len,
        save_path,
        metadata_dir_name="metadata",
        decontam_metadata_dir_regex="^decontam-.+-metadata$",
        splits_metadata_dir_refex="^map-.+-metadata$",
        splits_metadata_extension=".done.txt",
        index_stats_file_name="indexing_stats.jsonl",
        per_doc_decontam_stats_file_name="per_doc_decontam_stats.jsonl",
        avg_decontam_stats_file_name="avg_decontam_stats.jsonl",
        token_units=10**9,
        sample_units=10**6,
    ):
        self.dataset_dirs = dataset_dirs
        self.context_len = context_len
        self.save_path = save_path
        self.metadata_dir_name = metadata_dir_name
        self.decontam_metadata_dir_regex = decontam_metadata_dir_regex
        self.splits_metadata_dir_regex = splits_metadata_dir_refex
        self.splits_metadata_extension = splits_metadata_extension
        
        self.index_stats_file_name = index_stats_file_name
        self.per_doc_decontam_stats_file_name = per_doc_decontam_stats_file_name
        self.avg_decontam_stats_file_name = avg_decontam_stats_file_name
            
        self.token_units = token_units
        self.sample_units = sample_units
        
        self.s3 = S3UploadDownload()
        
    def _df_from_jsonl_files(self, path_list):
        df_list = []
        for path in tqdm(path_list, "creating df from jsonl files"):
            with smart_open.open(path) as f:
                df = pd.read_json(f, lines=True)
                df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        return df

    def is_decontam_successful(self, decontam_metadata_dir):
        index_stats_paths = find_s3_file_recursive(self.s3, decontam_metadata_dir, self.index_stats_file_name)
        index_stats_df = self._df_from_jsonl_files(index_stats_paths)

        decontam_stats_paths = find_s3_file_recursive(self.s3, decontam_metadata_dir, self.per_doc_decontam_stats_file_name)
        per_doc_decontam_stats_df = self._df_from_jsonl_files(decontam_stats_paths)

        did_index_all = sum(index_stats_df["delta"]) == 0
        did_decontam_all = all(per_doc_decontam_stats_df["success"].values)
        did_decontam_and_index_all = did_index_all and did_decontam_all
        return did_decontam_and_index_all

    def get_decontam_stats(self, decontam_metadata_dir):
        avg_decontam_stats_paths = find_s3_file_recursive(self.s3, decontam_metadata_dir, self.avg_decontam_stats_file_name)
        avg_stats_df = self._df_from_jsonl_files(avg_decontam_stats_paths)
        num_docs_dropped = sum(avg_stats_df["num_docs_dropped"].values)
        num_docs_total = sum(avg_stats_df["num_docs_total"].values)
        percent_docs_dropped = num_docs_dropped / num_docs_total * 100.0
        return {"num_docs_dropped": num_docs_dropped, "num_docs_total": num_docs_total, "percent_docs_dropped": percent_docs_dropped}

    def get_num_docs_by_split(self, splits_metadata_dir):
        metadata_pattern = os.path.join(splits_metadata_dir, f"*{self.splits_metadata_extension}")
        split_metadata_paths = self.s3.get_s3_files_matching_pattern(metadata_pattern)
        split_stats_df = self._df_from_jsonl_files(split_metadata_paths)

        split_size_dicts = split_stats_df["split_to_size"]
        num_train_docs = sum([a_dict["train"] for a_dict in split_size_dicts])
        num_dev_docs = sum([a_dict["dev"] for a_dict in split_size_dicts])
        num_total_docs = num_train_docs + num_dev_docs
        return {"num_train_docs": num_train_docs, "num_dev_docs": num_dev_docs, "num_total_docs": num_total_docs}

    def get_name_of_child_dirs(self, parent_dir):
        subdirs = set()
        for path in self.s3.ls_bucket(parent_dir):
            rel_path = os.path.relpath(path, parent_dir)
            if not rel_path.startswith(os.pardir) and rel_path != '.':
                first_component = rel_path.split(os.sep)[0]
                subdirs.add(first_component)
        return subdirs

    def get_data_dict_for_dataset(self, root_dir):
        metadata_dir = os.path.join(root_dir, self.metadata_dir_name)
        subdir_names = self.get_name_of_child_dirs(metadata_dir) 

        decontam_metadata_dir_name = [s for s in subdir_names if re.match(self.decontam_metadata_dir_regex, s)]
        assert len(decontam_metadata_dir_name) == 1, f"multiple metadata dirs {decontam_metadata_dir_name} \n metadata dir {metadata_dir}"
        decontam_metadata_dir = os.path.join(metadata_dir, decontam_metadata_dir_name[0])

        splits_metadata_dir_name = [s for s in subdir_names if re.match(self.splits_metadata_dir_regex, s)]
        assert len(splits_metadata_dir_name) == 1
        splits_metadata_dir = os.path.join(metadata_dir, splits_metadata_dir_name[0])

        data_dict = {
            "train": {
                "idx": os.path.join(root_dir, "train", "index.json"),
            },
            "dev": {
                "idx": os.path.join(root_dir, "dev", "index.json"),
            },
            "decontam_metadata_dir": decontam_metadata_dir,
            "splits_metadata_dir": splits_metadata_dir,
        }
        return data_dict

    def read_json(self, path):
        with smart_open.open(path) as f:
            json_dict = json.load(f)
        return json_dict

    def num_samples(self, index_dict):
        total = sum(shard["samples"] for shard in index_dict["shards"])
        return total

    def __call__(self):
        columns = [
            "dataset",
            "dev_tokens",
            "train_tokens",
            "total_tokens",
            "dev_samples",
            "train_samples",
            "total_samples",
            "train_docs",
            "dev_docs",
            "total_docs",
            "percent_contaminated_docs",
            "contaminated_docs",
            "total_docs_pre_decontam", 
            "decontam_successful",
            "avg_samples_per_shard",
        ]
        df_dict = {c: [] for c in columns}
        for root_dir in self.dataset_dirs:
            logger.info(f"getting stats for {root_dir}")
            data_dict = self.get_data_dict_for_dataset(root_dir)
            
            train_idx = self.read_json(data_dict["train"]["idx"])
            dev_idx = self.read_json(data_dict["dev"]["idx"])
            
            train_samples = self.num_samples(train_idx) 
            dev_samples = self.num_samples(dev_idx)
            
            train_tokens = train_samples * self.context_len / self.token_units
            dev_tokens = dev_samples * self.context_len / self.token_units
            
            train_samples = train_samples / self.sample_units
            dev_samples =  dev_samples / self.sample_units
            
            avg_samples_per_shard = np.mean([d["samples"] for d in train_idx["shards"]])

            doc_count_dict = self.get_num_docs_by_split(data_dict["splits_metadata_dir"])
            is_success = self.is_decontam_successful(data_dict["decontam_metadata_dir"])
            decontam_stats_dict = self.get_decontam_stats(data_dict["decontam_metadata_dir"])

            df_dict["dataset"].append(root_dir)
            df_dict["train_tokens"].append(train_tokens)
            df_dict["dev_tokens"].append(dev_tokens)
            df_dict["train_samples"].append(train_samples)
            df_dict["dev_samples"].append(dev_samples)
            df_dict["total_tokens"].append(train_tokens + dev_tokens)
            df_dict["total_samples"].append(train_samples + dev_samples)
            df_dict["avg_samples_per_shard"].append(avg_samples_per_shard)
            df_dict["train_docs"].append(doc_count_dict["num_train_docs"])
            df_dict["dev_docs"].append(doc_count_dict["num_dev_docs"])
            df_dict["total_docs"].append(doc_count_dict["num_total_docs"])
            df_dict["decontam_successful"].append(is_success)
            df_dict["contaminated_docs"].append(decontam_stats_dict["num_docs_dropped"])
            df_dict["percent_contaminated_docs"].append(decontam_stats_dict["percent_docs_dropped"])
            df_dict["total_docs_pre_decontam"].append(decontam_stats_dict["num_docs_total"])
        
        df = pd.DataFrame(df_dict)
        df.to_csv(self.save_path)
        return df[columns]
