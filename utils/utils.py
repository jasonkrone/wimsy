from typing import Optional
import os
import re
import yaml
import glob
import shutil
import tempfile
import functools
import time
from time import sleep
from fnmatch import fnmatch
from contextlib import nullcontext
from filelock import FileLock
from collections.abc import MutableMapping
from collections import defaultdict
from string import Template
from multiprocessing import Manager
from threading import Thread

import boto3
from tqdm import tqdm

import torch
import torch.amp as amp
from torch.distributed.fsdp import MixedPrecision

from utils.logger import logger


def copy_source_to_dest(source, dest):
    if os.path.exists(dest):
        logger.info(f"Copy destination: {dest} for file {source} already exists. Skipping copy.")
    if os.path.isdir(source):
        os.makedirs(dest, exist_ok=True)
        shutil.copytree(source, dest, dirs_exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(source, dest)
    return dest


def group_docs(glob_path, group_regex):
    """
    Groups the files returned by glob.glob(glob_path) in the group captured by group_regex
    """
    group_to_docs = defaultdict(list)
    for path in glob.glob(glob_path):
        group = re.search(group_regex, path).group(1)
        group_to_docs[group].append(path)
    return group_to_docs


def get_value_at_jsonpath(json_dict, jsonpath_expr, default=None):
    value = default
    if jsonpath_expr is not None: 
        matches = jsonpath_expr.find(json_dict)
        values = [m.value for m in matches]
        if len(values) > 0:
            value = values[0]
    return value


def split_path_with_compound_extension(path):
    """ 
    Written by GPT-4o
    """
    base, ext = os.path.splitext(path)
    extensions = [ext]
    while ext:
        base, ext = os.path.splitext(base)
        if ext:
            extensions.append(ext)
    full_extension = ''.join(reversed(extensions))
    return base, full_extension


def timeit(_func=None, *, fn_name=None):
    """
    Written by o1-preview
    """
    def decorator_timeit(func):
        @functools.wraps(func)
        def wrapper_timeit(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time

            name_to_log = fn_name or func.__name__
            print(f"Function '{name_to_log}' took {run_time:.4f} seconds to execute.")
            return result
        return wrapper_timeit

    if _func is None:
        # Decorator used with arguments
        return decorator_timeit
    else:
        # Decorator used without arguments
        return decorator_timeit(_func)


class ParallelTQDM(object):
    """
    Utility wrapper to use TQDM in a parallel setting
    """
    def __init__(self):
        self.manager = Manager()
        self.pbar_queue = self.manager.Queue()

    def start_pbar(self, total, desc, mininterval=30):
        self.pbar_thread = Thread(
            target=ParallelTQDM.pbar_listener, 
            args=(self.pbar_queue, total, desc, mininterval)
        )
        self.pbar_thread.start()
        return self.pbar_queue

    def stop_pbar(self):
        self.pbar_queue.put(None)
        self.pbar_thread.join()
        self.manager.shutdown()

    @classmethod
    def pbar_listener(cls, q, total, desc, mininterval):
        """
        Written by GPT-4    
        """
        pbar = tqdm(total=total, desc=desc, mininterval=mininterval)
        for item in iter(q.get, None):
            pbar.update(item)
        pbar.close()


class Config(MutableMapping):

    INCLUDE_REGEX = r'^include:\s*(.+)\s*$'
    INCLUDE_STR = "include:"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def from_yaml(cls, yaml_path):
        yaml_dict = None
        with open(yaml_path, "r") as file:
            yaml_str = file.read()
            yaml_str = cls.include_dependencies(yaml_str, yaml_path)
            yaml_dict = yaml.full_load(yaml_str)
            yaml_dict = cls.substitute_values_for_variables(yaml_str, yaml_dict)
        yaml_dict["config_path"] = yaml_path
        return cls.from_dict(yaml_dict)
    
    @classmethod
    def substitute_values_for_variables(cls, yaml_str, yaml_dict):
        # we allow string formatting via variables listed in substitutions
        substitutions = yaml_dict.get("substitutions")
        if substitutions:
            template = Template(yaml_str)
            yaml_substituted = template.safe_substitute(**substitutions)
            yaml_dict = yaml.full_load(yaml_substituted)
        return yaml_dict

    @classmethod
    def include_dependencies(cls, yaml_str, yaml_path):
        while cls.INCLUDE_STR in yaml_str:
            re_match = re.search(cls.INCLUDE_REGEX, yaml_str, re.MULTILINE)
            include_path = re_match.group(1)
            dir_path, basename = os.path.split(include_path)
            if dir_path == "":
                include_path = os.path.join(os.path.dirname(yaml_path), basename)
            include_path = include_path.strip()
            with open(include_path, "r") as include_file:
                str_to_include = include_file.read()
            yaml_str = yaml_str.replace(re_match.group(0), str_to_include)
        return yaml_str

    @classmethod
    def from_dict(cls, a_dict):
        for k, v in a_dict.items():
            if isinstance(v, dict):
                a_dict[k] = cls.from_dict(v)
        return cls(**a_dict)

    def update_with_arg_strs(self, arg_strs):
        assert len(arg_strs) % 2 == 0
        for key, value in zip(arg_strs[::2], arg_strs[1::2]):
            key = key.lstrip("--")
            if hasattr(self, key) and getattr(self, key):
                arg_type = type(getattr(self, key))
                value = arg_type(value)
            setattr(self, key, value)

    def as_dict(self):
        def convert(obj):
            if isinstance(obj, Config):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, MutableMapping):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            else:
                return obj
        return convert(self)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def items(self):
        return self.__dict__.items()

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.as_dict())

    def __len__(self):
        return len(self.as_dict())

    def __setattr__(self, name, value):
        if "." in name:
            name_list = name.split(".")
            obj = getattr(self, name_list[0])
            for attr_name in name_list[1:-1]:
                obj = getattr(obj, attr_name)
            setattr(obj, name_list[-1], value)
        else:
            super().__setattr__(name, value)


class Precision(object):

    PRECISION_NAME_TO_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    PRECISION_NAME_TO_FSDP_MIXED_PRECISION = {
        # taken from https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
        # duplicate here is intentional for pure_bf16 to make config simple
        "pure_bf16": MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        "bf16": MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            
            # TODO: temp
        ),
        "fp32": MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
    }

    @classmethod
    def get_grad_scaler(cls, precision):
        return torch.cuda.amp.GradScaler(enabled=precision == "fp16")

    @classmethod
    def get_amp_context(cls, precision, device):
        dtype = cls.PRECISION_NAME_TO_DTYPE[precision]
        device_type = "cuda" if "cuda" in device else device
        amp_context = nullcontext() if precision == "fp32" else amp.autocast(device_type=device_type, dtype=dtype)
        return amp_context


class Registry(object):

    registry = {}

    @classmethod
    def register(cls, name: str, constructor: Optional[str] = None):

        def add_to_registry(registrable):
            if name in cls.registry:
                raise ValueError(f"name {name} already in registry")
            cls.registry[name] = (registrable, constructor)
            return registrable

        return add_to_registry

    @classmethod
    def get(cls, name: str):
        out = cls.registry.get(name)
        if out is not None:
            subclass, constructor = out
            if constructor is not None:
                out = getattr(subclass, constructor)
            else:
                out = subclass
        return out


class EarlyStopper(object):

    def __init__(self, patience: int, lower_is_better: bool = True) -> None:
        self.patience = patience
        self.lower_is_better = lower_is_better
        self.scores = []
        self.score_to_beat = float("inf")
        self.patience_spent = 0

    def __call__(self, score: float) -> bool:
        if not self.lower_is_better:
            score = -score
        self.scores.append(score)
        if score < self.score_to_beat:
            self.patience_spent = 0
            self.score_to_beat = score
        else:
            self.patience_spent += 1
            logger.info(f"Score: {score} did not beat {self.score_to_beat}. Patience spent: {self.patience_spent}")
        return self.stop_early()

    def stop_early(self):
        return self.patience_spent >= self.patience


class S3UploadDownload(object):

    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None):
        if aws_secret_access_key is None:
            aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if aws_access_key_id is None:
            aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        self.client = boto3.client(
            "s3",
            aws_secret_access_key=aws_secret_access_key,
            aws_access_key_id=aws_access_key_id,
        )
        self.temp_files = []

    def cleanup(self):
        for temp_path in self.temp_files:
            if os.path.isdir(temp_path):
                shutil.rmtree(temp_path)
            elif os.path.isfile(temp_path):
                os.remove(temp_path)

    def download_if_on_s3(self, path, local_dir=None, show_progress=False):
        local_path = path
        if self.is_s3_path(path):
            # create a temporary directory to download the contents to
            if local_dir is None:
                local_dir = tempfile.mkdtemp()
                self.temp_files.append(local_dir)
            # if it's a path to a s3 bucket, download the entire bucket
            if "/" not in path[:5]:
                # TODO: need to do something here w/ all the temp files
                self.download_bucket_recursive(path, local_dir, show_progress)
                local_path = local_dir
            # otherwise, download the file
            else:
                local_path = os.path.join(local_dir, os.path.basename(path))
                self.download_file(path, local_path)
                self.temp_files.append(local_path)
        return local_path

    def download_file(self, remote_path, local_path):
        bucket_name, file_key = self._get_bucket_and_key_from_s3_path(remote_path)
        if not os.path.exists(local_path):
            lock_path = f"{local_path}.lock"
            with FileLock(lock_path):
                self.client.download_file(bucket_name, file_key, local_path)
            os.remove(lock_path)

    @staticmethod
    def wait_for_download_to_complete(local_path, tick=0.5):
        lock_path = f"{local_path}.lock"
        while not os.path.exists(local_path) or os.path.exists(lock_path):
            sleep(tick)

    def download_dir(self, path, save_dir, show_progress=False):
        file_list = self.ls_bucket(path)
        lock_path = f"{save_dir}.lock"
        with FileLock(lock_path):
            for file in tqdm(file_list, desc=f"downloading files from {path}", disable=not show_progress):
                file_name = os.path.basename(file)
                local_path = os.path.join(save_dir, file_name)
                self.download_file(file, local_path)
        os.remove(lock_path)

    def download_bucket_recursive(self, path, save_dir, show_progress=False):

        if self.is_s3_path(path):
            path = path[5:]

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        bucket, key = self._get_bucket_and_key_from_s3_path(path)
        bucket_contents = self.client.list_objects_v2(Bucket=bucket, Prefix=key).get('Contents', [])

        lock_path = f"{save_dir}.lock"
        with FileLock(lock_path):
            for file in tqdm(bucket_contents, desc=f"downloading files from {bucket_name}", disable=not show_progress):
                file_key = file["Key"]
                file_path = os.path.join(save_dir, file_key)
                file_name = os.path.basename(file_path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                if os.path.exists(file_path):
                    continue
                try:
                    logger.info(f"Downloading file: {file_name} from s3 bucket: {bucket_name} to {file_path}")
                    self.client.download_file(bucket_name, file_key, file_path)
                except Exception as e:
                    logger.info(f"Download of file: {file_name} failed: {e}")
        os.remove(lock_path)

    def upload_dir(self, dir_path, bucket_name, show_progress=False):
        file_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
        self.upload_files(file_paths, bucket_name, show_progress)

    def upload_files(self, file_paths, bucket_name, show_progress=False):
        prefix = None

        if self.is_s3_path(bucket_name):
            bucket_name = bucket_name[5:]

        if "/" in bucket_name:
            prefix = "/".join(bucket_name.split("/")[1:])
            bucket_name = bucket_name.split("/")[0]

        if not self.bucket_exists(bucket_name):
            self.client.create_bucket(Bucket=bucket_name)

        for path in tqdm(file_paths, desc="uploading files to s3", disable=not show_progress):
            key = os.path.basename(path)
            try:
                if prefix is not None:
                    key = os.path.join(prefix, key)
                logger.info(f"uploading file: {key} to s3 bucket: {bucket_name}")
                self.client.upload_file(path, bucket_name, key)
            except Exception as e:
                logger.info(f"Upload failed: {e}")

    def delete_files(self, file_paths):
        for path in file_paths:
            bucket, key = self._get_bucket_and_key_from_s3_path(path)
            self.client.delete_object(Bucket=bucket, Key=key)

    @staticmethod
    def is_s3_path(path):
        return path[:5] == "s3://"

    def _get_bucket_and_key_from_s3_path(self, path):
        if self.is_s3_path(path):
            path = path[5:]
        bucket_name, file_key = path.split("/", 1)
        return bucket_name, file_key

    def bucket_exists(self, bucket_name):
        exists = False
        if self.is_s3_path(bucket_name):
            bucket_name = bucket_name[5:]
        try:
            self.client.head_bucket(Bucket=bucket_name)
            exists = True
        except Exception as e:
            # not a bucket does not exist error, re-raise the exception
            if e.response["Error"]["Code"] != "404":
                raise e
        return exists

    def ls_bucket(self, path):
        matching_file_paths = []
        bucket, prefix = self._get_bucket_and_key_from_s3_path(path)
        paginator = self.client.get_paginator("list_objects_v2")
        response_iter = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for response in response_iter:
            if "Contents" in response:
                for item in response["Contents"]:
                    file_path = f"s3://{bucket}/{item['Key']}"
                    if file_path != f"{path}/":
                        matching_file_paths.append(file_path)
        return matching_file_paths

    def get_s3_files_matching_pattern(self, pattern):
        file_list = self.ls_bucket(os.path.dirname(pattern))
        matches = [file for file in file_list if fnmatch(file, pattern)]
        return matches

    def get_s3_buckets_matching_pattern(self, pattern):
        bucket_names = [f"s3://{bucket['Name']}" for bucket in self.client.list_buckets()['Buckets']]
        matches = [bucket for bucket in bucket_names if fnmatch(bucket, pattern)]
        return matches
