import os
import subprocess
import tempfile
import shutil
from collections import defaultdict

import numpy as np
import smart_open

from preprocessing.constants import SkypilotConstants
from utils import Registry, logger, S3UploadDownload, split_path_with_compound_extension


@Registry.register("artifact_uploader")
class ArtifactUploader(object):

    def __init__(self, s3_destination, dirs_to_upload):
        """
        :param s3_dir: The S3 directory to upload to (e.g., 's3://mydataset/metadata')
        :param local_dirs: A list of local directories to upload from
        """
        s3 = S3UploadDownload()
        bucket, _ = s3._get_bucket_and_key_from_s3_path(s3_destination)
        assert s3.bucket_exists(bucket)
        self.s3_destination = s3_destination.rstrip("/")
        self.dirs_to_upload = [dir.rstrip("/") for dir in dirs_to_upload]

    def __call__(self):
        """
        Perform the upload using s5cmd, preserving the directory structure from the basename down.
        """
        for local_dir in self.dirs_to_upload:
            basename = os.path.basename(local_dir)
            s3_target = os.path.join(self.s3_destination, basename)
            cmd = ["s5cmd", "cp", f"{local_dir}/", f"{s3_target}/"]
            subprocess.run(cmd, check=True)


@Registry.register("downloader")
class Downloader(object):

    def __init__(
        self,
        urls_path,
        destination_prefix,
        num_shards,
        num_processes,
        cleanup,
        is_distributed=False,
        rank=None,
        world_size=None,
        src_location="url",
    ):
        self.urls_path = urls_path
        self.src_location = src_location 
        self.destination_prefix = destination_prefix
        self.num_shards = num_shards
        self.num_processes = num_processes
        self.cleanup = cleanup
        self.is_distributed = is_distributed
        self.rank = SkypilotConstants.get_rank(default=rank)
        self.world_size = SkypilotConstants.get_world_size(default=world_size)
        if self.num_shards is None and self.world_size:
            self.num_shards = self.world_size
        # convert rank to int if urls path is a dict with str rank keys
        if isinstance(self.urls_path, dict) and isinstance(next(iter(self.urls_path.keys())), str):
            self.urls_path = {int(key): value for key, value in self.urls_path.items()}

    def __call__(self):
        rank_to_shards = defaultdict(dict)
        # urls path is a file containing all urls to download
        if isinstance(self.urls_path, str):
            # split urls into shards to download & save
            with smart_open.open(self.urls_path, "rt") as urls_file:
                url_lines = urls_file.readlines()
                url_shards = np.array_split(url_lines, self.num_shards)
            # name shards and assign them to each rank
            for i, shard in enumerate(url_shards):
                shard_num = i + 1
                shard_name = f"shard-{shard_num}-of-{self.num_shards}"
                if not self.is_distributed:
                    rank_to_shards[self.rank][shard_name] = shard
                elif shard_num % self.world_size == self.rank:
                    assert self.num_shards >= self.world_size
                    rank_to_shards[self.rank][shard_name] = shard
        # urls_path is a dict mapping rank => shard name => path 
        elif isinstance(self.urls_path, dict):
            assert len(self.urls_path) == self.world_size
            for shard_name, shard_path in self.urls_path[self.rank].items():
                with smart_open.open(shard_path, "rt") as urls_file:
                    url_lines = urls_file.readlines()
                    rank_to_shards[self.rank][shard_name] = url_lines
            
        for shard_name, shard_urls in rank_to_shards[self.rank].items():
            if self.src_location == "url":
                self._download_from_urls(shard_urls, shard_name)
            elif self.src_location == "s3":
                self._download_from_s3(shard_urls, shard_name)
            else:
                raise ValueError(f"Invalid source location: {self.src_location}")

    def _download_from_s3(self, s3_paths, shard_name):
        """
        Written by GPT 1o-preview
        """
        commands = []
        dst_dir = os.path.join(self.destination_prefix, shard_name)
        os.makedirs(dst_dir, exist_ok=True)
        # we name the files as idx.extension to avoid naming issues given we 
        # could be copying from nested directories
        for i, s3_path in enumerate(s3_paths):
            s3_path = s3_path.strip()
            _, extension = split_path_with_compound_extension(s3_path) 
            dst_path = os.path.join(dst_dir, f"{i}{extension}")
            commands.append(f"cp {s3_path} {dst_path}")
        command_str = "\n".join(commands)
        # Prepare s5cmd command with the specified number of workers
        s5cmd_command = ["s5cmd", "--numworkers", str(self.num_processes), "run"]
        # Execute s5cmd with commands piped via stdin
        result = subprocess.run(s5cmd_command, input=command_str, text=True, capture_output=True)
        # Check for errors
        if result.returncode != 0:
            raise Exception(f"s5cmd failed with error code {result.returncode}:\n{result.stderr}")
     
    def _download_from_urls(self, urls, shard_name):
        s3 = S3UploadDownload()
        has_s3_dst = s3.is_s3_path(self.destination_prefix)

        if has_s3_dst:
            local_dst = tempfile.mkdtemp(prefix=os.path.basename(local_dst))
        else:
            local_dst = os.path.join(self.destination_prefix, shard_name)
        os.makedirs(local_dst, exist_ok=True)

        if os.path.exists(local_dst) and len(os.listdir(local_dst)) > 0:
            logger.info(f"skipping download because destination: {local_dst} already exists and contains files {os.listdir(local_dst)}")
            return

        # create "shard" of urls file containing the sub-set of urls to download
        url_shard_path = os.path.join(local_dst, "urls.txt")
        with open(url_shard_path, "w") as f:
            f.writelines(urls)

        # download files
        logger.info(f"downloading files to {local_dst}")
        download_cmd = f"cat {f.name} | xargs -n 1 -P {self.num_processes} wget -P {local_dst} -q"
        subprocess.run(download_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # upload to S3 if necessary
        if has_s3_dst:
            bucket_name = f"{self.output_prefix}-{shard_name}"
            s3.upload_dir(local_dst, bucket_name, show_progress=True)
            if self.cleanup:
                shutil.rmtree(local_dst)

