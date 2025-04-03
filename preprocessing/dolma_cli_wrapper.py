import os
import glob
import yaml
import shutil
import tempfile
import subprocess

import smart_open

from utils import Registry, logger, group_docs


class DolmaCLIWrapper(object):

    @staticmethod
    def _run_command(cmd):
        cmd = f"RUST_BACKTRACE=full {cmd}"
        out = subprocess.run(cmd, shell=True, check=True, text=True)
        return out

    def cleanup(self):
        shutil.rmtree(self.temp_dir)

    def _make_config_for_dict(self, kwargs):
        self.temp_dir = tempfile.mkdtemp(prefix="dolma_cli_configs")
        self.config_path = os.path.join(self.temp_dir, "dolma_config.yaml")
        with open(self.config_path, "w") as f:
            yaml.dump(kwargs, f)


@Registry.register("document_tagger")
class DocumentTagger(DolmaCLIWrapper):

    def __init__(self, source_prefix, taggers, num_processes):
        self.source_prefix = source_prefix
        self.taggers = taggers
        self.num_processes = num_processes

    def __call__(self, **kwargs):
        tagger_str = " ".join(self.taggers)
        logger.info(f"tagging documents with taggers: {tagger_str}")
        self._run_command(f"dolma tag --documents {self.source_prefix} --taggers {tagger_str} --processes {self.num_processes}")


@Registry.register("document_filter")
class DocumentFilter(DolmaCLIWrapper):
    """
    Wrapper around dolma tag and mix CLI tools
    """

    FILE_EXTENSION = "json.gz"

    def __init__(self, num_processes, documents_glob=None, shard_from_source_regex=None, **kwargs):
        self.config_path = None
        self.shard_to_documents = None
        self.shard_to_output = None
        self.config_dict = kwargs
        self.num_processes = num_processes

        if documents_glob is not None:
            assert len(self.config_dict["streams"]) == 1
            self.shard_to_documents = group_docs(documents_glob, shard_from_source_regex)
            output_dir = self.config_dict["streams"][0]["output"]["path"]
            self.shard_to_output = {shard: os.path.join(output_dir, shard) for shard in self.shard_to_documents}
        else:
            self._make_config_for_dict(self.config_dict)

    def __call__(self, cleanup, **kwargs):
        output_dirs = []
        documents_to_cleanup = []

        if self.config_path:
            self._run_command(f"dolma -c {self.config_path} mix --processes {self.num_processes}")
            documents_to_cleanup += [doc for stream in self.config_dict["streams"] for doc in stream["documents"]]
            output_dirs += [stream["output"]["path"] for stream in self.config_dict["streams"]]
        else:
            for shard, output_dir in self.shard_to_output.items():
                self.config_dict["streams"][0]["documents"] = self.shard_to_documents[shard]
                self.config_dict["streams"][0]["output"]["path"] = output_dir
                self._make_config_for_dict(self.config_dict)
                self._run_command(f"dolma -c {self.config_path} mix --processes {self.num_processes}")
                documents_to_cleanup += self.shard_to_documents[shard]
                output_dirs.append(output_dir)

        for dir in output_dirs:
            self.remove_empty_files(os.path.join(dir, f"*.{self.FILE_EXTENSION}"))

        if cleanup:
            self.cleanup()
            self._cleanup_documents(documents_to_cleanup)

    @classmethod
    def remove_empty_files(cls, glob_path):
        empty_files = [path for path in glob.glob(glob_path) if cls.is_empty_file(path)]
        logger.info(f"removing empty files: {empty_files}")
        for path in empty_files:
            os.remove(path)

    @classmethod
    def is_empty_file(cls, path):
        is_empty = False
        with smart_open.open(path, "rt") as f:
            is_empty = f.readline() == ""
        return is_empty

    @staticmethod
    def _cleanup_documents(documents):
        parent_dirs = set([os.path.dirname(doc) for doc in documents])
        for dir in parent_dirs:
            shutil.rmtree(dir)


@Registry.register("bloom_filter")
class BloomFilter(DolmaCLIWrapper):
    """
    Wrapper around the dolma bloom filter CLI tool
    """
    def __init__(self, documents_glob=None, **kwargs):
        """
        Kwargs are all written to the bloom filter yaml config file
        """
        # we support multiple wildcard symbols via documents_glob
        if documents_glob is not None:
            paths = glob.glob(documents_glob)
            kwargs["documents"] = paths
        self._make_config_for_dict(kwargs)

    def __call__(self, **kwargs):
        self._run_command(f"dolma -c {self.config_path} dedupe")
        self.cleanup()

