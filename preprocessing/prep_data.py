import pdb
import os
import argparse
from multiprocessing import set_start_method

from tokenizer import *
from preprocessing.transform import *
from preprocessing.decontaminate import *
from preprocessing.writer import *
from preprocessing.upload_and_download import *
from preprocessing.dolma_cli_wrapper import *
from preprocessing.postprocess import *
from utils import Registry, Config, logger

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to data preprocess config file in yaml format")
parser.add_argument("--num_processes", type=int, default=64)


def main(config):
    config = config.as_dict()
    for step_config in config["pipeline"]:
        step_id = step_config["id"]
        logger.info(f"Running step: {step_id}")
        step_cls = Registry.get(step_id)
        step = step_cls(**step_config["args"])
        step(**step_config.get("call_kwargs", {}))


if __name__ == "__main__":
    # spawn required by dolma utils and must be set before other imports
    # TODO: this causes an error when using the humaneval dataset
    set_start_method("spawn")
    args = parser.parse_args()
    data_config = Config.from_yaml(args.config)
    os.environ["NUMEXPR_MAX_THREADS"] = str(args.num_processes)
    main(data_config)
