import os
import argparse
import subprocess

from utils import Config, logger

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)


def main(config):
    for location_dict in config.data_sync_locations:
        fsx_path = location_dict["fsx_path"]
        data_dir = os.path.join(config.fsx_mountpoint, fsx_path.replace("/", ""))
        logger.info(f"making data dir: {data_dir}")
        subprocess.run(f"mkdir -p {data_dir}", shell=True, check=True, text=True)
        subprocess.run(f"chmod -R 777 {data_dir}", shell=True, check=True, text=True)
        subprocess.run(f"chown -R 1000:0 {data_dir}", shell=True, check=True, text=True)


if __name__ == "__main__":
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    main(config)
