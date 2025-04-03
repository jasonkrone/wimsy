import os
import time
import json
import argparse

import matplotlib.pyplot as plt

from utils import Config
from preprocessing.prep_data import main as run_preprocessing

MICRO_BATCH_SIZES = [
    2,
    4,
    8, 
    16,
    32,
    64,
    128,
    256,
    512,
]

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to data preprocess config file in yaml format")
parser.add_argument("--save_dir", type=str, default="/local_out/sweep_micro_batch_size")


def plot_batch_size_vs_run_time(batch_sizes, run_times, save_path):
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, run_times, marker='o')

    # Add labels and title
    plt.xlabel('Micro Batch Size')
    plt.ylabel('Run Time (seconds)')
    plt.title('Run Time vs Micro Batch Size')
    
    # Add grid
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Save figure
    plt.savefig(save_path)
    plt.close()


def main(config, save_dir):

    run_durations = []
    for micro_batch_size in MICRO_BATCH_SIZES:
        config["pipeline"][0]["args"]["micro_batch_size"] = micro_batch_size
        start = time.time()
        run_preprocessing(config)
        end = time.time()
        delta = end - start
        run_durations.append(delta)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "micro_batch_size_sweep.png")
    plot_batch_size_vs_run_time(MICRO_BATCH_SIZES, run_durations, save_path)

    exp_path = os.path.join(save_dir, "outputs.json")
    exp_dict = {"durations": run_durations, "batch_sizes": MICRO_BATCH_SIZES}
    with open(exp_path, "w") as f:
        json.dump(exp_dict, f)


if __name__ == "__main__":
    # spawn required by dolma utils and must be set before other imports
    # TODO: this causes an error when using the humaneval dataset
    args = parser.parse_args()
    data_config = Config.from_yaml(args.config)
    main(data_config, args.save_dir)

