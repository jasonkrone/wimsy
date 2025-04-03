import os
import json
import pickle
from collections import Counter
from multiprocessing import Pool, Manager
from threading import Thread
from glob import glob

from tqdm import tqdm

from utils import S3UploadDownload


def pbar_listener(q, total):
    """
    Written by GPT-4    
    """
    pbar = tqdm(total=total, desc="Aggregating DCLM scores", mininterval=30)
    for item in iter(q.get, None):
        pbar.update(item)
    pbar.close()


def scores_for_file(path, pbar_queue):
    with open(path, "r") as f:
        json_dict = json.load(f)
    file_scores = [score_tuple[0] for score_tuple, count in json_dict["counts"] for _ in range(count)]
    pbar_queue.put(1)
    return file_scores


def aggregate_scores(path_list):
    # Start a thread to update the progress bar
    manager = Manager()
    pbar_queue = manager.Queue()
    pbar_thread = Thread(target=pbar_listener, args=(pbar_queue, len(path_list)))
    pbar_thread.start()

    args_list = [(path, pbar_queue) for path in path_list]

    with Pool(processes=64) as pool:
        results = pool.starmap(scores_for_file, args_list)

    # Signal the pbar thread to terminate
    pbar_queue.put(None)
    pbar_thread.join()
    manager.shutdown()

    # Flatten the list of lists into a single list
    scores = [score for partial_scores in results for score in partial_scores]
    return scores


def main():
    metadata_dir = "/local_out/dclm/count-metadata/"
    glob_path = f"{metadata_dir}/*/*.done.txt"
    count_paths = glob(glob_path) 
    score_list = aggregate_scores(count_paths)

    with open("/local_out/dclm/dclm_scores.pkl", "wb") as f:
        pickle.dump(score_list, f)


if __name__ == "__main__":
    main()
