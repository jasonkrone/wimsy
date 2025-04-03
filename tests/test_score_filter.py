import os
import shutil
import json

import pytest
import smart_open
import pandas as pd
import pandas.testing as pdt

from utils import Config
from preprocessing.prep_data import main as run_data_prep


TEST_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(TEST_DIR, "artifacts", "score_filter")
FILTER_INPUT_PATH = os.path.join(ARTIFACTS_DIR, "inputs", "shard-1-of-1", "input_docs.json.gz")
FILTER_OUTPUT_DIR = os.path.join(ARTIFACTS_DIR, "outputs")
FILTER_OUTPUT_PATH = os.path.join(FILTER_OUTPUT_DIR, "shard-1-of-1", "filtered-0000.json.gz")
EXPECTED_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "expected_output.jsonl")


@pytest.fixture(scope="module")
def run_sampler_pipeline(request):
    config_path = os.path.join(ARTIFACTS_DIR, "filter_config.yaml")
    config = Config.from_yaml(config_path)
    run_data_prep(config)

    def delete_test_outputs():
        shutil.rmtree(FILTER_OUTPUT_DIR)

    #request.addfinalizer(delete_test_outputs)
    return config 


def read_jsonl(path):
    json_list = []
    with smart_open.open(path) as f:
        lines = f.readlines()
        for l in lines:
            json_list.append(json.loads(l))
    return json_list


def df_for_jsonl(path):
    return pd.DataFrame(read_jsonl(path))


def test_sampler(run_sampler_pipeline):
    actual_df = df_for_jsonl(FILTER_OUTPUT_PATH)
    expected_df = df_for_jsonl(EXPECTED_OUTPUT_PATH)

    actual_df.drop("metadata", axis=1, inplace=True)
    actual_df = actual_df.reset_index(drop=True)
    actual_df = actual_df[sorted(actual_df.columns)]

    expected_df.drop("metadata", axis=1, inplace=True)
    expected_df = expected_df.reset_index(drop=True)
    expected_df = expected_df[sorted(expected_df.columns)]

    pdt.assert_frame_equal(actual_df, expected_df)
