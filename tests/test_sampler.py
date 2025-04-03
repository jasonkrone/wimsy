import os

import pytest
import pandas as pd
import pandas.testing as pdt

from utils import Config
from preprocessing.prep_data import main as run_data_prep


TEST_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(TEST_DIR, "artifacts", "sampler")
SAMPLER_OUTPUT_PATH = os.path.join(ARTIFACTS_DIR, "temp_sampler_input.jsonl")


@pytest.fixture(scope="module")
def run_sampler_pipeline(request):
    config_path = os.path.join(ARTIFACTS_DIR, "sampler_config.yaml")
    config = Config.from_yaml(config_path)
    run_data_prep(config)

    def delete_decontaination_test_outputs():
        os.remove(SAMPLER_OUTPUT_PATH)
        sampler_metadata_path = os.path.join(ARTIFACTS_DIR, "sampler_input.jsonl.done.txt")
        os.remove(sampler_metadata_path)

    request.addfinalizer(delete_decontaination_test_outputs)
    return config 


def test_sampler(run_sampler_pipeline):
    actual_samples_df = pd.read_json(SAMPLER_OUTPUT_PATH, lines=True)

    expected_samples_path = os.path.join(ARTIFACTS_DIR, "sampler_expected_output.jsonl")
    expected_samples_df = pd.read_json(expected_samples_path, lines=True)

    pdt.assert_frame_equal(actual_samples_df, expected_samples_df)
