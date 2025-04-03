import os
import shutil

import pytest
import pandas as pd
import pandas.testing as pdt
from elasticsearch import Elasticsearch

from utils import Config
from preprocessing.prep_data import main as run_data_prep

TEST_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(TEST_DIR, "artifacts", "decontamination")
OUTPUT_DIR = os.path.join(TEST_DIR, "temp_output", "decontamination")


@pytest.fixture(scope="module")
def run_decontamination_pipeline(request):
    # run decontam pipeline
    config_path = os.path.join(ARTIFACTS_DIR, "decontamination_config.yaml")
    config = Config.from_yaml(config_path)
    run_data_prep(config)

    def delete_decontaination_test_outputs():
        shutil.rmtree(OUTPUT_DIR)
        parent_dir = os.path.dirname(OUTPUT_DIR)
        if not os.listdir(parent_dir):
            os.rmdir(parent_dir)
        decontam_docs_path = os.path.join(ARTIFACTS_DIR, "temp_input_docs.jsonl")
        os.remove(decontam_docs_path)

    def delete_index():
        es = Elasticsearch("http://elasticsearch:9200")
        es.indices.delete(index=config.pipeline[0]["args"]["index_name"])

    request.addfinalizer(delete_decontaination_test_outputs)
    request.addfinalizer(delete_index)

    return config 


def test_contaminated_span_tagger(run_decontamination_pipeline):
    actual_spans_path = os.path.join(OUTPUT_DIR, "contaminated_spans.jsonl")
    actual_spans_df = pd.read_json(actual_spans_path, lines=True)

    expected_spans_path = os.path.join(ARTIFACTS_DIR, "expected_contaminated_spans.jsonl")
    expected_spans_df = pd.read_json(expected_spans_path, lines=True)

    pdt.assert_frame_equal(actual_spans_df, expected_spans_df)


def test_decontaminator(run_decontamination_pipeline):
    actual_out_path = os.path.join(ARTIFACTS_DIR, "temp_input_docs.jsonl")
    actual_out_df = pd.read_json(actual_out_path, lines=True)

    expected_out_path = os.path.join(ARTIFACTS_DIR, "expected_output_docs.jsonl")
    expected_out_df = pd.read_json(expected_out_path, lines=True)

    pdt.assert_frame_equal(actual_out_df, expected_out_df)


