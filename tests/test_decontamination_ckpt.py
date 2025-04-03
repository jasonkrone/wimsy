import os

from preprocessing.decontaminate import ContaminatedSpanTagger

TEST_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(TEST_DIR, "artifacts", "decontamination_ckpt")
NGRAMS_PATH = os.path.join(ARTIFACTS_DIR, "ngrams.jsonl")


def test_load_all_ngrams():
    i = 0
    expected_outputs = [
        ["1", "2", "3"],
        ["4", "5", "6", "7"],
        ["8", "9", "10", "11"],
        ["12"],
        ["13"],
    ]
    ngram_generator = ContaminatedSpanTagger.read_ngrams(NGRAMS_PATH, ckpt_iter=0)
    for ngram_list in ngram_generator:
        assert expected_outputs[i] == ngram_list
        i += 1


def test_load_ngrams_from_ckpt():
    i = 0
    expected_outputs = [
        ["12"],
        ["13"],
    ]
    ngram_generator = ContaminatedSpanTagger.read_ngrams(NGRAMS_PATH, ckpt_iter=11)
    for ngram_list in ngram_generator:
        assert expected_outputs[i] == ngram_list
        i += 1
