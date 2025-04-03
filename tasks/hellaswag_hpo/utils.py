import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        assert len(doc["endings"]) == 4
        doc["query"] = preprocess(ctx)
        return doc

    return dataset.map(_process_doc)


def doc_to_decontamination_query(doc):
    prefix = preprocess(doc["ctx_a"] + " " + doc["ctx_b"].capitalize())
    ending = doc["endings"][int(doc["label"])]
    return f"{prefix} {ending}"


def doc_to_target(doc):
    return doc["endings"][int(doc["label"])]


def doc_to_choice(doc):
    return doc["endings"]

