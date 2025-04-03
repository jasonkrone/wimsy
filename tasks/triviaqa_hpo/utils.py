"""
Taken from https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
"""
import string
import re


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def normalize_triviaqa_predictions(resps, docs):
    out_list = []
    for inst in resps:
        out = []
        for r in inst:
            r_normalized = normalize_answer(r)
            out.append(r_normalized)
        out_list.append(out)
    return out_list


def normalize_triviaqa_targets(doc):
    return doc["answer"]["normalized_aliases"]


def doc_to_target(doc):
    return doc["answer"]["aliases"][0]


def doc_to_choice(doc):
    return [doc["answer"]["aliases"][0]]


