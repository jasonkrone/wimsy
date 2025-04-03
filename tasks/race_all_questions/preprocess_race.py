import pdb
import re


def clean_question(question):
    """
    Removes extra punctuation in the question str from fill in the blank questions not ending in "?"
    """
    re_str = "[a-zA-Z](?=[^a-zA-Z]*$)"
    match = re.search(re_str, question)
    if question[match.end():].strip() != "?":
        question = question[0:match.end()]
    return question


def doc_to_choice(doc):
    return ["A", "B", "C", "D"]


def doc_to_text(doc):
    background = doc["article"]
    question = clean_question(doc["question"])
    choices = doc["options"]
    assert len(choices) == 4
    a, b, c, d = choices
    text = f"Background: {background}\n\nQuestion: {question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
    return text


def doc_to_target(doc):
    letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
    return letter_to_num[doc["answer"]]

