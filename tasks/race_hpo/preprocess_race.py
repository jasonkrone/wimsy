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


def doc_to_target_text(doc):
    target = None
    answer = doc["answer"]
    choices = doc["options"]
    assert len(choices) == 4
    a, b, c, d = choices
    if answer == "A":
        target = a
    elif answer == "B":
        target = b
    elif answer == "C":
        target = c
    elif answer == "D":
        target = d
    else:
        raise ValueError(f"RACE answer {answer} unsupported")
    return target


def doc_to_target(doc):
    return doc["answer"]


def doc_to_decontamination_query(doc):
    text = doc_to_text(doc)
    target = doc_to_target(doc)
    return f"{text} {target}"

