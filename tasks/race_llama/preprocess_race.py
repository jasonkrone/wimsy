import ast
import re


def process_ast(string):
    return ast.literal_eval(string)


def last_problem(doc):
    return process_ast(doc["problems"])[-1]


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
    problem = last_problem(doc)
    # this is a list
    choices = problem["options"]
    assert len(choices) == 4
    question = clean_question(problem["question"])
    a, b, c, d = choices
    text = f"Background: {background}\n\nQuestion: {question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
    return text


def doc_to_target(doc):
    return last_problem(doc)["answer"]

