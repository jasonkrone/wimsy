
def doc_to_target(doc):
    """
    We need to the function b/c otherwise the type of numerical answers get cast
    """
    choices = doc["choices"]["text"]
    target_key = doc["answerKey"]
    target_idx = doc["choices"]["label"].index(target_key)
    target = choices[target_idx]
    return target