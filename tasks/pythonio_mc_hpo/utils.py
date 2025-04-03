

def doc_to_decontamination_query(doc):
    """
    Remove beginning Program: and trailing Input: Output: for decontamination query
    """
    question = doc["Question"]
    return "\n".join(question.split("\n")[1:-3])
