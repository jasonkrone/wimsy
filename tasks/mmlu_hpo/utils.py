

def doc_to_target(doc):
    choices = doc["choices"]
    target_idx = doc["answer"]
    return choices[target_idx]