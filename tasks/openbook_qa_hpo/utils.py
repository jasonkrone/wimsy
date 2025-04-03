
def doc_to_target(doc):
    answer_key = doc["answerKey"].strip()
    answer_idx = doc["choices"]["label"].index(answer_key)
    return doc["choices"]["text"][answer_idx]
