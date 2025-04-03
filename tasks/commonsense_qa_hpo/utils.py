

def doc_to_target(doc):
    answer_id = doc["answerKey"]
    answer_idx = doc["choices"]["label"].index(answer_id)
    answer_str = doc["choices"]["text"][answer_idx]
    return answer_str


