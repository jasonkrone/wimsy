

def doc_to_choice(doc):
    a, b, c, d = doc["A"], doc["B"], doc["C"], doc["D"]
    return [a, b, c, d]

def doc_to_target(doc):
    key = doc["Answer"]
    return doc[key]
