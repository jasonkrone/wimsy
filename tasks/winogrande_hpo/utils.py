import difflib


def doc_to_decontamination_query(doc):
    s1, s2 = doc["choices"]
    s1_words = s1.split()
    s2_words = s2.split()
    matcher = difflib.SequenceMatcher(None, s1_words, s2_words)
    result_words = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            result_words.extend(s1_words[a0:a1])
        else:
            result_words.append('_')
    return ' '.join(result_words)
