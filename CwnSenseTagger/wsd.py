from .preprocess import preprocess
from .tokenize_test import tokenize
from .test import test

def senseTag(data):
    cwn_batch = preprocess(data)
    tokenize_data = tokenize(cwn_batch)
    ans_idx = test(tokenize_data)

    all_ans = []
    for batch, sentence_idx  in zip(cwn_batch, ans_idx):
        sentence_ans = []
        for word, idx in zip(batch, sentence_idx):
            if idx == -1:
                word_ans = [word[idx]['test_word'], word[idx]['test_pos'], "", ""]
            else:
                word_ans = [word[idx]['test_word'], word[idx]['test_pos'], word[idx]['cwn_sense_id'], word[idx]['cwn_definition']]
            sentence_ans.append(word_ans)
        all_ans.append(sentence_ans)
    return all_ans
