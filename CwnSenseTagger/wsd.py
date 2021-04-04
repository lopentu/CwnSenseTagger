from .preprocess import preprocess
from .tokenize_test import tokenize
from .test import test, test_batched

def senseTag_batched(data, batch_size=8):
    cwn_batch = preprocess(data)
    tokenize_data = tokenize(cwn_batch)
    ans_idx = test_batched(tokenize_data, batch_size)
    
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

def senseTag(data, batch_size=8):
    cwn_batch = preprocess(data)
    tokenize_data = tokenize(cwn_batch)
    ans_idx = test(tokenize_data, batch_size)

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
