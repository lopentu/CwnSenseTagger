import json
import logging
import numpy as np

from transformers import BertTokenizer
from .config import BERT_MODEL

def tokenize(test_data):
    all_instance = []
    logging.info("Tokenize data by %s"%(BERT_MODEL))
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    for sentence in test_data:
        one_sentence = []
        for word in sentence:
            one_instance = []
            for i in word:
                if "cwn_definition" not in i:
                    one_instance.append([])
                    continue
                instance = {
                'test_word': i['test_word'],
                'test_word_id': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(i['test_word'])),
                'test_sentence': i['test_sentence'],
                'test_sentence_id': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(i['test_sentence'])),
                'cwn_definition': i['cwn_definition'],
                'cwn_definition_id': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(i['cwn_definition'])),
                'cwn_sentence': i['cwn_sentence'],
                'cwn_sentence_id': tokenizer.convert_tokens_to_ids(tokenizer.tokenize(i['cwn_sentence'])),
                'label': bool(i['label']),
                }
                one_instance.append(instance)
            one_sentence.append(one_instance)
        all_instance.append(one_sentence) 
    logging.info("Done")
    return all_instance

if __name__ == "__main__":
    tokenize()
