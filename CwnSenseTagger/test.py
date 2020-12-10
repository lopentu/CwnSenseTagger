import logging
import numpy as np
import os
import torch
import warnings

from argparse import ArgumentParser
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, random_split

from .config import CLS, SEP, COMMA, PAD, USE_CUDA
from .model import WSDBertClassifer
from .util import positive_weight, accuracy
from .download import get_model_path

wsd_model = None

def batch_generation(batch_size, data):
    idx = 0
    target_idx = min(len(data), idx+batch_size)
    all_batch = []
    while idx < len(data):
        target_idx = min(len(data), idx+batch_size)
        max_length = min(max([len(data[i]['test_sentence_id'])+len(data[i]['test_word_id']) + len(data[i]['cwn_definition_id']) + len(data[i]['cwn_sentence_id']) + 5 for i in range(idx, target_idx) ]), 512)
        context = []
        attention_mask = []
        token_type_id = []
        label = []
        for i in range(idx, target_idx):
            attention = np.ones(max_length)
            token_type = np.zeros(max_length)
            text = [CLS] + data[i]['test_sentence_id'] + [SEP] + data[i]['test_word_id'] + [COMMA] + data[i]['cwn_definition_id'] + [COMMA] + data[i]['cwn_sentence_id'] + [SEP]

            pad_length = max(max_length-len(text),0)
            token_type[len(data[i]['test_sentence_id'])+2:len(text)] = 1
            if pad_length > 0:
                attention[-pad_length:] = 0
                text += [PAD for i in range(pad_length)]
            context.append(text)
            attention_mask.append(attention)
            token_type_id.append(token_type)
            label.append(data[i]['label'])
        batch = {
            "context": torch.tensor(context),
            "attention_mask": torch.tensor(attention_mask),
            "token_type_ids": torch.tensor(token_type_id),
            "label": torch.tensor(label)
            }
        all_batch.append(batch)
        idx += batch_size

    return all_batch

def warmup():
    global wsd_model
    if not wsd_model:
        device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
        logging.info('Device type is %s'%(device))
        logging.info("Prepare Dataset")
        wsd_model = WSDBertClassifer.from_pretrained(get_model_path())
        wsd_model.to(device)
        wsd_model.eval()

@torch.no_grad()
def test(all_json, batch_size=8):
    global wsd_model

    device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
    warnings.filterwarnings("ignore")
    warmup()

    # model.load_state_dict(checkpoint['state_dict'])
    all_ans = []
    logging.info("Start Inference")
    for sentence in all_json:
        sentence_ans =[]
        for word in sentence:
            if word[0] == []:
                sentence_ans.append(-1)
                continue

            batches = batch_generation(batch_size, word)
            one_label = []
            one_predict = []
            for b in batches:
                context = b['context'].to(device)
                attention_mask = b['attention_mask'].to(device)
                token_type_ids = b['token_type_ids'].to(torch.long).to(device)

                logits, _ = wsd_model(context, attention_mask=attention_mask, token_type_ids=token_type_ids)
                one_predict += logits.squeeze(1).tolist()

            sentence_ans.append(np.argmax(one_predict))
        all_ans.append(sentence_ans)
    logging.info("Done")
    return all_ans

