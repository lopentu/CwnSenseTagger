import logging
import numpy as np
import os
import torch
import warnings
from time import time

from argparse import ArgumentParser
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, random_split

from .config import CLS, SEP, COMMA, PAD
from . import config
from .model import WSDBertClassifer
from .util import positive_weight, accuracy
from .download import get_model_path

wsd_model = None        
pred_timers = np.zeros(4, dtype=np.double)
model_timers = np.zeros(3, dtype=np.double)
profile_dbg = {"concat_batch_size": []}

def reset_profiler():
    global pred_timers, model_timers, profile_dbg
    pred_timers = np.zeros(4, dtype=np.double)
    model_timers = np.zeros(3, dtype=np.double)
    profile_dbg = {"concat_batch_size": []}

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

def collate_batches(batches, device="cpu"):
    if not batches:
        return {}
    concat_batch = {}
    device = torch.device(device)

    for k in batches[0]:
        if batches[0][k].dim() == 1:
            concat_batch[k] = torch.cat([x[k] for x in batches])
            continue
        N = sum(x[k].size(0) for x in batches)
        max_seqlen = max(x[k].size(-1) for x in batches)
        padded = torch.full((N, max_seqlen), fill_value=PAD)
        N_offset = 0
        for batch_x in batches:
            n_x = batch_x[k].size(0)
            padded[N_offset: N_offset+n_x,
                   :batch_x[k].size(-1)] = batch_x[k]
            N_offset += n_x
        concat_batch[k] = padded.to(device)
    return concat_batch

def group_by_batch(data, batch_size=128):
    if not data:
        return
    N = data[list(data.keys())[0]].size(0)
    for i in range(0, N, batch_size):
        batch = {}
        for k in data:
            batch[k] = data[k][i:i+batch_size]
        yield batch

def warmup():
    global wsd_model
    if not wsd_model:
        device = torch.device('cuda' if torch.cuda.is_available() and config.USE_CUDA else 'cpu')
        logging.info('Device type is %s'%(device))
        logging.info("Prepare Dataset")
        wsd_model = WSDBertClassifer.from_pretrained(get_model_path())
        wsd_model.to(device)
        wsd_model.eval()

@torch.no_grad()
def test_batched(all_json, batch_size=8, profile=False):
    global wsd_model

    device = torch.device('cuda' if torch.cuda.is_available() and config.USE_CUDA else 'cpu')
    warnings.filterwarnings("ignore")
    warmup()

    # model.load_state_dict(checkpoint['state_dict'])
    all_ans = [[]] * len(all_json)
    logging.info("Start Inference")
    batches = []
    word_idxs = []
    sent_idxs = []

    if profile:
        global pred_timers, model_timers, profile_dbg
        t0 = time()

    for sent_idx, sentence in enumerate(all_json):
        sentence_ans = [-1] * len(sentence)
        for word_idx, word in enumerate(sentence):
            if len(word) == 0:            
                continue

            if word[0] == []:
                sentence_ans[word_idx] = -1
                continue

            if len(word) == 1:
                sentence_ans[word_idx] = 0
                continue

            batch_x = batch_generation(len(word), word)[0]
            batches.append(batch_x)
            n_batch_x = batch_x["context"].size(0)
            word_idxs.extend([word_idx] * n_batch_x)
            sent_idxs.extend([sent_idx] * n_batch_x)
        all_ans[sent_idx] = sentence_ans
    
    if profile: t1 = time()

    concat_batch = collate_batches(batches)
    word_idxs = np.array(word_idxs)
    sent_idxs = np.array(sent_idxs)    

    if profile: 
        t2 = time()
        if concat_batch:
            profile_dbg["concat_batch_size"].append(concat_batch["context"].size(0))

    one_predict = []

    # run model through batches
    for b in group_by_batch(concat_batch, batch_size):
        if profile: t21 = time()
        context = b['context'].to(device)
        attention_mask = b['attention_mask'].to(device)
        token_type_ids = b['token_type_ids'].to(torch.long).to(device)

        if profile: t22 = time()       
        logits, _ = wsd_model(context, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if profile: t23 = time()
        one_predict += logits.squeeze(1).tolist()

        if profile: 
            t24 = time()
            model_timers += np.array([t22-t21, t23-t22, t24-t23])

    if profile: t3 = time()
    # collect predictions by words
    one_predict = np.array(one_predict)
    uniq_cursor = np.unique(np.vstack([sent_idxs, word_idxs]), axis=1)

    for uniq_i in range(uniq_cursor.shape[1]):
        sent_idx, word_idx = uniq_cursor[:, uniq_i].tolist()
        sentence_ans = all_ans[sent_idx]
        mask = (sent_idxs == sent_idx) & (word_idxs == word_idx)
        logits_word = one_predict[mask]
        sentence_ans[word_idx] = np.argmax(logits_word)

    if profile:
        t4 = time()
        pred_timers += np.array([t1-t0, t2-t1, t3-t2, t4-t3])

    logging.info("Done")
    return all_ans


@torch.no_grad()
def test(all_json, batch_size=8):
    global wsd_model

    device = torch.device('cuda' if torch.cuda.is_available() and config.USE_CUDA else 'cpu')
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

            if len(word) == 1:
                sentence_ans.append(0)
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

