import argparse
import csv
import json
import logging
import os

from .cwn_base import CwnBase
from .download import get_model_path

__cwn_inst = None
__cwn_cache = {}

def simplify_pos(pos):
    relation = {
        "A": "OTHER",
        "Caa": "OTHER",
        "Cab": "OTHER",
        "Cba": "OTHER",
        "Cbb": "OTHER",
        "Da": "OTHER",
        "Dfa": "OTHER",
        "Dfb": "OTHER",
        "Di": "OTHER",
        "Dk": "OTHER",
        "D": "OTHER",
        "Na": "N",
        "Nb": "Nb",
        "Nc": "N",
        "Ncd": "N",
        "Nd": "N",
        "Neu": "OTHER",
        "Nes": "OTHER",
        "Nep": "OTHER",
        "Neqa": "OTHER",
        "Neqb": "OTHER",
        "Nf": "OTHER",
        "Ng": "OTHER",
        "Nh": "N",
        "Nv": "N",
        "I": "OTHER",
        "P": "OTHER",
        "T": "OTHER",
        "nom,VA": "V",
        "VH,D": "V",
        "VA": "V",
        "VAC": "V",
        "VB": "V",
        "VC": "V",
        "VCL": "V",
        "VD": "V",
        "VE": "V",
        "VF": "V",
        "VG": "V",
        "VH": "V",
        "VHC": "V",
        "VI": "V",
        "VJ": "V",
        "VK": "V",
        "VL": "V",
        "COLONCATEGORY": "OTHER",
        "COMMACATEGORY": "OTHER",
        "DASHCATEGORY": "OTHER",
        "ETCCATEGORY": "OTHER",
        "EXCLAMATIONCATEGORY": "OTHER",
        "PARENTHESISCATEGORY": "OTHER",
        "PAUSECATEGORY": "OTHER",
        "PERIODCATEGORY": "OTHER",
        "QUESTIONCATEGORY": "OTHER",
        "SEMICOLONCATEGORY": "OTHER",
        "SPCHANGECATEGORY": "OTHER",
    }
    try:
        return relation[pos]
    except KeyError as e:
        return "OTHER"


#Search and filter senses from CWN
def get_cwn_senses(cwn, word_info):
    word = word_info["word"]
    if word in __cwn_cache:
        senses = __cwn_cache[word]
    else:
        senses = cwn.find_all_senses(word)
        __cwn_cache[word] = senses

    if len(senses) == 0:
        return []
    elif word_info['pos'] == "":
        return senses
    else:
        simple_pos = simplify_pos(word_info["pos"])
        same_pos_senses = list(filter(lambda x:simplify_pos(x.pos)==simple_pos, senses))
        return same_pos_senses


#Generate original sentence with tag on target word
def generate_sentence(target, sentence_info):
    sentence = ""
    for i in range(len(sentence_info)):
        if i == target:
            sentence += "<"+sentence_info[i][0]+">"
        else:
            sentence += sentence_info[i][0]
    return sentence


#Generate a row of training data
def generate_row(word_info, cwn_sense):
    row = {}
    row['test_word'] = word_info["word"]
    row['test_pos'] = word_info["pos"]
    row['test_sense_id'] = word_info["sense_id"]
    row['test_definition'] = word_info["definition"]
    row['test_sentence'] = word_info["sentence"]
    row['cwn_sense_id'] = cwn_sense.id
    row['cwn_definition'] = cwn_sense.definition
    tmp = [x for x in cwn_sense.all_examples() if x.strip()]
    if len(tmp) > 0:
        row['cwn_sentence'] = tmp[0]
    else:
        return None
    row['label'] = True if word_info["sense_id"] == cwn_sense.id else False
    return row

def get_cwn_inst():
    global __cwn_inst
    if not __cwn_inst:
        logging.info("Preprocessing data")        
        logging.info("loading CWN")        
        __cwn_inst = CwnBase()
        logging.info("Done")
    return __cwn_inst

# data: a list of sentences
def preprocess(data):
    
    all_data = []
    cwn = get_cwn_inst()
    logging.info("Query CWN graph to build test data")

    for sentence_info in data:
        one_sentence = []
        for idx, i in enumerate(sentence_info):
            one_instance = []
            word_info = {}
            word_info["word"] = i[0]
            word_info["pos"] = i[1]
            word_info["sense_id"] = ""
            word_info["definition"] = ""
            
            word_info["sentence"] = generate_sentence(idx, sentence_info)                          
            cwn_senses = get_cwn_senses(cwn, word_info)
            
            if len(cwn_senses) == 0:
                one_instance.append({'test_word': word_info["word"], 'test_pos' :word_info["pos"]} )
            for sense in cwn_senses:
                row = generate_row(word_info, sense)
                if row is not None:
                    one_instance.append(row)
            one_sentence.append(one_instance)
        all_data.append(one_sentence)
    logging.info("Done")
    return all_data

