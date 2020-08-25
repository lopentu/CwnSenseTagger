import argparse
import csv
import json
import logging
import os

from .cwn_base import CwnBase


def simplify_pos(pos):
    relation = {
        "A": "A",
        "Caa": "C",
        "Cab": "POST",
        "Cba": "POST",
        "Cbb": "C",
        "Da": "ADV",
        "Dfa": "ADV",
        "Dfb": "ADV",
        "Di": "ASP",
        "Dk": "ADV",
        "D": "ADV",
        "Na": "N",
        "Nb": "Nb",
        "Nc": "N",
        "Ncd": "N",
        "Nd": "N",
        "Neu": "DET",
        "Nes": "DET",
        "Nep": "DET",
        "Neqa": "DET",
        "Neqb": "POST",
        "Nf": "M",
        "Ng": "POST",
        "Nh": "N",
        "Nv": "Nv",
        "I": "T",
        "P": "P",
        "T": "T",
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
        "COLONCATEGORY": "S",
        "COMMACATEGORY": "S",
        "DASHCATEGORY": "S",
        "ETCCATEGORY": "S",
        "EXCLAMATIONCATEGORY": "S",
        "PARENTHESISCATEGORY": "S",
        "PAUSECATEGORY": "S",
        "PERIODCATEGORY": "S",
        "QUESTIONCATEGORY": "S",
        "SEMICOLONCATEGORY": "S",
        "SPCHANGECATEGORY": "S",
    }
    try:
        return relation[pos]
    except KeyError as e:
        return pos

#Search and filter senses from CWN
def get_cwn_senses(cwn, word_info):
    senses = cwn.find_senses("^{}$".format(word_info["word"]))
    if len(senses) == 0:
        return []

    simple_pos = simplify_pos(word_info["pos"])
    same_pos_senses = list(filter(lambda x:simplify_pos(x.pos)==simple_pos, senses))

    if len(same_pos_senses) < 2:
        return senses
    
    # To prevent the correct sense from being filtered
    if word_info['sense_id'] is not "" and len(senses) != 0 and len(list(filter(lambda x:x.id == word_info["sense_id"], same_pos_senses))) == 0:
        same_pos_senses.append(list(filter(lambda x:x.id == word_info["sense_id"], senses))[0])
        
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


# data is list of sentences

def preprocess(data):
    logging.info("Preprocessing data")
    logging.info("Install CWN graph")
    CwnBase.install_cwn(os.path.join(os.path.dirname(__file__),"data","cwn_graph.pyobj"))
    logging.info("Done")
    cwn = CwnBase()
    all_data = []
    logging.info("Query CWN graph to build test data")

    for sentence_info in data:
        one_sentence = []
        for idx, i in enumerate(sentence_info):
            one_instance = []
            word_info = {}
            word_info["word"] = i[0]
            word_info["pos"] = i[1]
            word_info["sense_id"] = i[2]
            word_info["definition"] = i[3]
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

