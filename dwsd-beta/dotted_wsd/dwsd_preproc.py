import re
from functools import lru_cache
from typing import Tuple, List
from .cwn_pos_map import cwn_pos_map

Word = str
Pos = str
Token = Tuple[Word, Pos]

def is_pos_matched(in_pos, ref_pos):
    if in_pos in ref_pos:
        return True
    else:
        return cwn_pos_map.get(in_pos, 'OTHER')==cwn_pos_map.get(ref_pos, 'OTHER')

def simplify_pos(pos):
    try:
        return cwn_pos_map[pos]
    except KeyError as e:
        poses = [x.strip(' ') for x in pos.split(',') if x.lower() != 'nom']
        if poses:
            return cwn_pos_map.get(poses[0], 'OTHER')
        return 'OTHER'

def make_input_text(tok_idx, sentence: List[Token]):
    words = [x[0] for x in sentence]
    words[tok_idx] = f"<{words[tok_idx]}>"
    return "".join(words)

def get_target_word(input_text:str) -> str:
    target_word = re.findall(r"<(.+?)>", input_text)    
    if len(target_word) < 1:
        raise ValueError("There is no marked target in input_sentence")
    target_word = target_word[0]
    return target_word

@lru_cache(maxsize=1000)
def find_candidate_senses(cwn, target_word, target_pos):
    senses = cwn.find_all_senses(target_word)
    if target_pos:        
        candid_senses = []
        for sense_x in senses:
            avail_sentences = [x 
                        for x in sense_x.all_examples()
                        if x.strip()]
            if (is_pos_matched(target_pos, sense_x.pos) and
                len(avail_sentences) > 0):
                candid_senses.append(sense_x)        
    else:
        candid_senses = [x for x in senses 
                         if len([ex 
                                 for ex in x.all_examples() 
                                 if ex.strip()])>0]        
    return candid_senses