
import json
from pathlib import Path
from CwnGraph import CwnImage
from typing import List, Dict, Optional
from tqdm.auto import tqdm
import gdown
import shutil
import logging

import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer

from .dwsd_model import DottedWSD
from .dwsd_dataset import DottedWsdDataset, DataCollatorForDottedWSD
from .dwsd_prediction import ExamplePrediction, InstancePredictions
from .dwsd_preproc import (
    Token,
    find_candidate_senses, 
    get_target_word,
    make_input_text
)

class DottedWsdTagger:
    def __init__(self, use_cuda=True):
        self.device = "cuda" if (torch.cuda.is_available() and use_cuda) \
                      else "cpu"

        self.cwn = CwnImage.latest()
        (self.tokenizer,
         self.model,
         self.gloss_dict) = self.load_model()

    def download_model(self):
        home_dir = Path("~/.CwnSenseTagger").expanduser()
        home_dir.mkdir(exist_ok=True, parents=True)
        model_path = home_dir / "dotted-wsd"
        if not model_path.exists():
            MODEL_GDRIVE_ID = "14Ea1KtIC7zBQ9lFh-vwbKLuiftpabIoo" 
            outfile = gdown.download(id=MODEL_GDRIVE_ID)            
            model_dst = home_dir / outfile
            shutil.move(outfile, str(model_dst))
            gdown.extractall(str(model_dst))
            model_dst.unlink()
        return model_path

    def load_model(self):
        model_path = self.download_model()
        logging.info("Loading Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        logging.info("Loading model")
        model = DottedWSD.from_pretrained(model_path)        
        # >> debug
        # model, tokenizer = None, None
        model.to(self.device)
        with open(Path(__file__).parent / "glossdict.json", "r", encoding="UTF-8") as fin:
            gloss_dict = json.load(fin)
        return tokenizer, model, gloss_dict

    def sense_tag(self, sentences: List[List[Token]], show_progress=False):
        if isinstance(sentences[0][0], str):
            raise ValueError("Expecting a list of sentence as input")

        tagged_outputs = []        
        for sent_x in sentences:
            out = self.sense_tag_per_sentence(sent_x, show_progress)
            tagged_outputs.append(out)
        return tagged_outputs

    def sense_tag_per_sentence(self, sentence: List[Token], show_progress=False):
        TokenId = int
        ExampleId: int
        exid_maps: Dict[TokenId, ExampleId] = {}

        # collect instances
        all_instances = []
        pred_maps: Dict[TokenId, ExamplePrediction] = {}

        for tok_i, (word, pos) in enumerate(sentence):
            input_text = make_input_text(tok_i, sentence)                
            example_id = len(exid_maps)
            exid_maps[tok_i] = example_id
            instances = self.generate_wsd_instances(input_text, example_id, target_pos=pos)
            if (len(instances)==0 and
                pos in ("Nb", "Nc")):
                rp_instances = self.generate_rp_instances(input_text, example_id)
                instances.extend(rp_instances)
            
            if len(instances)==1:
                pred_maps[tok_i] = ExamplePrediction(1., instances[0])
            else:
                all_instances.extend(instances)
        
        # model inference
        logits = self.predict(all_instances, show_progress)
        predictions, _ = self.decode_examples(all_instances, logits)
        pred_maps.update(predictions)

        # prepare output
        out = []
        for tok_i, tok in enumerate(sentence):
            if tok_i in pred_maps:
                prediction = pred_maps[tok_i].prediction()
            else:
                prediction = ""
            out.append((*tok, prediction))
        return out
    
    def wsd_tag(self, input_text, hint: Optional[str]=None):
        instances = self.generate_wsd_instances(input_text, 1, hint)
        logits = self.predict(instances)
        predictions, by_example = self.decode_examples(instances, logits)

        return predictions[1], by_example[1]
    
    def rp_tag(self, input_text, hint: Optional[str]=None):
        instances = self.generate_rp_instances(input_text, 1, hint)
        logits = self.predict(instances)
        predictions, by_example = self.decode_examples(instances, logits)
        
        return predictions[1], by_example[1]        
        
    def dotted_tag(self, input_text, hint: Optional[str]=None):
        if hint is None:
            instances = self.generate_wsd_instances(input_text, 1, hint)
            if not instances:
                instances = self.generate_rp_instances(input_text, 1, hint)            
        elif "*" in hint:
            instances = self.generate_rp_instances(input_text, 1, hint)
        else: 
            instances = self.generate_wsd_instances(input_text, 1, hint)
        
        logits = self.predict(instances)
        predictions, by_example = self.decode_examples(instances, logits)

        return predictions[1], by_example[1]
        

    def generate_wsd_instances(self, input_text: str, ex_id: int, target_pos=None):

        # check whether target_word is in input_sentence
        target_word = get_target_word(input_text)

        # find candidate senses given the word and pos
        candid_senses = find_candidate_senses(self.cwn, target_word, target_pos)

        # generate WSD instances
        instances = []
        for sense_x in candid_senses:
            avail_examples = [x 
                        for x in sense_x.all_examples()
                        if x.strip()]
            instance = {
                "example_id": ex_id,
                "example_type": "wsd",
                "target_word": target_word,
                "probe": input_text,
                "sense_id": sense_x.id,
                "target_pos": target_pos,
                "cwn_pos": sense_x.pos,
                "simplified_pos": target_pos,
                "sense_def": sense_x.definition,
                "sense_refex": avail_examples[0],
            }
            instances.append(instance)

        return instances

    def generate_rp_instances(self, input_text: str, ex_id: int, rp_type=None):
        # check whether target_word is in input_sentence
        target_word = get_target_word(input_text)

        # get candidate dotted-types
        if rp_type:
            candid_types = [(x, self.gloss_dict[x])
                            for x in rp_type.split("*")
                            if x in self.gloss_dict]
        else:
            candid_types = list(self.gloss_dict.items())

        instances = []
        for type_en, gloss in candid_types:
            instance = {
                "example_id": ex_id,
                "example_type": "rp",   
                "target_word": target_word,             
                "probe": input_text,
                "typeclass_en": type_en,
                "typeclass_zh": gloss["zh_trans"],
                "typeclass_gloss_zh": gloss["zh_gloss"],
            }
            instances.append(instance)

        return instances

    def predict(self, instances, show_progress=False):
        model = self.model
        dataset = DottedWsdDataset(instances)
        data_collator = DataCollatorForDottedWSD(self.tokenizer)
        loader = DataLoader(dataset, 
                    batch_size=16,
                    shuffle=False,
                    collate_fn=data_collator)                    
        
        if show_progress:
            loader = tqdm(iter(loader))

        all_logits = []
        with torch.no_grad():
            for batch in loader:
                batch.to(self.device)
                out = model(**batch)
                logits = out.logits.detach().tolist()
                all_logits.extend(logits)
        assert len(all_logits) == len(instances)
        return np.array(all_logits)
            

    def decode_examples(self, instances, logits):        
        # groupby example_ids
        by_examples = {}
        for inst_x, logit_x in zip(instances, logits):
            ex_id = inst_x["example_id"]            
            ex_item = by_examples.setdefault(ex_id, {})
            ex_item.setdefault("logits", []).append(logit_x)            
            ex_item.setdefault("instances", []).append(inst_x)            
        
        ##  compute by-example metric
        ExampleID = int          
        example_pred_map: Dict[ExampleID, ExamplePrediction] = {}
        inst_preds_map: Dict[ExampleID, InstancePredictions] = {}
        for ex_id, ex_item in by_examples.items():        
            ex_logits = ex_item["logits"]          
            ex_insts = ex_item["instances"]
            ex_probs = np.exp(ex_logits) / np.exp(ex_logits).sum()
            inst_preds = InstancePredictions(ex_probs, ex_insts)            
            
            inst_preds_map[ex_id] = inst_preds
            example_pred_map[ex_id] = inst_preds.top()

        return example_pred_map, inst_preds_map