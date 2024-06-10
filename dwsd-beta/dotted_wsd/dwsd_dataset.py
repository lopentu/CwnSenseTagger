
import torch
from torch.utils.data import Dataset

class DottedWsdDataset(Dataset):
    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst_x = self.instances[idx]
        out = {
            "example_id": inst_x["example_id"],
            "context": inst_x["probe"],
            "label": None,
        }

        if inst_x["example_type"] == "wsd":
            out["candidate"] = '{},{},{}'.format(
                inst_x["target_word"],
                inst_x["sense_def"],
                inst_x["sense_refex"])
                
        elif inst_x["example_type"] == "rp":
            out["candidate"] = '{},{},{}'.format(
                inst_x["target_word"], 
                inst_x["typeclass_zh"], 
                inst_x["typeclass_gloss_zh"])
        else:
            raise ValueError("Unknonw ex_type: " +  inst_x["ex_type"])

        return out

class DataCollatorForDottedWSD:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding = True
        self.max_len = 320
        self.pad_to_multiple_of = 16
    
    def __call__(self, examples):        
        flat_contexts = [x['context'] for x in examples]
        flat_candidates = [x['candidate'] for x in examples]        
        
        batch = self.tokenizer(
            flat_contexts, flat_candidates, 
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_len,   
            truncation=True,
            return_tensors='pt')
        batch["example_ids"] = torch.tensor([x["example_id"] for x in examples], dtype=torch.int32)

        return batch