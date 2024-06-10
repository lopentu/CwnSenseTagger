from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class ExamplePrediction:
    prob: float
    instance: Dict[str, any]
    
    def __repr__(self):
        return ("<ExamplePrediction: {}>"
                .format(self.prediction()))
    
    @property
    def pred_class(self):
        inst = self.instance
        if inst.get("example_type") == "rp":
            return inst.get("typeclass_en", "--")            
        else:
            return inst.get("sense_id", "----")                            
        
    def prediction(self):
        inst = self.instance
        if inst.get("example_type") == "rp":
            return "[RP:{}] {} ({:.4f})".format(
                inst.get("typeclass_en", "--"),
                inst.get("typeclass_gloss_zh", "--"),
                self.prob
            )
        else:
            return "[{}] {} ({:.4f})".format(
                inst.get("sense_id", "----"),
                inst.get("sense_def", "----"),
                self.prob
            )

class InstancePredictions:
    def __init__(self, probs, instances):
        self.probs = probs
        self.ex_preds = []
        for prob_x, inst_x in zip(probs, instances):
            self.ex_preds.append(
                ExamplePrediction(prob_x, inst_x)
            )            
    
    def __repr__(self):
        return f"<InstancePredictions: {len(self.probs)} class(es)>"
    
    def __len__(self):
        return len(self.ex_preds)
    
    def predictions(self):
        preds = []
        for pred_x in self.ex_preds:
            preds.append(pred_x.prediction)
    
    def top(self) -> ExamplePrediction:
        return self.top_k()[0]
    
    def top_k(self, k=1) -> List[ExamplePrediction]:
        sorted_idxs = np.argsort(-self.probs)
        return [self.ex_preds[i] for i in sorted_idxs[:k]]
