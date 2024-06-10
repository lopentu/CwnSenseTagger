import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from dataclasses import dataclass

@dataclass
class DottedWSDModelOutput:    
    loss: torch.Tensor
    logits: torch.Tensor
    example_ids: np.ndarray
    labels: np.ndarray

class DottedWSD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)        
        self.loss_func = nn.BCEWithLogitsLoss()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)        
        self.init_weights()
    
    def forward(self, input_ids, example_ids,
                attention_mask = None, 
                token_type_ids = None,                 
                labels = None,              
                output_attentions=None,
                output_hidden_states=None, **kwargs):

        outputs = self.bert(
            input_ids = input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,            
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, **kwargs
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).squeeze(1)   

        if labels is not None:
            loss = self.loss_func(logits, labels)
        else:
            loss = None

        return DottedWSDModelOutput(            
            loss=loss,
            logits=logits,            
            example_ids=example_ids,
            labels=labels
        )
