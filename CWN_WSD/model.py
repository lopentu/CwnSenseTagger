import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel

class WSDBertClassifer(BertPreTrainedModel):
    def __init__(self, config, pos=1):
        super(WSDBertClassifer, self).__init__(config)
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos))
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifer = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
    
    def forward(self, context, attention_mask = None, token_type_ids = None, labels = None):
        seq_out, pool_out = self.bert(context, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pool_out = self.dropout(pool_out)
        logits = self.classifer(pool_out)
        loss = 0
        if labels is not None:
            loss = self.loss_function(logits.squeeze(1), labels)
        logits = self.sigmoid(logits)
        return logits, loss
