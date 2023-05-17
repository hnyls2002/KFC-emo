from torch import nn
from transformers import BertModel

class BertSentimentAnalysis(nn.Module):
    def __init__(self, config, num_labels=28, inference_head=None):
        super(BertSentimentAnalysis, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if inference_head is None:
            self.inference_head = nn.Linear(config.hidden_size, num_labels)
        else:
            self.inference_head = inference_head

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.inference_head(pooled_output)
        return logits

