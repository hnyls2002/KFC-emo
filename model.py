from torch import nn


class BertSentimentAnalysis(nn.Module):
    def __init__(self, pretrained_model, hidden_size, dropout_prob, num_labels, dense_num=1):
        super(BertSentimentAnalysis, self).__init__()
        self.num_labels = num_labels
        self.bert = pretrained_model.bert
        self.dropout = nn.Dropout(dropout_prob)

        if dense_num == 1:
            self.inference_head = nn.Linear(hidden_size, num_labels)
        else:
            self.inference_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, num_labels)
            )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.inference_head(pooled_output)
        return logits
