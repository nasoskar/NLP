import torch
import torch.nn as nn
from transformers import AutoModel

class FINBERT(nn.Module):
    def __init__(self, dropout):
        super(FINBERT, self).__init__()
        self.bert = AutoModel.from_pretrained("ProsusAI/finbert") #load pre-trained FinBERT model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = output.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits