import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, out_features, dropout):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.decoder = nn.Linear(hidden_size, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, (hidden, cell) = self.rnn(embedded)
        last_hidden = output[:, -1, :]   
        last_hidden = self.dropout(last_hidden)
        logits = self.decoder(last_hidden)
        
        return logits