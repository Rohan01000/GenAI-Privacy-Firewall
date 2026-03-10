import torch
import torch.nn as nn
from .embeddings import CustomEmbeddings

class ScratchNERModel(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embeddings = CustomEmbeddings(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        embeds = self.embeddings(input_ids)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits