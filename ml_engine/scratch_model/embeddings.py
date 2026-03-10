import torch
import torch.nn as nn

class CustomEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)