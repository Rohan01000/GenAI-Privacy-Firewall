import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, texts, tags, vocab, tag2idx, max_len=50):
        self.texts = texts
        self.tags = tags
        self.vocab = vocab
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx].split()
        tags = self.tags[idx]

        # Truncate
        words = words[:self.max_len]
        tags = tags[:self.max_len]

        # Convert to indices
        word_indices = [self.vocab.get(w.lower(), self.vocab["<UNK>"]) for w in words]
        tag_indices = [self.tag2idx[t] for t in tags]

        # Pad
        pad_len = self.max_len - len(word_indices)
        word_indices.extend([self.vocab["<PAD>"]] * pad_len)
        tag_indices.extend([self.tag2idx["O"]] * pad_len)
        
        mask = [1] * len(words) + [0] * pad_len

        return {
            "input_ids": torch.tensor(word_indices, dtype=torch.long),
            "labels": torch.tensor(tag_indices, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool)
        }