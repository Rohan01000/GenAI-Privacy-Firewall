"""
data_loader.py  —  Dataset utilities for BiLSTM-CRF training
==============================================================
Reads pii_ner_dataset.csv and provides:
  • Vocabulary / label mapping builders
  • PIIDataset  (torch Dataset)
  • collate_fn  (for DataLoader batching with padding)
"""

import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

PAD     = "<PAD>"
UNK     = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load raw sentences from CSV
# ─────────────────────────────────────────────────────────────────────────────
def load_sentences(csv_path: str) -> List[List[Tuple[str, str]]]:
    """
    Returns a list of sentences.
    Each sentence is a list of (word, label) tuples.
    """
    sentences: List[List[Tuple[str, str]]] = []
    current_id = None
    current_sentence: List[Tuple[str, str]] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["sentence_id"]
            if sid != current_id:
                if current_sentence:
                    sentences.append(current_sentence)
                current_sentence = []
                current_id = sid
            current_sentence.append((row["word"], row["label"]))

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build vocabularies
# ─────────────────────────────────────────────────────────────────────────────
def build_vocabs(sentences: List[List[Tuple[str, str]]]):
    word_freq: Dict[str, int] = defaultdict(int)
    char_freq: Dict[str, int] = defaultdict(int)
    label_set = set()

    for sentence in sentences:
        for word, label in sentence:
            word_freq[word.lower()] += 1
            for ch in word:
                char_freq[ch] += 1
            label_set.add(label)

    # Word vocab: keep words appearing ≥ 1 time
    word_vocab = {PAD: PAD_IDX, UNK: UNK_IDX}
    for w in sorted(word_freq):
        word_vocab[w] = len(word_vocab)

    # Char vocab
    char_vocab = {PAD: PAD_IDX, UNK: UNK_IDX}
    for c in sorted(char_freq):
        char_vocab[c] = len(char_vocab)

    # Label vocab  — keep O at index 0 for clarity
    sorted_labels = sorted(label_set - {"O"})
    label_vocab = {"O": 0}
    for lbl in sorted_labels:
        label_vocab[lbl] = len(label_vocab)

    return word_vocab, char_vocab, label_vocab


def save_vocabs(word_vocab, char_vocab, label_vocab, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "word_vocab.json"), "w") as f:
        json.dump(word_vocab, f, indent=2)
    with open(os.path.join(out_dir, "char_vocab.json"), "w") as f:
        json.dump(char_vocab, f, indent=2)
    with open(os.path.join(out_dir, "label_vocab.json"), "w") as f:
        json.dump(label_vocab, f, indent=2)
    print(f"✅  Saved vocabs to {out_dir}/")
    print(f"   Words: {len(word_vocab)}  Chars: {len(char_vocab)}  Labels: {len(label_vocab)}")


def load_vocabs(model_dir: str):
    with open(os.path.join(model_dir, "word_vocab.json"))  as f: word_vocab  = json.load(f)
    with open(os.path.join(model_dir, "char_vocab.json"))  as f: char_vocab  = json.load(f)
    with open(os.path.join(model_dir, "label_vocab.json")) as f: label_vocab = json.load(f)
    return word_vocab, char_vocab, label_vocab


# ─────────────────────────────────────────────────────────────────────────────
# 3. Encoding helpers
# ─────────────────────────────────────────────────────────────────────────────
MAX_WORD_LEN = 30

def encode_word(word: str, word_vocab: dict) -> int:
    return word_vocab.get(word.lower(), UNK_IDX)

def encode_chars(word: str, char_vocab: dict) -> List[int]:
    ids = [char_vocab.get(c, UNK_IDX) for c in word[:MAX_WORD_LEN]]
    ids += [PAD_IDX] * (MAX_WORD_LEN - len(ids))
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# 4. PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PIIDataset(Dataset):
    def __init__(self, sentences, word_vocab, char_vocab, label_vocab):
        self.samples = []
        for sentence in sentences:
            words  = [encode_word(w, word_vocab)          for w, _ in sentence]
            chars  = [encode_chars(w, char_vocab)         for w, _ in sentence]
            labels = [label_vocab.get(l, 0)               for _, l in sentence]
            self.samples.append((words, chars, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Collate function (pad variable-length sequences)
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    words_list, chars_list, labels_list = zip(*batch)
    max_len = max(len(w) for w in words_list)

    padded_words  = []
    padded_chars  = []
    padded_labels = []
    masks = []

    for words, chars, labels in zip(words_list, chars_list, labels_list):
        L = len(words)
        pad = max_len - L
        padded_words.append(words  + [PAD_IDX] * pad)
        padded_chars.append(chars  + [[PAD_IDX] * MAX_WORD_LEN] * pad)
        padded_labels.append(labels + [0] * pad)
        masks.append([1] * L + [0] * pad)

    return (
        torch.tensor(padded_words,  dtype=torch.long),    # (B, S)
        torch.tensor(padded_chars,  dtype=torch.long),    # (B, S, max_word_len)
        torch.tensor(padded_labels, dtype=torch.long),    # (B, S)
        torch.tensor(masks,         dtype=torch.float),   # (B, S)
    )
