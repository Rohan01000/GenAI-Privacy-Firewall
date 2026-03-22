"""
bilstm_crf.py  —  BiLSTM-CRF model for PII Named-Entity Recognition
=====================================================================
Architecture
  ┌──────────────┐
  │ Token Embeds │  word-level embeddings  (vocab × embed_dim)
  └──────┬───────┘
         │  concat
  ┌──────┴───────┐
  │  Char CNN    │  character-level features per token
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │   BiLSTM     │  bidirectional LSTM with dropout
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │  Linear/CRF  │  emit tag scores → CRF decoding
  └──────────────┘
"""

import torch
import torch.nn as nn

# ── CRF layer ──────────────────────────────────────────────────────────────────
class CRF(nn.Module):
    """
    Conditional Random Field layer.
    Supports forward (Viterbi) and neg-log-likelihood loss.
    """
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        # Transition matrix: transitions[i][j] = score of going from tag i → tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # Special START / STOP tags
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions   = nn.Parameter(torch.randn(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _log_partition(self, emissions, mask):
        """Compute log partition function (forward algorithm)."""
        seq_len, batch, num_tags = emissions.shape
        # shape: (batch, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_len):
            # (batch, num_tags, 1) + transitions (num_tags, num_tags) + (batch, 1, num_tags)
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score = torch.logsumexp(next_score, dim=1)
            # Apply mask
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)  # (batch,)

    def _score_sentence(self, emissions, tags, mask):
        """Compute score of a given tag sequence."""
        seq_len, batch, _ = emissions.shape
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch), tags[0]]

        for i in range(1, seq_len):
            active = mask[i]
            score += self.transitions[tags[i - 1], tags[i]] * active
            score += emissions[i, torch.arange(batch), tags[i]] * active

        # End transition for last *real* token
        last_tag_idx = mask.long().sum(0) - 1
        last_tags = tags[last_tag_idx, torch.arange(batch)]
        score += self.end_transitions[last_tags]
        return score  # (batch,)

    def neg_log_likelihood(self, emissions, tags, mask):
        """Negative log-likelihood loss (averaged over batch)."""
        # emissions: (seq_len, batch, num_tags)
        # tags:      (seq_len, batch)
        # mask:      (seq_len, batch) — 1 for real tokens
        log_Z = self._log_partition(emissions, mask)
        score = self._score_sentence(emissions, tags, mask)
        return (log_Z - score).mean()

    def decode(self, emissions, mask):
        """Viterbi decoding. Returns list of tag-id lists."""
        seq_len, batch, num_tags = emissions.shape
        viterbi_score = self.start_transitions + emissions[0]
        viterbi_path  = []

        for i in range(1, seq_len):
            broadcast = viterbi_score.unsqueeze(2)
            trans = broadcast + self.transitions
            best_scores, best_tags = trans.max(dim=1)
            next_score = best_scores + emissions[i]
            active = mask[i].unsqueeze(1)
            viterbi_score = torch.where(active, next_score, viterbi_score)
            viterbi_path.append(best_tags)

        viterbi_score += self.end_transitions
        best_last = viterbi_score.argmax(dim=1)  # (batch,)

        # Backtrack
        lengths = mask.long().sum(0)  # (batch,)
        best_paths = []
        for b in range(batch):
            L = lengths[b].item()
            path = [best_last[b].item()]
            for step in reversed(viterbi_path[:L - 1]):
                path.append(step[b][path[-1]].item())
            path.reverse()
            best_paths.append(path)
        return best_paths


# ── Character-level CNN ────────────────────────────────────────────────────────
class CharCNN(nn.Module):
    def __init__(self, char_vocab_size: int, char_embed_dim: int = 30, out_channels: int = 50):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_embed_dim, out_channels, kernel_size=3, padding=1)
        self.out_channels = out_channels

    def forward(self, chars):
        # chars: (batch, seq_len, max_word_len)
        B, S, W = chars.shape
        chars_flat = chars.view(B * S, W)
        emb = self.embedding(chars_flat)           # (B*S, W, char_embed_dim)
        emb = emb.permute(0, 2, 1)                # (B*S, char_embed_dim, W)
        conv_out = torch.relu(self.conv(emb))      # (B*S, out_channels, W)
        pooled = conv_out.max(dim=2).values        # (B*S, out_channels)
        return pooled.view(B, S, self.out_channels)


# ── Main model ─────────────────────────────────────────────────────────────────
class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size:      int,
        char_vocab_size: int,
        num_tags:        int,
        word_embed_dim:  int = 100,
        char_embed_dim:  int = 30,
        char_cnn_out:    int = 50,
        hidden_size:     int = 256,
        num_lstm_layers: int = 2,
        dropout:         float = 0.3,
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        self.char_cnn = CharCNN(char_vocab_size, char_embed_dim, char_cnn_out)

        lstm_input_dim = word_embed_dim + char_cnn_out
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_size * 2, num_tags)
        self.crf = CRF(num_tags)

    def _get_emissions(self, words, chars):
        # words:  (batch, seq_len)
        # chars:  (batch, seq_len, max_word_len)
        word_emb = self.dropout(self.word_embedding(words))   # (B, S, word_embed_dim)
        char_feat = self.char_cnn(chars)                       # (B, S, char_cnn_out)
        combined = torch.cat([word_emb, char_feat], dim=2)     # (B, S, combined)
        lstm_out, _ = self.lstm(combined)                      # (B, S, hidden*2)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)                  # (B, S, num_tags)
        return emissions.permute(1, 0, 2)                      # (S, B, num_tags)

    def loss(self, words, chars, tags, mask):
        emissions = self._get_emissions(words, chars)
        tags_t = tags.permute(1, 0)    # (S, B)
        mask_t = mask.permute(1, 0).bool()
        return self.crf.neg_log_likelihood(emissions, tags_t, mask_t)

    def predict(self, words, chars, mask):
        with torch.no_grad():
            emissions = self._get_emissions(words, chars)
            mask_t = mask.permute(1, 0).bool()
            return self.crf.decode(emissions, mask_t)
