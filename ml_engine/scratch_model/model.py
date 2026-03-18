import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from ml_engine.scratch_model.embeddings import TokenEmbedder

class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across all timesteps in a sequence,
    which is essential for recurrent networks.
    """
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        # x shape: (batch, seq_len, feature_dim)
        mask = torch.empty(x.size(0), 1, x.size(2), device=x.device).bernoulli_(1 - self.p)
        return mask * x / (1 - self.p)

class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        num_labels: int,
        label2idx: Dict[str, int],
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.num_labels = num_labels
        self.pad_idx = pad_idx

        # 1. ENCODER COMPONENTS
        self.embedder = TokenEmbedder(
            vocab_size=vocab_size, 
            char_vocab_size=char_vocab_size
        )
        
        self.var_dropout = VariationalDropout(p=dropout)
        
        # BiLSTM Encoder
        self.lstm = nn.LSTM(
            input_size=320,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Emission Projection Layer
        self.hidden2tag = nn.Linear(hidden_size * 2, num_labels)

        # 2. CRF COMPONENTS
        # transitions[i, j] is the score of transitioning FROM label j TO label i
        self.transitions = nn.Parameter(torch.empty(num_labels, num_labels))
        self.start_transitions = nn.Parameter(torch.empty(num_labels))
        self.end_transitions = nn.Parameter(torch.empty(num_labels))

        self._initialize_weights(label2idx)

    def _initialize_weights(self, label2idx: Dict[str, int]):
        # CRF Initialization
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        # Enforce Impossible CRF Transitions
        with torch.no_grad():
            # PAD cannot transition to/from any label
            self.transitions[self.pad_idx, :] = -10000.0
            self.transitions[:, self.pad_idx] = -10000.0
            self.start_transitions[self.pad_idx] = -10000.0
            self.end_transitions[self.pad_idx] = -10000.0

            for next_label, next_idx in label2idx.items():
                for prev_label, prev_idx in label2idx.items():
                    if next_label == "<PAD>" or prev_label == "<PAD>":
                        continue
                    
                    if next_label.startswith("I-"):
                        next_type = next_label[2:]
                        if prev_label == "O":
                            self.transitions[next_idx, prev_idx] = -10000.0
                        elif prev_label != "O":
                            prev_type = prev_label[2:]
                            if prev_type != next_type:
                                self.transitions[next_idx, prev_idx] = -10000.0

        # Encoder Initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)

    def _forward_algorithm(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_labels = emissions.shape
        
        # Initialize with start transitions + emissions of first token
        score = self.start_transitions + emissions[:, 0, :]  # Shape: (batch, num_labels)
        
        for t in range(1, seq_len):
            # Broadcast score: (batch, num_labels_prev, 1)
            broadcast_score = score.unsqueeze(2)
            
            # Broadcast emissions: (batch, 1, num_labels_next)
            broadcast_emissions = emissions[:, t, :].unsqueeze(1)
            
            # Transpose transitions to shape: (num_labels_prev, num_labels_next)
            trans = self.transitions.transpose(0, 1)
            
            # Combine scores -> Shape: (batch, num_labels_prev, num_labels_next)
            next_score = broadcast_score + trans + broadcast_emissions
            
            # LogSumExp to marginalize out the previous label dimension
            next_score = torch.logsumexp(next_score, dim=1)  # Shape: (batch, num_labels_next)
            
            # Update score if token is valid, else keep previous score
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
            
        # Add end transitions
        score = score + self.end_transitions
        
        return torch.logsumexp(score, dim=1)  # Shape: (batch,)

    def _score_sentence(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        batch_idx = torch.arange(batch_size, device=emissions.device)
        
        # Start transitions
        score = self.start_transitions[tags[:, 0]]
        score += emissions[batch_idx, 0, tags[:, 0]]
        
        for t in range(1, seq_len):
            prev_tags = tags[:, t-1]
            curr_tags = tags[:, t]
            
            # Transition from prev to curr
            trans_score = self.transitions[curr_tags, prev_tags]
            emit_score = emissions[batch_idx, t, curr_tags]
            
            # Only accumulate if it's a valid token
            step_score = trans_score + emit_score
            score += torch.where(mask[:, t], step_score, torch.zeros_like(score))
            
        # Add end transitions corresponding to the last valid token of each sequence
        seq_lengths = mask.sum(dim=1).long() - 1
        last_tags = tags[batch_idx, seq_lengths]
        score += self.end_transitions[last_tags]
        
        return score  # Shape: (batch,)

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        forward_score = self._forward_algorithm(emissions, mask)
        gold_score = self._score_sentence(emissions, tags, mask)
        loss = (forward_score - gold_score).mean()
        return loss

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> Tuple[List[List[int]], List[float]]:
        batch_size, seq_len, num_labels = emissions.shape
        
        viterbi_score = self.start_transitions + emissions[:, 0, :]
        backpointers = []
        
        for t in range(1, seq_len):
            broadcast_score = viterbi_score.unsqueeze(2)  # (batch, num_labels_prev, 1)
            trans = self.transitions.transpose(0, 1)      # (num_labels_prev, num_labels_next)
            
            # (batch, num_labels_prev, num_labels_next)
            next_score = broadcast_score + trans + emissions[:, t, :].unsqueeze(1)
            
            # Find best previous tag paths
            best_score, best_tag = next_score.max(dim=1)  # (batch, num_labels_next)
            
            viterbi_score = torch.where(mask[:, t].unsqueeze(1), best_score, viterbi_score)
            backpointers.append(best_tag)
            
        viterbi_score += self.end_transitions
        
        # Backtracking
        seq_lengths = mask.sum(dim=1).long()
        best_paths = []
        best_scores = []
        
        for b in range(batch_size):
            length = seq_lengths[b].item()
            best_final_tag = viterbi_score[b].argmax().item()
            best_scores.append(viterbi_score[b, best_final_tag].item())
            
            path = [best_final_tag]
            curr_best_tag = best_final_tag
            
            # Backtrack from (length-2) down to 0
            for t in range(length - 2, -1, -1):
                curr_best_tag = backpointers[t][b, curr_best_tag].item()
                path.insert(0, curr_best_tag)
                
            best_paths.append(path)
            
        return best_paths, best_scores

    def forward(self, word_ids: torch.Tensor, char_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Step 1: Embeddings
        embeddings = self.embedder(word_ids, char_ids)
        embeddings = self.var_dropout(embeddings)
        
        # Step 2: BiLSTM Encoder
        # PyTorch packs natively if sequences are varying, but standard padded handling works too.
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.var_dropout(lstm_out)
        
        # Step 3: Emission Projection Layer
        emissions = self.hidden2tag(lstm_out)
        
        return emissions

    def predict(self, word_ids: torch.Tensor, char_ids: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        self.eval()
        with torch.no_grad():
            emissions = self.forward(word_ids, char_ids, mask)
            tag_sequences, _ = self._viterbi_decode(emissions, mask)
        return tag_sequences

    def count_parameters(self) -> int:
        total = 0
        print(f"\n{'='*40}\nMODEL PARAMETERS\n{'='*40}")
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                print(f"{name:<30}: {num_params:,}")
                total += num_params
        print(f"{'='*40}\nTotal Trainable Parameters: {total:,}\n{'='*40}\n")
        return total


if __name__ == "__main__":
    # 1. Create dummy label2idx with exactly 21 labels
    entity_types = [
        "PERSON", "EMAIL", "PHONE", "SSN", "API_KEY", 
        "PASSWORD", "CREDIT_CARD", "IP_ADDRESS", 
        "ORG_INTERNAL", "PROPRIETARY_CODE"
    ]
    
    dummy_label2idx = {"<PAD>": 0, "O": 1}
    idx = 2
    for ent in entity_types:
        dummy_label2idx[f"B-{ent}"] = idx
        idx += 1
        # To maintain exactly 21 total labels, drop one I- tag
        if idx < 21:  
            dummy_label2idx[f"I-{ent}"] = idx
            idx += 1
            
    assert len(dummy_label2idx) == 21, f"Expected 21 labels, got {len(dummy_label2idx)}"

    # 2. Instantiate Model
    model = BiLSTMCRF(
        vocab_size=500,
        char_vocab_size=100,
        num_labels=21,
        label2idx=dummy_label2idx
    )

    # 3. Create dummy batch data
    batch, seq_len, max_char_len = 4, 15, 20
    word_ids = torch.randint(0, 500, (batch, seq_len))
    char_ids = torch.randint(0, 100, (batch, seq_len, max_char_len))
    tags = torch.randint(0, 21, (batch, seq_len))
    
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    mask[0, 10:] = False
    mask[2, 12:] = False

    # 4. Run Forward Pass
    emissions = model(word_ids, char_ids, mask)
    print(f"Emissions Shape     : {emissions.shape}")
    assert emissions.shape == (batch, seq_len, 21), "Emissions shape mismatch!"

    # 5. Run NLL Loss
    loss = model.neg_log_likelihood(emissions, tags, mask)
    print(f"NLL Loss Value      : {loss.item():.4f}")
    assert loss.dim() == 0, "Loss must be a scalar tensor!"

    # 6. Run Viterbi Predict
    predictions = model.predict(word_ids, char_ids, mask)
    print(f"Predicted Sequences : {predictions}")
    
    # Assert variable sequence lengths match valid tokens in mask
    expected_lengths = mask.sum(dim=1).tolist()
    pred_lengths = [len(seq) for seq in predictions]
    assert expected_lengths == pred_lengths, f"Expected lengths {expected_lengths}, got {pred_lengths}"

    # 7. Count Parameters
    model.count_parameters()
    
    print("All tests passed!")