import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        word_embed_dim: int = 128,
        char_embed_dim: int = 32,
        num_filters: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Internal flag used strictly for the test block to print shapes
        self._debug_shapes = False

        # 1. WORD-LEVEL EMBEDDINGS
        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_embed_dim,
            padding_idx=0
        )

        # 2. CHARACTER-LEVEL CNN EMBEDDINGS
        self.char_embeddings = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_embed_dim,
            padding_idx=0
        )

        # Three parallel Conv1d filters
        self.conv1d_2 = nn.Conv1d(
            in_channels=char_embed_dim,
            out_channels=num_filters,
            kernel_size=2
        )
        self.conv1d_3 = nn.Conv1d(
            in_channels=char_embed_dim,
            out_channels=num_filters,
            kernel_size=3
        )
        self.conv1d_4 = nn.Conv1d(
            in_channels=char_embed_dim,
            out_channels=num_filters,
            kernel_size=4
        )

        self.char_dropout = nn.Dropout(dropout)

        # 3. COMBINED REPRESENTATION
        # Word dim (128) + 3 * Conv1d filters (64 * 3 = 192) = 320
        combined_dim = word_embed_dim + (num_filters * 3)
        self.layer_norm = nn.LayerNorm(combined_dim)
        self.combined_dropout = nn.Dropout(dropout)

    def forward(self, word_ids, char_ids):
        batch_size, seq_len = word_ids.shape
        _, _, max_char_len = char_ids.shape

        # --- Process Word Embeddings ---
        word_embeds = self.word_embeddings(word_ids)
        if self._debug_shapes:
            print(f"after word embedding: {word_embeds.shape}")

        # --- Process Character CNN Embeddings ---
        # Reshape to (batch * seq_len, max_char_len)
        char_ids_reshaped = char_ids.view(batch_size * seq_len, max_char_len)
        
        # Embed characters: (batch * seq_len, max_char_len, 32)
        char_embeds = self.char_embeddings(char_ids_reshaped)
        if self._debug_shapes:
            print(f"after char embedding: {char_embeds.shape}")

        # Transpose for Conv1d: (batch * seq_len, 32, max_char_len)
        char_embeds_transposed = char_embeds.transpose(1, 2)

        # Apply each Conv1d + ReLU
        conv2_out = F.relu(self.conv1d_2(char_embeds_transposed))
        if self._debug_shapes:
            print(f"after Conv1d (kernel=2): {conv2_out.shape}")
            
        conv3_out = F.relu(self.conv1d_3(char_embeds_transposed))
        if self._debug_shapes:
            print(f"after Conv1d (kernel=3): {conv3_out.shape}")
            
        conv4_out = F.relu(self.conv1d_4(char_embeds_transposed))
        if self._debug_shapes:
            print(f"after Conv1d (kernel=4): {conv4_out.shape}")

        # Apply MaxPool1d over the time dimension and squeeze the last dimension
        pool2_out = F.max_pool1d(conv2_out, conv2_out.shape[2]).squeeze(2)
        if self._debug_shapes:
            print(f"after MaxPool (kernel=2): {pool2_out.shape}")
            
        pool3_out = F.max_pool1d(conv3_out, conv3_out.shape[2]).squeeze(2)
        if self._debug_shapes:
            print(f"after MaxPool (kernel=3): {pool3_out.shape}")
            
        pool4_out = F.max_pool1d(conv4_out, conv4_out.shape[2]).squeeze(2)
        if self._debug_shapes:
            print(f"after MaxPool (kernel=4): {pool4_out.shape}")

        # Concatenate the three pooled outputs: (batch * seq_len, 192)
        char_cnn_out = torch.cat([pool2_out, pool3_out, pool4_out], dim=1)
        if self._debug_shapes:
            print(f"after concatenation of three filters: {char_cnn_out.shape}")

        # Apply dropout to the 192-dim output
        char_cnn_out = self.char_dropout(char_cnn_out)

        # Reshape back to sequence format: (batch, seq_len, 192)
        char_cnn_out_reshaped = char_cnn_out.view(batch_size, seq_len, -1)
        if self._debug_shapes:
            print(f"after reshape back to (batch, seq_len, 192): {char_cnn_out_reshaped.shape}")

        # --- Combined Representation ---
        # Concatenate word embeddings and char CNN outputs along the last dimension
        combined = torch.cat([word_embeds, char_cnn_out_reshaped], dim=2)
        if self._debug_shapes:
            print(f"after final concatenation: {combined.shape}")

        # Apply LayerNorm
        normalized = self.layer_norm(combined)
        if self._debug_shapes:
            print(f"after LayerNorm: {normalized.shape}")

        # Apply Dropout
        final_output = self.combined_dropout(normalized)
        if self._debug_shapes:
            print(f"final output: {final_output.shape}")

        return final_output


if __name__ == "__main__":
    # Test Block
    batch = 4
    seq_len = 10
    max_char_len = 15
    
    # Create dummy tensors
    word_ids = torch.randint(0, 100, (batch, seq_len))
    char_ids = torch.randint(0, 50, (batch, seq_len, max_char_len))
    
    # Instantiate model
    model = TokenEmbedder(vocab_size=100, char_vocab_size=50)
    
    # Enable debug mode to print intermediate shapes
    model._debug_shapes = True
    
    print("--- Forward Pass Shape Trace ---")
    # Run forward pass
    output = model(word_ids, char_ids)
    print("--------------------------------")
    
    # Verify final shape
    assert output.shape == (batch, seq_len, 320), f"Expected shape (4, 10, 320), got {output.shape}"
    print("All shapes correct!")