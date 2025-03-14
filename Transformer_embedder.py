import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformerEmbedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers):
        super(SimpleTransformerEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, token_indices):
        embedded = self.embedding(token_indices)
        transformed = self.transformer_encoder(embedded)
        pooled_embedding = transformed.mean(dim=1)
        return F.normalize(pooled_embedding, p=2, dim=1)

# Example usage:
vocab_size = 30522  # typical vocabulary size
embedding_dim = 256
num_heads = 8
num_layers = 2

model = SimpleTransformerEmbedder(vocab_size, embedding_dim, num_heads, num_layers)

# Sample tokenized input (batch_size=1, sequence_length=5)
sample_input = torch.tensor([[101, 2054, 2003, 1996, 102]])  # Example token IDs

# Generate embedding
embedding_output = model(sample_input)
print(embedding_output)
