import torch
import torch.nn as nn

class AttentionAdapter(nn.Module):
    def __init__(self, d_features, n_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_features, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, embedding_feature):
        x = x + embedding_feature.permute(0, 2, 3, 1)  # Add embedding feature
        batch_size, seq_len, d_features = x.shape
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        attn_output = self.dropout(attn_output)
        x = x + attn_output  # Skip connection
        return x