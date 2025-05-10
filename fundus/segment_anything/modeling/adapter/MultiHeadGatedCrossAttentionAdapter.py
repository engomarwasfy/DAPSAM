import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadGatedCrossAttentionAdapter(nn.Module):
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        self.attention_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_dropout = nn.Dropout(dropout)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, H*W, C) or (B, seq_len, C)

        B, N, C = x.shape

        # Linear projections
        q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)    # (B, num_heads, N, head_dim)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)    # (B, num_heads, N, head_dim)

        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.head_dim**-0.5  # (B, num_heads, N, N)

        # Softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        # Compute attention output
        attention_output = torch.matmul(attention_probs, v)  # (B, num_heads, N, head_dim)

        # Concatenate heads and apply final linear projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, N, C) # (B, N, C)
        attention_output = self.proj(attention_output)
        attention_output = self.proj_dropout(attention_output)

        # Compute gate values
        gate_values = self.gate(x)

        # Apply gating and add residual connection
        # Here we apply the gate to the attention output and add it to the input
        output = x + gate_values * attention_output

        return output