import torch

import torch.nn as nn

class SelfAttentionAdapter(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1, scale_factor=2):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # Get the shape of the input tensor
        batch_size, seq_len, input_dim = x.shape
        
        # Project the input to queries, keys and values
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape to handle the multiple attention heads
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2) # (batch_size, num_heads, seq_len, dim/num_heads)
        k = k.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2) # (batch_size, num_heads, seq_len, dim/num_heads)
        v = v.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2) # (batch_size, num_heads, seq_len, dim/num_heads)
        
        # Compute the attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5) # (batch_size, num_heads, seq_len, seq_len)
        
        # Normalize the scores with softmax and apply dropout
        attn_probs = torch.softmax(attn_scores, dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        attn_probs = self.dropout(attn_probs) # (batch_size, num_heads, seq_len, seq_len)
        
        # Compute the context with the attention probabilities and values
        context = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # (batch_size, seq_len, dim)
        output = self.out(context) # (batch_size, seq_len, dim)
        
        return output / self.scale_factor + x # residual connection