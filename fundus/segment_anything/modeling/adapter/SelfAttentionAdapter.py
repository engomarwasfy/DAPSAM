import torch
import torch.nn as nn

class SelfAttentionAdapter(nn.Module):
    """Self attention adapter.
    """
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
        """
        Compute the self attention.

        Args:
            x (tensor): input tensor.
        Returns:
            tensor: output tensor.
        """
        batch_size, seq_len, _ = x.shape
        q = self.query(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out(context)

        return output / self.scale_factor + x