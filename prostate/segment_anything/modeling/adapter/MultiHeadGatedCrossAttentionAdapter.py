import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadGatedCrossAttentionAdapter(nn.Module):
    def __init__(self, input_dim, num_heads=8, dropout=0.1, gate_mlp_ratio=0.25):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.proj = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

        # Gating mechanism
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, int(input_dim * gate_mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(input_dim * gate_mlp_ratio), input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, H, W, C) or (B, N, C) where N = H*W
        # Assuming x is flattened to (B, N, C) before passing to the adapter if needed by ImageEncoderViT
        # If x is (B, H, W, C), reshape to (B, H*W, C) for attention
        B, _, C = x.shape
        # Assume N is the sequence length for attention
        N = x.shape[1]


        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # q, k, v shape: (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)

        # Compute gate
        gate = self.gate_mlp(x)

        # Apply gate and add residual connection
        output = x + gate * x_attn

        return output