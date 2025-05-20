import torch
import torch.nn as nn

from typing import Type

from segment_anything.modeling.adapter.Filter import ChannelFilter


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True, dropout_rate=0.1, num_attention_heads=8):
        super().__init__()
        self.channel_filter = ChannelFilter()

        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.norm = nn.LayerNorm(D_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn = nn.MultiheadAttention(D_features, num_attention_heads, batch_first=True)

    def forward(self, x, embedding_feature):#
        x = x + embedding_feature.permute(0, 2, 3, 1)
        x = self.channel_filter(x)

        # x is (BT, HW+1, D)
        # Apply self-attention
        B, H, W, D = x.shape
        attn_input = x.reshape(B, H*W, D)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        x = attn_output.reshape(B, H, W, D)

        # LayerNorm before MLP
        x = self.norm(x)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.dropout(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x