
import torch
import torch.nn as nn

from typing import Type

from segment_anything.modeling.adapter.Filter import ChannelFilter


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True, num_heads=8):
        super().__init__()
        self.channel_filter = ChannelFilter()

        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_features = D_features
        self.num_heads = num_heads
        head_dim = D_features // num_heads
        self.scale = head_dim ** -0.5
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

        # Attention layers
        self.q_proj = nn.Linear(D_features, D_features)
        self.k_proj = nn.Linear(D_features, D_features)
        self.v_proj = nn.Linear(D_features, D_features)
        self.out_proj = nn.Linear(D_features, D_features)

    def forward(self, x, embedding_feature):#
        x = x + embedding_feature.permute(0, 2, 3, 1)
        x_filtered = self.channel_filter(x)

        # Multi-head Self Attention
        B, N, C = x_filtered.shape # Assuming input is [B, N, D_features]
        q = self.q_proj(x_filtered).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x_filtered).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x_filtered).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_output = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        attn_output = self.out_proj(attn_output)

        # MLP path
        mlp_output = self.D_fc2(self.act(self.D_fc1(x_filtered)))

        # Combine and skip connect
        combined_output = attn_output + mlp_output
        if self.skip_connect:
            output = x + combined_output
        else:
            output = combined_output
        return output