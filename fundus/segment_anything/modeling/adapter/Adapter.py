import torch
import torch.nn as nn

from typing import Type

from segment_anything.modeling.adapter.Filter import ChannelFilter


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.35, act_layer=nn.GELU, skip_connect=True, dropout_rate=0.1):
        super().__init__()
        self.channel_filter = ChannelFilter()

        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, embedding_feature):#
        x = x + embedding_feature.permute(0, 2, 3, 1)
        x = self.channel_filter(x)

        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.dropout(xs)  # Add dropout after activation
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + self.residual_scale * xs
        else:
            x = xs
        return x