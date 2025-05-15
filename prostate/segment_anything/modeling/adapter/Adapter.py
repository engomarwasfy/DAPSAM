import torch
import torch.nn as nn

from typing import Type

from segment_anything.modeling.adapter.Filter import ChannelFilter


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True, num_layers=1):
        super().__init__()
        self.channel_filter = ChannelFilter()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        
        # Create 10 layers of MLP blocks
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D_features, D_hidden_features),
                act_layer(),
                nn.Linear(D_hidden_features, D_features)
            ) for _ in range(num_layers)
        ])

    def forward(self, x, embedding_feature):
        x = x + embedding_feature.permute(0, 2, 3, 1)
        x = self.channel_filter(x)

        # Apply all layers sequentially with skip connections
        for layer in self.layers:
            if self.skip_connect:
                x = x + layer(x)
            else:
                x = layer(x)
        
        return x