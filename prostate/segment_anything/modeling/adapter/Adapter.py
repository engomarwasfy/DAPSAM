
import torch
import torch.nn as nn

from typing import Type

from segment_anything.modeling.adapter.Filter import ChannelFilter


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=1.0, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.channel_filter = ChannelFilter()

        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        # Five-layer MLP
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_hidden_features)
        self.D_fc3 = nn.Linear(D_hidden_features, D_hidden_features)
        self.D_fc4 = nn.Linear(D_hidden_features, D_hidden_features)
        self.D_fc5 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x, embedding_feature):#
        # Note: The line below was causing a RuntimeError due to dimension mismatch.
        # If embedding_feature is intended to be used, its integration should be handled differently.
        x = x + embedding_feature.permute(0, 2, 3, 1)
        x_filtered = self.channel_filter(x)

        # Five-layer MLP calculation
        mlp_output = self.D_fc5(self.act(self.D_fc4(self.act(self.D_fc3(self.act(self.D_fc2(self.act(self.D_fc1(x_filtered)))))))))
        if self.skip_connect:
            output = x_filtered + mlp_output  # Skip connection with filtered features
        else:
            output = mlp_output
        return output