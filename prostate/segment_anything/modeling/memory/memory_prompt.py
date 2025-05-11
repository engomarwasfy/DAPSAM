import math

import numpy as np
import torch
from numpy import random
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Type

device = "cuda" if torch.cuda.is_available() else "cpu"
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim=256, fea_dim=256):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if len(input.shape) == 1:
            input = torch.unsqueeze(input, 0)
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return output # output


class EnhancedMemoryUnit(nn.Module):
    def __init__(self, num_prototypes=8, mem_dim=256, fea_dim=256):
        super(EnhancedMemoryUnit, self).__init__()
        self.num_prototypes = num_prototypes
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        # Changed dimensions to support multiple prototypes
        self.weight = nn.Parameter(torch.Tensor(self.num_prototypes, self.mem_dim, self.fea_dim))  # P x M x C
        self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2)) # Use fea_dim for initialization
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if len(input.shape) == 1:
            input = torch.unsqueeze(input, 0)
        batch_size = input.shape[0]

        # Calculate attention weights using einsum
        # input shape: (batch_size, fea_dim)
        # self.weight shape: (num_prototypes, mem_dim, fea_dim)
        # att_weight shape: (batch_size, num_prototypes, mem_dim)
        att_weight = torch.einsum('bc,pmc->bpm', input, self.weight)
        att_weight = F.softmax(att_weight, dim=2)  # B x P x M

        output = torch.einsum('bpm,pmc->bpc', att_weight, self.weight) # B x P x C
        return output

class PrototypePromptGenerate(nn.Module):
    def __init__(self, mem_dim=256, embed_dim=256, image_embedding_size=24):
        num_prototypes = 4  # Define the number of prototypes
        super(PrototypePromptGenerate, self).__init__()
        self.memory_bank = EnhancedMemoryUnit(num_prototypes=num_prototypes, mem_dim=mem_dim, fea_dim=embed_dim) # Use EnhancedMemoryUnit
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.image_embedding_size = (image_embedding_size, image_embedding_size)

        self.fuse_conv = nn.Conv2d(512 + num_prototypes * embed_dim, 256, 1) # Adjust input channels for multiple prototypes and cosine similarities
    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, feature):
        N, C, H, W = feature.shape
        feature_proto_avg = F.avg_pool2d(input=feature, kernel_size=feature.shape[-2:])#b x c x 1 x 1
        feature_proto_max = F.max_pool2d(input=feature, kernel_size=feature.shape[-2:])#b x c x 1 x 1
        feature_proto = (feature_proto_avg + feature_proto_max)
        feature_proto = feature_proto.squeeze()
        di_proto = self.memory_bank(feature_proto)# B x num_prototypes x C
        
        # Expand multiple refined prototypes to the spatial dimensions
        info_protos = di_proto.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, H, W) # B x num_prototypes x C x H x W
        
        # Calculate cosine similarity map for each prototype
        cos_sim_maps = F.cosine_similarity(info_protos, feature.unsqueeze(1), dim=2, eps=1e-7) # B x num_prototypes x H x W

        # Concatenate features, expanded prototypes, and cosine similarity maps
        concatenated_features = torch.cat([feature, info_protos.view(N, -1, H, W), cos_sim_maps], dim=1)

        prompt = self.fuse_conv(concatenated_features)

        sparse_embeddings = torch.empty((1, 0, C), device=device)
        return sparse_embeddings, prompt

