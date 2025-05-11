# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch.nn as nn

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from .modeling.memory.memory_prompt import PrototypePromptGenerate # Import PrototypePromptGenerate directly

import torch.nn as nn

def build_sam_vit_h(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None,):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


def build_sam_vit_b(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

from .modeling.adapter.MultiHeadGatedCrossAttentionAdapter import MultiHeadGatedCrossAttentionAdapter


def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        num_classes,
        image_size,
        pixel_mean,
        pixel_std,
        adapter=None,
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    sam = Sam(
        # Instantiate PrototypePromptGenerate
        prompt_encoder=PrototypePromptGenerate(
            mem_dim=256,
            embed_dim=prompt_embed_dim,
            image_embedding_size=image_embedding_size
        ),
        # The original PromptEncoder is commented out, and PrototypePromptGenerate acts as the prompt encoder here.
        # We will pass the image features to it in the wrapper function.

        image_encoder=ImageEncoderViT(
            adapter=adapter,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            
            window_size=14, 
            out_chans=prompt_embed_dim,
        ),
        mask_decoder=MaskDecoder(
            # num_multimask_outputs=3,
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )

    # Define a new PyTorch nn.Module class inside _build_sam
    class SamWithEnhancedPrompts(nn.Module):
        def __init__(self, sam_model, prototype_prompt_generator):
            super().__init__()
            self.sam = sam_model
            self.prototype_prompt_generator = prototype_prompt_generator

        def forward(self, image, multimask_output, args_img_size, point_coords=None, point_labels=None, boxes=None, masks=None):
            # 1. Pass the image through image_encoder to get image embeddings
            image_embeddings = self.sam.image_encoder(image)

            # 2. Pass image embeddings to PrototypePromptGenerate to get the dense prompt
            # We pass the image embeddings to the prototype prompt generator
            # The prototype_prompt_generator is now set as self.sam.prompt_encoder during Sam instantiation
            sparse_embeddings_generated, dense_embeddings_generated = self.sam.prompt_encoder(image_embeddings)

            # Generate positional encoding for the image embeddings
            B, C, H, W = image_embeddings.shape
            # Create coordinate grids for H and W
            grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
            # Normalize coordinates
            grid_x = (grid_x / (W - 1)) * 2 - 1
            grid_y = (grid_y / (H - 1)) * 2 - 1
            # Combine coordinates and repeat to match channel dimension
            pos_encoding = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(1, C // 2, 1, 1).to(image_embeddings.device)
            # Handle odd channel dimension by padding with zeros or similar
            N, C, H, W = image_embeddings.shape
            feature_proto_avg = F.avg_pool2d(input=image_embeddings, kernel_size=image_embeddings.shape[-2:])
            feature_proto_max = F.max_pool2d(input=image_embeddings, kernel_size=image_embeddings.shape[-2:])
            feature_proto = (feature_proto_avg + feature_proto_max).squeeze() # Shape (B, C)

            sparse_embeddings, dense_prompt = self.prototype_prompt_generator(feature_proto)

            # 3. Pass image embeddings, positional encoding, sparse prompts, and dense prompt to mask_decoder
            masks, iou_preds, low_res_masks = self.sam.mask_decoder(
 image_embeddings=image_embeddings,
 image_pe=pos_encoding, # Pass the generated positional encoding
 sparse_prompt_embeddings=sparse_embeddings_generated,
 dense_prompt_embeddings=dense_embeddings_generated,
 multimask_output=multimask_output,
            )

            return masks, iou_preds, low_res_masks

        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)
            sam.load_state_dict(new_state_dict)

        for name, value in sam.image_encoder.named_parameters(): 
            if "adapter" not in name:  # Adapter
                value.requires_grad = False

        if adapter is not None:
            sam.image_encoder.adapter = adapter(input_dim=encoder_embed_dim)

    return SamWithEnhancedPrompts(sam, sam.prompt_encoder), image_embedding_size # Return instance of the wrapper class



def load_from(sam, state_dict, image_size, vit_patch_size):
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict
