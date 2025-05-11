from .modeling.adapter.MultiHeadGatedCrossAttentionAdapter import MultiHeadGatedCrossAttentionAdapter

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from .modeling.memory.memory_prompt import PrototypePromptGenerate, EnhancedMemoryUnit # Ensure EnhancedMemoryUnit is also imported if needed elsewhere, but it's used within PrototypePromptGenerate


def build_sam_vit_h(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        adapter=MultiHeadGatedCrossAttentionAdapter,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        num_classes=num_classes,
        checkpoint=checkpoint,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None, ):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        num_classes=num_classes,
        checkpoint=checkpoint,
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
    # Define the wrapper class inside the build function
    class SamWithEnhancedPrompts(nn.Module):
        def __init__(self, sam_model, prototype_prompt_generator):
            super().__init__()
            self.sam = sam_model
            self.prototype_prompt_generator = prototype_prompt_generator

        def forward(self, image, point_coords=None, point_labels=None, mask_input=None, multimask_output=False):
            # Pass the image through the image encoder
            image_embeddings = self.sam.image_encoder(image)

            # Generate dense prompt from image embeddings using PrototypePromptGenerate
            # PrototypePromptGenerate returns sparse_embeddings and dense_embeddings
            sparse_embeddings_generated, dense_embeddings_generated = self.prototype_prompt_generator(image_embeddings)

            # Combine generated sparse embeddings with input sparse prompts (if any)
            # For simplicity, we'll primarily use the generated dense prompt and input sparse prompts
            # If point_coords and point_labels are provided, they will create sparse prompts via the original PromptEncoder logic
            # Since original PromptEncoder is commented out, we pass input sparse prompts directly if available,
            # and the generated sparse_embeddings_generated (which is currently empty)
            # If your task relies on input point/box prompts, you'll need to handle their encoding here.
            # Assuming for now that input sparse prompts are handled elsewhere or not used with this enhanced setup.
            # Let's assume the mask_decoder expects (image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings, multmask_output)
            # Note: The original SAM MaskDecoder forward signature includes point_coords, point_labels, and mask_input.
            # We need to align with that or ensure the mask_decoder can handle None for sparse prompts if only dense is used.
            masks, iou_preds, low_res_masks = self.sam.mask_decoder(
                image_embeddings, sparse_embeddings_generated, dense_embeddings_generated, point_coords, point_labels, mask_input, multimask_output)
            return masks, iou_preds, low_res_masks # Return the standard outputs
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    sam = Sam(
        # The image encoder outputs the image embeddings
        image_encoder=ImageEncoderViT(
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
            adapter = adapter,
        ),
        # The standard PromptEncoder is commented out, so we'll use PrototypePromptGenerate
        # prompt_encoder=PromptEncoder(
        #     embed_dim=prompt_embed_dim,
        #     image_embedding_size=(image_embedding_size, image_embedding_size),
        #     input_image_size=(image_size, image_size),
        #     mask_in_chans=16,
        # ),
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

    # Instantiate the PrototypePromptGenerate module
    prototype_prompt_generator = PrototypePromptGenerate(mem_dim=256, embed_dim=prompt_embed_dim, image_embedding_size=image_embedding_size)

    # Wrap the SAM model with the enhanced prompt generation
    sam.train()

    # Create a wrapper function to handle the forward pass
    def wrapped_sam_forward(image, point_coords, point_labels, mask_input=None, multimask_output=False):
        image_embeddings = sam.image_encoder(image)
        # Generate dense prompt from image embeddings using PrototypePromptGenerate
        sparse_embeddings, dense_embeddings = prototype_prompt_generator(image_embeddings)
        masks, iou_preds, low_res_masks = sam.mask_decoder(image_embeddings, sparse_embeddings, dense_embeddings, point_coords, point_labels, mask_input, multimask_output)
        return masks, iou_preds, low_res_masks

    if checkpoint is not None: # This part handles loading checkpoints into the original SAM model structure
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)
            sam.load_state_dict(new_state_dict)

        for name, value in sam.image_encoder.named_parameters():
            # freeze backbone except adapter
            if "adapter" not in name:#Adapter
                 value.requires_grad = False # Keep frozen except adapter

    # Return the wrapper class instance instead of the function
    return SamWithEnhancedPrompts(sam, prototype_prompt_generator), image_embedding_size


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
