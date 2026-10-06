# -*- coding: utf-8 -*-
"""
正确的 VesSAM 模型构建（修正官方 build_Ves_SAM.py 与组件签名不同步的问题）
==========================================================================
官方的 build_Ves_SAM.py 是为标准 SAM 的 PromptEncoder 写的，但仓库里的
PromptEncoder 是 VesSAM 自定义版，参数不匹配，导致直接调用会报错。

这里按仓库各组件真实的构造函数签名重新组装 Ves_SAM，并处理 SAM 权重
在 512 分辨率下的位置编码插值，保证能正确加载 image_encoder 的预训练参数。
"""
import os
from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional as F

from Models.image_encoder import ImageEncoderViT
from Models.Prompt_encoder import PromptEncoder
from Models.mask_decoder import MaskDecoder
from Models.Mytransformer import TwoWayTransformer
from Models.Ves_SAM import Ves_SAM


def build_vessam_model(image_size=512, checkpoint=None, adapter_train=False):
    prompt_embed_dim = 256

    image_encoder = ImageEncoderViT(
        img_size=image_size,
        patch_size=16,
        in_chans=3,
        embed_dim=768,          # vit_b
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_chans=prompt_embed_dim,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=14,
        global_attn_indexes=[2, 5, 8, 11],
        adapter_train=adapter_train,
    )

    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        img_size=image_size,
        base_chans=32,
        activation=nn.GELU,
        num_heads=8,
        num_layers=2,
        graph_layers=2,
        point_emb_chans=prompt_embed_dim,
    )

    mask_decoder = MaskDecoder(
        transformer_dim=prompt_embed_dim,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            num_heads=8,
            mlp_dim=2048,
        ),
        num_multimask_outputs=3,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    model = Ves_SAM(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if checkpoint is not None and os.path.exists(checkpoint):
        _load_pretrained(model, checkpoint, image_size)
    return model


def _load_pretrained(model, checkpoint, image_size):
    """加载官方 SAM 权重中的 image_encoder 部分（512 分辨率下插值 pos_embed）"""
    sd = torch.load(checkpoint, map_location="cpu")
    if "model" in sd:
        sd = sd["model"]

    prefix = "image_encoder."
    image_sd = {k: v for k, v in sd.items() if k.startswith(prefix)}

    token_size = image_size // 16
    # pos_embed 插值到当前分辨率
    if "image_encoder.pos_embed" in image_sd:
        pe = image_sd["image_encoder.pos_embed"]
        if pe.shape[1] != token_size:
            pe = pe.permute(0, 3, 1, 2)
            pe = F.interpolate(pe, (token_size, token_size), mode="bilinear", align_corners=False)
            pe = pe.permute(0, 2, 3, 1)
            image_sd["image_encoder.pos_embed"] = pe
    # 分辨率相关 rel_pos 尺寸不匹配，直接丢弃（保持随机/零初始化），避免加载报错
    image_sd = {k: v for k, v in image_sd.items() if "rel_pos" not in k}

    model.load_state_dict(image_sd, strict=False)
    print(f"[权重] 已加载 image_encoder 预训练参数（{len(image_sd)} 项，512 分辨率适配）")
