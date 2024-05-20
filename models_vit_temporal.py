# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed

from util.pos_embed import (
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid_torch,
    get_2d_sincos_pos_embed_with_resolution,
)


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, 345 + 1, kwargs["embed_dim"] - 384)
        )
        self.token_pos_embed = nn.Parameter(
            torch.zeros(1, 345 + 1, 640),
            requires_grad=False,
        )

        self.patch_embed_1 = PatchEmbed(224, 16, 3, 1024)
        self.patch_embed_2 = PatchEmbed(160, 16, 3, 1024)
        self.patch_embed_3 = PatchEmbed(112, 16, 3, 1024)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward(self, x, timestamps):
        x = self.forward_features(x, timestamps)
        x = self.head(x)
        return x

    def forward_features(self, x, timestamps):

        B = x[0].shape[0]
        x1 = self.patch_embed_1(x[0])
        x2 = self.patch_embed_2(x[1])
        x3 = self.patch_embed_3(x[2])
        x = torch.cat([x1, x2, x3], dim=1)

        mock_res_1 = torch.ones(x.shape[0])
        mock_res_2 = torch.ones(x.shape[0]) * 1.4
        mock_res_3 = torch.ones(x.shape[0]) * 2

        ts_embed = torch.cat(
            [
                get_1d_sincos_pos_embed_from_grid_torch(
                    128, timestamps.reshape(-1, 3)[:, 0].float()
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    128, timestamps.reshape(-1, 3)[:, 1].float()
                ),
                get_1d_sincos_pos_embed_from_grid_torch(
                    128, timestamps.reshape(-1, 3)[:, 2].float()
                ),
            ],
            dim=1,
        ).float()
        ts_embed = ts_embed.reshape(-1, 3, ts_embed.shape[-1]).unsqueeze(2)
        ts_embed_1 = ts_embed[:, 0, :, :].expand(-1, x1.shape[1], -1)
        ts_embed_2 = ts_embed[:, 1, :, :].expand(-1, x2.shape[1], -1)
        ts_embed_3 = ts_embed[:, 2, :, :].expand(-1, x3.shape[1], -1)

        ts_embed = torch.cat(
            [ts_embed_1, ts_embed_2, ts_embed_3],
            dim=1,
        ).float()

        ts_embed = torch.cat(
            [
                torch.zeros(
                    (ts_embed.shape[0], 1, ts_embed.shape[2]), device=ts_embed.device
                ),
                ts_embed,
            ],
            dim=1,
        )

        pos_embed = torch.cat(
            [
                get_2d_sincos_pos_embed_with_resolution(
                    640,
                    int(196**0.5),
                    mock_res_1,
                    cls_token=True,
                    device=x1.device,
                )[:, 1:, :],
                get_2d_sincos_pos_embed_with_resolution(
                    640,
                    int(100**0.5),
                    mock_res_2,
                    cls_token=True,
                    device=x2.device,
                )[:, 1:, :],
                get_2d_sincos_pos_embed_with_resolution(
                    640,
                    int(49**0.5),
                    mock_res_3,
                    cls_token=True,
                    device=x3.device,
                )[:, 1:, :],
            ],
            dim=1,
        ).float()

        pos_embed = torch.cat(
            [
                self.token_pos_embed[:, :1, :].repeat(x.shape[0], 1, 1),
                pos_embed,
            ],
            dim=1,
        )

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + torch.cat([pos_embed, ts_embed], dim=-1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# from models_vit import vit_large_patch16
# vit_large_patch16_nontemp = vit_large_patch16
