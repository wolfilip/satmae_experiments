# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from UPerNet.FPN_fuse import FPN_fuse
from UPerNet.PSPModule import PSPModule
from util.pos_embed import get_2d_sincos_pos_embed


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # Added by Samar, need default pos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.model_size = "large"
        if self.model_size == "small":
            self.feat_extr = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        if self.model_size == "small_reg":
            self.feat_extr = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14_reg"
            )
        if self.model_size == "base":
            self.feat_extr = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        if self.model_size == "base_reg":
            self.feat_extr = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14_reg"
            )
        if self.model_size == "large":
            self.feat_extr = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # feature_channels = [1024, 1024, 1024, 1024]
        feature_channels = [1024, 1024, 1024, 2048]

        fpn_out = 1024
        self.input_size = (224, 224)

        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, self.num_classes, kernel_size=3, padding=1)
        self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        # self.up_3 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        # self.up_4 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)
        # self.sigmoid = nn.Sigmoid()

        # self.conv = nn.Conv2d(
        #     in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1
        # )  # 3x3 kernel, stride 2, padding 1
        # self.bn = nn.BatchNorm2d(256)
        # self.relu = nn.ReLU()

        # Convolutional layers to transform from [B, 3, 224, 224] to [B, 1024, 56, 56]
        # self.conv_layers = nn.Sequential(
        #     # Conv1: Input [B, 3, 224, 224] -> Output [B, 64, 112, 112]
        #     nn.Conv2d(
        #         in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        #     ),  # Kernel size 7x7, stride 2, padding 3
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     # Conv2: Input [B, 64, 112, 112] -> Output [B, 128, 56, 56]
        #     nn.Conv2d(
        #         in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        #     ),  # Kernel size 3x3, stride 2, padding 1
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     # Conv3: Input [B, 128, 56, 56] -> Output [B, 256, 56, 56]
        #     nn.Conv2d(
        #         in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        #     ),  # Kernel size 3x3, stride 1, padding 1
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     # Conv4: Input [B, 256, 56, 56] -> Output [B, 512, 56, 56]
        #     nn.Conv2d(
        #         in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        #     ),  # Kernel size 3x3, stride 1, padding 1
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     # Conv5: Input [B, 512, 56, 56] -> Output [B, 1024, 56, 56]
        #     # nn.Conv2d(
        #     #     in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
        #     # ),  # Kernel size 3x3, stride 1, padding 1
        #     # nn.BatchNorm2d(1024),
        #     # nn.ReLU(),
        # )

        # self.conv_layers_small = nn.Sequential(
        #     # Conv1: Input [B, 3, 224, 224] -> Output [B, 64, 112, 112]
        #     nn.Conv2d(
        #         in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=3
        #     ),  # Kernel size 7x7, stride 2, padding 3
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     # Conv2: Input [B, 64, 112, 112] -> Output [B, 128, 56, 56]
        #     nn.Conv2d(
        #         in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
        #     ),  # Kernel size 3x3, stride 2, padding 1
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     # Conv3: Input [B, 128, 56, 56] -> Output [B, 256, 56, 56]
        #     # nn.Conv2d(
        #     #     in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        #     # ),  # Kernel size 3x3, stride 1, padding 1
        #     # nn.BatchNorm2d(64),
        #     # nn.ReLU(),
        #     # Conv4: Input [B, 256, 56, 56] -> Output [B, 512, 56, 56]
        #     # nn.Conv2d(
        #     #     in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        #     # ),  # Kernel size 3x3, stride 1, padding 1
        #     # nn.BatchNorm2d(128),
        #     # nn.ReLU(),
        #     # Conv5: Input [B, 512, 56, 56] -> Output [B, 1024, 56, 56]
        #     # nn.Conv2d(
        #     #     in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
        #     # ),  # Kernel size 3x3, stride 1, padding 1
        #     # nn.BatchNorm2d(1024),
        #     # nn.ReLU(),
        # )

    def encoder_conv(self, x):

        conv_embeds = self.conv_layers_small(x)

        return conv_embeds

    def encoder_forward(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(  # type: ignore
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x[:, 1:]

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in [23]:
                # if i in [3, 9, 17, 23]:
                # if i in [3, 8, 13, 18, 23]:
                outs.append(x)

        return outs

    def get_features(self, imgs):
        # layer = self.layer_num[0] # TODO: make it a list
        # layers = []
        with torch.no_grad():
            # if self.layer_num == "last":
            patch = self.feat_extr.get_intermediate_layers(imgs, (3, 9, 17, 23))  # type: ignore
            # layers.append(patch)
            # out = self.feat_extr.forward_features(imgs)  # type: ignore
            # patch = out["x_norm_patchtokens"]
            # layers.append(patch)
            # cls = out["x_norm_clstoken"]
            # elif self.layer_num == "first":

        # elif self.layer_num == "avg":
        #     pass
        out = patch
        return out

    def decoder_upernet(self, features):

        # conv_1 = self.relu(self.bn(self.conv(conv_embeds)))
        # conv_2 = self.relu(self.bn(self.conv(conv_1)))
        # conv_3 = self.relu(self.bn(self.conv(conv_2)))

        features[0] = torch.unflatten(features[0], dim=1, sizes=(16, 16))
        features[1] = torch.unflatten(features[1], dim=1, sizes=(16, 16))
        features[2] = torch.unflatten(features[2], dim=1, sizes=(16, 16))
        features[3] = torch.unflatten(features[3], dim=1, sizes=(16, 16))
        # features[4] = torch.unflatten(features[4], dim=1, sizes=(14, 14))
        features[0] = torch.permute(features[0], (0, 3, 1, 2))
        features[1] = torch.permute(features[1], (0, 3, 1, 2))
        features[2] = torch.permute(features[2], (0, 3, 1, 2))
        features[3] = torch.permute(features[3], (0, 3, 1, 2))
        # features[4] = torch.permute(features[4], (0, 3, 1, 2))

        features[-1] = F.interpolate(
            features[-1], scale_factor=0.5, mode="bilinear", align_corners=True
        )
        # features[2] = self.up_1(features[2])
        features[1] = self.up_1(features[1])
        features[0] = self.up_2(features[0])

        # features[0] = torch.cat((features[0], conv_embeds), 1)
        # features[1] = torch.cat((features[1], conv_1), 1)
        # features[2] = torch.cat((features[2], conv_2), 1)
        # features[3] = torch.cat((features[3], conv_3), 1)

        # features[0] = features[0] + conv_embeds

        features[-1] = self.PPN(features[-1])
        # x = self.head(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=self.input_size, mode="bilinear")
        return x

    def decoder_upernet_2(self, conv_embeds):

        conv_1 = self.relu(self.bn(self.conv(conv_embeds)))
        conv_2 = self.relu(self.bn(self.conv(conv_1)))
        conv_3 = self.relu(self.bn(self.conv(conv_2)))

        conv_3 = self.PPN(conv_3)
        # x = self.head(features[-1])
        x = self.head(self.FPN([conv_embeds, conv_1, conv_2, conv_3]))

        x = F.interpolate(x, size=self.input_size, mode="bilinear")
        return x

    def forward(self, x):
        # conv_embeds = self.encoder_conv(x)
        features_dino = self.get_features(x)
        features_mae = self.encoder_forward(x)
        features = []
        features.append(features_dino[0])
        features.append(features_dino[1])
        features.append(features_dino[2])
        features.append(torch.cat((features_dino[3], features_mae[0]), 2))
        # for feature1, feature2 in zip(features_dino, features_mae):
        #     features.append(torch.cat((feature1, feature2), 2))
        x = self.decoder_upernet(features)
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
