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
from UPerNet.UPerNetHead import UperNetHead
from util.linear_calssifier import LinearClassifier
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

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        for param in self.patch_embed.parameters():
            param.requires_grad = False

        self.conv_size = 0

        feature_channels = [
            self.embed_dim + self.conv_size,
            self.embed_dim,
            self.embed_dim,
            self.embed_dim,
        ]

        self.input_size = kwargs["img_size"]
        self.num_patches = int(kwargs["img_size"] / kwargs["patch_size"])

        if self.input_size % 16 != 0:
            self.do_interpolation = True

        config = {
            "pool_scales": [1, 2, 3, 6],
            "hidden_size": 512,
            "num_labels": kwargs["num_classes"],
            "initializer_range": 0.02,
        }

        self.upernet_head = UperNetHead(config, feature_channels)

        self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        # self.up_3 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        # self.up_4 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)
        # self.sigmoid = nn.Sigmoid()

        # self.classifier = LinearClassifier(
        #     self.embed_dim, self.num_patches, self.num_patches, self.num_classes
        # )

        if self.conv_size > 0:
            self.up = nn.Upsample(size=(372, 372), mode="bilinear", align_corners=True)

        # self.conv = nn.Conv2d(
        #     in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1
        # )  # 3x3 kernel, stride 2, padding 1
        # self.bn = nn.BatchNorm2d(256)
        # self.relu = nn.ReLU()

        # Convolutional layers to transform from [B, 3, 224, 224] to [B, 1024, 56, 56]
        if self.conv_size == 32:
            self.conv_layers = nn.Sequential(
                # Conv1: Input [B, 3, 224, 224] -> Output [B, 64, 112, 112]
                nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=3
                ),  # Kernel size 7x7, stride 2, padding 3
                nn.BatchNorm2d(16),
                nn.ReLU(),
                # Conv2: Input [B, 64, 112, 112] -> Output [B, 128, 56, 56]
                nn.Conv2d(
                    in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
                ),  # Kernel size 3x3, stride 2, padding 1
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # nn.Conv2d(
                #     in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
                # ),  # Kernel size 3x3, stride 2, padding 1
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.Upsample(size=(144, 144), mode="bilinear", align_corners=False),
            )
        elif self.conv_size == 256:
            self.conv_layers = nn.Sequential(
                # Conv1: Input [B, 3, 224, 224] -> Output [B, 64, 112, 112]
                nn.Conv2d(
                    in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
                ),  # Kernel size 7x7, stride 2, padding 3
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # Conv2: Input [B, 64, 112, 112] -> Output [B, 128, 56, 56]
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
                ),  # Kernel size 3x3, stride 2, padding 1
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # Conv3: Input [B, 128, 56, 56] -> Output [B, 256, 56, 56]
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # Kernel size 3x3, stride 1, padding 1
                nn.BatchNorm2d(256),
                nn.ReLU(),
                # Conv4: Input [B, 256, 56, 56] -> Output [B, 512, 56, 56]
                # nn.Conv2d(
                #     in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
                # ),  # Kernel size 3x3, stride 1, padding 1
                # nn.BatchNorm2d(512),
                # nn.ReLU(),
                # Conv5: Input [B, 512, 56, 56] -> Output [B, 1024, 56, 56]
                # nn.Conv2d(
                #     in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
                # ),  # Kernel size 3x3, stride 1, padding 1
                # nn.BatchNorm2d(1024),
                # nn.ReLU(),
            )

    def encoder_conv(self, x):

        conv_embeds = self.conv_layers(x)
        if conv_embeds.shape[-1] == 375:
            conv_embeds = self.up(conv_embeds)

        return conv_embeds

    def encoder_forward(self, x):

        with torch.no_grad():
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
                # if i in [3, 11]:
                if i in [3, 9, 17, 23]:
                    # if i in [3, 8, 13, 18, 23]:
                    outs.append(x)

        return outs

    def decoder_upernet(self, features, conv_embeds):

        new_features = []

        # features.append(torch.clone(feature_list[0]))
        # features.append(torch.clone(feature_list[1]))
        # # features.append(torch.clone(feature_list[2]))
        # # features.append(torch.clone(feature_list[3]))

        # conv_1 = self.relu(self.bn(self.conv(conv_embeds)))
        # conv_2 = self.relu(self.bn(self.conv(conv_1)))
        # conv_3 = self.relu(self.bn(self.conv(conv_2)))

        new_features.append(
            torch.unflatten(
                features[0], dim=1, sizes=(self.num_patches, self.num_patches)
            )
        )
        new_features.append(
            torch.unflatten(
                features[1], dim=1, sizes=(self.num_patches, self.num_patches)
            )
        )
        new_features.append(
            torch.unflatten(
                features[2], dim=1, sizes=(self.num_patches, self.num_patches)
            )
        )
        new_features.append(
            torch.unflatten(
                features[3], dim=1, sizes=(self.num_patches, self.num_patches)
            )
        )
        # features[4] = torch.unflatten(features[4], dim=1, sizes=(14, 14))
        new_features[0] = torch.permute(new_features[0], (0, 3, 1, 2))
        new_features[1] = torch.permute(new_features[1], (0, 3, 1, 2))
        new_features[2] = torch.permute(new_features[2], (0, 3, 1, 2))
        new_features[3] = torch.permute(new_features[3], (0, 3, 1, 2))
        # features[4] = torch.permute(features[4], (0, 3, 1, 2))

        new_features[-1] = F.interpolate(
            new_features[-1], scale_factor=0.5, mode="bilinear", align_corners=True
        )
        # features[2] = self.up_1(features[2])
        new_features[1] = self.up_1(new_features[1])
        new_features[0] = self.up_2(new_features[0])

        if self.conv_size > 0:
            new_features[0] = torch.cat((new_features[0], conv_embeds), 1)
        # features[1] = torch.cat((features[1], conv_1), 1)
        # features[2] = torch.cat((features[2], conv_2), 1)
        # features[3] = torch.cat((features[3], conv_3), 1)

        # features[0] = features[0] + conv_embeds

        x = self.upernet_head(new_features)

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

    def decoder_linear(self, x, conv_embeds):
        x = self.classifier(x)
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x):

        # chunks = torch.split(x, [1, 3, x.shape[1] - 4], dim=1)

        conv_embeds = 0
        if self.conv_size > 0:
            conv_embeds = self.encoder_conv(x)
        features = self.encoder_forward(x)
        # x = self.decoder_upernet(features, conv_embeds)
        # x = self.encoder_forward(x)
        # print(x.shape, features[0].shape, conv_embeds.shape)
        x = self.decoder_upernet(features, conv_embeds)
        # x = self.decoder_linear(features[-1], conv_embeds)
        return x, (conv_embeds, features[-1])


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
