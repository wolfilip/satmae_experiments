import torch
import torch.nn as nn
import torch.nn.functional as F

from UPerNet.FPN_fuse import FPN_fuse
from UPerNet.PSPModule import PSPModule
from UPerNet.UPerNetHead import UperNetHead

from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentation,
)


class SwinModel(nn.Module):

    def __init__(self, args, device) -> None:
        super().__init__()

        self.swin_backbone = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-cityscapes-semantic"
        ).model.pixel_level_module.encoder

        self.swin_backbone.to(device)

        self.input_size = (args.input_size, args.input_size)

        config = {
            "pool_scales": [1, 2, 3, 6],
            "hidden_size": 512,
            "num_labels": args.nb_classes,
            "initializer_range": 0.02,
        }

        self.feature_channels = [192, 384, 768, 1536]

        self.upernet_head = UperNetHead(config, self.feature_channels)

        self.num_patches = [
            int(self.input_size[0] / 4),
            int(self.input_size[0] / 8),
            int(self.input_size[0] / 16),
            int(self.input_size[0] / 32),
        ]

    def decoder_upernet(self, features):

        x = self.upernet_head(features)

        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def decoder_linear(self, x, conv_embeds):
        x = self.classifier(x)
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(x, conv_embeds)
        chunks = torch.split(x, [3, 7], dim=1)

        features = self.swin_backbone(chunks[0])
        # x = self.encoder_forward(x)
        x = self.decoder_upernet(features["feature_maps"])
        # new_features = []

        # new_features.append(
        #     features[0].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        # )
        # new_features.append(
        #     features[1].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        # )
        # new_features.append(
        #     features[2].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        # )
        # new_features.append(
        #     features[3].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        # )

        # new_features[0] = torch.permute(new_features[0], (0, 3, 1, 2))
        # new_features[1] = torch.permute(new_features[1], (0, 3, 1, 2))
        # new_features[2] = torch.permute(new_features[2], (0, 3, 1, 2))
        # new_features[3] = torch.permute(new_features[3], (0, 3, 1, 2))
        # x = self.upernet_head(new_features)
        # x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

        # x = self.classification_head(features)
        # x = self.decoder_linear(features[-1], conv_embeds)

        # x = self.decoder_upernet(x[1])

        return x, features
