import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simdino_models.utils import load_pretrained_weights
from torchvision import models as torchvision_models
from UPerNet.UPerNetHead import UperNetHead
from models.terrafm import terrafm_base

from .simdino_models import vision_transformer as vits


class TerraFMModel(nn.Module):

    def __init__(self, args, device) -> None:
        super().__init__()
        self.model_size = args.model.split("_")[1]
        self.conv_size = 0

        self.feat_extr = terrafm_base()
        checkpoint_model = torch.load(args.finetune, map_location="cpu")
        self.feat_extr.load_state_dict(checkpoint_model)

        self.feat_extr.eval()  # type: ignore
        self.feat_extr.to(device)  # type: ignore

        for p in self.feat_extr.parameters():
            p.requires_grad = False

        feature_channels = [768, 768, 768, 768]

        self.input_size = (args.input_size, args.input_size)

        config = {
            "pool_scales": [1, 2, 3, 6],
            "hidden_size": 512,
            "num_labels": args.nb_classes,
            "initializer_range": 0.02,
        }

        self.upernet_head = UperNetHead(config, feature_channels)

        self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        # self.channel_project = nn.Linear(3, 10)  # Define a learnable linear layer

        # self.classification_head = nn.Linear(feature_channels[-1], args.nb_classes)

    def decoder_upernet(self, features):

        # conv_1 = self.relu(self.bn(self.conv(conv_embeds)))
        # conv_2 = self.relu(self.bn(self.conv(conv_1)))
        # conv_3 = self.relu(self.bn(self.conv(conv_2)))
        # new_features = []

        # new_features.append(features[0].reshape(-1, 14, 14, 768))
        # new_features.append(features[1].reshape(-1, 14, 14, 768))
        # new_features.append(features[2].reshape(-1, 14, 14, 768))
        # new_features.append(features[3].reshape(-1, 14, 14, 768))
        # swin_embeds = conv_embeds[0].reshape(-1, 128, 128, 96)

        # new_features[0] = torch.permute(new_features[0], (0, 3, 1, 2))
        # new_features[1] = torch.permute(new_features[1], (0, 3, 1, 2))
        # new_features[2] = torch.permute(new_features[2], (0, 3, 1, 2))
        # new_features[3] = torch.permute(new_features[3], (0, 3, 1, 2))
        # swin_embeds = torch.permute(swin_embeds, (0, 3, 1, 2))

        features[-1] = F.interpolate(
            features[-1], scale_factor=0.5, mode="bilinear", align_corners=True
        )
        # features[2] = self.up_1(features[2])
        features[1] = self.up_1(features[1])
        features[0] = self.up_2(features[0])

        # new_features[0] = torch.cat((new_features[0], self.up(swin_embeds)), 1)
        # new_features[1] = torch.cat((new_features[1], conv_1), 1)
        # new_features[2] = torch.cat((new_features[2], conv_2), 1)
        # new_features[3] = torch.cat((new_features[3], conv_3), 1)

        # features[0] = features[0] + conv_embeds

        # new_features[-1] = self.PPN(new_features[-1])
        # x = self.head(features[-1])
        # x = self.head(self.FPN(new_features))

        x = self.upernet_head(features)

        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x):

        if x.shape[1] == 11:
            x = F.pad(x, (0, 0, 0, 0, 0, 1), "constant", 0)
        elif x.shape[1] == 3:
            x = torch.cat(
                [
                    torch.zeros_like(x[:, :1]),  # zero channel at index 0
                    x[:, :4],
                ],
                dim=1,
            )
            x = F.pad(x, (0, 0, 0, 0, 0, 8), "constant", 0)
        elif x.shape[1] == 4:
            x = torch.cat(
                [
                    torch.zeros_like(x[:, :1]),  # zero channel at index 0
                    x[:, :5],
                ],
                dim=1,
            )
            x = F.pad(x, (0, 0, 0, 0, 0, 7), "constant", 0)
        elif x.shape[1] == 13:
            x = torch.cat([x[:, :10], x[:, 11:]], dim=1)
        elif x.shape[1] == 10:
            x = torch.cat(
                [
                    torch.zeros_like(x[:, :1]),
                    x[:, :9],
                    torch.zeros_like(x[:, :1]),
                    x[:, 9:],
                ],
                dim=1,
            )

        features = self.feat_extr.extract_feature(x)
        # features = self.feat_extr(x)
        # x = self.classification_head(features)
        x = self.decoder_upernet(features)

        return x, (features, features[-1])
