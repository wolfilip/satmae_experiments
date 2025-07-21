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

        self.channel_project = nn.Linear(3, 10)  # Define a learnable linear layer

        if self.conv_size > 0:
            if args.dataset_type == "spacenet":
                self.up = nn.Upsample(
                    size=(64, 64), mode="bilinear", align_corners=True
                )
            elif (
                args.dataset_type == "sen1floods11"
                or args.dataset_type == "vaihingen"
                or args.dataset_type == "potsdam"
            ):
                self.up = nn.Upsample(
                    size=(144, 144), mode="bilinear", align_corners=True
                )
            elif args.dataset_type == "isaid":
                self.up = nn.Upsample(
                    size=(256, 256), mode="bilinear", align_corners=True
                )
            elif args.dataset_type == "mass_roads":
                self.up = nn.Upsample(
                    size=(428, 428), mode="bilinear", align_corners=True
                )
            # elif  args.dataset_type == "rgb":

        # self.conv = nn.Conv2d(
        #     in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1
        # )  # 3x3 kernel, stride 2, padding 1
        # self.bn = nn.BatchNorm2d(256)
        # self.relu = nn.ReLU()

        # self.conv = nn.Conv2d(
        #     in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
        # )
        # self.bn = nn.BatchNorm2d(32)
        # self.relu = nn.ReLU()

        # self.classifier = LinearClassifier(
        #     self.embed_dim, self.num_patches, self.num_patches, args.nb_classes
        # )

        if self.conv_size == 32:
            # Commenting out the original self.conv_layers definition
            self.conv_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=7, out_channels=16, kernel_size=7, stride=2, padding=3
                ),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )

            # Adding the new function to reduce image size by four
            # self.conv_layers = nn.Sequential(
            #     nn.Conv2d(
            #         in_channels=7, out_channels=16, kernel_size=3, stride=2, padding=1
            #     ),
            #     nn.BatchNorm2d(16),
            #     nn.ReLU(),
            #     nn.Conv2d(
            #         in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
            #     ),
            #     nn.BatchNorm2d(32),
            #     nn.ReLU(),
            #     nn.Conv2d(
            #         in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
            #     ),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(),
            #     nn.Conv2d(
            #         in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            #     ),
            #     nn.BatchNorm2d(128),
            #     nn.ReLU(),
            # )
        elif self.conv_size == 256:
            self.conv_layers = nn.Sequential(
                # Conv1: Input [B, 3, 224, 224] -> Output [B, 64, 112, 112]
                nn.Conv2d(
                    in_channels=7, out_channels=64, kernel_size=7, stride=2, padding=3
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

        if x.shape[1] == 3:
            x = F.pad(x, (0, 0, 0, 0, 0, 9), "constant", 0)

        features = self.feat_extr.extract_feature(x)
        x = self.decoder_upernet(features)

        return x, (features, features[-1])
