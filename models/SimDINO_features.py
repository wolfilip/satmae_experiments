import re
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from UPerNet.UPerNetHead import UperNetHead
from functools import partial

from models.simdino_models.utils import load_pretrained_weights
from .simdino_models import vision_transformer as vits
from torchvision import models as torchvision_models


class SimDINO(nn.Module):

    def __init__(self, args, device) -> None:
        super().__init__()
        # self.model_size = model_args["model_size"]
        self.model_size = args.model.split("_")[1]
        self.conv_size = 0

        # self.feat_extr = vits.__dict__[args.model](
        #     patch_size=args.patch_size, num_classes=0
        # )
        self.feat_extr = torchvision_models.__dict__[args.model]()

        self.ms_backbone = False

        if "ms" in args.finetune:
            self.feat_extr.features[0][0] = nn.Conv2d(
                10,
                self.feat_extr.features[0][0].out_channels,
                kernel_size=(4, 4),
                stride=(4, 4),
            )
            self.ms_backbone = True

        if args.finetune:
            load_pretrained_weights(
                self.feat_extr, args.finetune, "teacher", args.model, 16
            )

        self.feat_extr.eval()  # type: ignore
        self.feat_extr.to(device)  # type: ignore
        # self.device = device
        # self.patch_size = 16

        for p in self.feat_extr.parameters():
            p.requires_grad = False

        # upernet stuff
        # if self.model_size == "small" or self.model_size == "s":
        #     self.embed_dim = 384
        #     feature_channels = [
        #         self.embed_dim + self.conv_size,
        #         self.embed_dim,
        #     ]
        # elif self.model_size == "base" or self.model_size == "b":
        #     self.embed_dim = 768
        #     feature_channels = [
        #         self.embed_dim + self.conv_size,
        #         self.embed_dim,
        #     ]
        # else:
        #     self.embed_dim = 1024
        #     feature_channels = [
        #         self.embed_dim + self.conv_size,
        #         self.embed_dim,
        #         self.embed_dim,
        #         self.embed_dim,
        #     ]

        if args.model == "swin_b":
            feature_channels = [128, 256, 512, 1024]
        elif args.model == "swin_s":
            feature_channels = [96, 192, 384, 768]
        elif args.model == "swin_t":
            feature_channels = [96, 192, 384, 768]

        # fpn_out = self.embed_dim + self.conv_size
        self.input_size = (args.input_size, args.input_size)
        # self.num_patches = int(self.input_size[0] / self.patch_size)

        self.do_interpolation = False

        if args.input_size % 14 != 0:
            self.do_interpolation = True

        if args.dataset_type == "euro_sat" or args.dataset_type == "rgb":
            self.task = "classification"
            # self.classifier = LinearClassifier(
            #     self.embed_dim, self.num_patches, self.num_patches, args.nb_classes
            # )
            self.classification_head = nn.Linear(self.embed_dim, args.nb_classes)
        else:
            self.task = "segmentation"

        # self.PPN = PSPModule(feature_channels[-1])
        # self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        # self.head = nn.Conv2d(fpn_out, args.nb_classes, kernel_size=3, padding=1)
        # self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        config = {
            "pool_scales": [1, 2, 3, 6],
            "hidden_size": 512,
            "num_labels": args.nb_classes,
            "initializer_range": 0.02,
        }

        self.upernet_head = UperNetHead(config, feature_channels)

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

    def get_features(self, imgs):
        # layer = self.layer_num[0] # TODO: make it a list
        # layers = []
        if self.do_interpolation:
            if imgs.shape[-1] == 512:
                imgs = F.interpolate(
                    imgs, size=504, mode="bilinear", align_corners=True
                )
            elif imgs.shape[-1] == 64:
                imgs = F.interpolate(imgs, size=56, mode="bilinear", align_corners=True)
            elif imgs.shape[-1] == 256:
                imgs = F.interpolate(
                    imgs, size=252, mode="bilinear", align_corners=True
                )
            elif imgs.shape[-1] == 1500:
                imgs = F.interpolate(
                    imgs, size=1498, mode="bilinear", align_corners=True
                )

        with torch.no_grad():
            if self.task == "classification":
                out = self.feat_extr.forward_features(imgs)  # type: ignore
                cls = out["x_norm_clstoken"]
                out = cls
            else:
                # if self.layer_num == "last":
                if self.model_size == "base" or self.model_size == "small":
                    patch = self.feat_extr.get_intermediate_layers(imgs, (3, 5, 8, 11))  # type: ignore
                else:
                    patch = self.feat_extr.get_intermediate_layers(imgs, (3, 9, 17, 23))  # type: ignore
                out = patch

            # layers.append(patch)
            # layers.append(patch)
            # cls = out["x_norm_clstoken"]
            # elif self.layer_num == "first":

        # elif self.layer_num == "avg":
        #     pass
        return out

    def forward_swin(self, x):
        with torch.no_grad():
            features = []
            for i, layer in enumerate(self.feat_extr.features):
                x = layer(x)
                if i in [1, 3, 5, 7]:  # Specify the layers you want to extract
                    features.append(x)
        return features

    def encoder_conv(self, x):

        conv_embeds = self.conv_layers(x)
        # conv_embeds = self.up(conv_embeds)

        return conv_embeds

    def decoder_upernet_swin(self, swin_embeds, conv_embeds=None):

        x = self.upernet_head(swin_embeds)

        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

        return x

    def decoder_upernet(self, features, conv_embeds):

        # conv_1 = self.relu(self.bn(self.conv(conv_embeds)))
        # conv_2 = self.relu(self.bn(self.conv(conv_1)))
        # conv_3 = self.relu(self.bn(self.conv(conv_2)))
        new_features = []

        new_features.append(
            features[0].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        )
        new_features.append(
            features[1].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        )
        new_features.append(
            features[2].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        )
        new_features.append(
            features[3].reshape(-1, self.num_patches, self.num_patches, self.embed_dim)
        )

        new_features[0] = torch.permute(new_features[0], (0, 3, 1, 2))
        new_features[1] = torch.permute(new_features[1], (0, 3, 1, 2))
        new_features[2] = torch.permute(new_features[2], (0, 3, 1, 2))
        new_features[3] = torch.permute(new_features[3], (0, 3, 1, 2))

        new_features[-1] = F.interpolate(
            new_features[-1], scale_factor=0.5, mode="bilinear", align_corners=True
        )
        # features[2] = self.up_1(features[2])
        new_features[1] = self.up_1(new_features[1])
        new_features[0] = self.up_2(new_features[0])
        if self.conv_size > 0:
            new_features[0] = torch.cat((new_features[0], conv_embeds), 1)
            # new_features[1] = torch.cat((new_features[1], conv_1), 1)
            # new_features[2] = torch.cat((new_features[2], conv_2), 1)
            # new_features[3] = torch.cat((new_features[3], conv_3), 1)

        # features[0] = features[0] + conv_embeds

        # new_features[-1] = self.PPN(new_features[-1])
        # x = self.head(features[-1])
        # x = self.head(self.FPN(new_features))

        x = self.upernet_head(new_features)

        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def decoder_linear(self, x, conv_embeds):
        x = self.classifier(x)
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x):

        if not self.ms_backbone:
            chunks = torch.split(x, [3, 7], dim=1)
        conv_embeds = 0
        if self.conv_size > 0:
            conv_embeds = self.encoder_conv(x)
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(x, conv_embeds)
        if self.ms_backbone:
            swin_features = self.forward_swin(x)
        else:
            swin_features = self.forward_swin(chunks[0])  # type: ignore

        swin_features[0] = torch.permute(swin_features[0], (0, 3, 1, 2))
        swin_features[1] = torch.permute(swin_features[1], (0, 3, 1, 2))
        swin_features[2] = torch.permute(swin_features[2], (0, 3, 1, 2))
        swin_features[3] = torch.permute(swin_features[3], (0, 3, 1, 2))

        if self.conv_size > 0:
            swin_features[0] = torch.cat((swin_features[0], conv_embeds), 1)
        # features = self.get_features(x)
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(swin_features, conv_embeds)
        x = self.decoder_upernet_swin(swin_features, conv_embeds)
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

        return x, (conv_embeds, swin_features[-1])
