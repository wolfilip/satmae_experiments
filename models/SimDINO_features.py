from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simdino_models.utils import load_pretrained_weights
from torchvision import models as torchvision_models
from UPerNet.UPerNetHead import UperNetHead

from .simdino_models import vision_transformer as vits
from torchvision.ops.misc import Permute
from util.pos_embed import get_2d_sincos_pos_embed


class ConvStem(nn.Module):
    def __init__(self, channelt_size=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channelt_size, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class SimDINO(nn.Module):

    def __init__(self, args, device) -> None:
        super().__init__()
        # self.model_size = model_args["model_size"]
        self.model_size = args.model.split("_")[1]

        # self.feat_extr = vits.__dict__[args.model](
        #     patch_size=args.patch_size, num_classes=0
        # )

        self.ms_backbone = False
        self.model = args.model.split("_")[0]

        if "vit" in args.finetune:
            self.feat_extr = vits.__dict__[args.model]()
            self.feat_extr.patch_embed.proj = nn.Conv2d(
                10, 192, kernel_size=(16, 16), stride=(16, 16)
            )
        elif "dconv" in args.finetune:
            self.feat_extr = torchvision_models.__dict__[args.model]()
            del self.feat_extr.features[0]
            # self.feat_extr.conv_ms = ConvStem(10)
            # self.feat_extr.conv_rgb = ConvStem(3)
            norm_layer_ms = partial(nn.LayerNorm, eps=1e-5)
            norm_layer_rgb = partial(nn.LayerNorm, eps=1e-5)
            self.feat_extr.conv_ms = nn.Sequential(
                nn.Conv2d(
                    10,
                    self.feat_extr.features[0][0].norm1.normalized_shape[0],
                    kernel_size=(4, 4),
                    stride=(4, 4),
                ),
                Permute([0, 2, 3, 1]),
                norm_layer_ms(self.feat_extr.features[0][0].norm1.normalized_shape[0]),
            )
            # self.feat_extr.proj_ms = nn.Linear(768, 96)
            self.feat_extr.conv_rgb = nn.Sequential(
                nn.Conv2d(
                    3,
                    self.feat_extr.features[0][0].norm1.normalized_shape[0],
                    kernel_size=(4, 4),
                    stride=(4, 4),
                ),
                Permute([0, 2, 3, 1]),
                norm_layer_rgb(self.feat_extr.features[0][0].norm1.normalized_shape[0]),
            )
            # self.feat_extr.proj_rgb = nn.Linear(768, 96)
            # self.feat_extr.ms_process = torchvision_models.__dict__["swin_t"]()
            # self.feat_extr.rgb_process = torchvision_models.__dict__["swin_t"]()
            # del self.feat_extr.ms_process.features[0]
            # del self.feat_extr.rgb_process.features[0]
            self.ms_backbone = True
        elif "ms" in args.finetune:
            self.feat_extr = torchvision_models.__dict__[args.model]()
            del self.feat_extr.features[0]
            norm_layer_ms = partial(nn.LayerNorm, eps=1e-5)
            self.feat_extr.conv_ms = nn.Sequential(
                nn.Conv2d(
                    10,
                    self.feat_extr.features[0][0].norm1.normalized_shape[0],
                    kernel_size=(4, 4),
                    stride=(4, 4),
                ),
                Permute([0, 2, 3, 1]),
                norm_layer_ms(self.feat_extr.features[0][0].norm1.normalized_shape[0]),
            )
            # self.feat_extr.features[0][0] = nn.Conv2d(
            #     10,
            #     self.feat_extr.features[0][0].out_channels,
            #     kernel_size=(4, 4),
            #     stride=(4, 4),
            # )
            self.ms_backbone = True

        if args.finetune:
            load_pretrained_weights(
                self.feat_extr, args.finetune, "student", args.model, 16
            )

        # if args.dataset_type == "spacenet":
        #     conv1_weights = self.feat_extr.features[0][0].weight
        #     self.feat_extr.features[0][0].weight = nn.Parameter(
        #         conv1_weights[:, :3, :, :]
        #     )

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
        elif args.model == "vit_tiny":
            feature_channels = [192, 192, 192, 192]

        # fpn_out = self.embed_dim + self.conv_size
        self.input_size = (args.input_size, args.input_size)
        self.num_patches = int(self.input_size[0] / 16)

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

        self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        config = {
            "pool_scales": [1, 2, 3, 6],
            "hidden_size": 512,
            "num_labels": args.nb_classes,
            "initializer_range": 0.02,
        }

        self.upernet_head = UperNetHead(config, feature_channels)

        self.channel_project = nn.Linear(3, 10)  # Define a learnable linear layer

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

    # def forward_vit(self, x):
    #     with torch.no_grad():
    #         B = x.shape[0]
    #         x = self.feat_extr.patch_embed(x)

    #         cls_tokens = self.feat_extr.cls_token.expand(B, -1, -1)
    #         x = torch.cat((cls_tokens, x), dim=1)
    #         x = x + self.feat_extr.pos_embed
    #         x = self.feat_extr.pos_drop(x)
    #         x = x[:, 1:]

    #         outs = []
    #         for i, blk in enumerate(self.feat_extr.blocks):
    #             x = blk(x)
    #             # if i in [3, 11]:
    #             if i in [3, 5, 8, 11]:
    #                 # if i in [3, 8, 13, 18, 23]:
    #                 outs.append(x)

    def forward_swin(self, x):
        with torch.no_grad():
            features = []
            for i, layer in enumerate(self.feat_extr.features):
                if i == 0:
                    if x.shape[1] == 10:
                        x = self.feat_extr.conv_ms(x)
                    # for i, layer_ms in enumerate(
                    #     self.feat_extr.ms_process.features
                    # ):
                    #     x = layer_ms(x)
                    # x = self.feat_extr.ms_process.norm(x)
                    # x = self.feat_extr.proj_ms(x)
                    # x = x.permute(0, 3, 1, 2)
                    # x = F.interpolate(
                    #     x, size=(56, 56), mode="bilinear", align_corners=False
                    # )
                    # x = x.permute(0, 2, 3, 1)
                    else:
                        x = self.feat_extr.conv_rgb(x)
                # for i, layer_rgb in enumerate(
                #     self.feat_extr.rgb_process.features
                # ):
                #     x = layer_rgb(x)
                # x = self.feat_extr.rgb_process.norm(x)
                # x = self.feat_extr.proj_rgb(x)
                # x = x.permute(0, 3, 1, 2)
                # x = F.interpolate(
                #     x,
                #     size=(56, 56),
                #     mode="bilinear",
                #     align_corners=False,
                # )
                # x = x.permute(0, 2, 3, 1)
                x = layer(x)
                # if i in [1, 3, 5, 7]:
                if i in [0, 2, 4, 6]:
                    features.append(x)
        return features

    def encoder_conv(self, x):

        conv_embeds = self.conv_layers(x)
        # conv_embeds = self.up(conv_embeds)

        return conv_embeds

    def decoder_upernet_swin(self, swin_embeds):

        swin_embeds[0] = torch.permute(swin_embeds[0], (0, 3, 1, 2))
        swin_embeds[1] = torch.permute(swin_embeds[1], (0, 3, 1, 2))
        swin_embeds[2] = torch.permute(swin_embeds[2], (0, 3, 1, 2))
        swin_embeds[3] = torch.permute(swin_embeds[3], (0, 3, 1, 2))

        x = self.upernet_head(swin_embeds)

        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

        return x

    def decoder_upernet_vit(self, features):

        new_features = []

        new_features.append(
            features[0].reshape(-1, self.num_patches, self.num_patches, 192)
        )
        new_features.append(
            features[1].reshape(-1, self.num_patches, self.num_patches, 192)
        )
        new_features.append(
            features[2].reshape(-1, self.num_patches, self.num_patches, 192)
        )
        new_features.append(
            features[3].reshape(-1, self.num_patches, self.num_patches, 192)
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

        # if x.shape[1] == 4:
        #     x = F.pad(x, (0, 0, 0, 0, 0, 6), "constant", 0)

        if x.shape[1] == 4:
            x = torch.split(x, [3, x.shape[1] - 3], dim=1)[0]

        # if not self.ms_backbone and x.shape[1] != 3:
        #     chunks = torch.split(x, [3, 7], dim=1)
        # else:
        # if x.shape[1] == 3:
        #     x = F.pad(x, (0, 0, 0, 0, 0, 7), "constant", 0)  # Pad to 10 channels
        # elif x.shape[1] == 4:
        #     x = F.pad(x, (0, 0, 0, 0, 0, 6), "constant", 0)
        # x = x.permute(0, 2, 3, 1)
        # x = self.channel_project(x)
        # x = x.permute(0, 3, 1, 2)
        # print(x.shape)

        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(x, conv_embeds)
        if self.model == "swin":
            if self.ms_backbone:
                features = self.forward_swin(x)  # type: ignore
            else:
                if x.shape[1] != 3:
                    features = self.forward_swin(x)  # type: ignore
                else:
                    features = self.forward_swin(x)

            x = self.decoder_upernet_swin(features)
        else:
            features = self.feat_extr.get_intermediate_layers(x=x, n=[3, 5, 8, 11])
            x = self.decoder_upernet_vit(features)

        # features = self.get_features(x)
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(swin_features, conv_embeds)

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

        return x, (0, features[-1])
