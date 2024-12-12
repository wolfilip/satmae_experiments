import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from UPerNet.FPN_fuse import FPN_fuse
from UPerNet.PSPModule import PSPModule


class LinearClassifier(nn.Module):
    def __init__(self, in_channels, tokenW=16, tokenH=16, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        # embeddings = torch.cat((embeddings, conv_embeds), 1)

        return self.classifier(embeddings)


class DINOv2(nn.Module):

    def __init__(self, args, device) -> None:
        super().__init__()
        # self.model_size = model_args["model_size"]
        self.model_size = args.model.split("_")[0]
        self.conv_size = int(args.model.split("_")[1])

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
        self.feat_extr.eval()  # type: ignore
        self.feat_extr.to(device)  # type: ignore
        self.device = device
        self.patch_size = 14

        # upernet stuff
        if self.model_size == "small":
            self.embed_dim = 384
            feature_channels = [
                self.embed_dim + self.conv_size,
                self.embed_dim,
            ]
        elif self.model_size == "base":
            self.embed_dim = 768
            feature_channels = [
                self.embed_dim + self.conv_size,
                self.embed_dim,
            ]
        else:
            self.embed_dim = 1024
            feature_channels = [
                self.embed_dim + self.conv_size,
                self.embed_dim,
                self.embed_dim,
                self.embed_dim,
            ]

        fpn_out = self.embed_dim + self.conv_size
        self.input_size = (args.input_size, args.input_size)
        num_patches = int(self.input_size[0] / self.patch_size)

        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, args.nb_classes, kernel_size=3, padding=1)
        self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        if self.conv_size > 0:
            if args.dataset_type == "spacenet":
                self.up = nn.Upsample(
                    size=(64, 64), mode="bilinear", align_corners=True
                )
            else:
                self.up = nn.Upsample(
                    size=(144, 144), mode="bilinear", align_corners=True
                )

        # self.conv = nn.Conv2d(
        #     in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
        # )  # 3x3 kernel, stride 2, padding 1
        # self.bn = nn.BatchNorm2d(256)
        # self.relu = nn.ReLU()

        # self.conv = nn.Conv2d(
        #     in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
        # )
        # self.bn = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU()

        self.classifier = LinearClassifier(
            self.embed_dim, num_patches, num_patches, args.nb_classes
        )

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

    def get_features(self, imgs):
        # layer = self.layer_num[0] # TODO: make it a list
        # layers = []
        with torch.no_grad():
            if imgs.shape[-1] % 14 != 0:
                if imgs.shape[-1] == 512:
                    imgs = F.interpolate(
                        imgs, size=504, mode="bilinear", align_corners=True
                    )
            # if self.layer_num == "last":
            if self.model_size == "base" or self.model_size == "small":
                patch = self.feat_extr.get_intermediate_layers(imgs, (3, 11))  # type: ignore
            else:
                patch = self.feat_extr.get_intermediate_layers(imgs, (3, 9, 17, 23))  # type: ignore
            # layers.append(patch)
            # layers.append(patch)
            # cls = out["x_norm_clstoken"]
            # elif self.layer_num == "first":

        # elif self.layer_num == "avg":
        #     pass
        out = patch
        return out

    def encoder_conv(self, x):

        conv_embeds = self.conv_layers(x)
        conv_embeds = self.up(conv_embeds)

        return conv_embeds

    def decoder_upernet(self, features, conv_embeds):

        num_patches = int(self.input_size[0] / self.patch_size)

        # conv_1 = self.relu(self.bn(self.conv(conv_embeds)))
        # conv_2 = self.relu(self.bn(self.conv(conv_1)))
        # conv_3 = self.relu(self.bn(self.conv(conv_2)))
        new_features = []

        new_features.append(
            features[0].reshape(-1, num_patches, num_patches, self.embed_dim)
        )
        new_features.append(
            features[1].reshape(-1, num_patches, num_patches, self.embed_dim)
        )
        if self.model_size == "large":
            new_features.append(
                features[2].reshape(-1, num_patches, num_patches, self.embed_dim)
            )
            new_features.append(
                features[3].reshape(-1, num_patches, num_patches, self.embed_dim)
            )

        new_features[0] = torch.permute(new_features[0], (0, 3, 1, 2))
        new_features[1] = torch.permute(new_features[1], (0, 3, 1, 2))
        if self.model_size == "large":
            new_features[2] = torch.permute(new_features[2], (0, 3, 1, 2))
            new_features[3] = torch.permute(new_features[3], (0, 3, 1, 2))
        # features[4] = torch.permute(features[4], (0, 3, 1, 2))

        if self.model_size == "large":
            new_features[-1] = F.interpolate(
                new_features[-1], scale_factor=0.5, mode="bilinear", align_corners=True
            )
        # features[2] = self.up_1(features[2])
        new_features[1] = self.up_1(new_features[1])
        new_features[0] = self.up_2(new_features[0])
        if self.conv_size > 0:
            new_features[0] = torch.cat((new_features[0], conv_embeds), 1)
        # new_features[1] = torch.cat((new_features[1], conv_1), 1)
        # features[2] = torch.cat((features[2], conv_2), 1)
        # features[3] = torch.cat((features[3], conv_3), 1)

        # features[0] = features[0] + conv_embeds

        new_features[-1] = self.PPN(new_features[-1])
        # x = self.head(features[-1])
        x = self.head(self.FPN(new_features))

        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def decoder_linear(self, x, conv_embeds):
        x = self.classifier(x)
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        conv_embeds = 0
        if self.conv_size > 0:
            conv_embeds = self.encoder_conv(x)
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(x, conv_embeds)
        features = self.get_features(x)
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(features, conv_embeds)
        x = self.decoder_linear(features[-1], conv_embeds)

        # x = self.decoder_upernet(x[1])

        return x, (conv_embeds, features[-1])
