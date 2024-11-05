import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, model_args, args, device) -> None:
        super().__init__()
        self.model_size = model_args["model_size"]
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
        self.layer_num = model_args["layer"]
        self.feat_extr.eval()  # type: ignore
        self.feat_extr.to(device)  # type: ignore
        self.device = device
        self.patch_size = 14

        # upernet stuff

        feature_channels = [768, 768]

        fpn_out = 768
        self.input_size = (224, 224)

        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        # self.head = nn.Conv2d(fpn_out, args.nb_classes, kernel_size=3, padding=1)
        # self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.classifier = LinearClassifier(768, 16, 16, args.nb_classes)

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
        # )

    def get_features(self, imgs):
        # layer = self.layer_num[0] # TODO: make it a list
        # layers = []
        with torch.no_grad():
            # if self.layer_num == "last":
            # patch = self.feat_extr.get_intermediate_layers(imgs, (3, 11))  # type: ignore
            # layers.append(patch)
            out = self.feat_extr.forward_features(imgs)  # type: ignore
            patch = out["x_norm_patchtokens"]
            # layers.append(patch)
            # cls = out["x_norm_clstoken"]
            # elif self.layer_num == "first":

        # elif self.layer_num == "avg":
        #     pass
        out = patch
        return out

    def encoder_conv(self, x):

        conv_embeds = self.conv_layers_small(x)

        return conv_embeds

    def decoder_upernet(self, features):

        # conv_1 = self.relu(self.bn(self.conv(conv_embeds)))
        # conv_2 = self.relu(self.bn(self.conv(conv_1)))
        # conv_3 = self.relu(self.bn(self.conv(conv_2)))
        new_features = []

        new_features.append(features[0].reshape(-1, 16, 16, 768))
        new_features.append(features[1].reshape(-1, 16, 16, 768))
        # new_features.append(features[2].reshape(-1, 16, 16, 1024))
        # new_features.append(features[3].reshape(-1, 16, 16, 1024))

        new_features[0] = torch.permute(new_features[0], (0, 3, 1, 2))
        new_features[1] = torch.permute(new_features[1], (0, 3, 1, 2))
        # new_features[2] = torch.permute(new_features[2], (0, 3, 1, 2))
        # new_features[3] = torch.permute(new_features[3], (0, 3, 1, 2))
        # features[4] = torch.permute(features[4], (0, 3, 1, 2))

        # new_features[-1] = F.interpolate(
        #     new_features[-1], scale_factor=0.5, mode="bilinear", align_corners=True
        # )
        # features[2] = self.up_1(features[2])
        new_features[1] = self.up_1(new_features[1])
        new_features[0] = self.up_2(new_features[0])

        # features[0] = torch.cat((features[0], conv_embeds), 1)
        # features[1] = torch.cat((features[1], conv_1), 1)
        # features[2] = torch.cat((features[2], conv_2), 1)
        # features[3] = torch.cat((features[3], conv_3), 1)

        # features[0] = features[0] + conv_embeds

        new_features[-1] = self.PPN(new_features[-1])
        # x = self.head(features[-1])
        x = self.head(self.FPN(new_features))

        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        return x

    def decoder_linear(self, x):
        logits = self.classifier(x)
        logits = F.interpolate(logits, size=224, mode="bilinear", align_corners=False)
        return logits

    def forward(self, x):
        # conv_embeds = self.encoder_conv(x)
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(x, conv_embeds)
        features = self.get_features(x)
        # x = self.encoder_forward(x)
        x = self.decoder_linear(features)
        # x = self.decoder_upernet(x[1])

        return x, features
