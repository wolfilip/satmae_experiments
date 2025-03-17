import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=512):
        super(FPN_fuse, self).__init__()
        self.conv1x1 = nn.ModuleList(
            [
                nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                for ft_size in feature_channels[1:]
            ]
        )
        self.smooth_conv = nn.ModuleList(
            [nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
            * (len(feature_channels) - 1)
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(
                len(feature_channels) * fpn_out,
                fpn_out,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):

        features[1:] = [
            conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)
        ]
        P = [
            up_and_add(features[i], features[i - 1])
            for i in reversed(range(1, len(features)))
        ]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [
            F.interpolate(feature, size=(H, W), mode="bilinear", align_corners=True)
            for feature in P[1:]
        ]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


def up_and_add(x, y):
    return (
        F.interpolate(
            x, size=(y.size(2), y.size(3)), mode="bilinear", align_corners=True
        )
        + y
    )
