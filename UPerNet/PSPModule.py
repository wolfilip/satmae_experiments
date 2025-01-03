import torch
import torch.nn as nn
import torch.nn.functional as F


class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList(
            [self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + (out_channels * len(bin_sizes)),
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend(
            [
                F.interpolate(
                    stage(features), size=(h, w), mode="bilinear", align_corners=True
                )
                for stage in self.stages
            ]
        )
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output
