import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv_1 = DoubleConv(in_channels // 2 + 32, out_channels // 2)

    def forward(self, x, imgs_1):
        x = self.up(x)
        x = torch.cat([x, imgs_1], dim=1)
        x = self.conv_1(x)
        return x


class LiFT(nn.Module):
    def __init__(self, in_channels, patch_size, pre_shape=True, post_shape=True):
        super(LiFT, self).__init__()
        self.patch_size = patch_size
        self.pre_shape = pre_shape
        self.post_shape = post_shape

        self.up1 = Up(in_channels + 32, in_channels)
        self.outc = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.image_convs_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        if patch_size == 8:
            self.scale_adapter = nn.Identity()
        elif patch_size == 16:
            self.scale_adapter = nn.MaxPool2d(2, 2)
        else:
            print("ERROR: patch size %i not currently supported" % patch_size)
            exit()
        self.image_convs_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    # [B, T, C] --> [B, C, H, W]
    def run_pre_shape(self, imgs, x):
        H = int(imgs.shape[2] / self.patch_size)
        W = int(imgs.shape[3] / self.patch_size)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1, H, W)
        return x

    # [B, C, H, W] --> [B, T, C]
    def run_post_shape(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, imgs, x):
        if self.pre_shape:
            x = self.run_pre_shape(imgs, x)
        imgs_1 = self.image_convs_1(imgs)
        imgs_1 = self.scale_adapter(imgs_1)
        imgs_2 = self.image_convs_2(imgs_1)
        # Enable the following if working with both --imsize 56 and --patch_size 16
        # if(x.shape[2] != imgs_2.shape[2]):
        #     imgs_1 = self.image_convs_1(imgs[:,:,2:-2,2:-2])
        #     imgs_1 = self.scale_adapter(imgs_1)
        #     imgs_2 = self.image_convs_2(imgs_1)
        x = torch.cat([x, imgs_2], dim=1)
        x = self.up1(x, imgs_1)
        logits = self.outc(x)
        if self.post_shape:
            logits = self.run_post_shape(logits)
        return logits
