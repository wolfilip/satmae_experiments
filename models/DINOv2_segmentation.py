from math import e, sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from UPerNet.UPerNetHead import UperNetHead
from util.LiFT_module import LiFT
from util.convolutions import encoder


class LiteFusionSeg(nn.Module):
    def __init__(
        self, num_classes, token_dim=768, conv_dim=256, up_hw=256, mid_dim=256
    ):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, conv_dim)
        self.fuse_reduce = nn.Sequential(
            nn.Conv2d(conv_dim + conv_dim, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
        )

        # two depthwise-separable blocks
        def dw_sep(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, in_ch, 1, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
            )

        self.block1 = dw_sep(mid_dim)
        self.block2 = dw_sep(mid_dim)
        self.head = nn.Conv2d(mid_dim, num_classes, 1)

        self.up_hw = up_hw  # e.g., 256

    def forward(self, conv_feats, tokens):
        # conv_feats: (B, Cc, H, W) with H=W=256, Cc=256
        # tokens: (B, 256, Ct) with Ct=768
        B, _, H, W = conv_feats.shape
        t = self.token_proj(tokens)  # (B, 256, 256)
        t = t.permute(0, 2, 1).reshape(
            B, -1, int(sqrt(t.shape[1])), int(sqrt(t.shape[1]))
        )  # (B, 256, 16, 16)
        t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        x = torch.cat([conv_feats, t], dim=1)  # (B, 512, 256, 256)
        x = self.fuse_reduce(x)  # (B, mid_dim, 256, 256)
        x = self.block1(x)
        x = self.block2(x)
        logits = self.head(x)  # (B, num_classes, 256, 256)
        return logits


class DINOv2Segmenter(nn.Module):

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
            # self.feat_extr = torch.hub.load(
            #     "panopticon-FM/panopticon", "panopticon_vitb14"
            # )
        if self.model_size == "base_reg":
            self.feat_extr = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14_reg"
            )
        if self.model_size == "largev3sat":
            self.feat_extr = torch.hub.load(
                "/home/filip/dinov3",
                "dinov3_vitl16",
                source="local",
                weights="https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaWk0NzU3OGJhOThobmsweDQzMnkzYWxwIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjIzMzUwMTV9fX1dfQ__&Signature=hiwjT0RLVzPs9NXYXyISTcjdFvSqPvW8NybKkdy--%7Eg36qNpmHcBFaeBhHvQcaDnzhzimnDC4Dhcm08L05EUTimw1B0bf8tYB-ZSw17W9E39NttD0a2f7nnoEBu1FAoA1ZHZLQoHjrmX05fPBo-gpyCG-PkVoYW0Tsp5AqYXtZ8iWtButTeaeVCox3eGlnI4tLQfwfsn-V1oCowWW-v%7EcRgnaDNQXZpkPnAVdWIpc1nNHl6pqA5ITBmIk4HrXLUY1vpQ4FDtc1uOFWpTl2ioLEusuQ-jLzz63tFSuq%7EzMGxk7wjs7ryD55-WMQl4BJOrmRLGfIcNAgmsmISZXfp-Kg__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1156481202648884",
            )
        if self.model_size == "basev3":
            self.feat_extr = torch.hub.load(
                "/home/filip/dinov3",
                "dinov3_vitb16",
                source="local",
                weights="https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaWk0NzU3OGJhOThobmsweDQzMnkzYWxwIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjIzMzUwMTV9fX1dfQ__&Signature=hiwjT0RLVzPs9NXYXyISTcjdFvSqPvW8NybKkdy--%7Eg36qNpmHcBFaeBhHvQcaDnzhzimnDC4Dhcm08L05EUTimw1B0bf8tYB-ZSw17W9E39NttD0a2f7nnoEBu1FAoA1ZHZLQoHjrmX05fPBo-gpyCG-PkVoYW0Tsp5AqYXtZ8iWtButTeaeVCox3eGlnI4tLQfwfsn-V1oCowWW-v%7EcRgnaDNQXZpkPnAVdWIpc1nNHl6pqA5ITBmIk4HrXLUY1vpQ4FDtc1uOFWpTl2ioLEusuQ-jLzz63tFSuq%7EzMGxk7wjs7ryD55-WMQl4BJOrmRLGfIcNAgmsmISZXfp-Kg__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1156481202648884",
            )
        if self.model_size == "basev3conv":
            self.feat_extr = torch.hub.load(
                "/home/filip/dinov3",
                "dinov3_convnext_base",
                source="local",
                weights="https://dinov3.llamameta.net/dinov3_convnext_base/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZHIyaTlsMmJncjR0dzFrbXRydW5lMG9zIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTc0OTQxOTV9fX1dfQ__&Signature=fxQQBREKnOwqASyDbX4qnjp%7E4ivn1khbIDnS7aw%7EYYhisStCXF38PkuEcI2LTe0i6VKW6HnHKLnnqFTjAKLUT4FQwtCcVnuwQnPDcKsLjTYOZXLjSmf8S1%7Er07yFmljG06w26ZNaxl9pRq%7E6TjDdRcBv7TOAKtheH4xDYSvpYWENUzk8twsfgOtVdkxUAaQeJjbDnFcmRhMMIf5yx-fb5s2nPBZrodzTetMZIaJh2aca-HCOLqeDjjklnDGQob7tzg5qs6oMshoYiyi20y-HE%7EsQndgiisG-zUg-RajQ735IPfSPhfUg2ROOvuLHYSBuWaCXZEofe-o02v3HHn4mRg__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1965232937651613",
            )

        self.feat_extr.eval()  # type: ignore
        self.feat_extr.to(device)  # type: ignore
        self.device = device
        if self.model_size == "largev3sat" or self.model_size == "basev3":
            self.patch_size = 16
        else:
            self.patch_size = 14

        for p in self.feat_extr.parameters():  # type: ignore
            p.requires_grad = False

        # upernet stuff
        if self.model_size == "small":
            self.embed_dim = 384
            feature_channels = [
                self.embed_dim + self.conv_size,
                self.embed_dim,
            ]
        elif self.model_size == "base" or self.model_size == "basev3":
            self.embed_dim = 768
            feature_channels = [
                self.embed_dim + self.conv_size,
                self.embed_dim,
                self.embed_dim,
                self.embed_dim,
            ]
        elif self.model_size == "basev3conv":
            feature_channels = [128, 256, 512, 1024]
        else:
            self.embed_dim = 1024
            feature_channels = [
                self.embed_dim,
                self.embed_dim,
                self.embed_dim,
                self.embed_dim,
            ]

        self.input_size = (args.input_size, args.input_size)
        self.num_patches = int(self.input_size[0] / self.patch_size)

        self.do_interpolation = False

        if args.input_size % self.patch_size != 0:
            self.do_interpolation = True

        if (
            args.dataset_type == "geobench_eurosat"
            or args.dataset_type == "rgb"
            or args.dataset_type == "geobench_so2sat"
            or args.dataset_type == "geobench_bigearthnet"
        ):
            self.task = "classification"
            # self.classifier = LinearClassifier(
            #     self.embed_dim, self.num_patches, self.num_patches, args.nb_classes
            # )
            self.classification_head = nn.Linear(self.embed_dim, args.nb_classes)
        else:
            self.task = "segmentation"

            # self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

            # config = {
            #     "pool_scales": [1, 2, 3, 6],
            #     "hidden_size": 512,
            #     "num_labels": args.nb_classes,
            #     "initializer_range": 0.02,
            # }

            # self.upernet_head = UperNetHead(config, feature_channels)

            # # if self.conv_size > 0:
            # if args.dataset_type == "spacenet":
            #     self.up = nn.Upsample(
            #         size=(64, 64), mode="bilinear", align_corners=True
            #     )
            # elif (
            #     args.dataset_type == "sen1floods11"
            #     or args.dataset_type == "vaihingen"
            #     or args.dataset_type == "potsdam"
            # ):
            #     self.up = nn.Upsample(
            #         size=(144, 144), mode="bilinear", align_corners=True
            #     )
            # elif args.dataset_type == "isaid":
            #     self.up = nn.Upsample(
            #         size=(256, 256), mode="bilinear", align_corners=True
            #     )
            # elif args.dataset_type == "mass_roads":
            #     self.up = nn.Upsample(
            #         size=(428, 428), mode="bilinear", align_corners=True
            #     )

            self.encoder = encoder(
                3,
                256 // 2,
                kernel_size=1,
                ks_res=1,
                num_layers=2,
            )
            self.sem_encoder = encoder(
                3,
                256 // 2,
                kernel_size=3,
                ks_res=3,
                num_layers=2,
            )

            self.lite_segmenter = LiteFusionSeg(num_classes=args.nb_classes)
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

        ############# LiFT

        # lift_path = "/home/filip/lift/output/lift_fmow_trains_layer_3/vit_base_patch16_224_0.001_cosine_aug_256/lift_10.pth"

        # self.lift = LiFT(1024, 16)
        # state_dict = torch.load(lift_path)
        # self.lift.eval()

        # for k in list(state_dict.keys()):
        #     if k.startswith("module."):
        #         state_dict[k[7:]] = state_dict[k]
        #         del state_dict[k]

        # for p in self.lift.parameters():
        #     p.requires_grad = False

        # self.lift.load_state_dict(state_dict)
        # self.lift.to("cuda")
        # print("Loaded LiFT module from: " + lift_path)

        # lift_path = "/home/filip/lift/output/lift_fmow_rgb/dino_vits16_0.001_cosine_aug_448/lift.pth"

        # self.lift = LiFT(384, 16, pre_shape=False)
        # state_dict = torch.load(lift_path)

        # for k in list(state_dict.keys()):
        #     if k.startswith("module."):
        #         state_dict[k[7:]] = state_dict[k]
        #         del state_dict[k]

        # self.lift.load_state_dict(state_dict)
        # self.lift.to(device)
        # print("Loaded LiFT module from: " + lift_path)

        # self.lift.eval()

        # self.linear_transform_down = nn.Linear(768, 384)
        # self.linear_transform_up = nn.Linear(384, 768)

        ############# LiFT

        if self.conv_size == 32:
            self.conv_layers = nn.Sequential(
                # Conv1: Input [B, 3, 224, 224] -> Output [B, 64, 112, 112]
                nn.Conv2d(
                    in_channels=7, out_channels=16, kernel_size=7, stride=2, padding=3
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
        # if self.do_interpolation:
        # if imgs.shape[-1] == 500:
        #     imgs = F.interpolate(imgs, size=496, mode="bilinear", align_corners=True)
        # # elif imgs.shape[-1] == 512 or imgs.shape[-1] == 500:
        # #     imgs = F.interpolate(
        # #         imgs, size=504, mode="bilinear", align_corners=True
        # #     )
        if imgs.shape[-1] == 64:
            imgs = F.interpolate(imgs, size=56, mode="bilinear", align_corners=True)
        # elif imgs.shape[-1] == 256:
        #     imgs = F.interpolate(imgs, size=252, mode="bilinear", align_corners=True)
        # # elif imgs.shape[-1] == 1500:
        # #     imgs = F.interpolate(
        # #         imgs, size=1498, mode="bilinear", align_corners=True
        # #     )
        # if imgs.shape[-1] == 320:
        #     imgs = F.interpolate(imgs, size=308, mode="bilinear", align_corners=True)
        # elif imgs.shape[-1] == 128:
        #     imgs = F.interpolate(imgs, size=126, mode="bilinear", align_corners=True)
        if imgs.shape[-1] == 120:
            imgs = F.interpolate(imgs, size=112, mode="bilinear", align_corners=True)
        # elif imgs.shape[-1] == 32:
        #     imgs = F.interpolate(imgs, size=28, mode="bilinear", align_corners=True)

        with torch.no_grad():
            if self.task == "classification":
                out = self.feat_extr.forward_features(imgs)  # type: ignore
                cls = out["x_norm_clstoken"]
                out = cls
            else:
                # if self.layer_num == "last":
                if (
                    self.model_size == "base"
                    or self.model_size == "small"
                    or self.model_size == "basev3"
                ):
                    patch = self.feat_extr.get_intermediate_layers(x=imgs, n=(3, 5, 8, 11))  # type: ignore
                elif self.model_size == "basev3conv":
                    patch = self.feat_extr.get_intermediate_layers(imgs, n=(0, 1, 2, 3))  # type: ignore
                else:
                    patch = self.feat_extr.get_intermediate_layers(x=imgs, n=(3, 9, 17, 23))  # type: ignore
                out = patch

            # layers.append(patch)
            # layers.append(patch)
            # cls = out["x_norm_clstoken"]
            # elif self.layer_num == "first":

        # elif self.layer_num == "avg":
        #     pass
        return out

    def encoder_conv(self, x):

        conv_embeds = self.conv_layers(x)
        conv_embeds = self.up(conv_embeds)

        return conv_embeds

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
        # swin_embeds = conv_embeds[0].reshape(-1, 128, 128, 96)

        new_features[0] = torch.permute(new_features[0], (0, 3, 1, 2))
        new_features[1] = torch.permute(new_features[1], (0, 3, 1, 2))
        new_features[2] = torch.permute(new_features[2], (0, 3, 1, 2))
        new_features[3] = torch.permute(new_features[3], (0, 3, 1, 2))
        # swin_embeds = torch.permute(swin_embeds, (0, 3, 1, 2))

        new_features[-1] = F.interpolate(
            new_features[-1], scale_factor=0.5, mode="bilinear", align_corners=True
        )
        # features[2] = self.up_1(features[2])
        new_features[1] = self.up_1(new_features[1])
        new_features[0] = self.up_2(new_features[0])

        # new_features[0] = torch.cat((new_features[0], self.up(swin_embeds)), 1)

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

    def decoder_upernet_conv(self, features, conv_embeds):

        # conv_1 = self.relu(self.bn(self.conv(conv_embeds)))
        # conv_2 = self.relu(self.bn(self.conv(conv_1)))
        # conv_3 = self.relu(self.bn(self.conv(conv_2)))
        new_features = []

        new_features.append(
            features[0].reshape(
                -1,
                int(sqrt(features[0].shape[1])),
                int(sqrt(features[0].shape[1])),
                128,
            )
        )
        new_features.append(
            features[1].reshape(
                -1,
                int(sqrt(features[1].shape[1])),
                int(sqrt(features[1].shape[1])),
                256,
            )
        )
        new_features.append(
            features[2].reshape(
                -1,
                int(sqrt(int(features[2].shape[1]))),
                int(sqrt(int(features[2].shape[1]))),
                512,
            )
        )
        new_features.append(
            features[3].reshape(
                -1,
                int(sqrt(int(features[3].shape[1]))),
                int(sqrt(int(features[3].shape[1]))),
                1024,
            )
        )
        # swin_embeds = conv_embeds[0].reshape(-1, 128, 128, 96)

        new_features[0] = torch.permute(new_features[0], (0, 3, 1, 2))
        new_features[1] = torch.permute(new_features[1], (0, 3, 1, 2))
        new_features[2] = torch.permute(new_features[2], (0, 3, 1, 2))
        new_features[3] = torch.permute(new_features[3], (0, 3, 1, 2))
        # swin_embeds = torch.permute(swin_embeds, (0, 3, 1, 2))

        # new_features[-1] = F.interpolate(
        #     new_features[-1], scale_factor=0.5, mode="bilinear", align_corners=True
        # )
        # # features[2] = self.up_1(features[2])
        # new_features[1] = self.up_1(new_features[1])
        # new_features[0] = self.up_2(new_features[0])

        # new_features[0] = torch.cat((new_features[0], self.up(swin_embeds)), 1)

        # if self.conv_size > 0:
        #     new_features[0] = torch.cat((new_features[0], conv_embeds), 1)
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

        if x.shape[1] > 3:
            x = x[:, 1:4].flip(1)

        # conv_embeds = 0
        # if self.conv_size > 0:
        #     conv_embeds = self.encoder_conv(x)
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(x, conv_embeds)
        image_embeds_1 = self.encoder(x)
        image_embeds_2 = self.sem_encoder(x)
        image_embeds = torch.cat([image_embeds_1, image_embeds_2], dim=1)
        features = self.get_features(x)

        ######## LIFT ###########

        # new_features = []

        # new_features.append(
        #     self.linear_transform_down(
        #         features[0].reshape(
        #             -1, self.num_patches, self.num_patches, self.embed_dim
        #         )
        #     )
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

        # with torch.no_grad():
        #     new_features[0] = torch.permute(
        #         self.linear_transform_up(
        #             self.lift(chunks[0], new_features[0]).reshape(-1, 64, 64, 384)
        #         ),
        #         (0, 3, 1, 2),
        #     )

        # with torch.no_grad():
        #     features[0] = self.lift(x, features[0])

        ######## LIFT ###########

        # x = self.encoder_forward(x)
        # if self.task == "classification":
        #     x = self.classification_head(features)
        # else:
        #     x = self.decoder_upernet(features, conv_embeds)
        x = self.lite_segmenter(image_embeds, features[-1])
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

        # new_features[-1] = F.interpolate(
        #     new_features[-1], scale_factor=0.5, mode="bilinear", align_corners=True
        # )
        # features[2] = self.up_1(features[2])
        # new_features[1] = self.up_1(new_features[1])

        # x = self.upernet_head(new_features)
        # x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

        # x = self.classification_head(features)
        # x = self.decoder_linear(features[-1], conv_embeds)

        # x = self.decoder_upernet(x[1])

        return x, (0, features)
        # return x
