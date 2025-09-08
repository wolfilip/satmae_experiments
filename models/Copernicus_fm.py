import torch
import torch.nn as nn
import torch.nn.functional as F
from UPerNet.UPerNetHead import UperNetHead
from models.copernicus_fm_model.model_vit import vit_base_patch16


class CopernicusFM(nn.Module):

    def __init__(self, args, device) -> None:
        super().__init__()

        self.feat_extr = vit_base_patch16(
            num_classes=10, global_pool=False, intermediate_indices=(3, 9, 17, 23)
        )
        # load pre-trained weights
        path = args.finetune
        check_point = torch.load(path)
        if "model" in check_point:
            state_dict = check_point["model"]
        else:
            state_dict = check_point
        self.feat_extr.load_state_dict(state_dict, strict=False)

        self.feat_extr.eval()  # type: ignore
        self.feat_extr.to(device)  # type: ignore

        self.patch_size = 16

        for p in self.feat_extr.parameters():  # type: ignore
            p.requires_grad = False

        self.embed_dim = 768
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

    def get_features(self, imgs):
        # layer = self.layer_num[0] # TODO: make it a list
        # layers = []
        # if self.do_interpolation:
        # if imgs.shape[-1] == 500:
        #     imgs = F.interpolate(
        #         imgs, size=512, mode="bilinear", align_corners=True
        #     )
        # elif imgs.shape[-1] == 512 or imgs.shape[-1] == 500:
        #     imgs = F.interpolate(
        #         imgs, size=504, mode="bilinear", align_corners=True
        #     )
        # elif imgs.shape[-1] == 64:
        #     imgs = F.interpolate(imgs, size=56, mode="bilinear", align_corners=True)
        # elif imgs.shape[-1] == 256:
        #     imgs = F.interpolate(
        #         imgs, size=252, mode="bilinear", align_corners=True
        #     )
        # elif imgs.shape[-1] == 1500:
        #     imgs = F.interpolate(
        #         imgs, size=1498, mode="bilinear", align_corners=True
        #     )
        # if imgs.shape[-1] == 320:
        #     imgs = F.interpolate(
        #         imgs, size=308, mode="bilinear", align_corners=True
        #     )

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

    def decoder_upernet(self, features):

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

    def forward(self, x):

        meta = torch.full((1, 4), float("nan"))
        wvs = [490, 560, 665, 705, 740, 783, 842, 865, 945]
        bws = [65, 35, 30, 15, 15, 20, 115, 20, 20]
        language_embed = None
        kernel_size = 3
        input_mode = "spectral"

        features = self.feat_extr(
            x, meta, wvs, bws, language_embed, input_mode, kernel_size
        )

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
        x = self.decoder_upernet(features)
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

        return x, (conv_embeds, features[-1])
        # return x
