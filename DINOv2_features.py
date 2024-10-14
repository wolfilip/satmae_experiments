import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms.functional import resize


class LinearClassifier(nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

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
        self.img_size = args.input_size

        # for block in self.blocks:
        #     for param in block.parameters():
        #         param.requires_grad = False

        # for param in self.patch_embed.parameters():
        #     param.requires_grad = False

        # upernet stuff

        # feature_channels = [1024, 1024]

        # fpn_out = 1024
        # self.input_size = (224, 224)

        # self.PPN = PSPModule(feature_channels[-1])
        # self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        # self.head = nn.Conv2d(fpn_out, args.nb_classes, kernel_size=3, padding=1)
        # self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.classifier = LinearClassifier(1024, 16, 16, args.nb_classes)

    def get_features(self, imgs):
        # layer = self.layer_num[0] # TODO: make it a list
        with torch.no_grad():
            if self.layer_num == "last":
                out = self.feat_extr.forward_features(imgs)  # type: ignore
                patch = out["x_norm_patchtokens"]
                cls = out["x_norm_clstoken"]
            elif self.layer_num == "first":
                patch, cls = self.feat_extr.get_intermediate_layers(  # type: ignore
                    imgs, return_class_token=True
                )[0]
            elif self.layer_num == "avg":
                pass
        out = {"cls": cls, "patch": patch}
        return out

    def decoder_upernet(self, features):

        # conv_1 = self.relu(self.bn(self.conv(conv_embeds)))
        # conv_2 = self.relu(self.bn(self.conv(conv_1)))
        # conv_3 = self.relu(self.bn(self.conv(conv_2)))

        # features[0] = torch.unflatten(features[0], dim=1, sizes=(14, 14))
        # features[1] = torch.unflatten(features[1], dim=1, sizes=(14, 14))
        # features[2] = torch.unflatten(features[2], dim=1, sizes=(14, 14))
        # features[3] = torch.unflatten(features[3], dim=1, sizes=(14, 14))
        # features[4] = torch.unflatten(features[4], dim=1, sizes=(14, 14))
        # features[0] = torch.permute(features[0], (0, 3, 1, 2))
        # features[1] = torch.permute(features[1], (0, 3, 1, 2))
        # features[2] = torch.permute(features[2], (0, 3, 1, 2))
        # features[3] = torch.permute(features[3], (0, 3, 1, 2))
        # features[4] = torch.permute(features[4], (0, 3, 1, 2))

        # features[-1] = F.interpolate(
        #     features[-1], scale_factor=0.5, mode="bilinear", align_corners=True
        # )
        # features[2] = self.up_1(features[2])
        # features[1] = self.up_1(features[1])
        # features[0] = self.up_2(features[0])

        # features[0] = torch.cat((features[0], conv_embeds), 1)
        # features[1] = torch.cat((features[1], conv_1), 1)
        # features[2] = torch.cat((features[2], conv_2), 1)
        # features[3] = torch.cat((features[3], conv_3), 1)

        # features[0] = features[0] + conv_embeds

        features[-1] = self.PPN(features[-1])
        # x = self.head(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=self.input_size, mode="bilinear")
        return x

    def forward(self, x):
        # conv_embeds = self.encoder_conv(x)
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(x, conv_embeds)
        x = self.get_features(x)["patch"]
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(x)
        logits = self.classifier(x)
        logits = F.interpolate(logits, size=224, mode="bilinear", align_corners=False)
        return logits

    def visualize_features(self, x):
        x = self.get_features(x)["patch"].cpu()
        E_patch_norm = rearrange(x, "B L E -> (B L) E").to(torch.float64)
        _, _, V = torch.pca_lowrank(E_patch_norm)
        E_pca_1 = torch.matmul(E_patch_norm, V[:, :1])
        E_pca_1_norm = self.minmax_norm(E_pca_1)
        M_fg = E_pca_1_norm.squeeze() > 0.5
        _, _, V = torch.pca_lowrank(E_patch_norm[M_fg])
        E_pca_3_fg = torch.matmul(E_patch_norm[M_fg], V[:, :3])
        E_pca_3_fg = self.minmax_norm(E_pca_3_fg)
        B, L, _ = x.shape
        Z = B * L
        I_draw = torch.zeros(Z, 3).to(torch.float64)
        I_draw[M_fg] = E_pca_3_fg
        I_draw = rearrange(I_draw, "(B L) C -> B L C", B=B)
        I_draw = rearrange(
            I_draw,
            "B (h w) C -> B h w C",
            h=self.img_size // 14,
            w=self.img_size // 14,
        )
        image_1_pca = I_draw[0]
        image_2_pca = I_draw[1]
        image_1_pca = rearrange(image_1_pca, "H W C -> C H W")
        image_2_pca = rearrange(image_2_pca, "H W C -> C H W")
        image_1_pca = resize(image_1_pca, self.img_size)
        image_2_pca = resize(image_2_pca, self.img_size)
        return image_1_pca, image_2_pca
        # save_image(image_1_pca, "images/image_1_pca.png")
        # save_image(image_2_pca, "images/image_2_pca.png")

    def minmax_norm(self, x):
        """Min-max normalization"""
        return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)
