from segment_anything import sam_model_registry
import torch
import torch.nn as nn
import torch.nn.functional as F

from UPerNet.FPN_fuse import FPN_fuse
from UPerNet.PSPModule import PSPModule
from util.linear_calssifier import LinearClassifier


class SAMHQ(nn.Module):

    def __init__(self, args, device) -> None:
        super().__init__()
        # self.model_size = model_args["model_size"]
        self.model_size = args.model.split("_")[0]
        self.conv_size = int(args.model.split("_")[1])

        model_type = "vit_b"
        samhq_checkpoint = "/home/filip/sam-hq/pretrained_checkpoint/sam_hq_vit_b.pth"
        self.samhq_model = sam_model_registry[model_type](checkpoint=samhq_checkpoint)
        self.samhq_model.eval()
        self.samhq_model.to(device=device)
        # self.samhq_model.image_encoder.img_size = 224
        # self.predictor = SamPredictor(samhq_model)
        self.patch_size = 16

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
                self.embed_dim,
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
        self.num_patches = int(self.input_size[0] / self.patch_size)

        self.do_interpolation = False

        if args.input_size % 14 != 0:
            self.do_interpolation = True

        if args.dataset_type == "euro_sat" or args.dataset_type == "rgb":
            self.task = "classification"
            # self.classifier = LinearClassifier(
            #     self.embed_dim, self.num_patches, self.num_patches, args.nb_classes
            # )
            self.classification_head = nn.Linear(self.embed_dim, args.nb_classes)

        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, args.nb_classes, kernel_size=3, padding=1)
        self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.classifier = LinearClassifier(self.embed_dim, 64, 64, args.nb_classes)

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

    def get_features(self, x):
        # x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)

        with torch.no_grad():
            _, interm_features = self.samhq_model.image_encoder(x)

        return interm_features

    def encoder_conv(self, x):

        conv_embeds = self.conv_layers(x)
        # conv_embeds = self.up(conv_embeds)

        return conv_embeds

    def decoder_upernet(self, features, conv_embeds):

        # conv_1 = self.relu(self.bn(self.conv(conv_embeds)))
        # conv_2 = self.relu(self.bn(self.conv(conv_1)))
        # conv_3 = self.relu(self.bn(self.conv(conv_2)))
        new_features = []

        new_features.append(features[0])
        new_features.append(features[1])
        if self.model_size == "base":
            new_features.append(features[2])
            new_features.append(features[3])

        new_features[0] = torch.permute(new_features[0], (0, 3, 1, 2))
        new_features[1] = torch.permute(new_features[1], (0, 3, 1, 2))
        if self.model_size == "base":
            new_features[2] = torch.permute(new_features[2], (0, 3, 1, 2))
            new_features[3] = torch.permute(new_features[3], (0, 3, 1, 2))
        # features[4] = torch.permute(features[4], (0, 3, 1, 2))

        if self.model_size == "base":
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
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        conv_embeds = 0
        if self.conv_size > 0:
            conv_embeds = self.encoder_conv(x)

        # x = torch.permute(x.squeeze(), (1, 2, 0))
        features = self.get_features(x)

        # for i in range(x.shape[0]):
        #     self.predictor.set_image(x[i].squeeze())
        #     if i == 0:
        #         features = self.predictor.interm_features[3]
        #     else:
        #         features = torch.cat(
        #             (features, self.predictor.interm_features[3]), dim=0
        #         )

        # input_point = np.array([[512, 512]])
        # masks, scores, logits = self.predictor.predict(
        #     point_coords=input_point,
        #     point_labels=np.ones(input_point.shape[0]),
        #     box=None,
        #     multimask_output=False,
        #     hq_token_only=False,
        # )
        x = self.decoder_upernet(features, conv_embeds)

        return x, (0, 0)
