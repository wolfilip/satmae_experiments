import re
from collections import OrderedDict
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# from UPerNet.UPerNetHead import UperNetHead
# from functools import partial

# from transformers.models.mask2former.modeling_mask2former import (
#     Mask2FormerConfig,
#     Mask2FormerForUniversalSegmentation,
# )

# from transformers import (
#     AutoModelForImageClassification,
#     AutoFeatureExtractor,
#     AutoModel,
# )

# from torchvision import models as torchvision_models
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    RPNHead,
)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.image_list import ImageList


from util.LiFT_module import LiFT


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class DINOv2Detector(nn.Module):

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
        # if self.model_size == "base":
        #     self.feat_extr = torch.hub.load(
        #         "/home/filip/pretrained_weights/", "vitl16_reg4_SimDINOv2_100ep.pth"
        #     )
        if self.model_size == "base_reg":
            self.feat_extr = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14_reg"
            )
        if self.model_size == "large":

            # def revert_block_chunk_weight(state_dict):
            #     # convert blocks.chunkid.id.* to blocks.id.*: blocks.3.22. to blocks.22.
            #     return {
            #         re.sub(r"blocks\.(\d+)\.(\d+)\.", r"blocks.\2.", k): v
            #         for k, v in state_dict.items()
            #     }

            # ckpt = torch.load(
            #     "/home/filip/pretrained_weights/vitl16_reg4_SimDINOv2_100ep.pth",
            #     map_location="cpu",
            # )["teacher"]

            # ckpt = {
            #     k.removeprefix("backbone."): v
            #     for k, v in ckpt.items()
            #     if k.startswith("backbone")
            # }
            # ckpt = revert_block_chunk_weight(ckpt)
            # # ckpt = timm.models.vision_transformer.checkpoint_filter_fn(ckpt, model)

            # print(timm.list_models(pretrained=True))

            # self.feat_extr = timm.models.vision_transformer.VisionTransformer(
            #     embed_dim=1024,
            #     depth=24,
            #     num_heads=16,
            #     mlp_ratio=4,
            #     qkv_bias=True,
            #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # )
            # self.feat_extr.load_state_dict(ckpt)
            self.feat_extr = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            # self.feat_extr.load_state_dict(
            #     torch.load(
            #         "/home/filip/pretrained_weights/vitl16_reg4_SimDINOv2_100ep.pth"
            #     )
            # )
        # f self.model_size == "large":
        #     self.feat_extr = torch.hub.load(
        #         "/home/filip/pretrained_weights/",
        #         "vitl16_reg4_SimDINOv2_100ep.pth",
        #         source="local",
        #     )i
        self.feat_extr.eval()  # type: ignore
        self.feat_extr.to(device)  # type: ignore
        self.device = device
        self.patch_size = 14

        if self.model_size == "small":
            self.embed_dim = 384
        elif self.model_size == "base":
            self.embed_dim = 768
        else:
            self.embed_dim = 1024

        # self.detection_head = FasterRCNNHead(
        #     self.feat_extr, args.nb_classes, self.embed_dim
        # )
        self.input_size = (args.input_size, args.input_size)
        self.num_patches = int(self.input_size[0] / self.patch_size)

        self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        anchor_generator = AnchorGenerator(
            sizes=(
                (32,),  # Feature map "0"
                (64,),  # Feature map "1"
                (128,),  # Feature map "2"
                (256,),  # Feature map "3"
            ),
            aspect_ratios=(
                (0.5, 1.0, 2.0),  # Feature map "0"
                (0.5, 1.0, 2.0),  # Feature map "1"
                (0.5, 1.0, 2.0),  # Feature map "2"
                (0.5, 1.0, 2.0),  # Feature map "3"
            ),
        )
        rpn_head = RPNHead(
            self.embed_dim, anchor_generator.num_anchors_per_location()[0]
        )

        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_score_thresh = 0.0
        rpn_pre_nms_top_n_train = 2000
        rpn_pre_nms_top_n_test = 1000
        rpn_post_nms_top_n_train = 2000
        rpn_post_nms_top_n_test = 1000
        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test
        )
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test
        )

        self.rpn = RegionProposalNetwork(
            anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=[
                "0",
                "1",
                "2",
                "3",
            ],  # Use all feature maps from the backbone
            output_size=7,
            sampling_ratio=2,
        )

        resolution = roi_pooler.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(self.embed_dim * resolution**2, representation_size)

        representation_size = 1024
        box_predictor = FastRCNNPredictor(representation_size, args.nb_classes)

        box_score_thresh = 0.05
        box_nms_thresh = 0.5
        box_detections_per_img = 100
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = None

        self.roi_heads = RoIHeads(
            # Box
            roi_pooler,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

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

    def decoder_fasterrcnn(self, images, features, targets, conv_embeds):

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
        # # features[2] = self.up_1(features[2])
        new_features[1] = self.up_1(new_features[1])
        new_features[0] = self.up_2(new_features[0])

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

        images = ImageList(images, [(512, 512) for _ in range(len(images))])

        features = OrderedDict(
            {str(i): new_features[i] for i in range(len(new_features))}
        )

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        # x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return detector_losses

    def get_features(self, imgs):
        # layer = self.layer_num[0] # TODO: make it a list
        # layers = []
        if imgs.shape[-1] == 512:
            imgs = F.interpolate(imgs, size=504, mode="bilinear", align_corners=True)

        with torch.no_grad():
            if self.model_size == "base" or self.model_size == "small":
                patch = self.feat_extr.get_intermediate_layers(imgs, (3, 5, 8, 11))  # type: ignore
            else:
                patch = self.feat_extr.get_intermediate_layers(imgs, (3, 9, 17, 23))  # type: ignore
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

    def forward(self, x, targets):

        # chunks = torch.split(x, [3, 7], dim=1)

        # bla = self.linear_layer(chunks[1])

        # swin_embeds = self.swin_encoder(
        #     self.linear_7_to_3(chunks[1]), output_hidden_states=True
        # ).hidden_states
        # swin_embeds = 0

        conv_embeds = 0
        if self.conv_size > 0:
            conv_embeds = self.encoder_conv(x)
        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(x, conv_embeds)
        features = self.get_features(x)

        x = self.decoder_fasterrcnn(x, features, targets, conv_embeds)

        # detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

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

        ######## LIFT ###########

        # x = self.encoder_forward(x)
        # x = self.decoder_upernet(features, conv_embeds)
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

        # x = self.detection_head(x, targets)

        # return x, (conv_embeds, features[-1])
        return x


class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, out_channels):
        super(BackboneWithFPN, self).__init__()
        self.backbone = backbone
        self.out_channels = out_channels

    def forward(self, imgs):
        # Extract multiple feature maps from the backbone

        # if self.backbone.model_size == "base" or self.backbone.model_size == "small":
        features = self.backbone.get_intermediate_layers(imgs, (3, 5, 8, 11))  # type: ignore
        # else:
        #     features = self.backbone.get_intermediate_layers(imgs, (3, 9, 17, 23))  # type: ignore
        # features = self.backbone.get_features(x)
        return {
            "0": features[0],  # First feature map
            "1": features[1],  # Second feature map
            "2": features[2],  # Third feature map
            "3": features[3],  # Fourth feature map
        }  # Use all feature maps for multi-scale detection


class FasterRCNNHead(nn.Module):
    def __init__(self, backbone, num_classes, embed_dim):
        super(FasterRCNNHead, self).__init__()
        """
        Initializes the Faster R-CNN head with the given backbone and number of classes.

        Args:
            backbone (nn.Module): The feature extractor backbone (e.g., DINOv2).
            num_classes (int): The number of classes for object detection.
        """
        # Wrap the backbone to include out_channels and format the output
        self.out_channels = (
            embed_dim  # Set this to match the output channels of your backbone
        )
        self.backbone = BackboneWithFPN(backbone, self.out_channels)

        # Define the anchor generator with sizes and aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )

        # Define the region of interest (RoI) aligner
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=[
                "0",
                "1",
                "2",
                "3",
            ],  # Use all feature maps from the backbone
            output_size=7,
            sampling_ratio=2,
        )

        # Build the Faster R-CNN model
        self.model = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

    def forward(self, features, targets=None):
        """
        Forward pass for the Faster R-CNN head.

        Args:
            features (torch.Tensor): Backbone features to be processed.
            targets (Optional[List[Dict[str, torch.Tensor]]]): Ground truth boxes and labels for training.

        Returns:
            Dict[str, torch.Tensor] or List[Dict[str, torch.Tensor]]: The output of the Faster R-CNN head.
        """
        return self.model(features, targets)
