import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GFM_swin import build_swin
from UPerNet.UPerNetHead import UperNetHead
from scipy import interpolate


def remap_pretrained_keys_swin(model, checkpoint_model):
    state_dict = model.state_dict()

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:

                    src_size = int(L1**0.5)
                    dst_size = int(L2**0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r**n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = (
                            relative_position_bias_table_pretrained[:, i]
                            .view(src_size, src_size)
                            .float()
                            .numpy()
                        )
                        f_cubic = interpolate.interp2d(x, y, z, kind="cubic")
                        all_rel_pos_bias.append(
                            torch.Tensor(f_cubic(dx, dy))
                            .contiguous()
                            .view(-1, 1)
                            .to(relative_position_bias_table_pretrained.device)
                        )

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in checkpoint_model.keys() if "relative_position_index" in k
    ]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [
        k for k in checkpoint_model.keys() if "relative_coords_table" in k
    ]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model


def load_pretrained(
    finetune,
    model,
):
    checkpoint = torch.load(finetune, map_location="cpu")
    checkpoint_model = checkpoint["model"]

    checkpoint_model = {
        k.replace("encoder.", ""): v
        for k, v in checkpoint_model.items()
        if k.startswith("encoder.")
    }

    checkpoint = remap_pretrained_keys_swin(model, checkpoint_model)

    if (
        model.patch_embed.proj.weight.shape
        != checkpoint_model["patch_embed.proj.weight"].shape
    ):
        temp = model.patch_embed.proj.weight.data.cpu()
        if checkpoint_model["patch_embed.proj.weight"].shape[1] == 1:
            # greyscale pretrained model
            temp = checkpoint_model["patch_embed.proj.weight"].repeat(
                1, temp.shape[1], 1, 1
            )
        elif (
            checkpoint_model["patch_embed.proj.weight"].shape[1] == 12
            and temp.shape[1] == 3
        ):
            # For 12 band pretrained, the order is CGBR...
            temp[:, :, :, :] = checkpoint_model["patch_embed.proj.weight"][
                :, [3, 2, 1], :, :
            ]
        elif checkpoint_model["patch_embed.proj.weight"].shape[1] == 8:
            # SpaceNet superres pretrain
            min_channels = min(
                temp.shape[1], checkpoint_model["patch_embed.proj.weight"].shape[1]
            )
            temp[:, :min_channels, :, :] = checkpoint_model["patch_embed.proj.weight"][
                :, :min_channels, :, :
            ]
        else:
            temp[:, [3, 2, 1], :, :] = checkpoint_model["patch_embed.proj.weight"]
        checkpoint_model["patch_embed.proj.weight"] = temp
    msg = model.load_state_dict(checkpoint_model, strict=False)
    # if 'finetune' in config.PRETRAINED:
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     for param in model.head.parameters():
    #         param.requires_grad = True
    #     for n, param in model.named_parameters():
    #         layers = ['layers.0.blocks.0.attn.relative_position_index', 'layers.0.blocks.1.attn_mask', 'layers.0.blocks.1.attn.relative_position_index', 'layers.1.blocks.0.attn.relative_position_index',
    #             'layers.1.blocks.1.attn_mask', 'layers.1.blocks.1.attn.relative_position_index', 'layers.2.blocks.0.attn.relative_position_index', 'layers.2.blocks.1.attn_mask', 'layers.2.blocks.1.attn.relative_position_index',
    #              'layers.2.blocks.2.attn.relative_position_index', 'layers.2.blocks.3.attn_mask', 'layers.2.blocks.3.attn.relative_position_index', 'layers.2.blocks.4.attn.relative_position_index', 'layers.2.blocks.5.attn_mask']
    #         if 'layer3' in n or n in layers:
    #             param.requires_grad = True

    del checkpoint
    torch.cuda.empty_cache()


class GFMModel(nn.Module):

    def __init__(self, args, device) -> None:
        super().__init__()
        self.model_size = args.model.split("_")[1]
        self.conv_size = 0

        self.feat_extr = build_swin(args)
        load_pretrained(args.finetune, self.feat_extr)

        self.feat_extr.eval()  # type: ignore
        self.feat_extr.to(device)  # type: ignore

        for p in self.feat_extr.parameters():
            p.requires_grad = False

        feature_channels = [256, 512, 1024, 1024]

        self.input_size = (args.input_size, args.input_size)

        if (
            args.dataset_type == "geobench_eurosat"
            or args.dataset_type == "rgb"
            or args.dataset_type == "geobench_bigearthnet"
            or args.dataset_type == "geobench_forestnet"
            or args.dataset_type == "geobench_so2sat"
        ):
            self.task = "classification"
            # self.classifier = LinearClassifier(
            #     self.embed_dim, self.num_patches, self.num_patches, args.nb_classes
            # )
            self.classification_head = nn.Linear(feature_channels[-1], args.nb_classes)
        else:
            self.task = "segmentation"

            config = {
                "pool_scales": [1, 2, 3, 6],
                "hidden_size": 512,
                "num_labels": args.nb_classes,
                "initializer_range": 0.02,
            }

            self.upernet_head = UperNetHead(config, feature_channels)

        # self.channel_project = nn.Linear(3, 10)  # Define a learnable linear layer

        # self.classification_head = nn.Linear(feature_channels[-1], args.nb_classes)

    def decoder_upernet(self, features):

        new_features = []

        new_features.append(features[0].reshape(-1, 32, 32, 256))
        new_features.append(features[1].reshape(-1, 16, 16, 512))
        new_features.append(features[2].reshape(-1, 8, 8, 1024))
        new_features.append(features[3].reshape(-1, 8, 8, 1024))
        # swin_embeds = conv_embeds[0].reshape(-1, 128, 128, 96)

        new_features[0] = torch.permute(new_features[0], (0, 3, 1, 2))
        new_features[1] = torch.permute(new_features[1], (0, 3, 1, 2))
        new_features[2] = torch.permute(new_features[2], (0, 3, 1, 2))
        new_features[3] = torch.permute(new_features[3], (0, 3, 1, 2))

        x = self.upernet_head(new_features)

        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x):

        if x.shape[1] > 3:
            x = x[:, :3]

        x = F.interpolate(x, size=256, mode="bilinear", align_corners=True)

        # if x.shape[1] == 11:
        #     x = F.pad(x, (0, 0, 0, 0, 0, 1), "constant", 0)
        # elif x.shape[1] == 3:
        #     x = torch.cat(
        #         [
        #             torch.zeros_like(x[:, :1]),  # zero channel at index 0
        #             x[:, :4],
        #         ],
        #         dim=1,
        #     )
        #     x = F.pad(x, (0, 0, 0, 0, 0, 8), "constant", 0)
        # elif x.shape[1] == 4:
        #     x = torch.cat(
        #         [
        #             torch.zeros_like(x[:, :1]),  # zero channel at index 0
        #             x[:, :5],
        #         ],
        #         dim=1,
        #     )
        #     x = F.pad(x, (0, 0, 0, 0, 0, 7), "constant", 0)
        # elif x.shape[1] == 13:
        #     x = torch.cat([x[:, :10], x[:, 11:]], dim=1)
        # elif x.shape[1] == 10:
        #     x = torch.cat(
        #         [
        #             torch.zeros_like(x[:, :1]),
        #             x[:, :9],
        #             torch.zeros_like(x[:, :1]),
        #             x[:, 9:],
        #         ],
        #         dim=1,
        #     )
        if self.task == "classification":
            features = self.feat_extr.forward_features_cls(x)
            x = self.classification_head(features)
        else:
            features = self.feat_extr.forward_features_seg(x)
            x = self.decoder_upernet(features)

        return x, (features, features[-1])
