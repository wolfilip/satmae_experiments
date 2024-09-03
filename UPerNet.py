import math
from functools import partial
from typing import List, Optional

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import matplotlib.pyplot as plt
import numpy as np

from util.pos_embed import get_2d_sincos_pos_embed

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # Added by Samar, need default pos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        input_proj = []
        proj_norm = []
        atm_decoders = []
        for i in range(3):
            # FC layer to change ch
            proj = nn.Linear(1024, kwargs["embed_dim"])
            trunc_normal_(proj.weight, std=0.02)

            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            norm = nn.LayerNorm(kwargs["embed_dim"])

            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)
            # decoder layer
            decoder_layer = TPN_DecoderLayer(
                d_model=kwargs["embed_dim"],
                nhead=8,
                dim_feedforward=kwargs["embed_dim"] * 4,
            )
            decoder = TPN_Decoder(decoder_layer, 3)
            self.add_module("decoder_{}".format(i + 1), decoder)
            atm_decoders.append(decoder)

        self.q = nn.Embedding(self.num_classes, kwargs["embed_dim"])
        self.decoder = atm_decoders
        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.class_embed = nn.Linear(kwargs["embed_dim"], self.num_classes + 1)
        self.image_size_train = 384
        self.image_size_val = 384

        self.criterion = SetCriterion(2, losses=["masks", "labels"])

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

    def encoder_forward(self, conv_out, output_switch=None, seg_size=None):

        output_dict = {k: None for k in output_switch.keys()}
        # print(conv_out[-1].shape)

        conv5 = conv_out[-1]
        input_size = conv5.size()
        print(input_size)
        ppm_out = [conv5]
        roi = []  # fake rois, just used for pooling
        for i in range(input_size[0]):  # batch size
            roi.append(
                torch.Tensor([i, 0, 0, input_size[3], input_size[2]]).view(1, -1)
            )  # b, x0, y0, x1, y1
        roi = torch.cat(roi, dim=0).type_as(conv5)
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(
                pool_conv(
                    F.interpolate(
                        pool_scale(conv5, roi.detach()),
                        (input_size[2], input_size[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            )
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        if output_switch["scene"]:  # scene
            output_dict["scene"] = self.scene_head(f)

        if (
            output_switch["object"]
            or output_switch["part"]
            or output_switch["material"]
        ):
            fpn_feature_list = [f]
            for i in reversed(range(len(conv_out) - 1)):
                conv_x = conv_out[i]
                conv_x = self.fpn_in[i](conv_x)  # lateral branch

                f = F.interpolate(
                    f, size=conv_x.size()[2:], mode="bilinear", align_corners=False
                )  # top-down branch
                f = conv_x + f

                fpn_feature_list.append(self.fpn_out[i](f))
            fpn_feature_list.reverse()  # [P2 - P5]

            # material
            if output_switch["material"]:
                output_dict["material"] = self.material_head(fpn_feature_list[0])

            if output_switch["object"] or output_switch["part"]:
                output_size = fpn_feature_list[0].size()[2:]
                fusion_list = [fpn_feature_list[0]]
                for i in range(1, len(fpn_feature_list)):
                    fusion_list.append(
                        F.interpolate(
                            fpn_feature_list[i],
                            output_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                fusion_out = torch.cat(fusion_list, 1)
                x = self.conv_fusion(fusion_out)

                if output_switch["object"]:  # object
                    output_dict["object"] = self.object_head(x)
                if output_switch["part"]:
                    output_dict["part"] = self.part_head(x)

        if self.use_softmax:  # is True during inference
            # inference scene
            x = output_dict["scene"]
            x = x.squeeze(3).squeeze(2)
            x = F.softmax(x, dim=1)
            output_dict["scene"] = x

            # inference object, material
            for k in ["object", "material"]:
                x = output_dict[k]
                x = F.interpolate(
                    x, size=seg_size, mode="bilinear", align_corners=False
                )
                x = F.softmax(x, dim=1)
                output_dict[k] = x

            # inference part
            x = output_dict["part"]
            x = F.interpolate(x, size=seg_size, mode="bilinear", align_corners=False)
            part_pred_list, head = [], 0
            for idx_part, object_label in enumerate(broden_dataset.object_with_part):
                n_part = len(broden_dataset.object_part[object_label])
                _x = F.interpolate(
                    x[:, head : head + n_part],
                    size=seg_size,
                    mode="bilinear",
                    align_corners=False,
                )
                _x = F.softmax(_x, dim=1)
                part_pred_list.append(_x)
                head += n_part
            output_dict["part"] = part_pred_list

        else:  # Training
            # object, scene, material
            for k in ["object", "scene", "material"]:
                if output_dict[k] is None:
                    continue
                x = output_dict[k]
                x = F.log_softmax(x, dim=1)
                if k == "scene":  # for scene
                    x = x.squeeze(3).squeeze(2)
                output_dict[k] = x
            if output_dict["part"] is not None:
                part_pred_list, head = [], 0
                for idx_part, object_label in enumerate(
                    broden_dataset.object_with_part
                ):
                    n_part = len(broden_dataset.object_part[object_label])
                    x = output_dict["part"][:, head : head + n_part]
                    x = F.log_softmax(x, dim=1)
                    part_pred_list.append(x)
                    head += n_part
                output_dict["part"] = part_pred_list

        return output_dict


    def decoder_forward(self, inputs):
        x = []
        for stage_ in inputs[:3]:
            x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)
        x.reverse()
        bs = x[0].size()[0]

        laterals = []
        attns = []
        maps_size = []
        qs = []
        q = self.q.weight.repeat(bs, 1, 1).transpose(0, 1)

        for x_, proj_, norm_, decoder_ in zip(
            x, self.input_proj, self.proj_norm, self.decoder
        ):
            lateral = norm_(proj_(x_))
            laterals.append(lateral)

            q, attn = decoder_(q, lateral.transpose(0, 1))
            attn = attn.transpose(-1, -2)
            attn = self.d3_to_d4(attn)
            maps_size.append(attn.size()[-2:])
            qs.append(q.transpose(0, 1))
            attns.append(attn)
        qs = torch.stack(qs, dim=0)
        outputs_class = self.class_embed(qs)
        out = {"pred_logits": outputs_class[-1]}

        outputs_seg_masks = []
        size = maps_size[-1]

        for i_attn, attn in enumerate(attns):
            if i_attn == 0:
                outputs_seg_masks.append(
                    F.interpolate(attn, size=size, mode="bilinear", align_corners=False)
                )
            else:
                outputs_seg_masks.append(
                    outputs_seg_masks[i_attn - 1]
                    + F.interpolate(
                        attn, size=size, mode="bilinear", align_corners=False
                    )
                )
        if self.training:
            out["pred_masks"] = F.interpolate(
                outputs_seg_masks[-1],
                size=(self.image_size_train, self.image_size_train),
                mode="bilinear",
                align_corners=False,
            )
        else:
            out["pred_masks"] = F.interpolate(
                outputs_seg_masks[-1],
                size=(self.image_size_val, self.image_size_val),
                mode="bilinear",
                align_corners=False,
            )

        out["pred"] = self.semantic_inference(out["pred_logits"], out["pred_masks"])

        if self.training:
            # [l, bs, queries, embed]
            outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_seg_masks)
        else:
            return out

        return out

    def forward(self, x):
        x = self.encoder_forward(x)
        x = self.decoder_forward(x)
        return x