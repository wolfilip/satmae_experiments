# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

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
import numpy as np


from util.pos_embed import get_2d_sincos_pos_embed


class DiceLoss(nn.Module):
    """Dice Loss PyTorch
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """

    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight)  # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1)  # (N, C, *)
        target = target.view(N, C, -1)  # (N, 1, *)

        predict = F.softmax(predict, dim=1)  # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        # target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
        # target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, "weight"):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss


def up_and_add(x, y):
    return (
        F.interpolate(
            x, size=(y.size(2), y.size(3)), mode="bilinear", align_corners=True
        )
        + y
    )


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, losses, eos_coef=0.1):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        if src_masks.dim() != 4:
            return {"no_loss": 0}
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = self.nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": self.sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": self.dice_loss(src_masks, target_masks, num_masks),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets

        labels = [x["labels"] for x in targets]
        indices_new = []
        for label in labels:
            t = torch.arange(len(label))
            indices_new.append([label, t])
        indices = indices_new
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # use the indices as the last stage
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def nested_tensor_from_tensor_list(self, tensor_list: List[Tensor]):
        # TODO make this more general
        if tensor_list[0].ndim == 3:

            # TODO make it support different-sized images
            max_size = self._max_by_axis([list(img.shape) for img in tensor_list])
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = [len(tensor_list)] + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        else:
            raise ValueError("not supported")
        return NestedTensor(tensor, mask)

    def _max_by_axis(self, the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def sigmoid_focal_loss(
        self, inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2
    ):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_masks

    def dice_loss(self, inputs, targets, num_masks):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks


class TPN_Decoder(TransformerDecoder):
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        output = tgt
        # attns = []
        for mod in self.layers:
            output, attn = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            # attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size()
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = (
            self.q(xq)
            .reshape(B, Nq, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(xk)
            .reshape(B, Nk, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(xv)
            .reshape(B, Nv, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads


class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs["d_model"], num_heads=kwargs["nhead"], qkv_bias=True, attn_drop=0.1
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1)
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2


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
                nhead=2,
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
        self.image_size_train = 224
        self.image_size_val = 224

        self.criterion = SetCriterion(2, losses=["masks", "labels"])

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        feature_channels = [1024, 1024, 1024]

        fpn_out = 1024
        num_classes = 2 
        self.input_size = (224, 224)

        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)
        self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_3 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()

    def encoder_forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = x[:, 1:]

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in [7, 15, 23]:
                # if i in [5, 11, 17, 23]:
                # if i in [3, 8, 13, 18, 23]:
                outs.append(x)

        return outs

    def decoder_segvit(self, inputs):
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

    def decoder_upernet(self, features):

        features[0] = torch.unflatten(features[0], dim=1, sizes=(14, 14))
        features[1] = torch.unflatten(features[1], dim=1, sizes=(14, 14))
        features[2] = torch.unflatten(features[2], dim=1, sizes=(14, 14))
        # features[3] = torch.unflatten(features[3], dim=1, sizes=(14, 14))
        # features[4] = torch.unflatten(features[4], dim=1, sizes=(14, 14))
        features[0] = torch.permute(features[0], (0, 3, 1, 2))
        features[1] = torch.permute(features[1], (0, 3, 1, 2))
        features[2] = torch.permute(features[2], (0, 3, 1, 2))
        # features[3] = torch.permute(features[3], (0, 3, 1, 2))
        # features[4] = torch.permute(features[4], (0, 3, 1, 2))

        # features[3] = self.up_1(features[3])
        # features[2] = self.up_1(features[2])
        features[1] = self.up_1(features[1])
        features[0] = self.up_2(features[0])

        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=self.input_size, mode="bilinear")
        return x

    def forward(self, x):
        x = self.encoder_forward(x)
        x = self.decoder_upernet(x)
        return x

    def unpatchify(self, x, p, c):
        """
        x: (N, L, patch_size**2 *C)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        # if hw % 2 != 0:
        #     t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    def get_bce_loss(self, pred, mask):
        bce = nn.BCEWithLogitsLoss()
        # m = nn.Sigmoid()
        # pred = pred.argmax(1)
        # loss = F.binary_cross_entropy_with_logits(pred, mask)
        # loss = bce(torch.clamp(pred, min=0.0001, max=1.0), torch.clamp(mask, min=0.0001, max=1.0))
        loss = bce(pred, mask)
        return loss

    def get_iou(self, pred, target, nclass):
        # target = target.to(torch.float32)
        # pred = torch.sigmoid(pred)

        # print(torch.unique(pred[0].argmax(0)))
        # plt.imshow(target[0].argmax(0).cpu())
        # print(torch.unique(target[0].argmax(0).cpu()))
        # plt.savefig("foo_2.png")
        # plt.imshow(pred[0].argmax(0).cpu())
        # print(torch.unique(pred[0].argmax(0).cpu()))
        # plt.savefig("bar_2.png")
        # pred = torch.argmax(pred, 1)
        # pred = F.one_hot(pred, num_classes=nclass).permute(0, 3, 1, 2)
        # pred = pred.cpu().detach().numpy().astype(int)

        # target = target.cpu().detach().numpy().astype(int)
        bla_1 = (pred & target).sum()
        bla_2 = (pred | target).sum()
        iou = (pred & target).sum() / ((pred | target).sum() + 1e-6)
        # print(iou)
        return iou

    def loss(self, outputs, label):
        targets = self.prepare_targets(label)
        losses = self.criterion(outputs, targets)

        return losses

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            # gt_cls
            gt_cls = targets_per_image.unique()
            # gt_cls = gt_cls[gt_cls != self.ignore_index]
            masks = []
            for cls in gt_cls:
                masks.append(targets_per_image == cls)
            if len(gt_cls) == 0:
                masks.append(targets_per_image == self.ignore_index)

            masks = torch.stack(masks, dim=0)
            new_targets.append(
                {
                    "labels": gt_cls,
                    "masks": masks,
                }
            )
        return new_targets

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                "DeprecationWarning: ``efficient_test`` has been deprecated "
                "since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory "
                "friendly by default. "
            )

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results["gt_semantic_seg"]

    def intersect_and_union(
        self,
        pred_label,
        label,
        num_classes,
        ignore_index,
        label_map=dict(),
        reduce_zero_label=False,
    ):
        """Calculate intersection and Union.

        Args:
            pred_label (ndarray | str): Prediction segmentation map
                or predict result filename.
            label (ndarray | str): Ground truth segmentation map
                or label filename.
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.
            label_map (dict): Mapping old labels to new labels. The parameter will
                work only when label is str. Default: dict().
            reduce_zero_label (bool): Whether ignore zero label. The parameter will
                work only when label is str. Default: False.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        if isinstance(pred_label, str):
            pred_label = torch.from_numpy(np.load(pred_label))
        else:
            pred_label = torch.from_numpy((pred_label))

        if isinstance(label, str):
            label = torch.from_numpy(
                mmcv.imread(label, flag="unchanged", backend="pillow")
            )
        else:
            label = torch.from_numpy(label)

        if label_map is not None:
            for old_id, new_id in label_map.items():
                label[label == old_id] = new_id
        if reduce_zero_label:
            label[label == 0] = 255
            label = label - 1
            label[label == 254] = 255

        mask = label != ignore_index
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0, max=num_classes - 1
        )
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1
        )
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0, max=num_classes - 1
        )
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    def total_intersect_and_union(
        self,
        results,
        gt_seg_maps,
        num_classes,
        ignore_index,
        label_map=dict(),
        reduce_zero_label=False,
    ):
        """Calculate Total Intersection and Union.

        Args:
            results (list[ndarray] | list[str]): List of prediction segmentation
                maps or list of prediction result filenames.
            gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
                truth segmentation maps or list of label filenames.
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.
            label_map (dict): Mapping old labels to new labels. Default: dict().
            reduce_zero_label (bool): Whether ignore zero label. Default: False.

        Returns:
            ndarray: The intersection of prediction and ground truth histogram
                on all classes.
            ndarray: The union of prediction and ground truth histogram on all
                classes.
            ndarray: The prediction histogram on all classes.
            ndarray: The ground truth histogram on all classes.
        """
        total_area_intersect = torch.zeros((num_classes,), dtype=torch.float64)
        total_area_union = torch.zeros((num_classes,), dtype=torch.float64)
        total_area_pred_label = torch.zeros((num_classes,), dtype=torch.float64)
        total_area_label = torch.zeros((num_classes,), dtype=torch.float64)
        for result, gt_seg_map in zip(results, gt_seg_maps):
            area_intersect, area_union, area_pred_label, area_label = (
                self.intersect_and_union(
                    result,
                    gt_seg_map,
                    num_classes,
                    ignore_index,
                    label_map,
                    reduce_zero_label,
                )
            )
            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_pred_label += area_pred_label
            total_area_label += area_label
        return (
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
        )

    def total_area_to_metrics(
        self,
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
        metrics=["mIoU"],
        nan_to_num=None,
        beta=1,
    ):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (ndarray): The intersection of prediction and
                ground truth histogram on all classes.
            total_area_union (ndarray): The union of prediction and ground truth
                histogram on all classes.
            total_area_pred_label (ndarray): The prediction histogram on all
                classes.
            total_area_label (ndarray): The ground truth histogram on all classes.
            metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be replaced
                by the numbers defined by the user. Default: None.
        Returns:
            float: Overall accuracy on all images.
            ndarray: Per category accuracy, shape (num_classes, ).
            ndarray: Per category evaluation metrics, shape (num_classes, ).
        """
        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ["mIoU", "mDice", "mFscore"]
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError("metrics {} is not supported".format(metrics))

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({"aAcc": all_acc})
        for metric in metrics:
            if metric == "mIoU":
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics["IoU"] = iou
                ret_metrics["Acc"] = acc
            elif metric == "mDice":
                dice = (
                    2
                    * total_area_intersect
                    / (total_area_pred_label + total_area_label)
                )
                acc = total_area_intersect / total_area_label
                ret_metrics["Dice"] = dice
                ret_metrics["Acc"] = acc
            elif metric == "mFscore":
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor(
                    [f_score(x[0], x[1], beta) for x in zip(precision, recall)]
                )
                ret_metrics["Fscore"] = f_value
                ret_metrics["Precision"] = precision
                ret_metrics["Recall"] = recall

        ret_metrics = {metric: value.numpy() for metric, value in ret_metrics.items()}
        if nan_to_num is not None:
            ret_metrics = OrderedDict(
                {
                    metric: np.nan_to_num(metric_value, nan=nan_to_num)
                    for metric, metric_value in ret_metrics.items()
                }
            )
        return ret_metrics

    def eval_metrics(
        self,
        results,
        gt_seg_maps,
        num_classes,
        ignore_index,
        metrics=["mIoU"],
        nan_to_num=None,
        label_map=dict(),
        reduce_zero_label=False,
        beta=1,
    ):
        """Calculate evaluation metrics
        Args:
            results (list[ndarray] | list[str]): List of prediction segmentation
                maps or list of prediction result filenames.
            gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
                truth segmentation maps or list of label filenames.
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.
            metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be replaced
                by the numbers defined by the user. Default: None.
            label_map (dict): Mapping old labels to new labels. Default: dict().
            reduce_zero_label (bool): Whether ignore zero label. Default: False.
        Returns:
            float: Overall accuracy on all images.
            ndarray: Per category accuracy, shape (num_classes, ).
            ndarray: Per category evaluation metrics, shape (num_classes, ).
        """

        (
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
        ) = self.total_intersect_and_union(
            results,
            gt_seg_maps,
            num_classes,
            ignore_index,
            label_map,
            reduce_zero_label,
        )
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            metrics,
            nan_to_num,
            beta,
        )

        return ret_metrics


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
