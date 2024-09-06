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
import torch.autograd as ag
import collections

import torch
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

try:
    from os.path import join as pjoin, dirname
    from torch.utils.cpp_extension import load as load_extension

    root_dir = pjoin(dirname(__file__), "src")
    _prroi_pooling = load_extension(
        "_prroi_pooling",
        [
            pjoin(root_dir, "prroi_pooling_gpu.c"),
            pjoin(root_dir, "prroi_pooling_gpu_impl.cu"),
        ],
        verbose=True,
    )
except ImportError:
    raise ImportError("Can not compile Precise RoI Pooling library.")


from util.pos_embed import get_2d_sincos_pos_embed


class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.001, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine
        )

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

        # customed batch norm statistics
        self._moving_average_fraction = 1.0 - momentum
        self.register_buffer("_tmp_running_mean", torch.zeros(self.num_features))
        self.register_buffer("_tmp_running_var", torch.ones(self.num_features))
        self.register_buffer("_running_iter", torch.ones(1))
        self._tmp_running_mean = self.running_mean.clone() * self._running_iter
        self._tmp_running_var = self.running_var.clone() * self._running_iter

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input**2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(
                _ChildMessage(input_sum, input_ssum, sum_size)
            )
        else:
            mean, inv_std = self._slave_pipe.run_slave(
                _ChildMessage(input_sum, input_ssum, sum_size)
            )

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(
                inv_std * self.weight
            ) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)

        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2 : i * 2 + 2])))

        return outputs

    def _add_weighted(self, dest, delta, alpha=1, beta=1, bias=0):
        """return *dest* by `dest := dest*alpha + delta*beta + bias`"""
        return dest * alpha + delta * beta + bias

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert (
            size > 1
        ), "BatchNorm computes unbiased standard-deviation, which requires size > 1."
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self._tmp_running_mean = self._add_weighted(
            self._tmp_running_mean, mean.data, alpha=self._moving_average_fraction
        )
        self._tmp_running_var = self._add_weighted(
            self._tmp_running_var, unbias_var.data, alpha=self._moving_average_fraction
        )
        self._running_iter = self._add_weighted(
            self._running_iter, 1, alpha=self._moving_average_fraction
        )

        self.running_mean = self._tmp_running_mean / self._running_iter
        self.running_var = self._tmp_running_var / self._running_iter

        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class PrRoIPool2DFunction(ag.Function):
    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale):
        assert (
            "FloatTensor" in features.type() and "FloatTensor" in rois.type()
        ), "Precise RoI Pooling only takes float input, got {} for features and {} for rois.".format(
            features.type(), rois.type()
        )

        pooled_height = int(pooled_height)
        pooled_width = int(pooled_width)
        spatial_scale = float(spatial_scale)

        features = features.contiguous()
        rois = rois.contiguous()
        params = (pooled_height, pooled_width, spatial_scale)

        if features.is_cuda:
            output = _prroi_pooling.prroi_pooling_forward_cuda(features, rois, *params)
            ctx.params = params
            # everything here is contiguous.
            ctx.save_for_backward(features, rois, output)
        else:
            raise NotImplementedError(
                "Precise RoI Pooling only supports GPU (cuda) implememtations."
            )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, rois, output = ctx.saved_tensors
        grad_input = grad_coor = None

        if features.requires_grad:
            grad_output = grad_output.contiguous()
            grad_input = _prroi_pooling.prroi_pooling_backward_cuda(
                features, rois, output, grad_output, *ctx.params
            )
        if rois.requires_grad:
            grad_output = grad_output.contiguous()
            grad_coor = _prroi_pooling.prroi_pooling_coor_backward_cuda(
                features, rois, output, grad_output, *ctx.params
            )

        return grad_input, grad_coor, None, None, None


prroi_pool2d = PrRoIPool2DFunction.apply


class PrRoIPool2D(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super().__init__()

        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return prroi_pool2d(
            features, rois, self.pooled_height, self.pooled_width, self.spatial_scale
        )


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    @staticmethod
    def pixel_acc(pred, label, ignore_index=-1):
        _, preds = torch.max(pred, dim=1)
        valid = (label != ignore_index).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    @staticmethod
    def part_pixel_acc(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
        mask_object = gt_seg_object == object_label
        _, pred = torch.max(pred_part, dim=1)
        acc_sum = mask_object * (pred == gt_seg_part)
        acc_sum = torch.sum(acc_sum.view(acc_sum.size(0), -1), dim=1)
        acc_sum = torch.sum(acc_sum * valid)
        pixel_sum = torch.sum(mask_object.view(mask_object.size(0), -1), dim=1)
        pixel_sum = torch.sum(pixel_sum * valid)
        return acc_sum, pixel_sum

    @staticmethod
    def part_loss(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
        mask_object = gt_seg_object == object_label
        loss = F.nll_loss(pred_part, gt_seg_part * mask_object.long(), reduction="none")
        loss = loss * mask_object.float()
        loss = torch.sum(loss.view(loss.size(0), -1), dim=1)
        nr_pixel = torch.sum(mask_object.view(mask_object.shape[0], -1), dim=1)
        sum_pixel = (nr_pixel * valid).sum()
        loss = (loss * valid.float()).sum() / torch.clamp(sum_pixel, 1).float()
        return loss


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, loss_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit_dict = nn.ModuleDict()
        if loss_scale is None:
            self.loss_scale = {"object": 1, "part": 0.5, "scene": 0.25, "material": 1}
        else:
            self.loss_scale = loss_scale

        # criterion
        self.crit_dict["object"] = nn.NLLLoss(ignore_index=0)  # ignore background 0
        self.crit_dict["material"] = nn.NLLLoss(ignore_index=0)  # ignore background 0
        self.crit_dict["scene"] = nn.NLLLoss(ignore_index=-1)  # ignore unlabelled -1

    def forward(self, feed_dict, *, seg_size=None):
        if seg_size is None:  # training

            if feed_dict["source_idx"] == 0:
                output_switch = {
                    "object": True,
                    "part": True,
                    "scene": True,
                    "material": False,
                }
            elif feed_dict["source_idx"] == 1:
                output_switch = {
                    "object": False,
                    "part": False,
                    "scene": False,
                    "material": True,
                }
            else:
                raise ValueError

            pred = self.decoder(
                self.encoder(feed_dict["img"], return_feature_maps=True),
                output_switch=output_switch,
            )

            # loss
            loss_dict = {}
            if pred["object"] is not None:  # object
                loss_dict["object"] = self.crit_dict["object"](
                    pred["object"], feed_dict["seg_object"]
                )
            if pred["part"] is not None:  # part
                part_loss = 0
                for idx_part, object_label in enumerate(
                    broden_dataset.object_with_part
                ):
                    part_loss += self.part_loss(
                        pred["part"][idx_part],
                        feed_dict["seg_part"],
                        feed_dict["seg_object"],
                        object_label,
                        feed_dict["valid_part"][:, idx_part],
                    )
                loss_dict["part"] = part_loss
            if pred["scene"] is not None:  # scene
                loss_dict["scene"] = self.crit_dict["scene"](
                    pred["scene"], feed_dict["scene_label"]
                )
            if pred["material"] is not None:  # material
                loss_dict["material"] = self.crit_dict["material"](
                    pred["material"], feed_dict["seg_material"]
                )
            loss_dict["total"] = sum(
                [loss_dict[k] * self.loss_scale[k] for k in loss_dict.keys()]
            )

            # metric
            metric_dict = {}
            if pred["object"] is not None:
                metric_dict["object"] = self.pixel_acc(
                    pred["object"], feed_dict["seg_object"], ignore_index=0
                )
            if pred["material"] is not None:
                metric_dict["material"] = self.pixel_acc(
                    pred["material"], feed_dict["seg_material"], ignore_index=0
                )
            if pred["part"] is not None:
                acc_sum, pixel_sum = 0, 0
                for idx_part, object_label in enumerate(
                    broden_dataset.object_with_part
                ):
                    acc, pixel = self.part_pixel_acc(
                        pred["part"][idx_part],
                        feed_dict["seg_part"],
                        feed_dict["seg_object"],
                        object_label,
                        feed_dict["valid_part"][:, idx_part],
                    )
                    acc_sum += acc
                    pixel_sum += pixel
                metric_dict["part"] = acc_sum.float() / (pixel_sum.float() + 1e-10)
            if pred["scene"] is not None:
                metric_dict["scene"] = self.pixel_acc(
                    pred["scene"], feed_dict["scene_label"], ignore_index=-1
                )

            return {"metric": metric_dict, "loss": loss_dict}
        else:  # inference
            output_switch = {
                "object": True,
                "part": True,
                "scene": True,
                "material": True,
            }
            # print(self.encoder(feed_dict["img"]))
            pred = self.decoder(
                # self.encoder(feed_dict["img"], return_feature_maps=True),
                self.encoder(feed_dict["img"]),
                output_switch=output_switch,
                seg_size=seg_size,
            )
            return pred


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        pool_scales = (1, 2, 3, 6)
        fc_dim = 4096

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
            if i in [3, 8, 13, 18, 23]:
                outs.append(x)

        return outs

    def decoder_forward(self, conv_out, seg_size=None):

        output_switch = {
            "object": True,
            "part": False,
            "scene": False,
            "material": False,
        }

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

        # if output_switch["scene"]:  # scene
        #     output_dict["scene"] = self.scene_head(f)

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

            # # material
            # if output_switch["material"]:
            #     output_dict["material"] = self.material_head(fpn_feature_list[0])

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
                # if output_switch["part"]:
                #     output_dict["part"] = self.part_head(x)

        if self.use_softmax:  # is True during inference
            # inference scene
            # x = output_dict["scene"]
            # x = x.squeeze(3).squeeze(2)
            # x = F.softmax(x, dim=1)
            # output_dict["scene"] = x

            # inference object, material
            # for k in ["object", "material"]:
            x = output_dict["object"]
            x = F.interpolate(x, size=seg_size, mode="bilinear", align_corners=False)
            x = F.softmax(x, dim=1)
            output_dict["object"] = x

            # inference part
            # x = output_dict["part"]
            # x = F.interpolate(x, size=seg_size, mode="bilinear", align_corners=False)
            # part_pred_list, head = [], 0
            # for idx_part, object_label in enumerate(broden_dataset.object_with_part):
            #     n_part = len(broden_dataset.object_part[object_label])
            #     _x = F.interpolate(
            #         x[:, head : head + n_part],
            #         size=seg_size,
            #         mode="bilinear",
            #         align_corners=False,
            #     )
            #     _x = F.softmax(_x, dim=1)
            #     part_pred_list.append(_x)
            #     head += n_part
            # output_dict["part"] = part_pred_list

        else:  # Training
            # object, scene, material
            # for k in ["object", "scene", "material"]:
            #     if output_dict[k] is None:
            #         continue
            x = output_dict["object"]
            x = F.log_softmax(x, dim=1)
            # if k == "scene":  # for scene
            #     x = x.squeeze(3).squeeze(2)
            output_dict["object"] = x
            # if output_dict["part"] is not None:
            #     part_pred_list, head = [], 0
            #     for idx_part, object_label in enumerate(
            #         broden_dataset.object_with_part
            #     ):
            #         n_part = len(broden_dataset.object_part[object_label])
            #         x = output_dict["part"][:, head : head + n_part]
            #         x = F.log_softmax(x, dim=1)
            #         part_pred_list.append(x)
            #         head += n_part
            #     output_dict["part"] = part_pred_list

        return output_dict

    def forward(self, x):
        x = self.encoder_forward(x)
        x = self.decoder_forward(x)
        return x


# def decoder_forward(self, conv_out, seg_size=None):

#     output_switch = {
#         "object": True,
#         "part": False,
#         "scene": False,
#         "material": False,
#     }

#     output_dict = {k: None for k in output_switch.keys()}
#     # print(conv_out[-1].shape)

#     conv5 = conv_out[-1]
#     input_size = conv5.size()
#     print(input_size)
#     ppm_out = [conv5]
#     roi = []  # fake rois, just used for pooling
#     for i in range(input_size[0]):  # batch size
#         roi.append(
#             torch.Tensor([i, 0, 0, input_size[3], input_size[2]]).view(1, -1)
#         )  # b, x0, y0, x1, y1
#     roi = torch.cat(roi, dim=0).type_as(conv5)
#     ppm_out = [conv5]
#     for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
#         ppm_out.append(
#             pool_conv(
#                 F.interpolate(
#                     pool_scale(conv5, roi.detach()),
#                     (input_size[2], input_size[3]),
#                     mode="bilinear",
#                     align_corners=False,
#                 )
#             )
#         )
#     ppm_out = torch.cat(ppm_out, 1)
#     f = self.ppm_last_conv(ppm_out)

#     # if output_switch["scene"]:  # scene
#     #     output_dict["scene"] = self.scene_head(f)

#     if (
#         output_switch["object"]
#         or output_switch["part"]
#         or output_switch["material"]
#     ):
#         fpn_feature_list = [f]
#         for i in reversed(range(len(conv_out) - 1)):
#             conv_x = conv_out[i]
#             conv_x = self.fpn_in[i](conv_x)  # lateral branch

#             f = F.interpolate(
#                 f, size=conv_x.size()[2:], mode="bilinear", align_corners=False
#             )  # top-down branch
#             f = conv_x + f

#             fpn_feature_list.append(self.fpn_out[i](f))
#         fpn_feature_list.reverse()  # [P2 - P5]

#         # # material
#         # if output_switch["material"]:
#         #     output_dict["material"] = self.material_head(fpn_feature_list[0])

#         if output_switch["object"] or output_switch["part"]:
#             output_size = fpn_feature_list[0].size()[2:]
#             fusion_list = [fpn_feature_list[0]]
#             for i in range(1, len(fpn_feature_list)):
#                 fusion_list.append(
#                     F.interpolate(
#                         fpn_feature_list[i],
#                         output_size,
#                         mode="bilinear",
#                         align_corners=False,
#                     )
#                 )
#             fusion_out = torch.cat(fusion_list, 1)
#             x = self.conv_fusion(fusion_out)

#             if output_switch["object"]:  # object
#                 output_dict["object"] = self.object_head(x)
#             # if output_switch["part"]:
#             #     output_dict["part"] = self.part_head(x)

#     if self.use_softmax:  # is True during inference
#         # inference scene
#         # x = output_dict["scene"]
#         # x = x.squeeze(3).squeeze(2)
#         # x = F.softmax(x, dim=1)
#         # output_dict["scene"] = x

#         # inference object, material
#         # for k in ["object", "material"]:
#         x = output_dict["object"]
#         x = F.interpolate(x, size=seg_size, mode="bilinear", align_corners=False)
#         x = F.softmax(x, dim=1)
#         output_dict["object"] = x

#         # inference part
#         # x = output_dict["part"]
#         # x = F.interpolate(x, size=seg_size, mode="bilinear", align_corners=False)
#         # part_pred_list, head = [], 0
#         # for idx_part, object_label in enumerate(broden_dataset.object_with_part):
#         #     n_part = len(broden_dataset.object_part[object_label])
#         #     _x = F.interpolate(
#         #         x[:, head : head + n_part],
#         #         size=seg_size,
#         #         mode="bilinear",
#         #         align_corners=False,
#         #     )
#         #     _x = F.softmax(_x, dim=1)
#         #     part_pred_list.append(_x)
#         #     head += n_part
#         # output_dict["part"] = part_pred_list

#     else:  # Training
#         # object, scene, material
#         # for k in ["object", "scene", "material"]:
#         #     if output_dict[k] is None:
#         #         continue
#         x = output_dict["object"]
#         x = F.log_softmax(x, dim=1)
#         # if k == "scene":  # for scene
#         #     x = x.squeeze(3).squeeze(2)
#         output_dict["object"] = x
#         # if output_dict["part"] is not None:
#         #     part_pred_list, head = [], 0
#         #     for idx_part, object_label in enumerate(
#         #         broden_dataset.object_with_part
#         #     ):
#         #         n_part = len(broden_dataset.object_part[object_label])
#         #         x = output_dict["part"][:, head : head + n_part]
#         #         x = F.log_softmax(x, dim=1)
#         #         part_pred_list.append(x)
#         #         head += n_part
#         #     output_dict["part"] = part_pred_list

#     return output_dict
