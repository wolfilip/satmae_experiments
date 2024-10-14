# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import numpy as np

import torch.nn as nn

import torch
import torch.nn.functional as F
import wandb
from timm.data import Mixup
from timm.utils import accuracy

import util.lr_sched as lr_sched
import util.misc as misc
import cv2
import os
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
from torchmetrics.segmentation import MeanIoU
from torchmetrics import JaccardIndex

from models_vit_segmentaton import DiceLoss


def get_bce_loss(pred, mask):
    bce = nn.BCEWithLogitsLoss()
    # m = nn.Sigmoid()
    # pred = pred.argmax(1)
    # loss = F.binary_cross_entropy_with_logits(pred, mask)
    # loss = bce(torch.clamp(pred, min=0.0001, max=1.0), torch.clamp(mask, min=0.0001, max=1.0))
    loss = bce(pred, mask)
    return loss


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log(
                        {
                            "train_loss_step": loss_value_reduce,
                            "train_lr_step": max_lr,
                            "epoch_1000x": epoch_1000x,
                        }
                    )
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_temporal(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, resolutions, timestamps, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = [
            samples[0].to(device, non_blocking=True),
            samples[1].to(device, non_blocking=True),
            samples[2].to(device, non_blocking=True),
        ]
        resolutions = [
            resolutions[0].to(device, non_blocking=True),
            resolutions[1].to(device, non_blocking=True),
            resolutions[2].to(device, non_blocking=True),
        ]
        timestamps = timestamps.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # target_original = targets.clone()

        # if mixup_fn is not None:
        #     samples[0], targets = mixup_fn(samples[0], target_original)
        #     samples[1], targets = mixup_fn(samples[1], target_original)
        #     samples[2], targets = mixup_fn(samples[2], target_original)

        # targets = F.one_hot(targets, 62)

        with torch.cuda.amp.autocast():
            outputs = model(samples, timestamps)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log(
                        {
                            "train_loss_step": loss_value_reduce,
                            "train_lr_step": max_lr,
                            "epoch_1000x": epoch_1000x,
                        }
                    )
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_segmentation(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    cnt = 0

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if args.dataset_type == "spacenet":
        miou_metric = JaccardIndex(task="binary", zero_division=1.0)
    elif args.dataset_type == "loveda":
        miou_metric = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, zero_division=1.0
        )
    miou_metric = miou_metric.to(device)

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast("cuda"):
            # np.random.seed(233)
            # color_list = np.random.rand(1000, 3) * 0.7 + 0.3
            # outputs = model(samples)
            # object_result = outputs["pred"][0]
            # samples_0 = samples[0]
            # object_result  = torch.sigmoid(object_result )
            # object_result  = torch.argmax(object_result, 0)
            # object_result  = F.one_hot(object_result , num_classes=2).permute(2, 0, 1)
            # bla_1 = (255*samples_0).to(torch.uint8)
            # bla = draw_segmentation_masks(bla_1, object_result.to(torch.bool))
            # # normalized_masks = torch.nn.functional.softmax(object_result, dim=1)
            # # normalized_masks = object_result.argmax(dim=1)
            # # object_result_colored = maskrcnn_colorencode(samples, object_result.cpu().detach().numpy(), color_list)
            # cv2.imwrite(os.path.join("/home/filip/satmae_experiments", "object_result.png"), 255*bla.squeeze(0).cpu().detach().numpy().transpose())
            # cv2.imwrite(os.path.join("/home/filip/satmae_experiments", "object_result.png"), bla.cpu().detach().numpy().transpose(),)
            loss_value = calc_metrics(
                model, (samples, targets), miou_metric, device, epoch, cnt, args, 0, 0
            )

        # loss_value = loss_value.item()
        # print(miou)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss_value /= accum_iter
        loss_scaler(
            loss_value,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log(
                        {
                            "train_loss_step": loss_value_reduce,
                            "train_lr_step": max_lr,
                            "epoch_1000x": epoch_1000x,
                        }
                    )
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* IoU {iou:.4f} loss {losses.global_avg:.4f}".format(
            iou=miou_metric.compute(), losses=metric_logger.loss
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(acc1, acc5, flush=True)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_temporal(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    tta = False

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        timestamps = batch[2]
        target = batch[-1]

        batch_size = images[0].shape[0]
        # print(images.shape, timestamps.shape, target.shape)
        if tta:
            images = images.reshape(-1, 3, 3, 224, 224)
            timestamps = timestamps.reshape(-1, 3, 3)
            target = target.reshape(-1, 1)
        # images = images.reshape()
        # print('images and targets')
        images = [
            images[0].to(device, non_blocking=True),
            images[1].to(device, non_blocking=True),
            images[2].to(device, non_blocking=True),
        ]
        timestamps = timestamps.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, timestamps)

            if tta:
                # output = output.reshape(batch_size, 9, -1).mean(dim=1, keepdims=False)

                output = output.reshape(batch_size, 9, -1)
                sp = output.shape
                maxarg = output.argmax(dim=-1)

                output = F.one_hot(maxarg.reshape(-1), num_classes=1000).float()
                output = output.reshape(sp).mean(dim=1, keepdims=False)
                # print(output.shape)

                target = target.reshape(batch_size, 9)[:, 0]
            # print(target.shape)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(acc1, acc5, flush=True)

        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_segmentation(data_loader, model, device, epoch, args):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    cnt = 0
    if args.dataset_type == "spacenet":
        miou_metric = JaccardIndex(task="binary", zero_division=1.0)
    elif args.dataset_type == "loveda":
        miou_metric = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, zero_division=1.0
        )
    miou_metric = miou_metric.to(device)

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
            loss = calc_metrics(
                model, (images, target), miou_metric, device, epoch, cnt, args, 0, 0
            )
            # target = target.to(torch.float32)
            # pred = model(images)
            # loss = model.get_bce_loss(pred, target)
            # IoU = model.get_iou(pred, target, 2)

        cnt += images.shape[0]

        # batch_size = images.shape[0]
        metric_logger.update(loss=loss)
        # metric_logger.meters['IoU'].update(IoU, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    metric_logger.update(IoU=miou_metric.compute().item())

    print(
        "* IoU {iou.global_avg:.4f} loss {losses.global_avg:.4f}".format(
            iou=metric_logger.IoU, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def maskrcnn_colorencode(img, label_map, color_list):
    # do not modify original list
    label_map = np.array(np.expand_dims(label_map, axis=0), np.uint8)
    # label_map = label_map.transpose(1, 2, 0)
    label_list = list(np.unique(label_map))
    out_img = img.clone()
    for i, label in enumerate(label_list):
        if label == 0:
            continue
        this_label_map = label_map == label
        alpha = [0, 0, 0]
        o = i
        if o >= 6:
            o = np.random.randint(1, 6)
        o_lst = [o % 2, (o // 2) % 2, o // 4]
        for j in range(3):
            alpha[j] = np.random.random() * 0.5 + 0.45
            alpha[j] *= o_lst[j]
        out_img = MydrawMask(
            out_img,
            this_label_map,
            alpha=alpha,
            clrs=np.expand_dims(color_list[label], axis=0),
        )
    return out_img


def MydrawMask(img, masks, lr=(None, None), alpha=None, clrs=None, info=None):
    n, h, w = masks.shape[0], masks.shape[1], masks.shape[2]
    if lr[0] is None:
        lr = (0, n)
    if alpha is None:
        alpha = [0.4, 0.4, 0.4]
    alpha = [0.6, 0.6, 0.6]
    if clrs is None:
        clrs = np.zeros((n, 3)).astype(np.float64)
        for i in range(n):
            for j in range(3):
                clrs[i][j] = np.random.random() * 0.6 + 0.4

    for i in range(max(0, lr[0]), min(n, lr[1])):
        M = masks[i].reshape(-1)
        B = np.zeros(h * w, dtype=np.int8)
        ix, ax, iy, ay = 99999, 0, 99999, 0
        for y in range(h - 1):
            for x in range(w - 1):
                k = y * w + x
                if M[k] == 1:
                    ix = min(ix, x)
                    ax = max(ax, x)
                    iy = min(iy, y)
                    ay = max(ay, y)
                if M[k] != M[k + 1]:
                    B[k], B[k + 1] = 1, 1
                if M[k] != M[k + w]:
                    B[k], B[k + w] = 1, 1
                if M[k] != M[k + 1 + w]:
                    B[k], B[k + 1 + w] = 1, 1
        M.shape = (h, w)
        B.shape = (h, w)
        for j in range(3):
            O, c, a = img[:, :, j], clrs[i][j], alpha[j]
            am = a * M
            O = O - O * am + c * am * 255
            img[:, :, j] = O * (1 - B) + c * B
        # cv2.rectangle(img, (ix,iy), (ax,ay), (0,255,0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = ix - 1, iy - 1
        if x < 0:
            x = 0
        if y < 10:
            y += 7
        if int(img[y, x, 0]) + int(img[y, x, 1]) + int(img[y, x, 2]) > 650:
            col = (255, 0, 0)
        else:
            col = (255, 255, 255)
        # col = (255,0,0)
        # cv2.putText(img, id2class[info['category_id']]+': %.3f' % info['score'], (x, y), font, .3, col, 1)
    return img


def remove_small_mat(seg_mat, seg_obj, threshold=0.1):
    object_list = np.unique(seg_obj)
    seg_mat_new = np.zeros_like(seg_mat)
    for obj_label in object_list:
        obj_mask = seg_obj == obj_label
        mat_result = seg_mat * obj_mask
        mat_sum = obj_mask.sum()
        for mat_label in np.unique(mat_result):
            mat_area = (mat_result == mat_label).sum()
            if mat_area / float(mat_sum) < threshold:
                continue
            seg_mat_new += mat_result * (mat_result == mat_label)
        # sorted_mat_index = np.argsort(-np.asarray(mat_area))
    return seg_mat_new


def calc_metrics(
    model, data, miou_metric, device, epoch, cnt, args, epoch_loss, epoch_iou
):
    (data, mask) = data
    epoch_loss = epoch_loss
    epoch_iou = epoch_iou
    data = data.to(device)
    # viz_1, viz_2 = model.visualize_features(data)
    pred = model(data)

    if args.dataset_type == "spacenet":
        miou_temp = JaccardIndex(task="binary", zero_division=1.0)
        mask_one_hot = F.one_hot(mask, num_classes=args.nb_classes).permute(0, 3, 1, 2)

        if not os.path.exists(
            "satmae_experiments/spacenet_10pc_results/images/2_blocks_frozen_8/"
        ):
            os.makedirs(
                "satmae_experiments/spacenet_10pc_results/images/2_blocks_frozen_8/"
            )
        if not os.path.exists(
            "satmae_experiments/spacenet_10pc_results/per_image/2_blocks_frozen_8/"
        ):
            os.makedirs(
                "satmae_experiments/spacenet_10pc_results/per_image/2_blocks_frozen_8/"
            )
    elif args.dataset_type == "loveda":
        miou_temp = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, zero_division=1.0
        )
        mask = mask.squeeze(1)
        # print(mask.unique())
        mask_one_hot = F.one_hot(mask, num_classes=args.nb_classes).permute(0, 3, 1, 2)

        if not os.path.exists(
            "satmae_experiments/loveda_results/images/dinov2_l_linear-3/"
        ):
            os.makedirs("satmae_experiments/loveda_results/images/dinov2_l_linear-3/")
        if not os.path.exists(
            "satmae_experiments/loveda_results/per_image/dinov2_l_linear-3/"
        ):
            os.makedirs(
                "satmae_experiments/loveda_results/per_image/dinov2_l_linear-3/"
            )
    # dice_loss = DiceLoss()
    # miou = MeanIoU(include_background=False, num_classes=1)
    miou_temp = miou_temp.to(device)
    # plt.imshow(mask_one_hot.argmax(1).cpu()[0])
    # plt.savefig("satmae_experiments/loveda_results/images/vit_upernet_conv_small/img_" + str(cnt + 0) + ".png")
    # plt.close()
    # pred_bool = model.sigmoid(pred) >= 0.5
    # bla = pred['pred_masks'].argmax(1)
    # cv2.imwrite(os.path.join("/home/filip/satmae_experiments", "object_result.png"), 255*ppppred["pred"].cpu().detach().numpy().transpose(),)

    if not model.training:
        for i in range(data.shape[0]):
            if epoch == 400:
                f, axarr = plt.subplots(4)
                axarr[0].imshow(data.cpu()[i].permute(1, 2, 0))
                axarr[1].imshow(mask_one_hot.argmax(1).cpu()[i])
                axarr[2].imshow(pred.argmax(1).cpu()[i])
                # axarr[3].imshow(viz_1.permute(1, 2, 0))
                # plt.savefig("images/image_1_pca.png")
                plt.savefig(
                    "satmae_experiments/loveda_results/images/dinov2_l_linear-3/img_"
                    + str(cnt + i)
                    + ".png"
                )
                plt.close()
            # pred_bool = model.sigmoid(pred[i]) >= 0.5
            mIoU = miou_temp(pred.argmax(1)[i], mask[i]).item()
            if torch.all(mask[i] == 0) and torch.all(pred.argmax(1)[i] == 0):
                mIoU = 1.0
            f = open(
                "satmae_experiments/loveda_results/per_image/dinov2_l_linear-3/image_results_iou_"
                + str(epoch)
                + ".txt",
                "a",
            )
            f.write("img_" + str(cnt + i) + ": " + str(mIoU) + "\n")
            f.close()
        # return model.get_bce_loss(pred, mask_one_hot.float()) + dice_loss(pred, mask_one_hot.float()), miou_sum / data.shape[0]
        # if miou_sum / data.shape[0] > best_miou

    # loss_1 = model.get_bce_loss(pred, mask.float())
    loss_1 = get_bce_loss(pred, mask_one_hot.float())
    # loss_2 = dice_loss(pred, mask_one_hot.float())
    # loss = sum(loss.values())
    # loss = model.get_bce_loss(pred["pred_logits"], mask)
    # pred_bool = model.sigmoid(pred) >= 0.5
    # pred_bool = pred_bool[:, 0:1, :, :]
    miou_metric.update(pred, mask)
    # if torch.all(mask == 0) and torch.all(pred.argmax(1) == 0):
    #     mIoU = 1.0
    # IoU = model.get_iou(pred, mask, nclass)
    return loss_1
