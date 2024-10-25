# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import os
import sys
from argparse import ArgumentParser
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from timm.data import Mixup
from timm.utils import accuracy
from torch import device
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchmetrics import JaccardIndex

import util.lr_sched as lr_sched
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.visualize_features import visualize_features


def get_bce_loss(pred, mask):
    bce = nn.BCEWithLogitsLoss()
    # m = nn.Sigmoid()
    # pred = pred.argmax(1)
    # loss = F.binary_cross_entropy_with_logits(pred, mask)
    # loss = bce(torch.clamp(pred, min=0.0001, max=1.0), torch.clamp(mask, min=0.0001, max=1.0))
    loss = bce(pred, mask)
    return loss


def train_one_epoch(
    model: Module,
    criterion: Module,
    data_loader: Iterable,
    optimizer: Optimizer,
    device: device,
    epoch: int,
    loss_scaler: NativeScaler,
    log_writer,
    args: ArgumentParser,
    mixup_fn: Optional[Mixup] = None,
    max_norm: float = 1,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter  # type: ignore

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args  # type: ignore
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
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)  # type: ignore
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:  # type: ignore
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
    model: Module,
    criterion: Module,
    data_loader: Iterable,
    optimizer: Optimizer,
    device: device,
    epoch: int,
    loss_scaler: NativeScaler,
    log_writer,
    args: ArgumentParser,
    mixup_fn: Optional[Mixup] = None,
    max_norm: float = 1,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter  # type: ignore

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, resolutions, timestamps, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args  # type: ignore
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

        with torch.amp.autocast("cuda"):  # type: ignore
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
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)  # type: ignore
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:  # type: ignore
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
    model: Module,
    data_loader: Iterable,
    optimizer: Optimizer,
    device: device,
    epoch: int,
    loss_scaler: NativeScaler,
    log_writer,
    args: ArgumentParser,
    mixup_fn: Optional[Mixup] = None,
    max_norm: float = 1,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter  # type: ignore

    optimizer.zero_grad()

    cnt = 0

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if args.dataset_type == "spacenet":  # type: ignore
        miou_metric = JaccardIndex(task="binary", zero_division=1.0)
    elif args.dataset_type == "loveda":  # type: ignore
        miou_metric = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, zero_division=1.0  # type: ignore
        )
    miou_metric = miou_metric.to(device)

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args  # type: ignore
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):  # type: ignore
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
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)  # type: ignore
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:  # type: ignore
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
                output = output.reshape(sp).mean(dim=1, keepdims=False)  # type: ignore
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
        with torch.amp.autocast("cuda"):  # type: ignore
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


def calc_metrics(
    model, data, miou_metric, device, epoch, cnt, args, epoch_loss, epoch_iou
):
    (data, mask) = data
    epoch_loss = epoch_loss
    epoch_iou = epoch_iou
    data = data.to(device)
    pred, features = model(data)

    if args.visualize_features:
        if not os.path.exists(
            "satmae_experiments/feature_visualizations/mae_dinov2_upernet_last/"
        ):
            os.makedirs(
                "satmae_experiments/feature_visualizations/mae_dinov2_upernet_last/"
            )

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
            "satmae_experiments/loveda_results/images/dinov2_vit_4_blocks_vit_upernet-2/"
        ):
            os.makedirs(
                "satmae_experiments/loveda_results/images/dinov2_vit_4_blocks_vit_upernet-2/"
            )
        if not os.path.exists(
            "satmae_experiments/loveda_results/per_image/dinov2_vit_4_blocks_vit_upernet-2/"
        ):
            os.makedirs(
                "satmae_experiments/loveda_results/per_image/dinov2_vit_4_blocks_vit_upernet-2/"
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
                    "satmae_experiments/loveda_results/images/dinov2_vit_4_blocks_vit_upernet-2/img_"
                    + str(cnt + i)
                    + ".png"
                )
                plt.close()
            # pred_bool = model.sigmoid(pred[i]) >= 0.5
            mIoU = miou_temp(pred.argmax(1)[i], mask[i]).item()
            if torch.all(mask[i] == 0) and torch.all(pred.argmax(1)[i] == 0):
                mIoU = 1.0
            f = open(
                "satmae_experiments/loveda_results/per_image/dinov2_vit_4_blocks_vit_upernet-2/image_results_iou_"
                + str(epoch)
                + ".txt",
                "a",
            )
            f.write("img_" + str(cnt + i) + ": " + str(mIoU) + "\n")
            f.close()
            if args.visualize_features:
                viz_1, viz_2 = visualize_features(features)
                f, axarr = plt.subplots(2)
                axarr[0].imshow(data.cpu()[i].permute(1, 2, 0))
                axarr[1].imshow(viz_1.permute(1, 2, 0))
                plt.savefig(
                    "satmae_experiments/feature_visualizations/mae_dinov2_upernet_last/feature_"
                    + str(cnt + i)
                    + ".png"
                )
                plt.close()
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
