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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from timm.data import Mixup
from timm.utils import accuracy
from torch import device
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchmetrics import F1Score, JaccardIndex, Accuracy

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

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if args.dataset_type == "spacenet":  # type: ignore
        miou_metric = JaccardIndex(task="multiclass", num_classes=args.nb_classes)  # type: ignore
    else:
        miou_metric = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, average="micro"
        )
        miou_metric_2 = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, average="weighted"
        )
        f1_score = F1Score(
            task="multiclass", num_classes=args.nb_classes, average="micro"
        )
        overall_accuracy = Accuracy(
            task="multiclass", num_classes=args.nb_classes, average="weighted"
        )
        f1_score = f1_score.to(device)
        miou_metric_2 = miou_metric_2.to(device)
        overall_accuracy = overall_accuracy.to(device)
    miou_metric = miou_metric.to(device)

    if epoch == 0:
        if not os.path.exists(
            "satmae_experiments/"
            + args.dataset_type
            + "_"
            + args.dataset_split
            + "pc_results/images/"
            + args.method_name
        ):
            os.makedirs(
                "satmae_experiments/"
                + args.dataset_type
                + "_"
                + args.dataset_split
                + "pc_results/images/"
                + args.method_name
            )
        if not os.path.exists(
            "satmae_experiments/"
            + args.dataset_type
            + "_"
            + args.dataset_split
            + "pc_results/per_image/"
            + args.method_name
        ):
            os.makedirs(
                "satmae_experiments/"
                + args.dataset_type
                + "_"
                + args.dataset_split
                + "pc_results/per_image/"
                + args.method_name
            )

    for data_iter_step, (data, mask) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args  # type: ignore
            )

        data = data.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):  # type: ignore
            data = data.to(device)
            pred, _ = model(data)

            if args.dataset_type == "loveda" or args.dataset_type == "vaihingen" or args.dataset_type == "potsdam":  # type: ignore
                mask = mask.squeeze(1)

            mask_one_hot = F.one_hot(mask, num_classes=args.nb_classes).permute(  # type: ignore
                0, 3, 1, 2
            )

            loss_value = get_bce_loss(pred, mask_one_hot.float())
            # dice_loss = DiceLoss()
            # loss_2 = dice_loss(pred, mask_one_hot.float())
            miou_metric.update(pred.argmax(1), mask)
            if args.dataset_type != "spacenet":
                miou_metric_2.update(pred.argmax(1), mask)
                f1_score.update(pred.argmax(1), mask)
                overall_accuracy.update(pred.argmax(1), mask)

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
    if args.dataset_type == "spacenet":
        print(
            "* IoU {iou:.4f} loss {losses.global_avg:.4f}".format(
                iou=miou_metric.compute(), losses=metric_logger.loss
            )
        )
    else:
        print(
            "* IoU {iou:.4f} ioU 2 {iou2:.4f} F1 {f1:.4f} OA {oa:.4f} loss {losses.global_avg:.4f}".format(
                iou=miou_metric.compute(),
                iou2=miou_metric_2.compute(),
                f1=f1_score.compute(),
                oa=overall_accuracy.compute(),
                losses=metric_logger.loss,
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
def evaluate_segmentation(data_loader, model, device, epoch, max_iou, args):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    cnt = 0

    if args.eval:
        if not os.path.exists(
            "satmae_experiments/"
            + args.dataset_type
            + "_"
            + args.dataset_split
            + "pc_results/images/"
            + args.method_name
        ):
            os.makedirs(
                "satmae_experiments/"
                + args.dataset_type
                + "_"
                + args.dataset_split
                + "pc_results/images/"
                + args.method_name
            )
        if not os.path.exists(
            "satmae_experiments/"
            + args.dataset_type
            + "_"
            + args.dataset_split
            + "pc_results/per_image/"
            + args.method_name
        ):
            os.makedirs(
                "satmae_experiments/"
                + args.dataset_type
                + "_"
                + args.dataset_split
                + "pc_results/per_image/"
                + args.method_name
            )

    if args.dataset_type == "spacenet":  # type: ignore
        miou_metric = JaccardIndex(task="multiclass", num_classes=args.nb_classes)  # type: ignore
    else:
        miou_metric = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, average="micro"
        )
        miou_metric_2 = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, average="weighted"
        )
        f1_score = F1Score(
            task="multiclass", num_classes=args.nb_classes, average="micro"
        )
        overall_accuracy = Accuracy(
            task="multiclass", num_classes=args.nb_classes, average="weighted"
        )

        f1_score = f1_score.to(device)
        miou_metric_2 = miou_metric_2.to(device)
        overall_accuracy = overall_accuracy.to(device)

    miou_metric = miou_metric.to(device)

    miou_test = 0

    for batch in metric_logger.log_every(data_loader, 10, header):
        data = batch[0]
        mask = batch[-1]
        # print('images and targets')
        data = data.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.amp.autocast("cuda"):  # type: ignore
            data = data.to(device)
            pred, features = model(data)

            if (
                args.dataset_type == "loveda"
                or args.dataset_type == "vaihingen"
                or args.dataset_type == "potsdam"
            ):
                mask = mask.squeeze(1)

            mask_one_hot = F.one_hot(mask, num_classes=args.nb_classes).permute(
                0, 3, 1, 2
            )

            loss = get_bce_loss(pred, mask_one_hot.float())
            # dice_loss = DiceLoss()
            # loss_2 = dice_loss(pred, mask_one_hot.float())
            miou_metric.update(pred.argmax(1), mask)
            if args.dataset_type != "spacenet":
                miou_metric_2.update(pred.argmax(1), mask)
                f1_score.update(pred.argmax(1), mask)
                overall_accuracy.update(pred.argmax(1), mask)

            miou_test = save_results(
                data, mask, pred, device, epoch, cnt, miou_test, args
            )

        cnt += data.shape[0]

        # batch_size = images.shape[0]
        metric_logger.update(loss=loss)
        # metric_logger.meters['IoU'].update(IoU, n=batch_size)

    if args.dataset_type == "spacenet":
        print("miou test: " + str(miou_test * args.world_size / 1940))
    elif args.dataset_type == "vaihingen":
        print("miou test: " + str(miou_test * args.world_size / 398))
    elif args.dataset_type == "potsdam":
        print("miou test: " + str(miou_test * args.world_size / 2016))

    # gather the stats from all processes
    miou = miou_metric.compute().item()
    if args.dataset_type != "spacenet":
        miou_2 = miou_metric_2.compute().item()
        f1 = f1_score.compute().item()
        oa = overall_accuracy.compute().item()

    if args.save_images:
        cnt = 0
        if args.best_epoch:
            if miou > max_iou and epoch > 200:
                for batch in data_loader:
                    data = batch[0]
                    mask = batch[-1]
                    # print('images and targets')
                    data = data.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)

                    # print("before pass model")
                    # compute output
                    with torch.amp.autocast("cuda"):  # type: ignore
                        data = data.to(device)
                        pred, features = model(data)

                        if (
                            args.dataset_type == "loveda"
                            or args.dataset_type == "vaihingen"
                            or args.dataset_type == "potsdam"
                        ):
                            mask = mask.squeeze(1)

                        save_images(data, mask, pred, features, cnt, args)

                    cnt += data.shape[0]
        elif epoch == args.epochs - 1 or args.eval:
            for batch in data_loader:
                data = batch[0]
                mask = batch[-1]
                # print('images and targets')
                data = data.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                # print("before pass model")
                # compute output
                with torch.amp.autocast("cuda"):  # type: ignore
                    data = data.to(device)
                    pred, features = model(data)

                    if (
                        args.dataset_type == "loveda"
                        or args.dataset_type == "vaihingen"
                        or args.dataset_type == "potsdam"
                    ):
                        mask = mask.squeeze(1)

                    save_images(data, mask, pred, features, cnt, args)

                cnt += data.shape[0]

    max_iou = max(max_iou, miou)
    print(f"Max IoU: {max_iou:.4f}")

    metric_logger.synchronize_between_processes()

    metric_logger.update(IoU=miou)

    if args.dataset_type == "spacenet":
        print(
            "* IoU {iou:.4f} loss {losses.global_avg:.4f}".format(
                iou=miou, losses=metric_logger.loss
            )
        )
    else:
        metric_logger.update(f1=f1)
        metric_logger.update(oa=oa)
        print(
            "* IoU {iou:.4f} IoU 2 {iou2:.4f} F1 {f1:.4f} oa {oa:.4f} loss {losses.global_avg:.4f}".format(
                iou=miou, iou2=miou_2, f1=f1, oa=oa, losses=metric_logger.loss
            )
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, max_iou


def save_results(data, mask, pred, device, epoch, cnt, miou_test, args):

    if args.dataset_type == "spacenet":  # type: ignore
        miou_temp = JaccardIndex(task="multiclass", num_classes=args.nb_classes)  # type: ignore
    else:
        miou_temp = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, average="micro"
        )
    miou_temp = miou_temp.to(device)

    f = open(
        "satmae_experiments/"
        + args.dataset_type
        + "_"
        + args.dataset_split
        + "pc_results/per_image/"
        + args.method_name
        + "/image_results_iou_"
        + str(epoch)
        + ".txt",
        "a",
    )

    for i in range(data.shape[0]):
        mIoU = miou_temp(pred.argmax(1)[i], mask[i]).item()
        if torch.all(mask[i] == 0) and torch.all(pred.argmax(1)[i] == 0):
            mIoU = 1.0
        miou_test += mIoU
        f.write("img_" + str(cnt + i) + ": " + str(mIoU) + "\n")

    f.close()

    return miou_test


def save_images(data, mask, pred, features, cnt, args):

    for i in range(data.shape[0]):

        if args.visualize_features:
            _, axarr = plt.subplots(4)
            viz_1, _ = visualize_features(features)
        else:
            _, axarr = plt.subplots(3)

        for ax in axarr:
            ax.axis("off")

        axarr[0].imshow(data.cpu()[i].permute(1, 2, 0))

        if args.dataset_type == "spacenet":
            axarr[1].imshow(mask[i].cpu(), interpolation="none")
            axarr[2].imshow(pred.argmax(1).cpu()[i], interpolation="none")

        elif (
            args.dataset_type == "loveda"
            or args.dataset_type == "vaihingen"
            or args.dataset_type == "potsdam"
        ):
            mask_array_1 = np.array(mask[i].cpu())
            mask_array_2 = np.array(pred.argmax(1).cpu()[i])
            color_list = ["white", "red", "yellow", "blue", "violet", "green"]
            cmap_1 = matplotlib.colors.ListedColormap(
                [color_list[j] for j in np.unique(mask_array_1)]
            )
            cmap_2 = matplotlib.colors.ListedColormap(
                [color_list[j] for j in np.unique(mask_array_2)]
            )
            axarr[1].imshow(mask[i].cpu(), cmap=cmap_1, interpolation="none")
            axarr[2].imshow(pred.argmax(1).cpu()[i], cmap=cmap_2, interpolation="none")

        if args.visualize_features:
            axarr[3].imshow(viz_1.permute(1, 2, 0))

        plt.savefig(
            "satmae_experiments/"
            + args.dataset_type
            + "_"
            + args.dataset_split
            + "pc_results/images/"
            + args.method_name
            + "/img_"
            + str(cnt + i)
            + ".png",
            figsize=(3, 1),
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=600,
        )
        plt.close()
