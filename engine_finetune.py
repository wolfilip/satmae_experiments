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

matplotlib.use("Agg")
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
from torchmetrics import Accuracy, F1Score, JaccardIndex, Metric

import util.lr_sched as lr_sched
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.visualize_features import visualize_features


class SegPangaea(Metric):
    """
    SegPangaea is a class for evaluating segmentation models using a confusion matrix approach.

    Attributes:
        num_classes (int): Number of classes in the segmentation task
        ignore_index (int): Index value to ignore when computing metrics
        confusion_matrix (torch.Tensor): Matrix of shape (num_classes, num_classes) to store predictions

    Methods:
        update(pred, gt):
            Updates the confusion matrix with new predictions and ground truth.
            Args:
                pred (torch.Tensor): Model predictions
                gt (dict): Dictionary containing ground truth labels under 'label' key

        compute():
            Computes various metrics from the accumulated confusion matrix.
            Returns:
                dict: Dictionary containing the following metrics:
                    - mIoU: Mean Intersection over Union across all classes
                    - mF1: Mean F1 score across all classes
                    - mAcc: Mean pixel accuracy
    """

    def __init__(self, num_classes, ignore_index):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros(num_classes, num_classes)

    def update(self, pred, gt):
        label = gt.flatten(1, 2)
        pred = pred.flatten(1, 2)
        valid_mask = label != self.ignore_index
        pred, target = pred[valid_mask], label[valid_mask]
        count = torch.bincount(
            (pred * self.num_classes + target), minlength=self.num_classes**2
        )
        self.confusion_matrix = self.confusion_matrix.to(pred.device)
        self.confusion_matrix += count.view(self.num_classes, self.num_classes)

    def compute(self):
        # Calculate IoU for each class
        intersection = torch.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(dim=1)
            + self.confusion_matrix.sum(dim=0)
            - intersection
        )
        iou = intersection / (union + 1e-6)

        # Calculate precision and recall for each class
        precision = intersection / (self.confusion_matrix.sum(dim=0) + 1e-6)
        recall = intersection / (self.confusion_matrix.sum(dim=1) + 1e-6)

        # Calculate F1-score for each class
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        # Calculate mean IoU, mean F1-score, and mean Accuracy
        miou = iou.mean().item()
        mf1 = f1.mean().item()
        macc = (intersection.sum() / (self.confusion_matrix.sum() + 1e-6)).item()

        # Convert metrics to CPU and to Python scalars
        iou = iou.cpu()
        f1 = f1.cpu()
        precision = precision.cpu()
        recall = recall.cpu()

        # Prepare the metrics dictionary
        metrics = {
            "mIoU": miou,
            "mF1": mf1,
            "mAcc": macc,
        }

        return metrics


def get_bce_loss(pred, mask):
    bce = nn.BCEWithLogitsLoss()
    # m = nn.Sigmoid()
    # pred = pred.argmax(1)
    # loss = F.binary_cross_entropy_with_logits(pred, mask)
    # loss = bce(torch.clamp(pred, min=0.0001, max=1.0), torch.clamp(mask, min=0.0001, max=1.0))
    loss = bce(pred, mask)
    return loss


def get_bce_loss_ignore(pred, mask):
    # print(pred.unique(), mask.unique())
    # m = F.sigmoid(pred)
    bce = F.cross_entropy(pred, mask.long(), ignore_index=0)
    # print(bce)
    # m = nn.Sigmoid()
    # pred = pred.argmax(1)
    # loss = F.binary_cross_entropy_with_logits(pred, mask)
    # loss = bce(torch.clamp(pred, min=0.0001, max=1.0), torch.clamp(mask, min=0.0001, max=1.0))
    # loss = bce(pred, mask)
    return bce


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

        with torch.amp.autocast("cuda"):  # type: ignore
            outputs, _ = model(samples)
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
    print_freq = 50

    accum_iter = args.accum_iter  # type: ignore

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if args.dataset_type == "spacenet" or args.dataset_type == "mass_roads":  # type: ignore
        miou_metric = JaccardIndex(task="multiclass", num_classes=args.nb_classes)  # type: ignore
    elif args.dataset_type == "sen1floods11":  # type: ignore
        miou_metric = JaccardIndex(task="multiclass", num_classes=args.nb_classes, average="micro", ignore_index=0)  # type: ignore
        # miou_metric = SegPangaea(num_classes=args.nb_classes, ignore_index=0)
    elif args.dataset_type == "isaid":
        miou_metric = JaccardIndex(
            task="multiclass",
            num_classes=args.nb_classes,
            average="micro",
            ignore_index=0,
        )
        miou_metric_2 = JaccardIndex(
            task="multiclass",
            num_classes=args.nb_classes,
            average="weighted",
            ignore_index=0,
        )
        f1_score = F1Score(
            task="multiclass",
            num_classes=args.nb_classes,
            average="micro",
            ignore_index=0,
        )
        overall_accuracy = Accuracy(
            task="multiclass",
            num_classes=args.nb_classes,
            average="weighted",
            ignore_index=0,
        )

        f1_score = f1_score.to(device)
        miou_metric_2 = miou_metric_2.to(device)
        overall_accuracy = overall_accuracy.to(device)
    else:
        miou_metric = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, average="micro"  # type: ignore
        )
        miou_metric_2 = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, average="weighted"  # type: ignore
        )
        f1_score = F1Score(
            task="multiclass", num_classes=args.nb_classes, average="micro"  # type: ignore
        )
        overall_accuracy = Accuracy(
            task="multiclass", num_classes=args.nb_classes, average="weighted"  # type: ignore
        )
        f1_score = f1_score.to(device)
        miou_metric_2 = miou_metric_2.to(device)
        overall_accuracy = overall_accuracy.to(device)
    # miou_metric = miou_metric.to(device)

    if epoch == 0:
        if not os.path.exists(
            "satmae_experiments/"
            + args.dataset_type  # type: ignore
            + "_"
            + str(args.dataset_split)  # type: ignore
            + "pc_results/images/"
            + args.method_name  # type: ignore
        ):
            os.makedirs(
                "satmae_experiments/"
                + args.dataset_type  # type: ignore
                + "_"
                + str(args.dataset_split)  # type: ignore
                + "pc_results/images/"
                + args.method_name  # type: ignore
            )
        if not os.path.exists(
            "satmae_experiments/"
            + args.dataset_type  # type: ignore
            + "_"
            + str(args.dataset_split)  # type: ignore
            + "pc_results/per_image/"
            + args.method_name  # type: ignore
        ):
            os.makedirs(
                "satmae_experiments/"
                + args.dataset_type  # type: ignore
                + "_"
                + str(args.dataset_split)  # type: ignore
                + "pc_results/per_image/"
                + args.method_name  # type: ignore
            )

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args  # type: ignore
            )
        # if len(data) == 2:
        #     data_rgb = data[0].to(device, non_blocking=True)
        #     data_depth = data[1].to(device, non_blocking=True)
        # else:
        data = batch[0].to(device, non_blocking=True)
        mask = batch[-1].to(device, non_blocking=True)
        # if args.dataset_type == "sen1floods11":
        #     data_ms = batch[1].to(device, non_blocking=True)

        # print(mask.unique())

        with torch.amp.autocast("cuda"):  # type: ignore
            # data = data.to(device)
            # if len(data) == 2:
            #     pred, _ = model((data_rgb, data_depth))
            # else:
            if args.model_type == "croma":
                pred = model(data)
            else:
                pred, _ = model(data)
            # if args.dataset_type == "sen1floods11":
            #     pred_ms, _ = model(data, data_ms)

            if args.dataset_type == "loveda" or args.dataset_type == "vaihingen" or args.dataset_type == "potsdam":  # type: ignore
                mask = mask.squeeze(1)

            if args.dataset_type != "isaid":
                mask_one_hot = F.one_hot(mask, num_classes=args.nb_classes).permute(  # type: ignore
                    0, 3, 1, 2
                )
            # print(mask_one_hot.unique())

            if args.dataset_type == "sen1floods11" or args.dataset_type == "isaid":  # type: ignore
                loss_value = get_bce_loss_ignore(pred, mask)
            else:
                loss_value = get_bce_loss(pred, mask_one_hot.float())
            # print(loss_value)
            # dice_loss = DiceLoss()
            # loss_2 = dice_loss(pred, mask_one_hot.float())
            # miou_metric.update(pred.argmax(1), mask)
            if args.dataset_type != "spacenet" and args.dataset_type != "sen1floods11" and args.dataset_type != "mass_roads":  # type: ignore
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
    if args.dataset_type == "spacenet" or args.dataset_type == "sen1floods11" or args.dataset_type == "mass_roads":  # type: ignore
        print("* loss {losses.global_avg:.4f}".format(losses=metric_logger.loss))
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
        with torch.amp.autocast("cuda"):  # type: ignore
            output, _ = model(images)
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
            + str(args.dataset_split)
            + "pc_results/images/"
            + args.method_name
        ):
            os.makedirs(
                "satmae_experiments/"
                + args.dataset_type
                + "_"
                + str(args.dataset_split)
                + "pc_results/images/"
                + args.method_name
            )
        if not os.path.exists(
            "satmae_experiments/"
            + args.dataset_type
            + "_"
            + str(args.dataset_split)
            + "pc_results/per_image/"
            + args.method_name
        ):
            os.makedirs(
                "satmae_experiments/"
                + args.dataset_type
                + "_"
                + str(args.dataset_split)
                + "pc_results/per_image/"
                + args.method_name
            )

    if args.dataset_type == "spacenet" or args.dataset_type == "mass_roads":  # type: ignore
        miou_metric = JaccardIndex(task="multiclass", num_classes=args.nb_classes)  # type: ignore
    elif args.dataset_type == "sen1floods11":  # type: ignore
        miou_metric = JaccardIndex(task="multiclass", num_classes=args.nb_classes, average="micro", ignore_index=0)  # type: ignore
        # miou_metric = SegPangaea(num_classes=args.nb_classes, ignore_index=0)
    elif args.dataset_type == "isaid":
        miou_metric = JaccardIndex(
            task="multiclass",
            num_classes=args.nb_classes,
            average="micro",
            ignore_index=0,
        )
        # miou_metric_3 = JaccardIndex(
        #     task="multiclass",
        #     num_classes=args.nb_classes,
        #     average="micro",
        # )
        miou_metric_2 = JaccardIndex(
            task="multiclass",
            num_classes=args.nb_classes,
            average="macro",
            ignore_index=0,
        )
        miou_metric_4 = JaccardIndex(
            task="multiclass",
            num_classes=args.nb_classes,
            average="macro",
        )
        f1_score = F1Score(
            task="multiclass",
            num_classes=args.nb_classes,
            average="micro",
            ignore_index=0,
        )
        overall_accuracy = Accuracy(
            task="multiclass",
            num_classes=args.nb_classes,
            average="weighted",
            ignore_index=0,
        )

        f1_score = f1_score.to(device)
        miou_metric_2 = miou_metric_2.to(device)
        # miou_metric_3 = miou_metric_2.to(device)
        # miou_metric_4 = miou_metric_2.to(device)
        overall_accuracy = overall_accuracy.to(device)
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

    for batch in metric_logger.log_every(data_loader, 50, header):
        data = batch[0]
        mask = batch[-1]
        # print('images and targets')
        # if len(data) == 2:
        #     data_rgb = data[0].to(device, non_blocking=True)
        #     data_depth = data[1].to(device, non_blocking=True)
        # else:
        data = data.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.amp.autocast("cuda"):  # type: ignore
            # data = data.to(device)
            # print(data.shape)
            # if len(data) == 2:
            #     pred, _ = model((data_rgb, data_depth))
            # else:
            if args.model_type == "croma":
                pred = model(data)
            else:
                pred, _ = model(data)
            # pred = torch.full_like(
            #     mask, fill_value=mask.flatten().mode().values.item(), device=device
            # )  # Predict the majority class

            # _, axarr = plt.subplots(3)

            # axarr[0].imshow(data[0].cpu().squeeze().permute(1, 2, 0))
            # axarr[1].imshow(mask[0].cpu(), interpolation="none")
            # axarr[2].imshow(pred[0].argmax(0).cpu(), interpolation="none")

            # plt.savefig(
            #     "satmae_experiments/"
            #     + args.dataset_type
            #     + "_"
            #     + str(args.dataset_split)
            #     + "pc_results/images/"
            #     + args.method_name
            #     + "/img_"
            #     + str(cnt)
            #     + ".png",
            #     figsize=(3, 1),
            #     bbox_inches="tight",
            #     pad_inches=0.1,
            #     dpi=600,
            # )

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
            # loss = 2
            # dice_loss = DiceLoss()
            # loss_2 = dice_loss(pred, mask_one_hot.float())
            miou_metric.update(pred.argmax(1), mask)
            # miou_metric.update(pred, mask)
            if (
                args.dataset_type != "spacenet"
                and args.dataset_type != "sen1floods11"
                and args.dataset_type != "mass_roads"
            ):
                miou_metric_2.update(pred.argmax(1), mask)
                # miou_metric_3.update(pred.argmax(1), mask)
                # miou_metric_4.update(pred.argmax(1), mask)
                f1_score.update(pred.argmax(1), mask)
                overall_accuracy.update(pred.argmax(1), mask)

            miou_test = save_results(mask, pred, device, epoch, cnt, miou_test, args)

        cnt += args.batch_size

        # batch_size = images.shape[0]
        metric_logger.update(loss=loss)
        # metric_logger.meters['IoU'].update(IoU, n=batch_size)

    if args.dataset_type == "spacenet":
        print("miou test: " + str(miou_test * args.world_size / 1940))
    elif args.dataset_type == "vaihingen":
        print("miou test: " + str(miou_test * args.world_size / 398))
    elif args.dataset_type == "potsdam":
        print("miou test: " + str(miou_test * args.world_size / 2016))
    elif args.dataset_type == "sen1floods11":
        print("miou test: " + str(miou_test * args.world_size / 89))
    elif args.dataset_type == "isaid":
        print("miou test: " + str(miou_test * args.world_size / 11644))
    elif args.dataset_type == "mass_roads":
        print("miou test: " + str(miou_test * args.world_size / 49))

    # gather the stats from all processes
    miou = miou_metric.compute().item()

    if (
        args.dataset_type != "spacenet"
        and args.dataset_type != "sen1floods11"
        and args.dataset_type != "mass_roads"
    ):
        miou_2 = miou_metric_2.compute().item()
        # miou_3 = miou_metric_3.compute().item()
        # miou_4 = miou_metric_4.compute().item()
        f1 = f1_score.compute().item()
        oa = overall_accuracy.compute().item()

    if args.save_images:
        cnt = 0
        if args.best_epoch:
            if (miou > max_iou and epoch > 10) or epoch == -1:
                for batch in data_loader:
                    data = batch[0]
                    if args.dataset_type == "sen1floods11":
                        data_viz = batch[1]
                        data_viz = data_viz.to(device, non_blocking=True)
                    mask = batch[-1]
                    # print('images and targets')
                    data = data.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)

                    # print("before pass model")
                    # compute output
                    with torch.amp.autocast("cuda"):  # type: ignore
                        pred, features = model(data)

                        if (
                            args.dataset_type == "loveda"
                            or args.dataset_type == "vaihingen"
                            or args.dataset_type == "potsdam"
                        ):
                            mask = mask.squeeze(1)
                        if args.dataset_type == "sen1floods11":
                            save_images(data_viz, mask, pred, features, cnt, args)
                        else:
                            save_images(data, mask, pred, features, cnt, args)
                    cnt += data.shape[0]
        elif epoch == args.epochs - 1 or args.eval:
            for batch in data_loader:
                data = batch[0]
                if args.dataset_type == "sen1floods11":
                    data_viz = batch[1]
                    data_viz = data_viz.to(device, non_blocking=True)
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

                    if args.dataset_type == "sen1floods11":
                        save_images(data_viz, mask, pred, features, cnt, args)
                    else:
                        save_images(data, mask, pred, features, cnt, args)

                cnt += data.shape[0]

    max_iou = max(max_iou, miou)
    print(f"Max IoU: {max_iou:.4f}")

    metric_logger.synchronize_between_processes()

    metric_logger.update(IoU=miou)

    if (
        args.dataset_type == "spacenet"
        or args.dataset_type == "sen1floods11"
        or args.dataset_type == "mass_roads"
    ):
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
                iou=miou,
                iou2=miou_2,
                f1=f1,
                oa=oa,
                losses=metric_logger.loss,
            )
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, max_iou


def save_results(mask, pred, device, epoch, cnt, miou_test, args):

    if args.dataset_type == "spacenet":  # type: ignore
        miou_temp = JaccardIndex(task="multiclass", num_classes=args.nb_classes)  # type: ignore
    elif args.dataset_type == "sen1floods11":  # type: ignore
        miou_temp = JaccardIndex(task="multiclass", num_classes=args.nb_classes, average="micro", ignore_index=0)  # type: ignore
    else:
        miou_temp = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, average="micro"
        )
    miou_temp = miou_temp.to(device)

    f = open(
        "satmae_experiments/"
        + args.dataset_type
        + "_"
        + str(args.dataset_split)
        + "pc_results/per_image/"
        + args.method_name
        + "/image_results_iou_"
        + str(epoch)
        + ".txt",
        "a",
    )

    for i in range(pred.shape[0]):
        mIoU = miou_temp(pred.argmax(1)[i], mask[i]).item()
        if torch.all(mask[i] == 0) and torch.all(pred.argmax(1)[i] == 0):
            mIoU = 1.0
        miou_test += mIoU
        f.write("img_" + str(cnt + i) + ": " + str(mIoU) + "\n")

    f.close()

    return miou_test


def sentinel2_l2a_to_rgb(image):
    min_val = 0.0
    max_val = 0.3
    rgb_image = (image / 10000 - min_val) / (max_val - min_val)
    rgb_image[rgb_image < 0] = 0
    rgb_image[rgb_image > 1] = 1
    return rgb_image


def save_images(data, mask, pred, features, cnt, args):

    for i in range(data.shape[0]):

        if args.visualize_features:
            _, axarr = plt.subplots(5)
            if features[0] != 0:
                viz_conv_1, viz_conv_2 = visualize_features(features[0], True)
            viz_vit_1, viz_vit_2 = visualize_features(features[1], False)
        else:
            _, axarr = plt.subplots(3)

        for ax in axarr:
            ax.axis("off")

        # bla = sentinel2_l2a_to_rgb(data.cpu()[i])

        if args.dataset_type == "sen1floods11":
            axarr[0].imshow(sentinel2_l2a_to_rgb(data[i].cpu()).permute(1, 2, 0))
        else:
            axarr[0].imshow(data[i].cpu().permute(1, 2, 0))

        if (
            args.dataset_type == "spacenet"
            or args.dataset_type == "isaid"
            or args.dataset_type == "mass_roads"
        ):
            axarr[1].imshow(mask[i].cpu(), interpolation="none")
            axarr[2].imshow(pred.argmax(1).cpu()[i], interpolation="none")
        elif args.dataset_type == "sen1floods11":
            color_list = ["white", "grey", "blue"]
            mask_array_1 = np.array(mask[i].cpu())
            mask_array_2 = np.array(pred.argmax(1).cpu()[i])
            cmap_1 = matplotlib.colors.ListedColormap(
                [color_list[j] for j in np.unique(mask_array_1)]
            )
            cmap_2 = matplotlib.colors.ListedColormap(
                [color_list[j] for j in np.unique(mask_array_2)]
            )
            axarr[1].imshow(mask[i].cpu(), cmap=cmap_1, interpolation="none")
            axarr[2].imshow(pred.argmax(1).cpu()[i], cmap=cmap_2, interpolation="none")
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
            if i % 2 == 0:
                axarr[3].imshow(viz_vit_1.permute(1, 2, 0))
                if features[0] != 0:
                    axarr[4].imshow(viz_conv_1.permute(1, 2, 0))
            else:
                axarr[3].imshow(viz_vit_2.permute(1, 2, 0))
                if features[0] != 0:
                    axarr[4].imshow(viz_conv_2.permute(1, 2, 0))

            if features[0] == 0:
                plt.savefig(
                    "satmae_experiments/"
                    + args.dataset_type
                    + "_"
                    + str(args.dataset_split)
                    + "pc_results/images/"
                    + args.method_name
                    + "/img_"
                    + str(cnt + i)
                    + ".png",
                    figsize=(4, 1),
                    bbox_inches="tight",
                    pad_inches=0.1,
                    dpi=600,
                )
            else:
                plt.savefig(
                    "satmae_experiments/"
                    + args.dataset_type
                    + "_"
                    + str(args.dataset_split)
                    + "pc_results/images/"
                    + args.method_name
                    + "/img_"
                    + str(cnt + i)
                    + ".png",
                    figsize=(5, 1),
                    bbox_inches="tight",
                    pad_inches=0.1,
                    dpi=600,
                )

        else:
            plt.savefig(
                "satmae_experiments/"
                + args.dataset_type
                + "_"
                + str(args.dataset_split)
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


def train_one_epoch_frcnn(
    model, data_loader, optimizer, device, epoch, log_writer=None, args=None
):
    model.train()
    total_loss = 0.0

    for images, targets in data_loader:
        # images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Pass resized images and transformed ta0rgets
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

        # print(f"Epoch [{epoch}] - Loss: {losses:.4f}")

    avg_loss = total_loss / len(data_loader)
    return {"loss": avg_loss}


@torch.no_grad()
def evaluate_frcnn(data_loader, model, device, class_names=None):
    model.eval()

    total_loss = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    num_samples = 0
    coco_predictions = []

    # For each batch
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Temporarily switch to train to get loss (eval doesn't compute it)
        model.train()
        loss_dict = model(images, targets)
        for k in total_loss:
            total_loss[k] += loss_dict[k].item()
        num_samples += 1

        # Switch back to eval for predictions
        model.eval()
        predictions = model(images)

        for i, pred in enumerate(predictions):
            image_id = int(targets[i]["image_id"].item())  # must match GT JSON
            boxes = pred["boxes"].cpu()
            scores = pred["scores"].cpu()
            labels = pred["labels"].cpu()

            for box, score, label in zip(boxes, scores, labels):
                if score < 0.05:
                    continue
                x1, y1, x2, y2 = box.tolist()
                coco_predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

    # Generate COCO-style GT annotations
    from torch import distributed as dist

    # Only rank 0 creates the JSON
    if misc.is_main_process():
        coco_gt_path = create_coco_json_from_dataset(
            data_loader.dataset, save_dir=tempfile.gettempdir()
        )
        dist.barrier()  # Let other processes wait
    else:
        dist.barrier()  # Wait for rank 0
        coco_gt_path = os.path.join(tempfile.gettempdir(), "coco_gt.json")
    if len(coco_predictions) == 0:
        print("[❗] No predictions made — coco_predictions is empty.")
        return {
            "loss_classifier": 0.0,
            "loss_box_reg": 0.0,
            "loss_objectness": 0.0,
            "loss_rpn_box_reg": 0.0,
            "mAP_50": 0.0,
            "mAP_50_95": 0.0,
            "per_class_ap": {},
        }
    coco_gt = COCO(coco_gt_path)
    coco_gt = COCO(coco_gt_path)
    coco_dt = coco_gt.loadRes(coco_predictions)

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    mAP_50_95 = coco_eval.stats[0] * 100
    mAP_50 = coco_eval.stats[1] * 100

    # Per-class AP
    per_class_ap = {}
    if class_names:
        precisions = coco_eval.eval["precision"]  # [T, R, K, A, M]
        for idx, class_name in enumerate(class_names):
            cls_prec = precisions[:, :, idx, 0, 0]
            cls_prec = cls_prec[cls_prec > -1]
            ap = cls_prec.mean() if cls_prec.size > 0 else float("nan")
            per_class_ap[class_name] = round(ap * 100, 2)

    # Final loss averages
    avg_losses = {k: v / num_samples for k, v in total_loss.items()}

    return {
        **avg_losses,
        "mAP_50": mAP_50,
        "mAP_50_95": mAP_50_95,
        "per_class_ap": per_class_ap,
    }
