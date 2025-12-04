# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

from ast import arg
import math
import os
import sys
from argparse import ArgumentParser
from typing import Iterable, Optional

import matplotlib
import matplotlib.pyplot as plt

# from sklearn.metrics import precision_recall_fscore_support

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
from torchmetrics.functional.classification import (
    multilabel_average_precision,
    multilabel_f1_score,
)

# from util.visualize_features import visualize_features
from sklearn.decomposition import PCA


def visualize_features(args, features, is_swin=False):
    """
    Create DINOv2-style PCA feature visualizations for ViT or Swin Transformer features.

    Args:
        features: Feature tensor
            - ViT/DINOv2: (B, N, C) where N = num_patches (e.g., 196 for 14x14)
            - Swin: (B, C, H, W) or (B, N, C) depending on layer
        is_swin: Whether features are from Swin Transformer (may be in (B, C, H, W) format)

    Returns:
        Two visualization tensors (for alternating images) of shape (3, H, W)
    """
    # Check if features are None or a placeholder value (0 or scalar)

    if features is None:
        dummy = torch.zeros(3, 224, 224)
        return dummy, dummy

    # Check if features is a scalar 0 (used as placeholder in some models)
    if isinstance(features, (int, float)) and features == 0:
        dummy = torch.zeros(3, 224, 224)
        return dummy, dummy

    with torch.no_grad():
        features = (
            features.detach().cpu().float()
        )  # Convert to float for quantile calculation

        # # Print feature statistics
        # print("\n" + "="*60)
        # print("Feature Statistics:")
        # print("="*60)
        # print(f"Shape: {features.shape}")
        # print(f"Data type: {features.dtype}")
        # print(f"Min value: {features.min().item():.6f}")
        # print(f"Max value: {features.max().item():.6f}")
        # print(f"Mean: {features.mean().item():.6f}")
        # print(f"Std: {features.std().item():.6f}")
        # print(f"L2 norm (mean): {torch.norm(features, dim=-1).mean().item():.6f}")
        # print(f"% zeros: {(features == 0).float().mean().item() * 100:.2f}%")
        # print(f"% negative: {(features < 0).float().mean().item() * 100:.2f}%")
        # print("="*60 + "\n")

        # Remove outliers by clipping to percentiles
        # This makes PCA visualization more stable and interpretable
        # lower_percentile = torch.quantile(features, 0.01)
        # upper_percentile = torch.quantile(features, 0.99)
        # features = torch.clamp(features, min=lower_percentile, max=upper_percentile)

        # print("\n" + "=" * 60)
        # print("Feature Statistics:")
        # print("=" * 60)
        # print(f"Shape: {features.shape}")
        # print(f"Data type: {features.dtype}")
        # print(f"Min value: {features.min().item():.6f}")
        # print(f"Max value: {features.max().item():.6f}")
        # print(f"Mean: {features.mean().item():.6f}")
        # print(f"Std: {features.std().item():.6f}")
        # print(f"L2 norm (mean): {torch.norm(features, dim=-1).mean().item():.6f}")
        # print(f"% zeros: {(features == 0).float().mean().item() * 100:.2f}%")
        # print(f"% negative: {(features < 0).float().mean().item() * 100:.2f}%")
        # print("=" * 60 + "\n")

        # Handle different feature shapes
        if len(features.shape) == 4:
            # Format: (B, C, H, W) - typical for Swin or conv layers
            if args.model_type == "simdino":
                B, H, W, C = features.shape
                # B, C, H, W = features.shape
                # Reshape to (B*H*W, C) for PCA
                # features_flat = features.permute(0, 2, 3, 1).reshape(-1, C).numpy()
                features_flat = features.reshape(-1, C).numpy()
            elif args.model_type == "copernicusfm":
                B, C, H, W = features.shape
                # Reshape to (B*H*W, C) for PCA
                features_flat = features.permute(0, 2, 3, 1).reshape(-1, C).numpy()
        elif len(features.shape) == 3:
            # Format: (B, N, C) - typical for ViT/DINOv2/Swin patch features
            B, N, C = features.shape
            # Calculate spatial dimensions (assume square grid)
            H = W = int(np.sqrt(N))
            if H * W != N:
                # Handle non-square grids (e.g., Swin with shifted windows)
                # Try common aspect ratios
                for h in range(int(np.sqrt(N)), 0, -1):
                    if N % h == 0:
                        H = h
                        W = N // h
                        break
            # Reshape to (B*N, C) for PCA
            features_flat = features.reshape(-1, C).numpy()
        else:
            raise ValueError(
                f"Unexpected feature shape: {features.shape}. Expected (B, C, H, W) or (B, N, C)"
            )

        # PCA Visualization (DINOv2 style)
        # Reduce to 3 components for RGB visualization
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_flat)

        # Reshape back to spatial dimensions
        features_pca = features_pca.reshape(B, H, W, 3)

        # Normalize each channel to [0, 1] for visualization
        pca_viz = []
        for i in range(B):
            pca_img = features_pca[i].copy()
            # Normalize each RGB channel independently
            for c in range(3):
                channel = pca_img[:, :, c]
                pmin, pmax = channel.min(), channel.max()
                if pmax > pmin:
                    pca_img[:, :, c] = (channel - pmin) / (pmax - pmin)
                else:
                    pca_img[:, :, c] = 0

            # Convert to torch tensor (H, W, 3) -> (3, H, W)
            pca_tensor = torch.from_numpy(pca_img).permute(2, 0, 1).float()
            pca_viz.append(pca_tensor)

        # Ensure we have at least 2 visualizations (for alternating display)
        if len(pca_viz) == 1:
            return pca_viz[0], pca_viz[0]
        else:
            return pca_viz[0], pca_viz[1] if len(pca_viz) > 1 else pca_viz[0]


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
    bce = F.cross_entropy(pred.flatten(2), mask.long().flatten(1), ignore_index=-1)
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
    elif args.dataset_type == "sen1floods11" or args.dataset_type == "soca":  # type: ignore
        miou_metric = JaccardIndex(task="multiclass", num_classes=args.nb_classes, average="macro", ignore_index=-1)  # type: ignore
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
        miou_metric = miou_metric.to(device)

    if epoch == 0:
        if not os.path.exists(
            args.output_dir
            + "images/"
            + args.dataset_type  # type: ignore
            + "_"
            + str(args.dataset_split)  # type: ignore
            + "pc_results/images/"
            + args.method_name  # type: ignore
        ):
            os.makedirs(
                args.output_dir
                + "images/"
                + args.dataset_type  # type: ignore
                + "_"
                + str(args.dataset_split)  # type: ignore
                + "pc_results/images/"
                + args.method_name  # type: ignore
            )
        if not os.path.exists(
            args.output_dir
            + "images/"
            + args.dataset_type  # type: ignore
            + "_"
            + str(args.dataset_split)  # type: ignore
            + "pc_results/per_image/"
            + args.method_name  # type: ignore
        ):
            os.makedirs(
                args.output_dir
                + "images/"
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
        # label = batch[-1].to(device, non_blocking=True)
        # if args.dataset_type == "sen1floods11":
        #     data_ms = batch[1].to(device, non_blocking=True)

        # print(mask.unique())

        with torch.amp.autocast("cuda"):  # type: ignore
            # data = data.to(device)
            # if len(data) == 2:
            #     pred, _ = model((data_rgb, data_depth))
            # else:
            pred, _ = model(data)
            # if args.dataset_type == "sen1floods11":
            #     pred_ms, _ = model(data, data_ms)

            if args.dataset_type == "loveda" or args.dataset_type == "vaihingen" or args.dataset_type == "potsdam":  # type: ignore
                mask = mask.squeeze(1)

            if args.dataset_type != "isaid" and args.dataset_type != "geobench_eurosat" and args.dataset_type != "geobench_bigearthnet" and args.dataset_type != "geobench_forestnet" and args.dataset_type != "geobench_so2sat" and args.dataset_type != "soca":  # type: ignore
                mask_one_hot = F.one_hot(mask, num_classes=args.nb_classes).permute(  # type: ignore
                    0, 3, 1, 2
                )
            # print(mask_one_hot.unique())
            # if args.dataset_type == "sen1floods11" or args.dataset_type == "isaid" or "geobench" in args.dataset_type
            if args.dataset_type == "sen1floods11" or args.dataset_type == "isaid" or args.dataset_type == "soca":  # type: ignore
                loss_value = get_bce_loss_ignore(pred, mask)
            elif args.dataset_type == "geobench_eurosat" or args.dataset_type == "geobench_forestnet" or args.dataset_type == "geobench_so2sat":  # type: ignore
                loss_value = F.cross_entropy(pred, mask)
            elif args.dataset_type == "geobench_bigearthnet":
                loss_value = F.binary_cross_entropy_with_logits(pred, mask.float())
            else:
                loss_value = get_bce_loss(pred, mask_one_hot.float())
            # print(loss_value)
            # dice_loss = DiceLoss()
            # loss_2 = dice_loss(pred, mask_one_hot.float())
            # miou_metric.update(pred.argmax(1), mask)
            if (
                args.dataset_type != "spacenet"
                and args.dataset_type != "sen1floods11"
                and args.dataset_type != "mass_roads"
                and "geobench" not in args.dataset_type
                and args.dataset_type != "PASTIS"
                and args.dataset_type != "soca"
            ):
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
    if args.dataset_type == "spacenet" or args.dataset_type == "sen1floods11" or args.dataset_type == "mass_roads" or "geobench" in args.dataset_type or args.dataset_type == "PASTIS" or args.dataset_type == "soca":  # type: ignore
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
    # criterion = torch.nn.CrossEntropyLoss()
    oa_eurosat = Accuracy(
        task="multiclass",
        num_classes=10,
        average="micro",
    ).to(device)

    oa_so2sat = Accuracy(
        task="multiclass",
        num_classes=17,
        average="micro",
    ).to(device)

    f1_score = F1Score(
        task="multilabel",
        num_labels=43,
        average="micro",
    ).to(device)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        target = batch[-1]
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.amp.autocast("cuda"):  # type: ignore
            output, _ = model(images)
            # loss = criterion(output, target)

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # score = torch.sigmoid(output) > 0.5
        if output.shape[-1] == 43:
            f1_score.update(output, target.float())
        elif output.shape[-1] == 10:
            oa_eurosat.update(output, target.float())
        else:
            oa_so2sat.update(output, target.float())
        # score = torch.sigmoid(output).detach()
        # target = F.one_hot(target, num_classes=10)
        # acc1 = (
        #     multilabel_average_precision(score, target, num_labels=10, average="micro")
        #     * 100
        # )
        # print(acc1, acc5, flush=True)

        # acc1 = (
        #     multilabel_average_precision(score, target, num_labels=43, average="micro")
        #     * 100
        # )
        # acc5 = multilabel_f1_score(score, target, num_labels=43, average="micro") * 100
        # f1 = f1_score(score, target.float()) * 100

        # _, _, f1, _ = precision_recall_fscore_support(
        #     y_true=target.cpu().numpy(),
        #     y_pred=score.cpu().numpy(),
        #     average="micro",
        #     zero_division=0,
        # )
        # f1 = f1 * 100

    metric_logger.update(loss=0)
    metric_logger.meters["acc5"].update(0)
    if output.shape[-1] == 43:
        metric_logger.meters["acc1"].update(0)
        metric_logger.meters["f1"].update(f1_score.compute().item() * 100)
    elif output.shape[-1] == 10:
        metric_logger.meters["acc1"].update(oa_eurosat.compute().item() * 100)
        metric_logger.meters["f1"].update(0)
    else:
        metric_logger.meters["acc1"].update(oa_so2sat.compute().item() * 100)
        metric_logger.meters["f1"].update(0)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} F1 {f1.global_avg:.3f}".format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss,
            f1=metric_logger.f1,
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
def evaluate_segmentation(data_loader, is_test, model, device, epoch, max_iou, args):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    cnt = 0

    if args.eval:
        if not os.path.exists(
            args.output_dir
            + "images/"
            + args.dataset_type
            + "_"
            + str(args.dataset_split)
            + "pc_results/images/"
            + args.method_name
        ):
            os.makedirs(
                args.output_dir
                + "images/"
                + args.dataset_type
                + "_"
                + str(args.dataset_split)
                + "pc_results/images/"
                + args.method_name
            )
        if not os.path.exists(
            args.output_dir
            + "images/"
            + args.dataset_type
            + "_"
            + str(args.dataset_split)
            + "pc_results/per_image/"
            + args.method_name
        ):
            os.makedirs(
                args.output_dir
                + "images/"
                + args.dataset_type
                + "_"
                + str(args.dataset_split)
                + "pc_results/per_image/"
                + args.method_name
            )
        if not os.path.exists(
            args.output_dir
            + "images/"
            + args.dataset_type
            + "_"
            + str(args.dataset_split)
            + "pc_results/PCA/"
            + args.method_name
        ):
            os.makedirs(
                args.output_dir
                + "images/"
                + args.dataset_type
                + "_"
                + str(args.dataset_split)
                + "pc_results/PCA/"
                + args.method_name
            )

    if args.dataset_type == "spacenet" or args.dataset_type == "mass_roads":  # type: ignore
        miou_metric = JaccardIndex(task="multiclass", num_classes=args.nb_classes)  # type: ignore
    elif args.dataset_type == "sen1floods11" or args.dataset_type == "soca":  # type: ignore
        miou_metric = JaccardIndex(task="multiclass", num_classes=args.nb_classes, average="macro", ignore_index=-1)  # type: ignore
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
    elif "geobench" in args.dataset_type or args.dataset_type == "PASTIS":  # type: ignore
        miou_metric = JaccardIndex(
            task="multiclass",
            num_classes=args.nb_classes,
            average="macro",
            # ignore_index=0,
        )
        miou_metric_2 = JaccardIndex(
            task="multiclass",
            num_classes=args.nb_classes,
            average="micro",
            # ignore_index=0,
        )
        f1_score = F1Score(
            task="multiclass",
            num_classes=args.nb_classes,
            average="micro",
            # ignore_index=0,
        )
        overall_accuracy = Accuracy(
            task="multiclass",
            num_classes=args.nb_classes,
            average="weighted",
            # ignore_index=0,
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

            if args.dataset_type != "geobench_eurosat" and args.dataset_type != "geobench_bigearthnet" and args.dataset_type != "soca":  # type: ignore
                mask_one_hot = F.one_hot(mask, num_classes=args.nb_classes).permute(
                    0, 3, 1, 2
                )
            if (
                args.dataset_type == "geobench_eurosat"
                or args.dataset_type == "geobench_bigearthnet"
            ):
                loss = F.cross_entropy(pred, mask.long())
            elif args.dataset_type == "soca":
                loss = get_bce_loss_ignore(pred, mask)
            else:
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
                and args.dataset_type != "geobench_eurosat"
                and args.dataset_type != "geobench_bigearthnet"
                and args.dataset_type != "soca"
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
        and args.dataset_type != "soca"
    ):
        miou_2 = miou_metric_2.compute().item()
        # miou_3 = miou_metric_3.compute().item()
        # miou_4 = miou_metric_4.compute().item()
        f1 = f1_score.compute().item()
        oa = overall_accuracy.compute().item()

    if is_test:
        print(f"Test IoU: {miou:.4f}")
    else:
        max_iou = max(max_iou, miou)
        print(f"Max IoU: {max_iou:.4f}")

    if args.save_images or args.visualize_features:
        cnt = 0

        # Initialize feature accumulator if visualize_features is enabled
        feature_accumulator = None
        if args.visualize_features:
            feature_accumulator = {"features": [], "labels": [], "image_ids": []}

        # Create unique folder structure once at the start
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{args.method_name}_{timestamp}"

        base_output_dir = os.path.join(
            args.output_dir,
            "images",
            args.dataset_type + "_" + str(args.dataset_split) + "pc_results",
            "individual_images",
            folder_name,
        )

        # Create subdirectories for different image types
        os.makedirs(os.path.join(base_output_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(base_output_dir, "gt_mask"), exist_ok=True)
        os.makedirs(os.path.join(base_output_dir, "pred_mask"), exist_ok=True)
        if args.visualize_features:
            os.makedirs(os.path.join(base_output_dir, "features"), exist_ok=True)

        if args.best_epoch:
            if (miou > max_iou and epoch > -1) or epoch == -1:
                for batch in data_loader:
                    data = batch[0]
                    if (
                        args.dataset_type == "sen1floods11"
                        or "geobench" in args.dataset_type
                    ):
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
                        if (
                            args.dataset_type == "sen1floods11"
                            or "geobench" in args.dataset_type
                            or args.dataset_type == "soca"
                        ):
                            save_images(
                                data_viz,
                                mask,
                                pred,
                                features,
                                cnt,
                                args,
                                feature_accumulator,
                                base_output_dir,
                            )
                        else:
                            save_images(
                                data,
                                mask,
                                pred,
                                features,
                                cnt,
                                args,
                                feature_accumulator,
                                base_output_dir,
                            )
                    cnt += data.shape[0]

                # Save accumulated features once after all batches
                if args.visualize_features and feature_accumulator is not None:
                    # Concatenate all patch tokens from all images
                    # Each element in features list is (num_patches_i, feature_dim)
                    all_features = torch.cat(feature_accumulator["features"], dim=0)
                    all_labels = torch.cat(feature_accumulator["labels"], dim=0)

                    save_path = os.path.join(
                        args.output_dir,
                        "images",
                        args.dataset_type
                        + "_"
                        + str(args.dataset_split)
                        + "pc_results",
                        "PCA",
                        args.method_name,
                        f"class_features_epoch_{epoch}.npz",
                    )

                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    save_class_features_for_tsne(
                        features=(
                            all_features,
                        ),  # Wrap in tuple to match expected format
                        labels=all_labels,
                        image_ids=feature_accumulator["image_ids"],
                        save_path=save_path,
                    )

        elif args.eval or is_test:
            for batch in data_loader:
                data = batch[0]
                rgb_data = batch[1]
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
                        save_images(
                            data_viz,
                            mask,
                            pred,
                            features,
                            cnt,
                            args,
                            feature_accumulator,
                            base_output_dir,
                        )
                    elif args.dataset_type == "soca":
                        save_images(
                            data,
                            mask,
                            pred,
                            features,
                            cnt,
                            args,
                            feature_accumulator,
                            base_output_dir,
                        )
                    else:
                        save_images(
                            rgb_data,
                            mask,
                            pred,
                            features,
                            cnt,
                            args,
                            feature_accumulator,
                            base_output_dir,
                        )

                cnt += data.shape[0]

            # Save accumulated features once after all batches
            if args.visualize_features and feature_accumulator is not None:
                all_features = torch.cat(feature_accumulator["features"], dim=0)
                all_labels = torch.cat(feature_accumulator["labels"], dim=0)

                save_path = os.path.join(
                    args.output_dir,
                    "images",
                    args.dataset_type + "_" + str(args.dataset_split) + "pc_results",
                    "PCA",
                    args.method_name,
                    f"class_features_epoch_{epoch}.npz",
                )

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                save_class_features_for_tsne(
                    features=(all_features,),  # Wrap in tuple to match expected format
                    labels=all_labels,
                    image_ids=feature_accumulator["image_ids"],
                    save_path=save_path,
                )

    metric_logger.synchronize_between_processes()

    metric_logger.update(IoU=miou)

    if (
        args.dataset_type == "spacenet"
        or args.dataset_type == "sen1floods11"
        or args.dataset_type == "mass_roads"
        or args.dataset_type == "soca"
    ):
        print(
            "* IoU {iou:.4f} loss {losses.global_avg:.4f}".format(
                iou=miou, losses=metric_logger.loss
            )
        )
    else:
        metric_logger.update(f1=f1)
        metric_logger.update(oa=oa)
        metric_logger.update(iou2=miou_2)
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
    elif args.dataset_type == "sen1floods11" or args.dataset_type == "soca":  # type: ignore
        miou_temp = JaccardIndex(task="multiclass", num_classes=args.nb_classes, average="macro", ignore_index=-1)  # type: ignore
    else:
        miou_temp = JaccardIndex(
            task="multiclass", num_classes=args.nb_classes, average="micro"
        )
    miou_temp = miou_temp.to(device)

    f = open(
        args.output_dir
        + "images/"
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


def save_class_features_for_tsne(features, labels, image_ids, save_path):
    """
    Save patch token features (from model output) in NumPy format for t-SNE visualization.

    Args:
        features: Tensor or list containing patch features
                 Shape should be (total_patches, feature_dim) after concatenation
        labels: Patch-level or image-level labels (total_patches,)
        image_ids: List of image identifiers for each patch
        save_path: Path to save the .npz file

    The saved file can be loaded with:
        data = np.load('features.npz', allow_pickle=True)
        features = data['features']  # (N_patches, D) array
        labels = data['labels']      # (N_patches,) array
        image_ids = data['image_ids'] # (N_patches,) array of image identifiers
    """
    import numpy as np
    import torch

    # Extract features from list/tuple
    if isinstance(features, (list, tuple)):
        class_features = features[0]  # Features should be concatenated already
    else:
        class_features = features

    # Convert to numpy
    if isinstance(class_features, torch.Tensor):
        class_features = class_features.detach().cpu().numpy()

    # Process labels
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Ensure labels is 1D
    if len(labels.shape) > 1:
        labels = labels.flatten()

    # Save to npz file
    np.savez_compressed(
        save_path, features=class_features, labels=labels, image_ids=np.array(image_ids)
    )

    return class_features.shape


def save_images(
    data,
    mask,
    pred,
    features,
    cnt,
    args,
    feature_accumulator=None,
    base_output_dir=None,
):
    """
    Args:
        feature_accumulator: Optional dict to accumulate features across batches
                            {'features': [], 'labels': [], 'image_ids': []}
        base_output_dir: Base directory for saving images. If None, uses default path.
    """

    # Use provided base_output_dir or create default path
    if base_output_dir is None:
        base_output_dir = os.path.join(
            args.output_dir,
            "images",
            args.dataset_type + "_" + str(args.dataset_split) + "pc_results",
            "images",
            args.method_name,
        )

    # Accumulate features if visualize_features is enabled and accumulator is provided
    if args.visualize_features and feature_accumulator is not None:
        batch_size = data.shape[0]

        # Extract patch token features from the features list/tuple
        # Based on the model type, features may be in different positions
        if isinstance(features, (list, tuple)):
            if args.model_type == "simdino":
                # SimDINO can use either ViT or Swin backbone
                # Check if it's using Swin (features[1] is a list of feature maps)
                if isinstance(features[1], (list, tuple)) and len(features[1]) > 0:
                    # Swin backbone: features[1] is a list of 4 feature maps
                    # Use one of the intermediate feature maps (not too early, not too late)
                    patch_features = features[1][2]  # 3rd stage features
                else:
                    # ViT backbone: use features[1][2]
                    patch_features = features[1][2]
            elif (
                args.model_type == "copernicusfm"
                or args.model_type == "dinov2_segmentation"
            ):
                # Use intermediate layer features for ViT-based models
                patch_features = features[1][
                    3
                ]  # features[1][3] for copernicusfm/dinov2
            elif args.model_type == "swin":
                # Swin returns list of 4 feature maps from different stages
                # Use an intermediate stage for visualization
                patch_features = features[2]  # 3rd stage features
            else:
                patch_features = features[-1]  # Default: last element
        else:
            patch_features = features

        # Handle different feature shapes - extract patch tokens
        if len(patch_features.shape) == 3:
            # Shape: (batch_size, num_patches, feature_dim)
            # This is already in patch token format - keep all patches
            pass
        elif len(patch_features.shape) == 4:
            # Shape: (batch_size, feature_dim, H, W)
            # Reshape to (batch_size, H*W, feature_dim)
            B, C, H, W = patch_features.shape
            patch_features = patch_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # If shape is (batch_size, feature_dim), add patch dimension
        elif len(patch_features.shape) == 2:
            patch_features = patch_features.unsqueeze(1)  # (B, 1, D)

        # For each image in the batch, store its patches with labels
        for i in range(batch_size):
            img_patches = patch_features[i]  # (num_patches, feature_dim)
            num_patches = img_patches.shape[0]

            # Get the most common class for this image
            mask_flat = mask[i].cpu().flatten()
            unique, counts = torch.unique(mask_flat, return_counts=True)
            most_common_label = unique[counts.argmax()].item()

            # Store patches for this image
            feature_accumulator["features"].append(img_patches.detach().cpu())

            # Assign the same label to all patches of this image
            patch_labels = torch.full(
                (num_patches,), most_common_label, dtype=torch.long
            )
            feature_accumulator["labels"].append(patch_labels)

            # Store image ID for each patch
            img_id = f"img_{cnt + i}"
            patch_ids = [f"{img_id}_patch_{j}" for j in range(num_patches)]
            feature_accumulator["image_ids"].extend(patch_ids)

    for i in range(data.shape[0]):
        img_id = cnt + i

        # Save RGB image
        fig_rgb, ax_rgb = plt.subplots(1, 1, figsize=(8, 8))
        ax_rgb.axis("off")

        if args.dataset_type == "sen1floods11":
            ax_rgb.imshow(sentinel2_l2a_to_rgb(data[i].cpu()).permute(1, 2, 0))
        elif "geobench" in args.dataset_type:
            if "cashew" in args.dataset_type:
                ax_rgb.imshow(
                    sentinel2_l2a_to_rgb(data[i][:3, ...].cpu()).permute(1, 2, 0)
                )
            else:
                ax_rgb.imshow(data[i][:3, ...].cpu().permute(1, 2, 0))
        else:
            ax_rgb.imshow(data[i].cpu().permute(1, 2, 0))

        plt.savefig(
            os.path.join(base_output_dir, "rgb", f"img_{img_id}.png"),
            bbox_inches="tight",
            pad_inches=0.0,
            dpi=300,
        )
        plt.close(fig_rgb)

        # Prepare colormaps based on dataset type
        if (
            args.dataset_type == "spacenet"
            or args.dataset_type == "isaid"
            or args.dataset_type == "mass_roads"
        ):
            color_list = [
                "#000000",  # Background – black
                "#e31a1c",  # Building footprint – strong red
            ]
            mask_array_1 = np.array(mask[i].cpu())
            mask_array_2 = np.array(pred.argmax(1).cpu()[i])
            cmap_gt = matplotlib.colors.ListedColormap(
                [color_list[j] for j in np.unique(mask_array_1)]
            )
            cmap_pred = matplotlib.colors.ListedColormap(
                [color_list[j] for j in np.unique(mask_array_2)]
            )
        elif args.dataset_type == "sen1floods11":
            color_list = ["white", "grey", "blue"]
            mask_array_1 = np.array(mask[i].cpu())
            mask_array_2 = np.array(pred.argmax(1).cpu()[i])
            cmap_gt = matplotlib.colors.ListedColormap(
                [color_list[j] for j in np.unique(mask_array_1)]
            )
            cmap_pred = matplotlib.colors.ListedColormap(
                [color_list[j] for j in np.unique(mask_array_2)]
            )
        elif "geobench" in args.dataset_type:
            color_list = [
                "#000000",  # No Data – black
                "#1b9e77",  # Lucerne/Medics – teal green
                "#66a61e",  # Planted pastures (perennial) – olive green
                "#e6ab02",  # Fallow – mustard yellow
                "#7570b3",  # Wine grapes – violet
                "#d95f02",  # Weeds – orange
                "#a6761d",  # Small grain grazing – brownish ochre
                "#e7298a",  # Wheat – magenta
                "#1f78b4",  # Canola – medium blue
                "#b2df8a",  # Rooibos – light green
            ]
            cmap_gt = matplotlib.colors.ListedColormap(color_list[: args.nb_classes])
            cmap_pred = cmap_gt
        elif (
            args.dataset_type == "loveda"
            or args.dataset_type == "vaihingen"
            or args.dataset_type == "potsdam"
        ):
            mask_array_1 = np.array(mask[i].cpu())
            mask_array_2 = np.array(pred.argmax(1).cpu()[i])
            color_list = ["white", "red", "yellow", "blue", "violet", "green"]
            cmap_gt = matplotlib.colors.ListedColormap(
                [color_list[j] for j in np.unique(mask_array_1)]
            )
            cmap_pred = matplotlib.colors.ListedColormap(
                [color_list[j] for j in np.unique(mask_array_2)]
            )
        else:
            cmap_gt = None
            cmap_pred = None

        # Save ground truth mask
        fig_gt, ax_gt = plt.subplots(1, 1, figsize=(8, 8))
        ax_gt.axis("off")
        if "geobench" in args.dataset_type and cmap_gt is not None:
            ax_gt.imshow(
                mask[i].cpu(),
                cmap=cmap_gt,
                vmin=0,
                vmax=args.nb_classes - 1,
                interpolation="none",
            )
        elif cmap_gt is not None:
            ax_gt.imshow(mask[i].cpu(), cmap=cmap_gt, interpolation="none")
        else:
            ax_gt.imshow(mask[i].cpu(), interpolation="none")

        plt.savefig(
            os.path.join(base_output_dir, "gt_mask", f"img_{img_id}.png"),
            bbox_inches="tight",
            pad_inches=0.0,
            dpi=300,
        )
        plt.close(fig_gt)

        # Save prediction mask
        fig_pred, ax_pred = plt.subplots(1, 1, figsize=(8, 8))
        ax_pred.axis("off")
        if "geobench" in args.dataset_type and cmap_pred is not None:
            ax_pred.imshow(
                pred.argmax(1).cpu()[i],
                cmap=cmap_pred,
                vmin=0,
                vmax=args.nb_classes - 1,
                interpolation="none",
            )
        elif cmap_pred is not None:
            ax_pred.imshow(
                pred.argmax(1).cpu()[i], cmap=cmap_pred, interpolation="none"
            )
        else:
            ax_pred.imshow(pred.argmax(1).cpu()[i], interpolation="none")

        plt.savefig(
            os.path.join(base_output_dir, "pred_mask", f"img_{img_id}.png"),
            bbox_inches="tight",
            pad_inches=0.0,
            dpi=300,
        )
        plt.close(fig_pred)

        # Save feature visualizations if enabled
        if args.visualize_features:
            if args.model_type == "simdino":
                viz_feat1_2, _ = visualize_features(args, features[1][2])
            elif (
                args.model_type == "copernicusfm"
                or args.model_type == "dinov2_segmentation"
            ):
                viz_feat1_2, _ = visualize_features(args, features[1][3])

            fig_feat, ax_feat = plt.subplots(1, 1, figsize=(8, 8))
            ax_feat.axis("off")
            ax_feat.imshow(viz_feat1_2.permute(1, 2, 0))
            # ax_feat.set_title(f"PCA Features - img_{img_id}", fontsize=10)

            plt.savefig(
                os.path.join(base_output_dir, "features", f"img_{img_id}.png"),
                bbox_inches="tight",
                pad_inches=0.0,
                dpi=300,
            )
            plt.close(fig_feat)


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
