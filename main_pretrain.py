# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import time
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import wandb
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from torch.optim.adamw import AdamW

import models_mae
import models_mae_dinov2
import models_mae_temporal
import util.misc as misc
from engine_pretrain import train_one_epoch, train_one_epoch_temporal
from util.datasets import build_fmow_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.visualize_features import visualize_features


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        default=None,
        choices=["group_c", "temporal", "vanilla", "dinov2_mae"],
        help="Use channel model",
    )
    parser.add_argument(
        "--model",
        default="mae_vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--visualize_features",
        action="store_true",
        default=False,
        help="Visualize first three PCA components",
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--patch_size", default=14, type=int, help="images input size")

    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed patches).",
    )
    parser.add_argument(
        "--spatial_mask",
        action="store_true",
        default=False,
        help="Whether to mask all channels of a spatial location. Only for indp c model",
    )

    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument(
        "--target_size", nargs="*", type=int, help="images input size", default=[224]
    )
    parser.add_argument(
        "--source_size", nargs="*", type=int, help="images source size", default=[224]
    )
    parser.add_argument("--scale_min", default=0.2, type=float, help="Min RRC scale")
    parser.add_argument("--scale_max", default=1.0, type=float, help="Max RRC scale")

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--base_resolution",
        default=2.5,
        type=float,
        help="The base resolution to use for the period of the sin wave for positional embeddings",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )

    # Dataset parameters
    parser.add_argument(
        "--train_path",
        default="/home/train_62classes.csv",
        type=str,
        help="Train .csv path",
    )
    parser.add_argument(
        "--dataset_type",
        default="rgb",
        choices=["rgb", "temporal", "sentinel", "euro_sat", "naip"],
        help="Whether to use fmow rgb, sentinel, or other dataset.",
    )
    parser.add_argument(
        "--masked_bands",
        type=int,
        nargs="+",
        default=None,
        help="Sequence of band indices to mask (with mean val) in sentinel dataset",
    )
    parser.add_argument(
        "--dropped_bands",
        type=int,
        nargs="+",
        default=None,
        help="Which bands (0 indexed) to drop from sentinel data.",
    )
    parser.add_argument(
        "--grouped_bands",
        type=int,
        nargs="+",
        action="append",
        default=[],
        help="Bands to group for GroupC mae",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )

    parser.add_argument("--config", default="config.yaml", type=str, help="Config file")
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="Wandb project name, eg: sentinel_pretrain",
    )

    parser.add_argument(
        "--start_epoch", default=1, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local-rank", default=os.getenv("LOCAL_RANK", 0), type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_fmow_dataset(is_train=True, args=args)

    num_tasks = misc.get_world_size()
    print(num_tasks)
    global_rank = misc.get_rank()
    print(global_rank)
    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.model_type == "temporal":
        model = models_mae_temporal.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss
        )
    elif args.model_type == "dinov2_mae":
        model = models_mae_dinov2.__dict__[args.model](
            img_size=args.input_size,
            patch_size=args.patch_size,
            in_chans=dataset_train.in_c,
            norm_pix_loss=args.norm_pix_loss,
        )
    else:
        model = models_mae.__dict__[args.model](
            img_size=args.input_size,
            patch_size=args.patch_size,
            in_chans=dataset_train.in_c,
            norm_pix_loss=args.norm_pix_loss,
        )
    model.to(global_rank)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # if args.distributed:
    model = DistributedDataParallel(
        model, device_ids=[global_rank], find_unused_parameters=True
    )
    model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_layer_decay(
        model_without_ddp, args.weight_decay
    )
    optimizer = AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    # Set up wandb
    if global_rank == 0 and args.wandb is not None:
        wandb.init(project=args.wandb)
        wandb.config.update(args)
        wandb.watch(model)

    misc.save_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        epoch=0,
    )
    print("saving random init model")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        data_loader_train.sampler.set_epoch(epoch)  # type: ignore

        if args.model_type == "temporal":
            train_stats = train_one_epoch_temporal(
                model,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                log_writer=log_writer,
                args=args,
            )
        else:
            train_stats, samples = train_one_epoch(
                model,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                log_writer=log_writer,
                args=args,
            )

        # if args.visualize_features:  # type: ignore
        #     if not os.path.exists(
        #         "satmae_experiments/feature_visualizations/dinov2_mae_pretrain/"
        #     ):
        #         os.makedirs(
        #             "satmae_experiments/feature_visualizations/dinov2_mae_pretrain/"
        #         )
        #     for i in range(samples.shape[0]):
        #         viz_mae, _ = visualize_features(
        #             torch.unsqueeze(mae_features[i, :, :], 0)
        #         )
        #         viz_dino, _ = visualize_features(
        #             torch.unsqueeze(dinov2_features[i, :, :], 0)
        #         )
        #         _, axarr = plt.subplots(3)
        #         axarr[0].imshow(samples.cpu()[i].permute(1, 2, 0))
        #         axarr[1].imshow(viz_mae.detach().permute(1, 2, 0))
        #         axarr[2].imshow(viz_dino.detach().permute(1, 2, 0))
        #         plt.savefig(
        #             "satmae_experiments/feature_visualizations/dinov2_mae_pretrain/feature_"
        #             + str(epoch + i)
        #             + ".png"
        #         )
        #         plt.close()

        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

            # try:
            #     wandb.log(log_stats)
            # except ValueError:
            #     print(f"Invalid stats?")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
