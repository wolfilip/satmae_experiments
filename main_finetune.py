# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import datetime
import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import wandb
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import trunc_normal_
from torch.nn.parallel import DistributedDataParallel
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torch.utils.tensorboard.writer import SummaryWriter

from models.DINOv2_detection import DINOv2Detector
from models.DINOv2_segmentation import DINOv2Segmenter
from models.OmniSat import LTAE, Omni, OmniSat
from models.SimDINO_features import SimDINO
from models.SimDINOv2_features import SimDINOv2
import models.models_resnet as models_resnet

# from models.models_swin import SwinModel
from models.models_swin import SwinModel
import models.models_vit as models_vit
import models.models_vit_dinov2_segmentation as models_vit_dinov2_segmentation
import models.models_vit_group_channels as models_vit_group_channels
import models.models_vit_segmentation as models_vit_segmentation
import models.models_vit_temporal as models_vit_temporal
from models.simdinov2_models import build_model
from models.simdinov2_models.utils import load_pretrained_weights
import util.lr_decay as lrd
import util.misc as misc
from engine_finetune import (
    evaluate,
    evaluate_segmentation,
    evaluate_temporal,
    train_one_epoch,
    train_one_epoch_frcnn,
    train_one_epoch_segmentation,
    train_one_epoch_temporal,
)
from models import MAE_LiFT_model
from models.SAMHQ_model import SAMHQ
from util.datasets import build_fmow_dataset, collate_fn_dior
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed
from util.utils_swin import remap_pretrained_keys_swin


def get_args_parser():
    parser = ArgumentParser("MAE fine-tuning for image classification", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--visualize_features",
        action="store_true",
        default=False,
        help="Visualize first three PCA components",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        default=False,
    )
    # Model parameters
    parser.add_argument(
        "--model_type",
        default=None,
        choices=[
            "group_c",
            "resnet",
            "resnet_pre",
            "temporal",
            "vanilla",
            "segmentation",
            "dinov2_segmentation",
            "dinov2_detection",
            "samhq_segmentation",
            "lift_segmentation",
            "dinov2_classification",
            "dinov2_vit",
            "omnisat",
            "swin",
            "simdinov2",
            "simdino",
        ],
        help="Use channel model",
    )
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--patch_size", default=16, type=int, help="images input size")

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
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
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    # Dataset parameters
    # parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
    #                     help='dataset path')
    parser.add_argument(
        "--train_path",
        default="/home/train_62classes.csv",
        type=str,
        help="Train .csv path",
    )
    parser.add_argument(
        "--test_path",
        default="/home/val_62classes.csv",
        type=str,
        help="Test .csv path",
    )
    parser.add_argument(
        "--dataset_type",
        default="rgb",
        choices=[
            "rgb",
            "temporal",
            "sentinel",
            "euro_sat",
            "naip",
            "spacenet",
            "loveda",
            "vaihingen",
            "potsdam",
            "sen1floods11",
            "isaid",
            "mass_roads",
            "dior",
        ],
        help="Whether to use fmow rgb, sentinel, or other dataset.",
    )
    parser.add_argument(
        "--dataset_split",
        default="100",
        choices=["100", "20", "10"],
        help="What percentage split of the data to use.",
    )
    parser.add_argument(
        "--method_name",
        default="debug_method",
        help="Method name used for saving preocedures.",
    )
    parser.add_argument(
        "--best_epoch",
        action="store_true",
        default=False,
        help="Save images after every new best epoch.",
    )
    parser.add_argument(
        "--masked_bands",
        default=None,
        nargs="+",
        type=int,
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
        help="Bands to group for GroupC vit",
    )

    parser.add_argument(
        "--nb_classes", default=62, type=int, help="number of the classification types"
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="How frequently (in epochs) to save ckpt",
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="Wandb project name, eg: sentinel_finetune",
    )

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
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

    dataset_train = build_fmow_dataset(is_train=True, data_split="trainval", args=args)
    dataset_val = build_fmow_dataset(is_train=False, data_split="test", args=args)
    # dataset_test = build_fmow_dataset(is_train=False, data_split="test", args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:  # type: ignore
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )  # shuffle=True to reduce monitor bias

        sampler_test = DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )  # shuffle=True to reduce monitor bias
    else:
        sampler_val = SequentialSampler(dataset_val)  # type: ignore
        dataset_test = SequentialSampler(dataset_test)  # type: ignore

    if global_rank == 0 and args.log_dir is not None and not args.eval:
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
        collate_fn=collate_fn_dior,
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn_dior,
    )

    data_loader_test = DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn_dior,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    # Define the model
    if args.model_type == "group_c":
        # Workaround because action append will add to default list
        if len(args.grouped_bands) == 0:
            args.grouped_bands = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]
        print(f"Grouping bands {args.grouped_bands}")
        model = models_vit_group_channels.__dict__[args.model](
            patch_size=args.patch_size,
            img_size=args.input_size,
            in_chans=dataset_train.in_c,
            channel_groups=args.grouped_bands,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    elif args.model_type == "resnet" or args.model_type == "resnet_pre":
        pre_trained = args.model_type == "resnet_pre"
        model = models_resnet.__dict__[args.model](
            in_c=dataset_train.in_c, pretrained=pre_trained
        )
    elif args.model_type == "temporal":
        model = models_vit_temporal.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    elif args.model_type == "segmentation":
        model = models_vit_segmentation.__dict__[args.model](
            patch_size=args.patch_size,
            img_size=args.input_size,
            in_chans=3,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
        )
    elif args.model_type == "lift_segmentation":
        model = MAE_LiFT_model.__dict__[args.model](
            patch_size=args.patch_size,
            img_size=args.input_size,
            in_chans=dataset_train.in_c,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
        )
    elif (
        args.model_type == "dinov2_segmentation"
        or args.model_type == "dinov2_classification"
    ):
        model = DINOv2Segmenter(args, "cuda")
    elif args.model_type == "dinov2_detection":
        model = DINOv2Detector(args, "cuda")
    elif args.model_type == "samhq_segmentation":
        model = SAMHQ(args, "cuda")
    elif args.model_type == "omnisat":
        projectors = {"s2": LTAE.LTAE(["s2"])}
        encoder = Omni.OmniModule(projectors, ["s2"])
        model = OmniSat.Fine(encoder, "omnisat_state_dict.pth")
    elif args.model_type == "swin":
        # model_name = transformers.AutoModel.from_pretrained(args.finetune)
        model = SwinModel(args, device)
    elif args.model_type == "simdinov2":
        model = SimDINOv2(args, "cuda")
        if args.input_size is not None and args.input_size != 224:
            print(
                f"OPTIONS -- evaluation img size: resizing from 224 to {args.input_size}"
            )
            model.update_img_size(args.input_size)
    elif args.model_type == "simdino":
        model = SimDINO(args, "cuda")
    elif args.model_type == "dinov2_vit":
        model = models_vit_dinov2_segmentation.__dict__[args.model](
            patch_size=args.patch_size,
            img_size=args.input_size,
            in_chans=dataset_train.in_c,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
        )
    else:
        model = models_vit.__dict__[args.model](
            patch_size=args.patch_size,
            img_size=args.input_size,
            in_chans=dataset_train.in_c,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

    # model = torch.compile(model, dynamic=False)

    if args.finetune and args.model_type != "swin" and "simdino" not in args.model_type:
        checkpoint = torch.load(args.finetune, map_location="cpu")
        # print(checkpoint_model)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()

        if args.model_type == "swin":
            if any(
                [True if "encoder." in k else False for k in checkpoint_model.keys()]
            ):
                checkpoint_model = {
                    k.replace("encoder.", ""): v
                    for k, v in checkpoint_model.items()
                    if k.startswith("encoder.")
                }
                print("Detect pre-trained model, remove [encoder.] prefix.")
            else:
                print("Detect non-pre-trained model, pass without doing anything.")
            checkpoint_model = remap_pretrained_keys_swin(model, checkpoint_model)

            for k in [
                "head.weight",
                "head.bias",
            ]:
                if (
                    k in checkpoint_model
                    and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        else:
            for k in [
                "pos_embed",
                "patch_embed.proj.weight",
                "patch_embed.proj.bias",
                "head.weight",
                "head.bias",
            ]:
                if (
                    k in checkpoint_model
                    and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            if "cross_scale" in args.finetune:
                for key in list(checkpoint_model.keys()):
                    checkpoint_model[key.replace("encoder", "blocks")] = (
                        checkpoint_model.pop(key)
                    )

        # load pre-trained model

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # TODO: change assert msg based on patch_embed
        if args.global_pool:
            print(set(msg.missing_keys))
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            print(set(msg.missing_keys))
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    # for param in model.parameters():
    #     print(param)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    model = DistributedDataParallel(
        model, device_ids=[global_rank], find_unused_parameters=True
    )
    model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    if args.model_type is not None and (
        args.model_type.startswith("resnet")
        or args.model_type == "dinov2_classification"
        or args.model_type == "dinov2_segmentation"
        or args.model_type == "dinov2_detection"
        or args.model_type == "samhq_segmentation"
        or args.model_type == "lift_segmentation"
        or args.model_type == "swin"
        or "simdino" in args.model_type
    ):
        param_groups = model_without_ddp.parameters()
    else:
        param_groups = lrd.param_groups_lrd(
            model_without_ddp,
            args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay,
        )
    optimizer = AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

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

    if args.eval:
        if args.model_type == "temporal":
            test_stats = evaluate_temporal(data_loader_val, model, device)
        elif (
            args.model_type == "segmentation"
            or args.model_type == "dinov2_segmentation"
            or args.model_type == "samhq_segmentation"
            or args.model_type == "lift_segmentation"
            or args.model_type == "dinov2_vit"
            or args.model_type == "swin"
            or "simdino" in args.model_type
        ):
            test_stats, max_iou = evaluate_segmentation(
                data_loader_val, model, device, 0, 0, args
            )
        else:
            test_stats = evaluate(data_loader_val, model, device)

        if (
            args.model_type == "segmentation"
            or args.model_type == "dinov2_segmentation"
            or args.model_type == "samhq_segmentation"
            or args.model_type == "lift_segmentation"
            or args.model_type == "dinov2_vit"
            or args.model_type == "swin"
            or "simdino" in args.model_type
        ):
            print(
                f"mIoU of the network on the {len(dataset_val)} test images: {test_stats['IoU']:.4f}"  # type: ignore
            )
        else:
            print(
                f"Evaluation on {len(dataset_val)} test images- acc1: {test_stats['acc1']:.2f}%, "  # type: ignore
                f"acc5: {test_stats['acc5']:.2f}%"
            )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_iou = 0.0
    current_iou = 0.0
    # best_model = torch.clone(model)
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        data_loader_train.sampler.set_epoch(epoch)  # type: ignore

        if args.eval is False:
            if args.model_type == "temporal":
                train_stats = train_one_epoch_temporal(
                    model,
                    criterion,
                    data_loader_train,
                    optimizer,
                    device,
                    epoch,
                    loss_scaler,
                    log_writer,
                    args,
                    mixup_fn,
                )
            elif args.model_type == "dinov2_detection":
                train_stats = train_one_epoch_frcnn(
                    model,
                    data_loader_train,
                    optimizer,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                )
            elif (
                args.model_type == "segmentation"
                or args.model_type == "dinov2_segmentation"
                or args.model_type == "samhq_segmentation"
                or args.model_type == "lift_segmentation"
                or args.model_type == "dinov2_vit"
                or args.model_type == "swin"
                or "simdino" in args.model_type
            ):
                # test_stats = evaluate_segmentation(data_loader_val, model, device)
                # print(
                #     f"mIoU of the network on the {len(dataset_val)} test images: {test_stats['IoU']:.3f}"
                # )
                # max_iou = max(max_iou, test_stats["IoU"])
                # print(f"Max IoU: {max_iou:.3f}")

                # if log_writer is not None:
                #     log_writer.add_scalar("perf/test_iou", test_stats["IoU"], epoch)
                #     log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

                train_stats = train_one_epoch_segmentation(
                    model,
                    data_loader_train,
                    optimizer,
                    device,
                    epoch,
                    loss_scaler,
                    log_writer,
                    args,
                    mixup_fn,
                    args.clip_grad,
                )
            else:
                train_stats = train_one_epoch(
                    model,
                    criterion,
                    data_loader_train,
                    optimizer,
                    device,
                    epoch,
                    loss_scaler,
                    log_writer,
                    args,
                    mixup_fn,
                )

        if args.output_dir and (
            epoch % args.save_every == 0 or epoch + 1 == args.epochs
        ):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        if args.model_type == "temporal":
            test_stats = evaluate_temporal(data_loader_val, model, device)
        elif (
            args.model_type == "segmentation"
            or args.model_type == "dinov2_segmentation"
            or args.model_type == "samhq_segmentation"
            or args.model_type == "lift_segmentation"
            or args.model_type == "dinov2_vit"
            or args.model_type == "swin"
            or "simdino" in args.model_type
        ):
            test_stats, max_iou = evaluate_segmentation(
                data_loader_val, model, device, epoch, max_iou, args
            )

            if max_iou > current_iou:
                current_iou = max_iou
                print("Saving best model")
                misc.save_best_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
        elif args.model_type == "dinov2_detection":
            print(train_stats)
        else:
            test_stats = evaluate(data_loader_val, model, device)

        if (
            args.model_type == "segmentation"
            or args.model_type == "dinov2_segmentation"
            or args.model_type == "samhq_segmentation"
            or args.model_type == "lift_segmentation"
            or args.model_type == "dinov2_vit"
            or args.model_type == "swin"
            or "simdino" in args.model_type
        ):
            # print(
            #     f"mIoU of the network on the {len(dataset_val)} test images: {test_stats['IoU']:.4f}"  # type: ignore
            # )

            if log_writer is not None:
                log_writer.add_scalar("perf/val_iou", test_stats["IoU"], epoch)
                log_writer.add_scalar("perf/val_loss", test_stats["loss"], epoch)
                if (
                    args.dataset_type != "spacenet"
                    and args.dataset_type != "sen1floods11"
                    and args.dataset_type != "mass_roads"
                ):
                    log_writer.add_scalar("perf/val_f1", test_stats["f1"], epoch)

        else:
            print(
                f"Accuracy of the network on the {len(dataset_val)} val images: {test_stats['acc1']:.1f}%"  # type: ignore
            )
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f"Max accuracy: {max_accuracy:.2f}%")

            if log_writer is not None:
                log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
                log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
                log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

        if args.eval is False:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        else:
            log_stats = {
                **{f"val_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.wandb is not None:
                try:
                    wandb.log(log_stats)
                except ValueError:
                    print("Invalid stats?")

    misc.load_best_model(args, model)

    epoch = -1

    test_stats, max_iou = evaluate_segmentation(
        data_loader_test, model, device, epoch, max_iou, args
    )

    if log_writer is not None:
        log_writer.add_scalar("perf/test_iou", test_stats["IoU"], epoch)
        log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)
        if (
            args.dataset_type != "spacenet"
            and args.dataset_type != "sen1floods11"
            and args.dataset_type != "mass_roads"
        ):
            log_writer.add_scalar("perf/test_f1", test_stats["f1"], epoch)

    if args.eval is False:
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
    else:
        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(
            os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        if args.wandb is not None:
            try:
                wandb.log(log_stats)
            except ValueError:
                print("Invalid stats?")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
