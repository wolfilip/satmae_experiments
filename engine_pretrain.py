# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
import os
import sys
from typing import Iterable

import torch
import wandb
from matplotlib import pyplot as plt

import util.lr_sched as lr_sched
import util.misc as misc
from util.visualize_features import visualize_features


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,  # type: ignore
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter  # type: ignore

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args  # type: ignore
            )

        samples = samples.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):  # type: ignore
            loss_mae, loss_dino, _, _, _ = model(samples, mask_ratio=args.mask_ratio)  # type: ignore

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

        # if data_iter_step == 0:
        #     if not os.path.exists(
        #         "satmae_experiments/feature_visualizations/dinov2_mae_pretrain_reconstruction/epoch_"
        #         + str(epoch)
        #     ):
        #         os.makedirs(
        #             "satmae_experiments/feature_visualizations/dinov2_mae_pretrain_reconstruction/epoch_"
        #             + str(epoch)
        #         )
        #     for i in range(samples.shape[0]):
        #         _, axarr = plt.subplots(2)
        #         axarr[0].imshow(samples.cpu()[i].permute(1, 2, 0))
        #         axarr[1].imshow(
        #             reconstructed_image.cpu().detach()[i].permute(1, 2, 0).double()
        #         )
        #         plt.savefig(
        #             "satmae_experiments/feature_visualizations/dinov2_mae_pretrain_reconstruction/epoch_"
        #             + str(epoch)
        #             + "/feature_"
        #             + str(i)
        #             + ".png"
        #         )
        #         plt.close()

        loss_mae_value, loss_dino_value = loss_mae.item(), loss_dino.item()

        if not math.isfinite(loss_mae_value) or not math.isfinite(loss_dino_value):
            print("Loss is {}, stopping training".format(loss_mae_value))
            raise ValueError(f"Loss is {loss_mae_value}, stopping training")
            # sys.exit(1)

        loss = loss_mae + loss_dino
        loss_value = loss_mae_value + loss_dino_value

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_mae=loss_mae_value, loss_dino=loss_dino_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_mae_value_reduce = misc.all_reduce_mean(loss_mae_value)
        loss_dino_value_reduce = misc.all_reduce_mean(loss_dino_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)  # type: ignore
            log_writer.add_scalar("train_loss_mae", loss_mae_value_reduce, epoch_1000x)
            log_writer.add_scalar(
                "train_loss_dino", loss_dino_value_reduce, epoch_1000x
            )
            log_writer.add_scalar("train_loss_total", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

            # Wandb logging
            if args.local_rank == 0 and args.wandb is not None:  # type: ignore
                try:
                    wandb.log(
                        {
                            "train_loss_step": loss_value_reduce,
                            "train_lr_step": lr,
                            "epoch_1000x": epoch_1000x,
                        }
                    )
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        samples,
    )


def train_one_epoch_temporal(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,  # type: ignore
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
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
    # for data_iter_step, (samples, res, timestamps) in enumerate(
    #     metric_logger.log_every(data_loader, print_freq, header)
    # ):
    for data_iter_step, (samples, resolutions, timestamps, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # for data_iter_step, ((samples, res, targets, target_res, timesteps), metadata) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args  # type: ignore
            )

        samples = [
            samples[0].to(device, non_blocking=True),
            samples[1].to(device, non_blocking=True),
            # samples[2].to(device, non_blocking=True),
        ]
        # resolutions = [
        #     resolutions[0].to(device, non_blocking=True),
        #     resolutions[1].to(device, non_blocking=True),
        #     resolutions[2].to(device, non_blocking=True),
        # ]
        timestamps = timestamps.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # loss, y, mask, mean, var, pos_emb, pos_emb_decoder, samples = model(
            #     samples,
            #     mask_ratio=args.mask_ratio,
            #     timestamps=timestamps,
            # )
            # loss, _, _ = model(samples, timestamps, ratios, mask_ratio=args.mask_ratio)
            loss, _, _ = model(
                samples, resolutions, timestamps, mask_ratio=args.mask_ratio
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)  # type: ignore
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

            # Use wandb
            if args.local_rank == 0 and args.wandb is not None:  # type: ignore
                try:
                    wandb.log(
                        {
                            "train_loss_step": loss_value_reduce,
                            "train_lr_step": lr,
                            "epoch_1000x": epoch_1000x,
                        }
                    )
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_scale(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,  # type: ignore
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    scheduler=None,
    source_size_scheduler=None,
    fix_resolution_scheduler=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    accum_iter = args.accum_iter  # type: ignore

    optimizer.zero_grad()

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, ((samples, res, targets, target_res), _) in enumerate(
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
            target_size = scheduler.get_target_size(epoch)  # type: ignore
            source_size = source_size_scheduler.get_target_size(epoch)[0]  # type: ignore
            fix_decoding_size = fix_resolution_scheduler.get_target_size(epoch)  # type: ignore
            model.module.set_target_size(target_size)
            model.module.set_fix_decoding_size(fix_decoding_size)

            loss_mae, loss_dino, _, _, _, _, _, _, _ = model(
                samples,
                input_res=res,
                targets=targets,
                target_res=target_res,
                mask_ratio=args.mask_ratio,
                source_size=source_size,
            )

        # if data_iter_step % print_freq == 0:
        #     y = [
        #         model.module.unpatchify(y_i)[0].permute(1, 2, 0).detach().cpu()
        #         for y_i in y
        #     ]
        #     x = torch.einsum("nchw->nhwc", samples[:1]).detach().cpu()
        #     wandb_dump_input_output(
        #         x[0],
        #         y,
        #         epoch,
        #         f"target-size:{target_size}-output_size:{fix_decoding_size}",
        #     )
        #     if metadata:
        #         wandb_log_metadata(metadata)

        loss_mae_value, loss_dino_value = loss_mae.item(), loss_dino.item()

        if not math.isfinite(loss_mae_value) or not math.isfinite(loss_dino_value):
            print("Loss is {}, stopping training".format(loss_mae_value))
            raise ValueError(f"Loss is {loss_mae_value}, stopping training")
            # sys.exit(1)

        loss = loss_mae + loss_dino
        loss_value = loss_mae_value + loss_dino_value

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_mae=loss_mae_value, loss_dino=loss_dino_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_mae_value_reduce = misc.all_reduce_mean(loss_mae_value)
        loss_dino_value_reduce = misc.all_reduce_mean(loss_dino_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)  # type: ignore
            log_writer.add_scalar("train_loss_mae", loss_mae_value_reduce, epoch_1000x)
            log_writer.add_scalar(
                "train_loss_dino", loss_dino_value_reduce, epoch_1000x
            )
            log_writer.add_scalar("train_loss_total", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

            # Wandb logging
            if args.local_rank == 0 and args.wandb is not None:  # type: ignore
                try:
                    wandb.log(
                        {
                            "train_loss_step": loss_value_reduce,
                            "train_lr_step": lr,
                            "epoch_1000x": epoch_1000x,
                        }
                    )
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        samples,
    )
