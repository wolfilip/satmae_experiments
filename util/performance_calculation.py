import csv
import gc
import sys
import timeit

from tqdm import tqdm
from argparse import ArgumentParser

sys.path.append("../")

from models import models_vit_segmentation
from models.DINOv2_features import DINOv2
import torch


def get_args_parser():
    parser = ArgumentParser("MAE fine-tuning for image classification", add_help=False)
    parser.add_argument(
        "--model",
        default="base_0",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
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
        ],
        help="Whether to use fmow rgb, sentinel, or other dataset.",
    )
    parser.add_argument(
        "--nb_classes", default=62, type=int, help="number of the classification types"
    )
    return parser


def params():
    config = {
        "--nproc_per_node",
        "1",
        "--model",
        "base_0",
        "--input_size",
        "224",
        "--dataset_type",
        "spacenet",
    }
    # model = DINOv2(config, "cuda")

    model = models_vit_segmentation.__dict__[args.model](
        patch_size=16,
        img_size=args.input_size,
        in_chans=3,
        num_classes=args.nb_classes,
        drop_path_rate=0.1,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total params:", total_params, "Trainable params:", trainable_params)


def prepare_image(batched=False):
    if batched:
        img = torch.randn(64, 3, 224, 224, dtype=torch.float16)
    else:
        img = torch.randn(1, 3, 224, 224, dtype=torch.float16)

    return img


def prepare_model(args):
    # model = DINOv2(args, "cuda")

    model = models_vit_segmentation.__dict__[args.model](
        patch_size=16,
        img_size=args.input_size,
        in_chans=3,
        num_classes=args.nb_classes,
        drop_path_rate=0.1,
    )

    checkpoint = torch.load(
        "../../scale-mae/scalemae-vitlarge-800.pth", map_location="cpu"
    )
    msg = model.load_state_dict(checkpoint["model"], strict=False)

    model.to("cuda")
    model.to(torch.float16)
    model.eval()

    return model


@torch.no_grad()
def inference_speed(reps, args):
    model = prepare_model(args)
    img = prepare_image()

    # first - warmup
    for _ in tqdm(range(reps), desc="Warmup"):
        img = img.to("cpu")

        img = img.to("cuda")
        out = model(img)

    total_time = 0
    # next - real
    for _ in tqdm(range(reps), desc="Timing inference"):
        img = img.to("cpu")

        img = img.to("cuda")
        t0 = timeit.default_timer()
        out = model(img)
        t1 = timeit.default_timer()

        total_time += t1 - t0

    # * 1000 to get ms
    ms = total_time * 1000 / reps
    print("Speed in ms:", ms)
    return ms


@torch.no_grad()
def throughput(reps, args):
    model = prepare_model(args)
    img = prepare_image(batched=True)

    # first - warmup
    for _ in tqdm(range(reps), desc="Warmup"):
        img = img.to("cpu")

        img = img.to("cuda")
        _ = model(img)

    total_time = 0
    # next - real
    for _ in tqdm(range(reps), desc="Throughput"):
        img = img.to("cpu")

        img = img.to("cuda")
        t0 = timeit.default_timer()
        _ = model(img)
        t1 = timeit.default_timer()

        total_time += t1 - t0

    thru = 64 * reps / total_time
    print("Throughput:", thru)
    return thru


@torch.no_grad()
def memory(reps, args):
    model = prepare_model(args)
    img = prepare_image()
    img = img.to("cuda")

    # first - warmup
    for _ in tqdm(range(reps), desc="Warmup"):
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        _ = model(img)

        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_memory = 0
    # next - real
    for _ in tqdm(range(reps), desc="Memory calc"):
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        _ = model(img)

        total_memory += torch.cuda.max_memory_reserved()

        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # MB -> 10**6 bytes, then "reps" runs
    mbs = total_memory / (10**6) / reps
    print("Memory in MB:", mbs)
    return mbs


@torch.no_grad()
def flops(reps, args):
    model = prepare_model(args)
    img = prepare_image()
    img = img.to("cuda")

    # first - warmup
    _ = model(img)

    # real - don't need reps as the result is always same
    with torch.profiler.profile(with_flops=True) as prof:
        _ = model(img)
    tflops = sum(x.flops for x in prof.key_averages()) / 1e9
    print("TFLOPS:", tflops)

    return tflops


def main(args):
    cycles = 6
    reps = 1000

    torch.backends.cudnn.deterministic = True

    with open(f"perf_{args.model}.csv", "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["time", "throughput", "memory", "tflops"])
        for cyc in range(cycles):
            ms = inference_speed(reps, args)
            thru = throughput(reps, args)
            mbs = memory(reps, args)
            tflops = flops(reps, args)

            if cyc == 0:
                # skip first one, as the system is not warmed up and it's too fast
                continue

            writer.writerow([ms, thru, mbs, tflops])
            print("-" * 42)
            print("Speed [ms]:", ms)
            print("Throughput:", thru)
            print("Memory [MB]:", mbs)
            print("TFLOPS:", tflops)
            print("-" * 42)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    main(args)
    # params()
