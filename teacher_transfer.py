# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""Transfer learning of teacher models.

Basically, this file has been created to perform transfer learning on pretrained RegNetY models
in order to use them for other datasets (CIFAR10 for instance). We are using the same
parameters as in main.py, except that there is only a teacher model. The final linear layer of
the teacher model is modified and trained again. The checkpoint can then be used in main.py as
a new teacher model for training a student model.
"""


import datetime
import time

import json
from pathlib import Path

import argparse

import numpy as np

import torch
from torch import nn
from torch.backends import cudnn

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma


import utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler


def get_args_parser():
    """Parser getter."""

    parser = argparse.ArgumentParser("Script for training transfered models.", add_help=False)

    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Training batch size. Test batch size is 1.5 times this parameter. Default: 64."
    )
    parser.add_argument(
        "--epochs",
        default=300,
        type=int,
        help="Number of training epochs. Default: 300."
    )

    # Teacher model parameters
    parser.add_argument(
        "--teacher-model",
        default="regnety_160",
        type=str,
        metavar="MODEL",
        help="""Name of teacher model to train. Default: "regnety_160"."""
    )
    parser.add_argument(
        "--teacher-path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--input-size",
        default=224,
        type=int,
        help="Input size for the images. Default: 224."
    )

    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate. Default: 0.0."
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate. Default: 0.1."
    )

    parser.add_argument(
        "--model-ema",
        action="store_true"
    )
    parser.add_argument(
        "--no-model-ema",
        action="store_false",
        dest="model_ema"
    )
    parser.set_defaults(model_ema=True)
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99996,
        help="Decay for the EMA model. Default: 0.99996."
    )
    parser.add_argument(
        "--model-ema-force-cpu",
        action="store_true",
        default=False,
        help="Force CPU for EMA model."
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help="""Optimizer. Default: "adamw"."""
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Epsilon value for the optimizer, if needed. Default: 1e-8."
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Betas values for the optimizer, if needed. Default: None."
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm. Default: None."
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Momentum value for the optimizer, if needed. Default: 0.9."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay value for the optimizer, if needed. Default: 0.05."
    )

    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help="""LR scheduler. Default: "cosine"."""
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="Learning rate. Default: 5e-4."
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="Learning rate noise on/off epoch percentages. Default: None."
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="Learning rate noise limit percent. Default: 0.67."
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="Learning rate noise std. Default: 1.0."
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="Warmup learning rate. Default: 1e-6."
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="Lower bound for cyclic schedulers that hit 0. Default: 1e-5."
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="Epoch interval to decay LR. Default: 30."
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="Number of warmup epochs, if supported by the scheduler. Default: 5."
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="Epochs to cooldown LR at min_lr, after cyclic schedule ends. Default: 10."
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="Patience epochs for the plateau LR scheduler. Default: 10."
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="Decay rate for the LR scheduler. Default: 0.1."
    )

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="""
             Color jitter (random changes in brightness, contrast and saturation) factor.
             Default: 0.4.
             """
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help="""Use AutoAugment policy. "v0" or "original". Default: "rand-m9-mstd0.5-inc1"."""
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
        help="Label smoothing (uniform shift). Default: 0.1."
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help="""Training interpolation: "random", "bilinear" or "bicubic". Default: "bicubic"."""
    )

    parser.add_argument(
        "--repeated-aug",
        action="store_true"
    )
    parser.add_argument(
        "--no-repeated-aug",
        action="store_false",
        dest="repeated_aug"
    )
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase probability on images. Default: 0.25."
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help="""Random erase mode. Default: "pixel"."""
    )
    parser.add_argument(
        "--recount",
        type=int,
        default=1,
        help="Random erase count. Default: 1."
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split."
    )

    # * Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="Alpha value for MixUp. Enabled if > 0. Default: 0.8."
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="Alpha value for CutMix. Enabled if > 0. Default: 1.0."
    )
    parser.add_argument(
        "--cutmix-minmax",
        type=float,
        nargs="+",
        default=None,
        help="""
             Min/max ratio for CutMix. This parameter overrides alpha and enables CutMix if set.
             Default: None.
             """
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="""
             Probability of performing MixUp or CutMix when either/both is enabled. Default: 1.0.
             """
    )
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="""
             Probability of switching to CutMix when both MixUp and CutMix are enabled.
             Default: 0.5.
             """
    )
    parser.add_argument(
        "--mixup-mode",
        type=str,
        default="batch",
        help="""
             How to apply MixUp/CutMix params. Per "batch", "pair", or "elem". Default: "batch".
             """
    )

    # Dataset parameters
    parser.add_argument(
        "--data-path",
        default="/datasets01/imagenet_full_size/061417/",
        type=str,
        help="""Path to the dataset. Default: "/datasets01/imagenet_full_size/061417/"."""
    )
    parser.add_argument(
        "--data-set",
        default="IMNET",
        choices=["CIFAR10", "CIFAR100", "IMNET", "INAT", "INAT19"],
        type=str,
        help="""
             Name of the dataset to use: "CIFAR10", "CIFAR100", "IMNET", "INAT" or "INAT19".
             Default: "IMNET".
             """
    )
    parser.add_argument(
        "--inat-category",
        default="name",
        choices=["kingdom", "phylum", "class", "order", "supercategory", "family", "genus", "name"],
        type=str,
        help="""Semantic granularity value for INAT. Default: "name"."""
    )

    parser.add_argument(
        "--output_dir",
        default="",
        help="""Path where to save, empty for no saving. Default: ""."""
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="""Device to use for training and testing. Default: "cuda"."""
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int
    )
    parser.add_argument(
        "--resume",
        default="",
        help="""Resume from checkpoint. Default: ""."""
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="Start epoch. Default: 0."
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int
    )
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU."
    )
    parser.add_argument(
        "--no-pin-mem",
        action="store_false",
        dest="pin_mem",
        help=""
    )
    parser.set_defaults(pin_mem=True)

    return parser


def main(args: argparse.Namespace):
    """Main function to run.

    Parameters
    ----------
    args : `argparse.Namespace`
        Namespace of arguments as parsed by `argparse`.
    """

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, nb_classes_after = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    nb_classes_before = 1000  # ImageNet

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if args.repeated_aug:

        sampler_train = RASampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    else:

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=nb_classes_after
        )

    print(f"Creating teacher model: {args.teacher_model}")
    teacher_model = create_model(
        args.teacher_model,
        pretrained=True,
        num_classes=nb_classes_before,
        global_pool="avg",
    )

    # Change the final layer
    for param in teacher_model.parameters():
        param.requires_grad = False
    fc_in_features = teacher_model.head.fc.weight.shape[1]
    teacher_model.head.fc = nn.Linear(fc_in_features, nb_classes_after, bias=True)

    # Put on device
    teacher_model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP
        # wrapper
        model_ema = ModelEma(
            teacher_model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="")

    n_parameters = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    print("Number of params:", n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, teacher_model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # Smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    criterion = DistillationLoss(criterion, teacher_model, "none", 0., 0.)

    output_dir = Path(args.output_dir)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(
            teacher_model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=True
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / "teacher_checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:

                ema_state = None if model_ema is None else get_state_dict(model_ema)

                utils.save_on_master({
                    "teacher_model": teacher_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "model_ema": ema_state,
                    "scaler": loss_scaler.state_dict(),
                    "args": args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, teacher_model, device)
        msg = f"Accuracy of the network on the {len(dataset_val)}"
        msg += f""" test images: {test_stats["acc1"]:.1f}%"""
        print(msg)

        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / "teacher_best_checkpoint.pth"]
                for checkpoint_path in checkpoint_paths:

                    ema_state = None if model_ema is None else get_state_dict(model_ema)

                    utils.save_on_master({
                        "model": teacher_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "model_ema": ema_state,
                        "scaler": loss_scaler.state_dict(),
                        "args": args,
                    }, checkpoint_path)

        print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     **{f"test_{k}": v for k, v in test_stats.items()},
                     "epoch": epoch,
                     "n_parameters": n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Script for training transfered models.",
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
