# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""Main file for training models using distillation."""


import os

import datetime
import time

import json
from pathlib import Path

import argparse

import warnings

import numpy as np

import torch
from torch.backends import cudnn

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from tensorboardX import SummaryWriter


import models
import utils
from datasets import build_dataset
from engine import train_one_epoch_shrink, evaluate_shrink
from helpers import speed_test, get_macs
from losses import DistillationLoss
from samplers import RASampler


warn_msg = "Argument interpolation should be of type InterpolationMode instead of int"
warnings.filterwarnings("ignore", warn_msg)


def get_args_parser():
    """Parser getter."""
    parser = argparse.ArgumentParser("DeiT training and evaluation script", add_help=False)
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

    # Arguments related to the shrinking of inattentive tokens
    # Taken from https://github.com/youweiliang/evit
    parser.add_argument(
        "--test_speed",
        action="store_true",
        help="Also measure throughput of model."
    )
    parser.add_argument(
        "--only_test_speed",
        action="store_true",
        help="Only measure throughput of model."
    )
    parser.add_argument(
        "--fuse_token",
        action="store_true",
        help="Fuse the inattentive tokens."
    )
    parser.add_argument(
        "--base_keep_rate",
        type=float,
        default=0.7,
        help="Base keep rate. Default: 0.7."
    )
    parser.add_argument(
        "--shrink_epochs",
        default=0,
        type=int,
        help="""
             Number of epochs epochs to perform gradual shrinking of inattentive tokens.
             Default: 0.
             """
    )
    parser.add_argument(
        "--shrink_start_epoch",
        default=10,
        type=int,
        help="On which epoch to start shrinking of inattentive tokens. Default: 10."
    )
    parser.add_argument(
        "--drop_loc",
        default="(3, 6, 9)",
        type=str,
        help="""
             Indices of layers for shrinking inattentive tokens written as a tuple.
             Default: "(3, 6, 9)".
             """
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="deit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="""Name of model to train. Default: "deit_base_patch16_224"."""
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

    # Distillation parameters
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
        "--distillation-type",
        default="none",
        choices=["none", "soft", "hard"],
        type=str,
        help="""Name of the distillation to use. Default: "none"."""
    )
    parser.add_argument(
        "--distillation-alpha",
        default=0.5,
        type=float,
        help="""
             Alpha value for the loss when distillation is applied (lambda in the paper).
             Default: 0.5.
             """
    )
    parser.add_argument(
        "--distillation-tau",
        default=1.0,
        type=float,
        help="Tau value for the loss when distillation is applied. Default: 1.0."
    )

    # * Finetuning params
    parser.add_argument(
        "--finetune",
        default="",
        help="""Finetune from checkpoint. Default: ""."""
    )

    # Dataset parameters
    parser.add_argument(
        "--data-path",
        default="data/",
        type=str,
        help="""Path to the dataset. Default: "data/"."""
    )
    parser.add_argument(
        "--data-set",
        default="CIFAR10",
        choices=["CIFAR10", "CIFAR100", "IMNET", "INAT", "INAT19"],
        type=str,
        help="""
             Name of the dataset to use: "CIFAR10", "CIFAR100", "IMNET", "INAT" or "INAT19".
             Default: "CIFAR10".
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
        default="results/",
        help="""Path where to save, empty for no saving. Default: "results/"."""
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
        "--eval_gap",
        default=5,
        type=int,
        help="Number of epochs between each evaluation during training. Default: 5."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Perform evaluation only."
    )
    parser.add_argument(
        "--dist-eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation."
    )
    parser.add_argument(
        "--num_workers",
        default=8,
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

    # distributed training parameters
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="Number of processes. Default: 1."
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        help="""URL used to set up distributed training. Default: "env://"."""
    )

    return parser


def main(args: argparse.Namespace):
    """Main function to run.

    Parameters
    ----------
    args : `argparse.Namespace`
        Namespace of arguments as parsed by `argparse`.
    """
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != "none" and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # if True:  # args.distributed:

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

    if args.dist_eval:

        if len(dataset_val) % num_tasks != 0:

            warn_msg = "Warning: Enabling distributed evaluation with an eval dataset not"
            warn_msg += " divisible by process number. This will slightly alter validation"
            warn_msg += " results as extra duplicate entries are added to achieve equal num"
            warn_msg += " of samples per-process."
            print(warn_msg)

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    else:

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # else:

    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        base_keep_rate=args.base_keep_rate,
        drop_loc=eval(args.drop_loc),
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        fuse_token=args.fuse_token,
        img_size=(args.input_size, args.input_size)
    )

    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location="cpu")

        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size)
        pos_tokens = pos_tokens.permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model["pos_embed"] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    output_dir = Path(args.output_dir)

    # Taken from https://github.com/youweiliang/evit
    if args.test_speed and utils.is_main_process():
        # test model throughput for three times to ensure accuracy
        inference_speed = speed_test(model)
        print("inference_speed (inaccurate):", inference_speed, "images/s")
        inference_speed = speed_test(model)
        print("inference_speed:", inference_speed, "images/s")
        inference_speed = speed_test(model)
        print("inference_speed:", inference_speed, "images/s")
        MACs = get_macs(model)
        print("GMACs:", MACs * 1e-9)

        def log_func1(*arg, **kwargs):
            log1 = " ".join([f"{xx}" for xx in arg])
            log2 = " ".join([f"{key}: {v}" for key, v in kwargs.items()])
            log = log1 + "\n" + log2
            log = log.strip("\n") + "\n"
            if args.output_dir and utils.is_main_process():
                with (output_dir / "speed_macs.txt").open("a") as f:
                    f.write(log)
        log_func1(inference_speed=inference_speed, GMACs=MACs * 1e-9)
        log_func1(args=args)
    if args.only_test_speed:
        return

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP
        # wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # Taken from https://github.com/youweiliang/evit
    if args.test_speed and utils.is_main_process():
        log_func1(n_parameters=n_parameters * 1e-6)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
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

    teacher_model = None
    if args.distillation_type != "none":

        assert args.teacher_path, "need to specify teacher-path when using distillation"

        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool="avg",
        )
        if args.teacher_path.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location="cpu")

        teacher_model.load_state_dict(checkpoint["model"])
        teacher_model.to(device)
        teacher_model.eval()

    # Wrap the criterion in our custom DistillationLoss, which just dispatches to the original
    # criterion if args.distillation_type is "none"
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.
        distillation_tau
    )

    # Taken from https://github.com/youweiliang/evit
    if utils.is_main_process():
        print("output_dir:", args.output_dir)
        writer = SummaryWriter(os.path.join(args.output_dir, "runs"))
    else:
        writer = None

    output_dir = Path(args.output_dir)
    if args.resume:

        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])

        if (
            not args.eval and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and
            "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])

        lr_scheduler.step(args.start_epoch)

    if args.eval:

        test_stats = evaluate_shrink(data_loader_val, model, device)
        msg = f"Accuracy of the network on the {len(dataset_val)}"
        msg += f""" test images: {test_stats["acc1"]:.1f}%"""
        print(msg)

        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats, keep_rate = train_one_epoch_shrink(
            model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn, writer,
            set_training_mode=args.finetune == "",  # keep in eval mode during finetuning
            args=args
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:

                ema_state = None if model_ema is None else get_state_dict(model_ema)

                utils.save_on_master({
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "model_ema": ema_state,
                    "scaler": loss_scaler.state_dict(),
                    "args": args,
                }, checkpoint_path)

        # test_stats = evaluate(data_loader_val, model, device)
        # msg = f"Accuracy of the network on the {len(dataset_val)}"
        # msg += f""" test images: {test_stats["acc1"]:.1f}%"""
        # print(msg)

        # if max_accuracy < test_stats["acc1"]:
        #     max_accuracy = test_stats["acc1"]
        #     if args.output_dir:
        #         checkpoint_paths = [output_dir / "best_checkpoint.pth"]
        #         for checkpoint_path in checkpoint_paths:

        #             ema_state = None if model_ema is None else get_state_dict(model_ema)

        #             utils.save_on_master({
        #                 "model": model_without_ddp.state_dict(),
        #                 "optimizer": optimizer.state_dict(),
        #                 "lr_scheduler": lr_scheduler.state_dict(),
        #                 "epoch": epoch,
        #                 "model_ema": ema_state,
        #                 "scaler": loss_scaler.state_dict(),
        #                 "args": args,
        #             }, checkpoint_path)

        # print(f"Max accuracy: {max_accuracy:.2f}%")

        # log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
        #              **{f"test_{k}": v for k, v in test_stats.items()},
        #              "epoch": epoch,
        #              "n_parameters": n_parameters}

        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

        test_interval = args.eval_gap
        if epoch % test_interval == 0 or epoch == args.epochs - 1:
            test_stats = evaluate_shrink(data_loader_val, model, device, keep_rate)
            test_msg = f"Accuracy of the network on the {len(dataset_val)}"
            test_msg += f""" test images: {test_stats["acc1"]:.1f}%"""
            print(test_msg)
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f"Max accuracy: {max_accuracy:.2f}%")
            test_stats1 = {f"test_{k}": v for k, v in test_stats.items()}
        else:
            test_stats1 = {}

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     **test_stats1,
                     "epoch": epoch,
                     "n_parameters": n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if epoch % test_interval == 0 or epoch == args.epochs - 1:
                writer.add_scalar("test_acc1", test_stats["acc1"], epoch)
                writer.add_scalar("test_acc5", test_stats["acc5"], epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":

    parser = argparse.ArgumentParser("DeiT training and evaluation script.",
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
