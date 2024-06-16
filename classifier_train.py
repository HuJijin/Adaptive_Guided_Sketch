"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
import shutil
import random
import blobfile as bf
import torch as th
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.utils.tensorboard as tb

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    
    logger.configure(dir=args.log_path)
    with open(os.path.join(args.log_path, "config.txt"), "w") as f:
        config_dict = vars(args)
        for k,v in config_dict.items():
            f.write(f"{k} : {v}")
            f.write('\n')
        f.close()
        
    ##copy file
    shutil.copyfile(__file__,os.path.join(logger.get_dir(),os.path.basename(__file__)))


    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=True,
        deterministic=False,
    )
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            deterministic=True,
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    schedule = th.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=0.7, last_epoch=-1)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")
    
    if dist.get_rank() == 0:
        tb_path = os.path.join(args.log_path, "tensorboard")
        tb_logger = tb.SummaryWriter(log_dir=tb_path)     

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)  # batch.size([4,3,64,64])
        labels = extra["y"].to(dist_util.dev())  # labels.size([4])

        batch = batch.to(dist_util.dev())
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())  # t.size([4])
            # t in [0,steps-1], 即0-999
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):  # sub_batch.size([4,3,64,64])  sub_labels.size([4])  sub_t.size([4])
            logits = model(sub_batch, timesteps=sub_t)  # logits.size([4,250])
            loss = F.cross_entropy(logits, sub_labels, reduction="none")   # loss.size([4])

            losses = {}
            acc_1 = compute_top_k(         # acc_1.size([4])
                logits, sub_labels, k=1, reduction="none"
            )
            acc_5 = compute_top_k(       # acc_5.size([4])
                logits, sub_labels, k=5, reduction="none"
            )
            
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = acc_1
            losses[f"{prefix}_acc@5"] = acc_5
            log_loss_dict(diffusion, sub_t, losses)
            del losses
            loss = loss.mean()
            acc_1 = acc_1.mean()
            acc_5 = acc_5.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))
        
        return loss, acc_1, acc_5

    min_acc = 0
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        loss, acc_1, acc_5 = forward_backward_log(data)
        if dist.get_rank() == 0:
            tb_logger.add_scalar("loss", loss, global_step=step)
            tb_logger.add_scalar("acc_1", acc_1, global_step=step)
            tb_logger.add_scalar("acc_5", acc_5, global_step=step)
        logger.log(f"step: {step}, loss: {loss}, acc_1: {acc_1}, acc_5: {acc_5}")
        mp_trainer.optimize(opt)
        schedule.step()
        
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)
            
        if dist.get_rank() == 0:
            if acc_1 > min_acc:
                min_acc = acc_1
                th.save(
                    mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
                    os.path.join(logger.get_dir(), f"model_best.pt"),
                )
                th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt_best.pt"))

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        log_path="",
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=10000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=True,
        batch_size=64,
        step_size=50000,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=100,
        eval_interval=1000,
        save_interval=10000,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
