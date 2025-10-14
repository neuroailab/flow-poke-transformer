# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025 Stefan Baumann et al., CompVis @ LMU Munich

import os
from pathlib import Path
import logging
import random
import math
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention.flex_attention import create_block_mask
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange, repeat

from flow_poke.model import FlowPokeTransformer, FlowPokeTransformer_Base, query_causal_mask_mod
from flow_poke.data import TrackerShardsDataModule


def endless_iter(iterable):
    while True:
        yield from iterable


# Main training entry point
# Add arguments here to make them configurable via CLI
def train(
    # General
    out_dir="output",
    load_checkpoint: str | None = None,
    checkpoint_freq: int = 10_000,
    max_steps: int = 1_000_000,
    warmup_steps: int = 100_000,
    # Data
    data_tar_base: str = "data",
    # Training
    local_batch_size: int = 32,
    lr: float = 5e-5,
    clip_grad_norm: float = 1.0,
    # Misc
    compile: bool = False,
    autotune: bool = False,
    enable_wandb: bool = True,
):
    train_params = locals()
    # Output & logging setup
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "train.log"),
        ],
    )

    # Distributed init & handling of single-GPU case
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group()
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device_type = "cuda"
        device = torch.device(f"{device_type}:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(f"Running distributed. Local rank: {local_rank}, World size: {world_size}")

        rank0logger = logging.getLogger(__name__)
        if rank != 0:
            rank0logger.disabled = True
        barrier = dist.barrier
    else:
        rank = 0
        device_type = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
        logger.info(f"Running non-distributed on {device_type}")

        rank0logger = logger
        barrier = lambda: None

    # WandB setup
    if enable_wandb and rank == 0:
        import wandb

        wandb.init(
            project="flow-poke-transformer",
            config=train_params | {"global_batch_size": local_batch_size * world_size},
            dir=out_dir,
        )

    # Checkpoint loading pt1
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint)
        start_step = checkpoint["step"]
        rank0logger.info(f"Loaded checkpoint from {load_checkpoint} @ step {start_step}.")
    else:
        checkpoint = None
        start_step = 0

    # Seeding
    seed = 42 + rank + start_step
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Important setup stuff
    # If you want to change anything about what you train, you'll likely want to do it here and add it as a parameter to train()
    model: FlowPokeTransformer = FlowPokeTransformer_Base().to(device)
    data = TrackerShardsDataModule(tar_base=data_tar_base, batch_size=local_batch_size, shuffle=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), fused=compile)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=1e-8),
        ],
        milestones=[warmup_steps],
    )
    rank0logger.info(
        f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M"
        f" ({sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M trainable)"
    )

    # Checkpoint loading pt2: actually loading state
    if load_checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    # DDP wrapping
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], static_graph=True)  # type: ignore

    # Inner train step function
    def train_step_inner(
        batch: dict[str, torch.Tensor], train_block_mask, train_is_query, L_poke, compute_metrics: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Construct model inputs & target
        # Order: pokes, queries
        pos = torch.cat(
            [
                batch["pos_poke"],
                rearrange(batch["pos_query"], "b n_p n_q c -> b (n_p n_q) c"),
            ],
            dim=1,
        )
        flow_target = rearrange(batch["flow_query"], "b n_p n_q c -> b (n_p n_q) c")
        flow = torch.cat(
            [
                batch["flow_poke"],
                torch.zeros_like(flow_target),
            ],
            dim=1,
        )

        # Training forward
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            d_img = (model.module if is_distributed else model).embed_image(batch["x"])
            distribution = model(
                pos=pos,
                flow=flow,
                is_query=train_is_query,
                camera_static=batch["camera_static"],
                mask=train_block_mask,
                d_img=d_img,
            )  # [B, L_poke + L_query, C]

        # Loss is NLL of GT flow for queries
        loss = -distribution[:, L_poke:].log_prob(flow_target).mean()

        if not compute_metrics:
            return loss
        else:
            with torch.no_grad():
                metrics = {}
                samples = distribution[:, L_poke:].sample()
                epe = (samples - flow_target).norm(p=2, dim=-1)
                metrics["epe"] = epe.mean().detach()
                # PCK at different thresholds
                pck_thresholds = [0.1, 0.01, 0.001]
                for alpha in pck_thresholds:
                    metrics[f"pck@{alpha}"] = epe.le(alpha).float().mean().detach()
                metrics["flow_mag_gt"] = flow_target.norm(p=2, dim=-1).mean().detach()
                metrics["flow_mag_pred"] = samples.norm(p=2, dim=-1).mean().detach()
                metrics["frac_static_camera"] = batch["camera_static"].float().mean().detach()

            return loss, metrics

    # Compile if requested
    if compile:
        train_step_inner = torch.compile(
            train_step_inner, fullgraph=False, mode="max-autotune" if autotune else "default"
        )
        rank0logger.info("Model compiled with torch.compile.")

    barrier()

    # Training loop
    # We assume that the data loader always returns batches with consistent shapes, therefore we can cache some things
    caches_initialized, train_block_mask, train_is_query, L_poke = False, None, None, None
    if rank == 0:
        logger.info("Starting training...")
    for i, batch in enumerate(
        pbar := tqdm(endless_iter(data.make_loader()), desc="Training", initial=start_step, disable=rank != 0)
    ):
        try:
            if not caches_initialized:
                B, L_poke, L_query = (
                    batch["pos_poke"].size(0),
                    batch["pos_poke"].size(1),
                    math.prod(batch["pos_query"].shape[1:3]),
                )
                train_block_mask = create_block_mask(
                    query_causal_mask_mod(
                        sequence_length=L_poke,
                        n_query=batch["pos_query"].size(2),
                    ),
                    B=1,
                    H=1,
                    Q_LEN=L_poke + L_query,
                    KV_LEN=L_poke + L_query,
                    device=device,
                )
                train_is_query = repeat(
                    torch.cat(
                        [
                            torch.zeros(L_poke, dtype=torch.bool, device=device),
                            torch.ones(L_query, dtype=torch.bool, device=device),
                        ]
                    ),
                    "l -> b l",
                    b=B,
                )
                caches_initialized = True
            optimizer.zero_grad()
            loss, metrics = train_step_inner(
                {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()},
                train_block_mask=train_block_mask,
                train_is_query=train_is_query,
                L_poke=L_poke,
                compute_metrics=True,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            scheduler.step()

            avg_loss = (
                (dist_nn.all_reduce(loss.detach().clone(), op=dist.ReduceOp.SUM) / world_size)
                if is_distributed
                else loss.detach()
            )

            metrics = {
                k: (
                    dist_nn.all_reduce(v.detach(), op=dist.ReduceOp.SUM) / world_size if is_distributed else v.detach()
                ).item()
                for k, v in metrics.items()
            }
            train_meta = {
                "loss": avg_loss.item(),
                "grad_norm": grad_norm.item(),
                "lr": scheduler.get_last_lr()[0],
            } | metrics

            pbar.set_postfix(train_meta)
            if enable_wandb and rank == 0:
                wandb.log({f"train/{k}": v for k, v in train_meta.items()}, step=start_step + i)

            done = max_steps is not None and (start_step + i) >= max_steps
            if done:
                rank0logger.info(f"Reached max steps: {start_step + i} >= {max_steps}. Stopping training...")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping training...")
            done = True
        if done or (i % checkpoint_freq == 0 and rank == 0) and i > 0:
            # Save checkpoint
            checkpoint = {
                "model": (model.module if is_distributed else model).state_dict(),  # type: ignore
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": start_step + i,
            }
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, ckpt_dir / f"checkpoint_{start_step + i:07}.pt")
            rank0logger.info(f"Saved checkpoint at step {start_step + i}.")

        if done:
            break
    barrier()
    rank0logger.info("Training stopped.")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)

    # By launching with fire, all arguments become specifyable via the CLI
    # e.g. python train.py --data_tar_base /path/to/data --local_batch_size 32
    try:
        import fire

        fire.Fire(train)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
