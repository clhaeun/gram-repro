"""Evaluate a saved checkpoint on the test set and print metrics.

pretrain.py only sends eval metrics to wandb, so a run with WANDB_MODE=disabled
records nothing. This reruns evaluate() on a checkpoint and prints to stdout.

Usage (4 GPUs):
  torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
    eval_checkpoint.py <checkpoint_dir> <step_file> [key=value ...] [--out-name NAME]

Trailing key=value pairs are OmegaConf dotlist overrides applied on top of the
checkpoint's all_config.yaml -- e.g. to score a checkpoint on a different test set:
    data_paths=[data/sudoku-extreme-1k-aug-1000]
--out-name overrides the result filename so such a run does not clobber the default.
"""
import os
import sys
import json

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import pretrain as P


def main():
    argv = sys.argv[1:]
    out_name = None
    if "--out-name" in argv:
        i = argv.index("--out-name")
        out_name = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    ckpt_dir, ckpt_file = argv[0], argv[1]
    overrides = argv[2:]
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)

    RANK, WORLD_SIZE, CPU_PROCESS_GROUP = 0, 1, None
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")

    cfg = OmegaConf.load(os.path.join(ckpt_dir, "all_config.yaml"))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg.load_checkpoint = ckpt_path
    cfg.ema = False              # checkpoint is already the EMA copy; don't re-wrap
    cfg.eval_save_outputs = []
    cfg.evaluators = []
    config = P.PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))

    torch.random.manual_seed(config.seed + RANK)

    train_loader, train_metadata = P.create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader, eval_metadata = P.create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    train_state = P.init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)
    train_state.step = int(ckpt_file.split("_")[-1])
    train_state.model.eval()

    metrics = P.evaluate(config, train_state, eval_loader, eval_metadata, [],
                         rank=RANK, world_size=WORLD_SIZE, cpu_group=CPU_PROCESS_GROUP)

    if RANK == 0 and metrics is not None:
        flat = P.flatten_metrics(metrics)
        print("\n" + "=" * 60)
        print(f"EVAL RESULT  checkpoint={ckpt_file}  data_paths={list(config.data_paths)}")
        print("=" * 60)
        for k, v in sorted(flat.items()):
            print(f"  {k:35s} {v:.6f}")
        out = os.path.join(ckpt_dir, out_name or f"eval_{ckpt_file}.json")
        with open(out, "w") as f:
            json.dump(flat, f, indent=2)
        print(f"\nsaved -> {out}")

    if WORLD_SIZE > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
