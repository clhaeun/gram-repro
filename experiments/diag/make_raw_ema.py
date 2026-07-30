"""Reconstruct the *raw* (uncorrected) EMA shadow from a bias-corrected checkpoint.

Why this is needed
------------------
pretrain.py:688 saves `step_N` = the **bias-corrected** EMA copy
(pretrain.py:659 -> ema.ema_copy(model, step=train_state.step)).
The raw shadow buffer itself is never persisted, and `ema.w0` (the init
snapshot taken at ema.py:35-36) is not persisted either.  So "raw EMA" cannot
simply be read off disk.

It can be *reconstructed*, because the correction is invertible.  With
mt = mu**step:

    corrected = (shadow - mt*w0) / (1 - mt)          # ema.py:59
=>  shadow    = corrected*(1 - mt) + mt*w0

The only missing term is w0, the random init.  pretrain.py:594 seeds
`torch.random.manual_seed(config.seed + RANK)` immediately before the dataset
and model are built, so rank-0's w0 is reproducible by replaying that same
sequence.  This script replays it and reports whether the replay is stable
under the choices we had to make (world_size, whether dataloaders are built),
so the caveat can be stated honestly rather than assumed away.

Only parameters with requires_grad are in the shadow (ema.py:34,42) -- every
other tensor in the checkpoint is copied through untouched.
"""
import argparse
import copy
import os
import sys

import torch
from omegaconf import OmegaConf

# experiments/diag/ 로 한 단계 내려왔으므로 repo root 는 부모의 부모의 부모다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pretrain as P


def build_w0(ckpt_dir, world_size, make_loaders=True):
    """Replay pretrain.py:594-616 (rank 0) and return the fresh-init params."""
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "all_config.yaml"))
    cfg.load_checkpoint = None
    cfg.ema = False
    cfg.eval_save_outputs = []
    cfg.evaluators = []          # the original runs also had evaluators=[]
    config = P.PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))

    torch.random.manual_seed(config.seed + 0)          # pretrain.py:594, RANK=0

    train_metadata = None
    if make_loaders:                                    # pretrain.py:602-604
        _, train_metadata = P.create_dataloader(
            config, "train", test_set_mode=False, epochs_per_iter=config.eval_interval,
            global_batch_size=config.global_batch_size, rank=0, world_size=world_size)
        try:
            P.create_dataloader(
                config, "test", test_set_mode=True, epochs_per_iter=1,
                global_batch_size=config.global_batch_size, rank=0, world_size=world_size)
        except Exception:
            pass
    if train_metadata is None:
        _, train_metadata = P.create_dataloader(
            config, "train", test_set_mode=False, epochs_per_iter=1,
            global_batch_size=config.global_batch_size, rank=0, world_size=world_size)

    # pretrain.py:616 -> init_train_state -> create_model.  config.load_checkpoint
    # is None so nothing is loaded; the only other thing we must neutralise is the
    # rank broadcast at pretrain.py:144-147, which needs a process group.  It is
    # `dist.broadcast(param, src=0)` executed on rank 0 -- i.e. the identity for
    # the tensors we are reading -- so stubbing it out cannot change w0.
    real_broadcast = torch.distributed.broadcast
    torch.distributed.broadcast = lambda *a, **k: None
    try:
        model, _, _ = P.create_model(config, train_metadata, rank=0, world_size=world_size)
    finally:
        torch.distributed.broadcast = real_broadcast
    return {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}, config


def max_absdiff(a, b):
    keys = set(a) | set(b)
    worst, worst_k = 0.0, None
    for k in sorted(keys):
        if k not in a or k not in b:
            return float("inf"), k
        d = (a[k].float() - b[k].float()).abs().max().item()
        if d > worst:
            worst, worst_k = d, k
    return worst, worst_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True, help="e.g. step_17085")
    ap.add_argument("--mu", type=float, default=None, help="default: ema_rate from all_config")
    ap.add_argument("--world-size", type=int, default=4, help="world_size used in the original run")
    ap.add_argument("--out-suffix", default=".raw")
    ap.add_argument("--check-determinism", action="store_true")
    args = ap.parse_args()

    step_n = int(args.step.split("_")[-1])
    w0, config = build_w0(args.ckpt_dir, args.world_size)
    mu = args.mu if args.mu is not None else config.ema_rate
    mt = mu ** step_n
    print(f"[w0] {len(w0)} trainable tensors | mu={mu} step={step_n} mu**step={mt:.6f}")

    if args.check_determinism:
        w0b, _ = build_w0(args.ckpt_dir, args.world_size)
        d, k = max_absdiff(w0, w0b)
        print(f"[check] rerun same settings      : max|dw0|={d:.3e} (worst {k})")
        w0c, _ = build_w0(args.ckpt_dir, 1)
        d, k = max_absdiff(w0, w0c)
        print(f"[check] world_size {args.world_size} vs 1        : max|dw0|={d:.3e} (worst {k})")
        w0d, _ = build_w0(args.ckpt_dir, args.world_size, make_loaders=False)
        d, k = max_absdiff(w0, w0d)
        print(f"[check] with vs without loaders  : max|dw0|={d:.3e} (worst {k})")

    corrected = torch.load(os.path.join(args.ckpt_dir, args.step), map_location="cpu")
    raw = copy.copy(corrected)
    n_hit = n_miss = 0
    rel_num = rel_den = 0.0
    for k, v in corrected.items():
        if k in w0:
            c = v.float()
            r = c * (1.0 - mt) + mt * w0[k].float().cpu()
            rel_num += (r - c).pow(2).sum().item()
            rel_den += c.pow(2).sum().item()
            raw[k] = r.to(v.dtype)
            n_hit += 1
        else:
            n_miss += 1
    rel = (rel_num / rel_den) ** 0.5 if rel_den else float("nan")
    print(f"[raw] rebuilt {n_hit} tensors, passed through {n_miss} "
          f"(non-parameter / non-grad). ||raw-corrected||/||corrected|| = {rel:.6f}")

    out = os.path.join(args.ckpt_dir, args.step + args.out_suffix)
    torch.save(raw, out)
    print(f"[raw] wrote {out}")


if __name__ == "__main__":
    main()
