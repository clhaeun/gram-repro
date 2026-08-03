"""Decide empirically which KL reduction ACTLossHead actually applies.

losses.py carries a commented/uncommented pair of reduction lines, so the
active reduction cannot be read off a config key -- it is whichever line is
uncommented in the working tree.  This runs a real forward with the target
arch config, pulls the raw kl tensor out via return_keys=["kl"], and compares
the loss head's own kl_loss against both candidate reductions computed by hand.

    L-SUM   kl.sum(-1).sum(-1).sum() / B     <- gram_sudoku_halt_only (71.85%)
    L-MEAN  kl.sum(-1).mean(-1).sum() / B    <- gram_sudoku_lmean_v2  (1.8%)

They differ by exactly L = seq_len + puzzle_emb_len, so `kl_raw == kl_loss`
(diag/kl_raw is a per-position mean) is a positive detector of L-MEAN.

usage: CUDA_VISIBLE_DEVICES=3 python experiments/test_kl_reduction.py [--arch gram_nqueens8]
"""
import argparse
import json
import os
import sys

import torch
from omegaconf import OmegaConf

# experiments/legacy/diag/ 로 내려왔으므로 repo root 는 부모의 부모의 부모의 부모다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from utils.functions import load_model_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gram_nqueens8")
    ap.add_argument("--data", default="data/nqueens-8")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data, "train", "dataset.json")))
    arch = OmegaConf.to_container(OmegaConf.load(f"config/arch/{args.arch}.yaml"), resolve=True)
    arch_name, loss_cfg = arch.pop("name"), arch.pop("loss")

    model_cfg = dict(**arch, batch_size=args.batch_size, vocab_size=meta["vocab_size"],
                     seq_len=meta["seq_len"], num_puzzle_identifiers=meta["num_puzzle_identifiers"],
                     causal=False)

    with torch.device("cuda"):
        model = load_model_class(arch_name)(model_cfg)
        head = load_model_class(loss_cfg["name"])(
            model, **{k: v for k, v in loss_cfg.items() if k != "name"})
    head.train()   # posterior path -> kl is produced

    B, L_seq = args.batch_size, meta["seq_len"]
    torch.manual_seed(0)
    batch = {
        "inputs": torch.randint(1, meta["vocab_size"], (B, L_seq), device="cuda", dtype=torch.int32),
        "labels": torch.randint(1, meta["vocab_size"], (B, L_seq), device="cuda", dtype=torch.int32),
        "puzzle_identifiers": torch.zeros(B, device="cuda", dtype=torch.int32),
    }
    with torch.device("cuda"):          # pretrain.py wraps initial_carry the same way
        carry = head.initial_carry(batch)
    carry, loss, metrics, outs, _ = head(carry=carry, batch=batch, return_keys=["kl"])

    kl = outs["kl"].float()                      # [B, L, D]
    L = kl.shape[1]
    reported = float(metrics["kl_loss"]) / B     # train_batch divides *_loss by batch
    l_sum = float(kl.sum(-1).sum(-1).sum()) / B
    l_mean = float(kl.sum(-1).mean(-1).sum()) / B
    kl_raw = float(metrics.get("diag/kl_raw", float("nan")))

    print(f"\narch={args.arch}  kl tensor {tuple(kl.shape)}  L={L} "
          f"(= seq_len {L_seq} + puzzle_emb_len {L - L_seq})")
    print(f"  head-reported kl_loss/B = {reported:.6f}")
    print(f"  manual  L-SUM   /B      = {l_sum:.6f}   ratio to reported = {l_sum / reported:.4f}")
    print(f"  manual  L-MEAN  /B      = {l_mean:.6f}   ratio to reported = {l_mean / reported:.4f}")
    print(f"  diag/kl_raw             = {kl_raw:.6f}")
    print(f"  L-SUM / L-MEAN          = {l_sum / l_mean:.4f}  (must equal L={L})")

    tol = 1e-3
    is_sum = abs(reported - l_sum) / max(abs(l_sum), 1e-9) < tol
    is_mean = abs(reported - l_mean) / max(abs(l_mean), 1e-9) < tol
    verdict = "L-SUM" if is_sum else ("L-MEAN" if is_mean else "NEITHER")
    print(f"\n  VERDICT: active reduction is {verdict}")
    print(f"  kl_raw == kl_loss ? {abs(kl_raw - reported) / max(abs(reported), 1e-9) < tol}"
          f"   (True <=> L-MEAN)")

    res = dict(arch=args.arch, L=L, seq_len=L_seq, reported=reported, l_sum=l_sum,
               l_mean=l_mean, kl_raw=kl_raw, verdict=verdict)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}")
    return 0 if verdict != "NEITHER" else 1


if __name__ == "__main__":
    sys.exit(main())
