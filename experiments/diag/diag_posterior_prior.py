"""B1/B2 diagnostic: posterior vs prior on the N-Queens test set.

Decides whether the backbone can produce constraint-valid boards at all.

  posterior  eps ~ q(.|u,y)   -- labels injected, the training-time path
  prior      eps ~ p(.|u)     -- inference path, no labels

Both are run at K=1, to halting, over the full test split, and scored with the
project's own evaluators.nqueens.verify_boards (rule-based; no label compare).

WHY a bespoke script instead of gram_harness.rollout: gram_harness hardcodes
SEQ_LEN=81 (sudoku) in its padded-batch builder and would silently mangle the
64-token N-Queens boards.  Here seq_len comes from the data (64).

usage:
  CUDA_VISIBLE_DEVICES=3 python experiments/diag_posterior_prior.py \
      --ckpt-dir checkpoints/Nqueens-8-ACT-torch/gram_nq8_mean_nh512 --step step_3417
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# experiments/diag/ 로 한 단계 내려왔으므로 repo root 는 부모의 부모의 부모다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.gram_harness import load_model
from evaluators.nqueens import verify_boards


def load_test(data_path):
    d = os.path.join(data_path, "test")
    inp = np.load(os.path.join(d, "all__inputs.npy")).astype(np.int64)
    lab = np.load(os.path.join(d, "all__labels.npy")).astype(np.int64)
    return inp, lab


@torch.inference_mode()
def run_path(net, inputs, labels, use_posterior, n, batch_size=512, seed=0):
    """K=1 forward to halting over all puzzles. Returns preds [N, L] int64."""
    N, L = inputs.shape
    preds = np.zeros((N, L), dtype=np.int64)
    torch.random.manual_seed(seed)
    for lo in range(0, N, batch_size):
        hi = min(lo + batch_size, N)
        bs = hi - lo
        inp = np.zeros((batch_size, L), dtype=np.int32)
        lab = np.zeros((batch_size, L), dtype=np.int32)
        inp[:bs] = inputs[lo:hi]
        lab[:bs] = labels[lo:hi]
        batch = {
            "inputs": torch.from_numpy(inp).cuda(),
            "labels": torch.from_numpy(lab).cuda(),
            "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.int32).cuda(),
        }
        with torch.device("cuda"):
            carry = net.initial_carry(batch)
        p_final = torch.zeros(batch_size, L, dtype=torch.long, device="cuda")
        done = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
        while True:
            carry, _, _, out, all_finish = net(
                carry=carry, batch=batch, return_keys=["preds"],
                use_posterior=use_posterior)
            newly = carry.halted & (~done)
            if newly.any():
                p_final[newly] = out["preds"][newly].long()
            done |= carry.halted
            if all_finish:
                break
        preds[lo:hi] = p_final[:bs].cpu().numpy()
    return preds


def score(preds, inputs, n):
    r = verify_boards(preds, inputs, n)
    qc = r["queen_count"]
    N = len(preds)
    return {
        "n_eval": int(N),
        "valid_rate": float(r["valid"].mean()),
        "consistent_rate": float(r["consistent"].mean()),
        "strict_rate": float(r["strict"].mean()),
        "queen_count_mean": float(qc.mean()),
        "queen_count_min": int(qc.min()),
        "queen_count_max": int(qc.max()),
    }


def board_str(flat, n):
    g = np.asarray(flat).reshape(n, n)
    sym = {0: ".", 1: ".", 2: "Q"}   # pad and empty both render as .
    return "\n".join(" ".join(sym.get(int(v), "?") for v in row) for row in g)


def side_by_side(inp, pred, n):
    a = board_str(inp, n).split("\n")
    b = board_str(pred, n).split("\n")
    lines = [f"{a[i]}     {b[i]}" for i in range(n)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=5, help="boards to print")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    net, config = load_model(args.ckpt_dir, args.step)
    inputs, labels = load_test(args.data)
    print(f"loaded {args.ckpt_dir}/{args.step}  test={inputs.shape}  n={args.n}", flush=True)

    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "n": args.n}
    preds = {}
    for mode, up in [("posterior", True), ("prior", False)]:
        p = run_path(net, inputs, labels, up, args.n, args.batch_size, args.seed)
        preds[mode] = p
        res[mode] = score(p, inputs, args.n)
        s = res[mode]
        print(f"\n[{mode}] valid={s['valid_rate']*100:.2f}%  consistent={s['consistent_rate']*100:.2f}%  "
              f"strict={s['strict_rate']*100:.2f}%  queen_count mean={s['queen_count_mean']:.2f} "
              f"[{s['queen_count_min']},{s['queen_count_max']}]", flush=True)

    # B2: label reference (what the posterior was shown) + both predictions
    print("\n" + "=" * 60)
    print(f"BOARDS (left=input clues, right=prediction).  n={args.n}")
    for mode in ("posterior", "prior"):
        print(f"\n----- {mode} -----")
        for i in range(min(args.show, len(inputs))):
            r = verify_boards(preds[mode][i:i+1], inputs[i:i+1], args.n)
            print(f"\n#{i}  queens={int(r['queen_count'][0])}  valid={bool(r['valid'][0])} "
                  f"consistent={bool(r['consistent'][0])}")
            print(side_by_side(inputs[i], preds[mode][i], args.n))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
