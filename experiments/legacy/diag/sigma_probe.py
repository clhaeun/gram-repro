"""Measure a checkpoint's sigma statistics directly from its weights.

Task 1's AC asks for lmean_v2's final sigma std_mean "from the training log",
but that run predates the Task 0 sigma logging AND ran with WANDB_MODE=disabled,
so no such record exists.  This recovers the same quantity from the checkpoint
by running test puzzles through the model and intercepting reparameterize(),
which is the measurement the log would have contained.

Both latent paths are probed:
  prior     eps ~ p(.|u)      -- the inference-time path
  posterior eps ~ q(.|u,y)    -- the training-time path (labels injected)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# experiments/legacy/diag/ 로 내려왔으므로 repo root 는 부모의 부모의 부모의 부모다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from experiments.gram_harness import load_model, load_puzzles, capture_sigma, capture_kl
from experiments.eval_harness import _make_batch, _forward_once


def probe(net, inputs, labels, use_posterior, batch_size, seed=0):
    sig, kl = [], []
    torch.random.manual_seed(seed)
    batch = _make_batch(inputs, labels, 0, min(batch_size, len(inputs)), batch_size,
                        inputs.shape[1])
    with capture_sigma(net, sig), capture_kl(net, kl):
        with torch.inference_mode():
            with torch.device("cuda"):
                carry = net.initial_carry(batch)
            while True:
                carry, _, _, _, all_finish = net(
                    carry=carry, batch=batch, return_keys=[], use_posterior=use_posterior)
                if all_finish:
                    break
    means = [d["mean"] for d in sig]
    meds = [d["median"] for d in sig]
    out = {
        "n_reparameterize_calls": len(sig),
        "std_mean": float(np.mean(means)) if means else None,
        "std_p50": float(np.mean(meds)) if meds else None,
        "std_mean_min": float(np.min(means)) if means else None,
        "std_mean_max": float(np.max(means)) if means else None,
    }
    if kl:
        out["kl_per_sample_sum"] = float(np.mean(np.concatenate(kl)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", default="step_65100")
    ap.add_argument("--num-puzzles", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    net, config = load_model(args.ckpt_dir, args.step)
    inputs, labels = load_puzzles(config, args.num_puzzles, seed=args.seed)

    res = {
        "checkpoint": os.path.join(args.ckpt_dir, args.step),
        "n": int(len(inputs)),
        "source": "measured from checkpoint weights (no sigma record exists in the training log)",
        "prior": probe(net, inputs, labels, False, args.batch_size, args.seed),
        "posterior": probe(net, inputs, labels, True, args.batch_size, args.seed),
    }
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
