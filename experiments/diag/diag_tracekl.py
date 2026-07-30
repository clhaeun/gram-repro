"""Task 3.3: KL/position via gram.py's trace_kl hook, on a given checkpoint dir.

Sets inner.trace_kl=True, runs the posterior path once over test puzzles, and
reports the per-supervision-step KL (sum over D, mean over batch and L) that
gram.py records in inner._kl_trace.  This is the same quantity the training log
called kl_raw.
"""
import argparse, json, os, sys
import numpy as np, torch
# experiments/diag/ 로 한 단계 내려왔으므로 repo root 는 부모의 부모의 부모다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.gram_harness import load_model
from experiments.diag.diag_posterior_prior import load_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8")
    ap.add_argument("--num-puzzles", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    net, config = load_model(args.ckpt_dir, args.step)
    inner = net.model.inner
    inputs, labels = load_test(args.data)
    inputs, labels = inputs[:args.num_puzzles], labels[:args.num_puzzles]
    B, L = inputs.shape
    batch = {
        "inputs": torch.from_numpy(inputs.astype(np.int32)).cuda(),
        "labels": torch.from_numpy(labels.astype(np.int32)).cuda(),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32).cuda(),
    }
    inner.trace_kl = True
    torch.random.manual_seed(0)
    with torch.inference_mode(), torch.device("cuda"):
        carry = net.initial_carry(batch)
        # one supervision step is enough: trace_kl records every inner transition
        net(carry=carry, batch=batch, return_keys=[], use_posterior=True)
    trace = list(inner._kl_trace)
    inner.trace_kl = False

    res = {
        "checkpoint": os.path.join(args.ckpt_dir, args.step),
        "kl_per_position_by_transition": trace,
        "kl_per_position_final_transition": trace[-1] if trace else None,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
