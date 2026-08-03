"""Task 2: re-measure N-Queens with the (fixed) harness sampler.

Uses eval_harness.sample_dataset (prior path, seq_len from data) -- the SAME code
path as the earlier eval_harness runs -- and scores every candidate on BOTH
  valid   (constraints only)      <- paper Table 1 headline
  strict  (valid AND clue-consistent)
so the paper-headline number and our stricter number sit side by side.

Metrics per K: single(=K1), pass@K (any candidate correct), majority@K
(most-common board correct), distinct@K.  "correct" reported under both defs.
"""
import argparse, json, os, sys
from collections import Counter
import numpy as np, torch
# experiments/legacy/diag/ 로 내려왔으므로 repo root 는 부모의 부모의 부모의 부모다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from experiments.eval_harness import sample_dataset
from experiments.gram_harness import load_model
from evaluators.nqueens import verify_boards
from experiments.legacy.diag.diag_posterior_prior import load_test


def metrics_under(correct, cand, ks):
    """correct [n,N] bool, cand [n,N,L]. Returns per-K dict."""
    n, N = correct.shape
    out = {}
    for k in sorted({k for k in ks if k <= N} | {N}):
        maj = np.zeros(n, dtype=bool)
        distinct = np.zeros(n)
        for i in range(n):
            sub = cand[i, :k]
            c = Counter(tuple(r) for r in sub)
            board = np.array(c.most_common(1)[0][0])
            midx = int(np.flatnonzero((sub == board).all(-1))[0])
            maj[i] = correct[i, midx]
            distinct[i] = len(c)
        out[str(k)] = {
            "single_or_passfirst": float(correct[:, :k].any(1).mean()),
            "pass": float(correct[:, :k].any(1).mean()),
            "majority": float(maj.mean()),
            "distinct": float(distinct.mean()),
        }
    out["single"] = float(correct[:, 0].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net, config = load_model(args.ckpt_dir, args.step)
    inputs, labels = load_test(args.data)
    cand, v = sample_dataset(net, inputs, labels, args.k, sigma_scale=1.0, seed=args.seed)

    n, N, L = cand.shape
    valid = np.zeros((n, N), dtype=bool)
    strict = np.zeros((n, N), dtype=bool)
    for kk in range(N):
        r = verify_boards(cand[:, kk], inputs, args.n)
        valid[:, kk] = r["valid"]
        strict[:, kk] = r["strict"]

    ks = (1, 5, 10, 20)
    res = {
        "checkpoint": os.path.join(args.ckpt_dir, args.step),
        "n": n, "N": N, "seed": args.seed, "sigma_scale": 1.0,
        "headline_valid": metrics_under(valid, cand, ks),
        "headline_strict": metrics_under(strict, cand, ks),
    }
    print(f"\n{args.ckpt_dir}/{args.step}  n={n} N={N}")
    for name in ("headline_valid", "headline_strict"):
        m = res[name]
        print(f"\n[{name}]  single={m['single']*100:.2f}%")
        print(f"  {'K':>3} {'pass':>8} {'majority':>9} {'distinct':>9}")
        for k in ("1", "5", "10", "20"):
            if k in m:
                r = m[k]
                print(f"  {k:>3} {r['pass']*100:>7.2f}% {r['majority']*100:>8.2f}% {r['distinct']:>9.2f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
