"""KL sweep용 harness: prior N=20 + posterior 1샷을 한 번에, **valid 기준**으로 집계.

experiments/eval_harness.py 를 쓰지 않는 이유: 그쪽 NQueensTask.verify 는
`strict`(valid AND consistent)를 정답 기준으로 반환한다(eval_harness.py:120-126).
이번 지시는 valid 기준 majority/pass/verifier_selected 를 요구하므로 별도로 집계한다.
샘플링 RNG 스트림은 gram_harness.rollout 그대로다(seed*100003+k).

사용법:
    python experiments/sweep_harness.py --ckpt-dir <dir> --step step_6834 \
        --out experiments/results/sweep_<run>_<step>.json
"""
import argparse
import json
import os
import sys
from collections import Counter
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.gram_harness import load_model, rollout   # noqa: E402
from evaluators.nqueens import verify_boards               # noqa: E402


def pass_at_k(num_correct, n_samples, k):
    """eval_harness.py:248-253 과 동일한 불편추정량."""
    if n_samples - num_correct < k:
        return 1.0
    return 1.0 - comb(n_samples - num_correct, k) / comb(n_samples, k)


def majority_row(rows):
    """eval_harness.py:255-258 과 동일: 최빈 보드, 동률은 첫 등장."""
    return np.array(Counter(tuple(r) for r in rows).most_common(1)[0][0], dtype=np.int64)


def flat_stats(preds, inputs, n):
    """preds [K, B, L] -> valid/consistent/strict/queen_count 를 전 샘플 평균."""
    K, B, L = preds.shape
    r = verify_boards(preds.reshape(K * B, L), np.tile(inputs, (K, 1)), n)
    qc = r["queen_count"]
    return dict(valid=float(r["valid"].mean()), consistent=float(r["consistent"].mean()),
                strict=float(r["strict"].mean()), queen_count_mean=float(qc.mean()),
                queen_count_min=int(qc.min()), queen_count_max=int(qc.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8/test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net, _ = load_model(args.ckpt_dir, args.step)
    inputs = np.load(f"{args.data}/all__inputs.npy").astype(np.int64)
    labels = np.load(f"{args.data}/all__labels.npy").astype(np.int64)
    B = len(inputs)

    pri = rollout(net, inputs, labels, K=args.N, sigma_scale_=args.sigma_scale,
                  mode="prior", seed=args.seed, verbose=False)
    post = rollout(net, inputs, labels, K=1, sigma_scale_=args.sigma_scale,
                   mode="posterior", seed=args.seed, verbose=False)

    cand, v = pri["preds"], pri["v"]                       # [N, B, L], [N, B]
    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "n": B, "N": args.N,
           "seed": args.seed, "sigma_scale": args.sigma_scale, "tag": args.tag}

    res["prior_all_samples"] = flat_stats(cand, inputs, args.n)
    res["prior_single_k0"] = flat_stats(cand[:1], inputs, args.n)
    res["posterior"] = flat_stats(post["preds"], inputs, args.n)

    # per-puzzle valid 행렬 [B, N]
    vb = verify_boards(cand.reshape(args.N * B, -1), np.tile(inputs, (args.N, 1)), args.n)
    valid = vb["valid"].reshape(args.N, B).T                # [B, N]

    nc = valid.sum(1)
    res["pass_at"] = {str(k): float(np.mean([pass_at_k(int(c), args.N, k) for c in nc]))
                      for k in (1, 5, 10, 20) if k <= args.N}

    maj, sel, dis = [], [], []
    for b in range(B):
        rows = cand[:, b, :]
        maj.append(majority_row(rows))
        sel.append(rows[int(np.argmax(v[:, b]))])
        dis.append(len({tuple(r) for r in rows}))
    maj, sel = np.stack(maj), np.stack(sel)
    res["majority_at_N_valid"] = float(verify_boards(maj, inputs, args.n)["valid"].mean())
    res["verifier_selected_at_N_valid"] = float(verify_boards(sel, inputs, args.n)["valid"].mean())
    res["distinct_at_N_mean"] = float(np.mean(dis))
    res["puzzles_with_ge1_valid"] = float((nc > 0).mean())

    p = res["prior_all_samples"]
    print(f"{res['checkpoint']}")
    print(f"  prior  valid(all {args.N})={p['valid']:.4f} single_k0={res['prior_single_k0']['valid']:.4f} "
          f"consistent={p['consistent']:.4f} strict={p['strict']:.4f} "
          f"queens {p['queen_count_mean']:.2f} ({p['queen_count_min']}-{p['queen_count_max']})")
    q = res["posterior"]
    print(f"  post   valid={q['valid']:.4f} consistent={q['consistent']:.4f} strict={q['strict']:.4f} "
          f"queens {q['queen_count_mean']:.2f} ({q['queen_count_min']}-{q['queen_count_max']})")
    print(f"  pass@k {res['pass_at']}  majority@{args.N}={res['majority_at_N_valid']:.4f} "
          f"verifier_sel@{args.N}={res['verifier_selected_at_N_valid']:.4f} "
          f"distinct@{args.N}={res['distinct_at_N_mean']:.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  [out] {args.out}")


if __name__ == "__main__":
    main()
