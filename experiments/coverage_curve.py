"""논문 Figure 15: x=해 개수, y=coverage, 곡선=N∈{1,5,10,20}.  sweep_harness 와 같은
rollout 재사용, found = 앞 N개 표본 중 서로 다른 valid∧consistent 보드 수.
상한 min(N,total)/total 곡선 병기 (해 90개 + N=20 이면 완벽해도 22%)."""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gram_harness import load_model, rollout                      # noqa: E402
from sweep_harness import verify_boards, solution_counts          # noqa: E402


def coverage_by_n(cand, inputs, n, totals, n_list):
    """cand [K, B, L] -> {N: per-puzzle found 배열} (앞 N개 표본 prefix 기준)."""
    K, B, L = cand.shape
    r = verify_boards(cand.reshape(K * B, L), np.tile(inputs, (K, 1)), n)
    ok = (r["valid"] & r["consistent"]).reshape(K, B)
    found = {N: np.zeros(B, dtype=np.int64) for N in n_list}
    for b in range(B):
        seen, nxt = set(), 0
        for k in range(K):
            if ok[k, b]:
                seen.add(tuple(cand[k, b]))
            while nxt < len(n_list) and n_list[nxt] == k + 1:
                found[n_list[nxt]][b] = len(seen)
                nxt += 1
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8/test")
    ap.add_argument("--sidecar", default="data/nqueens-8/sidecar_8.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--N", type=int, default=20, help="rollout 표본 수 (곡선의 최대 N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-list", default="1,5,10,20")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_list = sorted(int(x) for x in args.n_list.split(","))
    assert n_list[-1] <= args.N, f"n_list 최대값 {n_list[-1]} 이 --N {args.N} 보다 크다"

    net, _ = load_model(args.ckpt_dir, args.step)
    inputs = np.load(f"{args.data}/all__inputs.npy").astype(np.int64)
    labels = np.load(f"{args.data}/all__labels.npy").astype(np.int64)

    pri = rollout(net, inputs, labels, K=args.N, mode="prior", seed=args.seed, verbose=False)
    cand = pri["preds"]                                    # [N, B, L]
    totals = solution_counts(inputs, args.sidecar)
    found = coverage_by_n(cand, inputs, args.n, totals, n_list)
    have = totals > 0

    strata = {}                                            # 해 개수별 층화 (Fig 15 의 x축)
    for t in sorted(set(int(x) for x in totals[have])):
        m = have & (totals == t)
        strata[t] = {
            "n_puzzles": int(m.sum()),
            "coverage": {str(N): float((found[N][m] / totals[m]).mean()) for N in n_list},
            "found_mean": {str(N): float(found[N][m].mean()) for N in n_list},
            "upper_bound": {str(N): float(min(N, t) / t) for N in n_list},
        }

    res = {
        "checkpoint": os.path.join(args.ckpt_dir, args.step),
        "n": args.n, "N": args.N, "seed": args.seed, "n_list": n_list,
        "n_puzzles": int(have.sum()), "n_puzzles_no_total": int((~have).sum()),
        "overall": {
            str(N): {
                "coverage": float((found[N][have] / totals[have]).mean()),
                "upper_bound": float(np.minimum(N, totals[have]).astype(float).mean()
                                     / totals[have].astype(float).mean()),
                "upper_bound_perpuzzle": float((np.minimum(N, totals[have]) / totals[have]).mean()),
            } for N in n_list
        },
        "by_total": strata,
        "per_puzzle": {"totals": totals.tolist(),
                       **{f"found_{N}": found[N].tolist() for N in n_list}},
    }
    with open(args.out, "w") as f:
        json.dump(res, f)

    line = " ".join(f"cov@{N}={res['overall'][str(N)]['coverage']*100:.2f}%" for N in n_list)
    print(f"{res['checkpoint']}  퍼즐 {res['n_puzzles']}  {line}  [out] {args.out}")


if __name__ == "__main__":
    main()
