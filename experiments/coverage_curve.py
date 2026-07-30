"""논문 Figure 15 와 맞대볼 수 있는 형태로 coverage 를 그린다.

x축 = 인스턴스의 해 개수, y축 = coverage, 곡선 = 샘플 수 N.

`sweep_harness.py` 의 coverage_at_20 은 N=20 한 점만, 그것도 굵은 버킷(1 / 2-5 / 6-20 / 21+)
으로만 낸다.  여기서는 같은 rollout 을 재사용해 N ∈ {1,5,10,20} 의 coverage 를 해 개수별로
층화한다.  coverage 정의는 논문 p7 과 동일하다 -- found / total valid solutions.

N 샘플의 coverage 상한은 min(N, total)/total 이라, 해가 많은 인스턴스는 모델이 완벽해도
상한에 걸린다(해 90 개 + N=20 이면 22%).  그래서 상한 곡선도 같이 낸다.

    python experiments/coverage_curve.py \
        --ckpt-dir checkpoints/Nqueens-10-ACT-torch/nq10_on_b520_s1 --step step_37466.live \
        --n 10 --data data/nqueens-10/test --sidecar data/nqueens-10/sidecar_10.json \
        --out experiments/results/covcurve_nq10_b520_s1.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gram_harness import load_model, rollout                      # noqa: E402
from sweep_harness import verify_boards, solution_counts          # noqa: E402


def coverage_by_n(cand, inputs, n, totals, n_list):
    """cand [K, B, L] -> {N: per-puzzle found 배열}.

    found = 앞 N 개 표본 중 서로 다른 valid&consistent 보드의 개수.
    """
    K, B, L = cand.shape
    r = verify_boards(cand.reshape(K * B, L), np.tile(inputs, (K, 1)), n)
    ok = (r["valid"] & r["consistent"]).reshape(K, B)

    found = {N: np.zeros(B, dtype=np.int64) for N in n_list}
    for b in range(B):
        seen = set()
        nxt = 0
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
    ap.add_argument("--tag", default=None)
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

    # 해 개수별 층화.  Figure 15 처럼 개별 해 개수 값을 그대로 x 축으로 쓴다.
    strata = {}
    for t in sorted(set(int(x) for x in totals[have])):
        m = have & (totals == t)
        strata[t] = {
            "n_puzzles": int(m.sum()),
            "coverage": {str(N): float((found[N][m] / totals[m]).mean()) for N in n_list},
            "found_mean": {str(N): float(found[N][m].mean()) for N in n_list},
            # 모델이 완벽해도 못 넘는 상한 = min(N, total) / total
            "upper_bound": {str(N): float(min(N, t) / t) for N in n_list},
        }

    res = {
        "checkpoint": os.path.join(args.ckpt_dir, args.step),
        "tag": args.tag, "n": args.n, "N": args.N, "seed": args.seed,
        "n_list": n_list,
        "n_puzzles": int(have.sum()),
        "n_puzzles_no_total": int((~have).sum()),
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

    print(f"{res['checkpoint']}   퍼즐 {res['n_puzzles']}")
    print(f"{'N':>4s} {'coverage':>9s} {'상한(퍼즐평균)':>13s} {'달성률':>7s}")
    for N in n_list:
        o = res["overall"][str(N)]
        print(f"{N:4d} {o['coverage']*100:8.2f}% {o['upper_bound_perpuzzle']*100:12.2f}% "
              f"{o['coverage']/o['upper_bound_perpuzzle']*100:6.2f}%")
    print(f"\n{'해수':>5s} {'n':>6s} " + " ".join(f"{'cov@'+str(N):>9s} {'ub@'+str(N):>7s}" for N in n_list))
    for t in sorted(strata):
        s = strata[t]
        row = f"{t:5d} {s['n_puzzles']:6d} "
        for N in n_list:
            row += f" {s['coverage'][str(N)]*100:8.2f}% {s['upper_bound'][str(N)]*100:6.1f}%"
        print(row)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
