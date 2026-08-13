"""[50 별건] found > n_solutions 의 기전 확인 (측정 전용, 산출물 CSV 아님).

sweep_harness.coverage_at_N:60 / fprm_harness 의 found 는
    len({tuple(cand[k, b]) for k in range(N) if ok[k, b]})
로 ★토큰 벡터★ 를 유일화한다.  반면 evaluators.nqueens.verify_boards 는
`preds == QUEEN(2)` 만 읽으므로 빈칸 토큰이 1 이든 0 이든 valid/consistent 판정이
같다.  따라서 퀸 배치가 동일해도 빈칸 토큰 하나가 다르면 서로 다른 해로 세어진다.

이 프로브는 실제 rollout 표본에서
  found_token   토큰 벡터 유일화 (기존 정의)
  found_queens  퀸 좌표집합 유일화
를 함께 세고, 예측 토큰의 값 분포를 찍는다.
"""
import argparse, hashlib, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.gram_harness import load_model, rollout        # noqa: E402
from experiments.sweep_harness import solution_counts           # noqa: E402
from evaluators.nqueens import verify_boards                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-10/test")
    ap.add_argument("--sidecar", default="data/nqueens-10/sidecar_10.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=1024, help="앞 L 행만 (기전 확인용 표본)")
    ap.add_argument("--halt-max-steps", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net, _ = load_model(args.ckpt_dir, args.step)
    net.model.config.halt_max_steps = args.halt_max_steps
    inputs = np.load(f"{args.data}/all__inputs.npy").astype(np.int64)[: args.limit]
    labels = np.load(f"{args.data}/all__labels.npy").astype(np.int64)[: args.limit]
    B = len(inputs)

    cand = rollout(net, inputs, labels, K=args.N, sigma_scale_=1.0, mu_scale_=1.0,
                   mode="prior", seed=args.seed, verbose=False)["preds"]
    r = verify_boards(cand.reshape(args.N * B, -1), np.tile(inputs, (args.N, 1)), args.n)
    ok = (r["valid"] & r["consistent"]).reshape(args.N, B)

    totals = solution_counts(inputs, args.sidecar)
    ftok = np.zeros(B, dtype=np.int64); fq = np.zeros(B, dtype=np.int64)
    for b in range(B):
        ks = [k for k in range(args.N) if ok[k, b]]
        ftok[b] = len({tuple(cand[k, b]) for k in ks})
        fq[b] = len({frozenset(np.flatnonzero(cand[k, b] == 2).tolist()) for k in ks})

    have = totals > 0
    vals, cnts = np.unique(cand, return_counts=True)
    ok_flat = ok.reshape(-1)
    vals_ok, cnts_ok = np.unique(cand.reshape(args.N * B, -1)[ok_flat], return_counts=True)

    ex = [{"row": int(i), "ihash": hashlib.sha1(inputs[i].astype(np.uint8).tobytes()).hexdigest()[:16],
           "n_solutions": int(totals[i]), "found_token": int(ftok[i]), "found_queens": int(fq[i])}
          for i in range(B) if have[i] and ftok[i] > totals[i]][:20]

    res = {
        "checkpoint": os.path.join(args.ckpt_dir, args.step), "N": args.N, "seed": args.seed,
        "limit": args.limit, "rows": B, "hms": net.model.config.halt_max_steps,
        "n_found_gt_total_token": int((ftok[have] > totals[have]).sum()),
        "n_found_gt_total_queens": int((fq[have] > totals[have]).sum()),
        "n_rows_token_gt_queens": int((ftok > fq).sum()),
        "sum_found_token": int(ftok[have].sum()), "sum_found_queens": int(fq[have].sum()),
        "sum_n_solutions": int(totals[have].sum()),
        "pred_token_value_counts_all": {int(v): int(c) for v, c in zip(vals, cnts)},
        "pred_token_value_counts_valid_consistent": {int(v): int(c) for v, c in zip(vals_ok, cnts_ok)},
        "examples_found_gt_total": ex,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "examples_found_gt_total"}, indent=2))
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
