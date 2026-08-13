"""논문 Figure 15: x=해 개수, y=coverage, 곡선=N∈{1,5,10,20}.  sweep_harness 와 같은
rollout 재사용, found = 앞 N개 표본 중 서로 다른 valid∧consistent 보드 수.
상한 min(N,total)/total 곡선 병기 (해 90개 + N=20 이면 완벽해도 22%)."""
import argparse, csv, hashlib, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gram_harness import load_model, rollout                      # noqa: E402
from sweep_harness import verify_boards, solution_counts          # noqa: E402
from evaluators.nqueens import queen_key                          # noqa: E402


def coverage_by_n(cand, inputs, n, totals, n_list, want_valid1=False):
    """cand [K, B, L] -> {N: per-puzzle found 배열} (앞 N개 표본 prefix 기준).

    want_valid1=True 면 (found, valid1) 을 돌려준다.  valid1 은 첫 표본 하나가
    제약을 만족하는지(0/1) 로, found 와 달리 consistent 를 요구하지 않는다 --
    Fig 4(right) 의 valid_n1 컬럼 정의가 '단일 샘플이 제약 만족' 이기 때문이다.

    ★ 유일화 키는 nqueens.queen_key (퀸 칸 인덱스).  토큰 벡터 유일화가 아니다.
    """
    K, B, L = cand.shape
    r = verify_boards(cand.reshape(K * B, L), np.tile(inputs, (K, 1)), n)
    ok = (r["valid"] & r["consistent"]).reshape(K, B)
    found = {N: np.zeros(B, dtype=np.int64) for N in n_list}
    for b in range(B):
        seen, nxt = set(), 0
        for k in range(K):
            if ok[k, b]:
                seen.add(queen_key(cand[k, b]))
            while nxt < len(n_list) and n_list[nxt] == k + 1:
                found[n_list[nxt]][b] = len(seen)
                nxt += 1
    if want_valid1:
        return found, r["valid"].reshape(K, B)[0].astype(np.int64)
    return found


def dump_perpuzzle(path, inputs, totals, found, valid1, n_list):
    """unique 입력 단위 CSV.  nq 데이터는 (입력,해) pair 로 행이 확장돼 있으므로
    같은 sha1[:16] 을 가진 행들을 1행으로 합친다 -- 같은 입력의 rollout 은 노이즈
    시드만 다를 뿐 입력이 동일하므로 ★첫 등장 행★ 을 대표로 쓴다."""
    ihash = [hashlib.sha1(v.astype(np.uint8).tobytes()).hexdigest()[:16] for v in inputs]
    seen, rows = set(), []
    for i, h in enumerate(ihash):
        if h in seen or totals[i] <= 0:
            continue
        seen.add(h)
        rows.append([int(totals[i]), int(valid1[i])] + [int(found[N][i]) for N in n_list])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_solutions", "valid_n1"] + [f"found_n{N}" for N in n_list])
        w.writerows(rows)
    return len(rows), len(ihash), len(set(ihash))


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
    ap.add_argument("--dump-perpuzzle", default=None,
                    help="unique 입력 단위 per-puzzle CSV 경로 (Fig 4 right / Fig 15)")
    args = ap.parse_args()

    n_list = sorted(int(x) for x in args.n_list.split(","))
    assert n_list[-1] <= args.N, f"n_list 최대값 {n_list[-1]} 이 --N {args.N} 보다 크다"

    net, _ = load_model(args.ckpt_dir, args.step)
    inputs = np.load(f"{args.data}/all__inputs.npy").astype(np.int64)
    labels = np.load(f"{args.data}/all__labels.npy").astype(np.int64)

    pri = rollout(net, inputs, labels, K=args.N, mode="prior", seed=args.seed, verbose=False)
    cand = pri["preds"]                                    # [N, B, L]
    totals = solution_counts(inputs, args.sidecar)
    found, valid1 = coverage_by_n(cand, inputs, args.n, totals, n_list, want_valid1=True)
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

    if args.dump_perpuzzle:
        n_rows, n_all, n_uniq = dump_perpuzzle(
            args.dump_perpuzzle, inputs, totals, found, valid1, n_list)
        meta = {
            "checkpoint": res["checkpoint"],
            "ckpt_dir": args.ckpt_dir, "step": args.step,
            "weight": "EMA (gram_harness.load_model 은 all_config.yaml + ema=False 로 로드하고 bare step 파일이 EMA 사본이다)",
            "data": args.data, "sidecar": args.sidecar,
            "n": args.n, "N": args.N, "seed": args.seed, "n_list": n_list,
            "rows_in_split": n_all, "unique_inputs_in_split": n_uniq,
            "rows_written": n_rows,
            "rows_per_input": (n_all / n_uniq) if n_uniq else None,
            "unit": "unique 입력 1행 (sha1(inputs)[:16] 첫 등장 행 대표)",
            "columns": {
                "n_solutions": "sidecar 의 해당 입력 GT 유효해 개수",
                "valid_n1": "첫 표본 1개가 제약 만족 (0/1, consistent 불요구)",
                **{f"found_n{N}": f"앞 {N}개 표본에서 찾은 서로 다른 valid&consistent 해 개수"
                   for N in n_list},
            },
            "note": "N=%d 로 한 번 rollout 한 뒤 앞에서 K개씩 잘라 found_nK 를 동시에 산출했다." % args.N,
            "excluded": "sidecar 미매칭(totals<=0) 행 제외",
        }
        with open(args.dump_perpuzzle.replace(".csv", ".meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[out] {args.dump_perpuzzle}  {n_rows} rows (unique {n_uniq} / 전체 {n_all})")


if __name__ == "__main__":
    main()
