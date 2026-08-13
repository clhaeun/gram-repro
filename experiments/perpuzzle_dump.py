"""[50] Fig 4 right / Fig 15 용 per-puzzle 덤프.  N 마다 독립 실행한다.

기존 coverage_curve.py 는 N=20 을 한 번 rollout 한 뒤 앞에서 K개씩 잘라
found_nK 를 동시에 만들었다.  여기서는 --N 하나만 받아 그 N 으로만 rollout 하고
행 전수를 CSV 로 덤프한다 (test 서브샘플 없음).

열 정의
  n_solutions           sidecar 의 해당 입력 GT 전체 해 개수 (라벨 개수 아님).
                        sweep_harness.solution_counts / experiments/solution_counts.py
                        와 같은 정의 -- len(sidecar["inputs"][sha1(x)[:16]]["solutions"]).
  n_valid_samples       N 표본 중 valid (제약 만족) 인 것의 개수.  consistent 불요구.
  n_valid_consistent    N 표본 중 valid ∧ consistent 인 것의 개수 (found 의 분모 축).
                        n_valid_samples 의 정의가 valid 단독인지 valid∧consistent
                        인지 갈리므로 둘 다 남긴다.
  found                 N 표본이 맞힌 서로 다른 valid ∧ consistent 보드 수.
                        유일화 키는 nqueens.queen_key (퀸 칸 인덱스) -- 빈칸 토큰 값은
                        무시한다.  sweep_harness.coverage_at_N 의 found 와 같은 정의.
  first_valid           0/1 -- 첫 표본(k=0) 하나가 valid 인가.  Fig 4 right 축.
                        coverage_curve.dump_perpuzzle 의 valid_n1 과 같은 정의.

행 단위 덤프다.  nq 데이터는 (입력, 해) pair 로 행이 확장돼 있어 같은 ihash 가 여러
행에 나온다 -- 퍼즐 축 집계는 downstream 에서 ihash 로 한다.

사용:
  python experiments/perpuzzle_dump.py --ckpt-dir <dir> --step step_N --N 5 \
      --data data/nqueens-8/test --sidecar data/nqueens-8/sidecar_8.json --n 8 \
      --out results/figdata/nq8_perpuzzle_N5.csv
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.gram_harness import load_model, rollout            # noqa: E402
from experiments.sweep_harness import solution_counts               # noqa: E402
from evaluators.nqueens import queen_key, verify_boards             # noqa: E402

COLS = ["run", "step", "weight", "harness_seed", "N", "hms", "sigma_scale",
        "row", "ihash", "n_solutions", "n_valid_samples", "n_valid_consistent",
        "found", "first_valid"]


def md5(path):
    import hashlib as _h
    return _h.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8/test")
    ap.add_argument("--sidecar", default="data/nqueens-8/sidecar_8.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    ap.add_argument("--mu-scale", type=float, default=1.0)
    ap.add_argument("--halt-max-steps", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--weight", default="ema", choices=["ema", "live"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net, _ = load_model(args.ckpt_dir, args.step)
    if args.halt_max_steps is not None:
        net.model.config.halt_max_steps = args.halt_max_steps   # inner 와 공유 객체
    hms = net.model.config.halt_max_steps

    inputs = np.load(f"{args.data}/all__inputs.npy").astype(np.int64)
    labels = np.load(f"{args.data}/all__labels.npy").astype(np.int64)
    B = len(inputs)

    pri = rollout(net, inputs, labels, K=args.N, sigma_scale_=args.sigma_scale,
                  mu_scale_=args.mu_scale, mode="prior", seed=args.seed,
                  batch_size=args.batch_size, verbose=False)
    cand = pri["preds"]                                          # [N, B, L]

    r = verify_boards(cand.reshape(args.N * B, -1), np.tile(inputs, (args.N, 1)), args.n)
    valid = r["valid"].reshape(args.N, B)
    ok = (r["valid"] & r["consistent"]).reshape(args.N, B)

    found = np.zeros(B, dtype=np.int64)
    for b in range(B):
        found[b] = len({queen_key(cand[k, b]) for k in range(args.N) if ok[k, b]})
    n_valid = valid.sum(0).astype(np.int64)
    n_ok = ok.sum(0).astype(np.int64)
    first_valid = valid[0].astype(np.int64)

    totals = solution_counts(inputs, args.sidecar)
    ihash = [hashlib.sha1(v.astype(np.uint8).tobytes()).hexdigest()[:16] for v in inputs]

    run = os.path.basename(args.ckpt_dir.rstrip("/"))
    step_num = int(args.step.replace(".live", "").split("_")[-1])
    const = [run, step_num, args.weight, args.seed, args.N, hms, args.sigma_scale]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for i in range(B):
            w.writerow(const + [i, ihash[i], int(totals[i]), int(n_valid[i]),
                                int(n_ok[i]), int(found[i]), int(first_valid[i])])

    # ---- 집계 (meta 에만 기록.  CSV 는 원자료) ----
    have = totals > 0
    row_macro = float((found[have] / totals[have]).mean())
    micro = float(found[have].sum() / totals[have].sum())
    agg = {}
    for h, t, fd, fv in zip(ihash, totals.tolist(), found.tolist(), first_valid.tolist()):
        if t > 0:
            a = agg.setdefault(h, [0, 0, 0, 0])
            a[0] += fd; a[1] = t; a[2] += fv; a[3] += 1
    puzzle_macro = float(np.mean([min(a[0], a[1]) / a[1] for a in agg.values()]))
    gt = [(ihash[i], int(totals[i]), int(found[i]), i)
          for i in range(B) if have[i] and found[i] > totals[i]]

    first_row = {}
    for i, h in enumerate(ihash):
        first_row.setdefault(h, i)
    uniq_idx = np.array(sorted(first_row.values()))
    uniq_have = uniq_idx[totals[uniq_idx] > 0]

    meta = {
        "checkpoint": os.path.join(args.ckpt_dir, args.step),
        "ckpt_dir": args.ckpt_dir, "run": run, "step": step_num,
        "weight": args.weight,
        "weight_detail": ("bare step 파일 = EMA 사본 (gram_harness.load_model 이 "
                          "all_config.yaml + ema=False 로 로드)"),
        "data": args.data, "sidecar": args.sidecar,
        "n": args.n, "N": args.N, "harness_seed": args.seed,
        "hms": hms, "sigma_scale": args.sigma_scale, "mu_scale": args.mu_scale,
        "batch_size": args.batch_size,
        "unit": "행 단위 (test 전수).  같은 ihash 가 여러 행에 나온다.",
        "rows_written": B,
        "unique_inputs": len(set(ihash)),
        "rows_per_input": B / len(set(ihash)),
        "n_rows_no_total": int((~have).sum()),
        "independent_run": ("이 프로세스는 K=--N 으로만 rollout 한다 (prefix 자르기 아님). "
                            "단 gram_harness.rollout:152 가 표본 k 마다 "
                            "torch.random.manual_seed(seed*100003+k) 로 재시드하므로 "
                            "같은 seed 면 표본 k 는 N 과 무관하게 동일하다."),
        "columns": {
            "n_solutions": "sidecar 의 해당 입력 GT 전체 해 개수 (라벨 개수 아님)",
            "n_valid_samples": "N 표본 중 valid (제약 만족) 개수, consistent 불요구",
            "n_valid_consistent": "N 표본 중 valid ∧ consistent 개수 (스펙 외 추가 열)",
            "found": "N 표본이 맞힌 서로 다른 valid ∧ consistent 보드 수",
            "first_valid": "첫 표본(k=0) 1개가 valid 인가 (0/1) -- Fig 4 right 축",
            "row": "split 내 행 인덱스", "ihash": "sha1(inputs)[:16]",
        },
        "aggregates": {
            "coverage_row_macro": row_macro,
            "coverage_micro_sum_found_over_sum_total": micro,
            "coverage_puzzle_macro": puzzle_macro,
            "found_mean": float(found[have].mean()),
            "total_mean": float(totals[have].mean()),
            "sum_found": int(found[have].sum()),
            "sum_n_solutions": int(totals[have].sum()),
            "n_found_gt_total": len(gt),
            "first_valid_mean_rows": float(first_valid[have].mean()),
            "first_valid_mean_unique_first_row": float(first_valid[uniq_have].mean()),
            "n_valid_samples_mean": float(n_valid[have].mean()),
            "acc_all_samples_valid": float(valid.mean()),
        },
        "found_gt_total_examples": [
            {"ihash": h, "row": i, "n_solutions": t, "found": fd} for h, t, fd, i in gt[:20]],
        "created": datetime.now(timezone.utc).isoformat(),
        "code_md5": {"dump": md5(os.path.abspath(__file__)),
                     "gram_harness": md5("experiments/gram_harness.py"),
                     "sweep_harness": md5("experiments/sweep_harness.py"),
                     "nqueens_eval": md5("evaluators/nqueens.py")},
        "argv": sys.argv,
    }
    try:
        meta["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip() or None
    except Exception:
        meta["git_head"] = None

    with open(args.out.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    a = meta["aggregates"]
    print(f"[out] {args.out}  rows={B}  N={args.N} hms={hms} sigma={args.sigma_scale}  "
          f"cov_micro={a['coverage_micro_sum_found_over_sum_total']:.6f} "
          f"cov_row_macro={a['coverage_row_macro']:.6f} "
          f"cov_puzzle_macro={a['coverage_puzzle_macro']:.6f} "
          f"first_valid={a['first_valid_mean_rows']:.6f} "
          f"gt_total={a['n_found_gt_total']}", flush=True)


if __name__ == "__main__":
    main()
