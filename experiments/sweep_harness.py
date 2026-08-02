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
import hashlib
import json
import os
import re
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


def solution_counts(inputs, sidecar_path):
    """문제별 ground-truth 해의 총 개수 total(x).

    라벨 개수가 아니다 -- 라벨은 x 하나당 1개만 저장돼 있다. sidecar 는 해를 전수
    열거해 두었으므로 그쪽의 per-x 해 개수를 쓴다.
    키: 'sha1(flat token board uint8 bytes).hexdigest()[:16]' (sidecar meta 참조).
    조회 실패한 문제는 -1.
    """
    tbl = json.load(open(sidecar_path))["inputs"]
    out = np.full(len(inputs), -1, dtype=np.int64)
    for i, row in enumerate(inputs):
        e = tbl.get(hashlib.sha1(row.astype(np.uint8).tobytes()).hexdigest()[:16])
        if e is not None:
            out[i] = len(e["solutions"])
    return out


# 논문 Fig 15 대응 버킷
SOL_BUCKETS = [("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21+", 21, 10 ** 9)]


def coverage_at_N(cand, inputs, n, totals):
    """논문 Table 1 의 coverage = "unique valid solutions discovered with 20 samples".

    문제 x 마다
      found(x) = N 샘플 중 **valid AND consistent** 인 보드의 unique 개수
                 (consistent 필수: 힌트를 어긴 valid 보드는 x 의 해집합에 없다.
                  우리 데이터는 해 전수 열거 기반이라 valid & consistent <=> x 의 정답해)
      total(x) = sidecar 의 per-x 해 개수 (라벨 개수 아님)
      coverage(x) = found(x) / total(x)
    반환: 전체 평균 + 해 개수 버킷별 분해 + 진단용 카운트.
    """
    N, B, L = cand.shape
    r = verify_boards(cand.reshape(N * B, L), np.tile(inputs, (N, 1)), n)
    ok = (r["valid"] & r["consistent"]).reshape(N, B)          # [N, B]

    found = np.zeros(B, dtype=np.int64)
    for b in range(B):
        found[b] = len({tuple(cand[k, b]) for k in range(N) if ok[k, b]})

    have = totals > 0
    cov = np.zeros(B, dtype=np.float64)
    cov[have] = found[have] / totals[have]

    res = dict(
        coverage=float(cov[have].mean()) if have.any() else None,
        n_puzzles_scored=int(have.sum()),
        n_puzzles_no_total=int((~have).sum()),
        found_mean=float(found[have].mean()) if have.any() else None,
        total_mean=float(totals[have].mean()) if have.any() else None,
        # found > total 이면 정의가 깨진 것(해집합 밖 보드가 통과). 0 이어야 정상.
        n_found_gt_total=int((found[have] > totals[have]).sum()),
    )
    buckets = []
    for name, lo, hi in SOL_BUCKETS:
        m = have & (totals >= lo) & (totals <= hi)
        buckets.append(dict(bucket=name, n_puzzles=int(m.sum()),
                            coverage=(float(cov[m].mean()) if m.any() else None),
                            found_mean=(float(found[m].mean()) if m.any() else None),
                            total_mean=(float(totals[m].mean()) if m.any() else None)))
    res["by_solution_bucket"] = buckets

    # 행 단위 원자료.  2026-08-03 추가 (기존 키는 하나도 안 건드렸다 -- 순수 추가다).
    #
    # 왜 필요한가: test 가 pair 확장(입력 x 가 해 개수 k 만큼 반복)되면서 위 `coverage`
    # 한 값이 어느 축인지 애매해졌다.  실제로는
    #     mean_row(found/total) = (1/R)·Σ_x k·(found_x/k) = Σ_x found_x / R  = **pair 축**
    # 이라 `coverage` 는 pair 가중이다.  입력(퍼즐) 축 mean_x(found_x/total_x) 은 같은 x 의
    # k 개 행을 먼저 묶어야 나오는데, 집계값만으로는 복원이 안 된다.  그래서 행 배열을 남긴다.
    # ihash 는 빌더/eval_harness 와 같은 sha1(flat uint8 board)[:16] 이다.
    res["per_row"] = {
        "ihash": [hashlib.sha1(v.astype(np.uint8).tobytes()).hexdigest()[:16] for v in inputs],
        "totals": totals.tolist(),
        "found": found.tolist(),
    }
    return res


# 결과 json 소비자가 지표를 헷갈리지 않도록 정의를 파일 자체에 박아둔다.
# 축: [가중치는 --ckpt-dir/--step 로, 경로는 이 스크립트=harness, 표본=N, 기준=valid/strict]
METRIC_DEFS = {
    "집계방식/단일시도": "퍼즐당 N 샘플을 각각 채점한 평균. '한 번 뽑아 맞을 확률'. pass@1 과 수학적으로 동일.",
    "prior_all_samples.valid": "단일시도, valid 기준(퀸 n개 + 행/열/양대각 중복 없음). 입력 힌트 보존은 보지 않음.",
    "prior_all_samples.consistent": "단일시도, 입력에 주어진 퀸을 모두 보존했는가.",
    "prior_all_samples.strict": "단일시도, valid AND consistent.",
    "prior_single_k0": "단일시도지만 k=0 샘플 1개만 사용(표본 1/N). 참고용.",
    "posterior": "정답 y 를 본 posterior 경로. prior 성능과 다른 축이라 같이 읽지 말 것.",
    "집계방식/pass@k": "퍼즐당 N 샘플 중 k 개를 뽑았을 때 1개 이상 valid 일 확률(불편추정량). k=N 이면 puzzles_with_ge1_valid 와 항상 동일.",
    "집계방식/selection": "N 샘플에서 1개를 고른 뒤 채점. 고르는 규칙이 majority / verifier.",
    "majority_at_N_valid": "selection. N 샘플의 최빈 보드 1개(동률은 첫 등장)를 valid 채점.",
    "verifier_selected_at_N_valid": "selection. 모델 자신의 lrpm_head 점수가 최고인 샘플 1개를 valid 채점. 정답 미사용.",
    "distinct_at_N_mean": "다양성 지표(정확도 아님). N 샘플 중 서로 다른 보드 수, 1~N.",
    "집계방식/coverage": "정답해 집합을 몇 % 발견했는가. 위 셋과 완전히 다른 축.",
    "coverage_at_20": "논문 Table 1 coverage. 퍼즐마다 found/total 을 구해 전체 평균. "
                      "found = N 샘플 중 valid AND consistent 인 보드의 unique 개수, "
                      "total = sidecar 의 per-x 전체 해 개수(라벨 개수 아님). N 은 --N (기본 20).",
    "coverage_at_20.by_solution_bucket": "논문 Fig 15 대응. total(x) 을 {1, 2-5, 6-20, 21+} 로 나눈 분해.",
}


def log_to_wandb(res, seed):
    """측정 결과 1점을 wandb 곡선에 얹는다. --wandb 를 줬을 때만 호출된다.

    학습 run 하나당 harness run 하나(`<run>__live_scan` 등)를 **결정적 id** 로 잡고
    resume="allow" 로 이어붙인다. 그래서 스캔을 step 마다 따로 실행해도 한 곡선에 모이고,
    사후 일괄 업로드(wandb_harness_upload.py --mode curves)와도 같은 run 을 가리킨다.

    x 축은 커스텀 metric `harness/step` 이다. 끝난 학습 run 에 resume 해서 과거 step 으로
    log 하면 wandb 가 _step 단조성 때문에 조용히 버리기 때문에 그 경로는 쓸 수 없다.

    주의: 같은 (학습 run, 가중치, seed) 를 **두 프로세스가 동시에** 올리면 같은 wandb run 에
    붙어 경합한다. 스캔은 run 당 GPU 1장 순차 실행이라 문제되지 않지만, 병렬로 돌릴 거면
    --wandb 를 끄고 끝난 뒤 wandb_harness_upload.py 로 일괄 업로드해라.
    """
    import wandb
    from experiments.wandb_harness_upload import harness_run_id, harness_run_name, run_meta

    m = re.match(r"checkpoints/([^/]+)/([^/]+)/step_(\d+)(\.live)?$", res["checkpoint"])
    if not m:
        print(f"  [wandb] checkpoint 경로가 예상 형식이 아니라 건너뜀: {res['checkpoint']}")
        return
    project, train_run, step, live = m.group(1), m.group(2), int(m.group(3)), bool(m.group(4))
    weights = "live" if live else "ckpt"

    wrun = wandb.init(project=project, group=train_run, job_type="harness",
                      name=harness_run_name(train_run, weights, seed),
                      id=harness_run_id(project, train_run, weights, seed),
                      resume="allow", reinit=True,
                      config=dict(source_run=train_run, weights=weights, harness_seed=seed,
                                  **run_meta(project, train_run)))
    wrun.define_metric("harness/step")
    wrun.define_metric("harness/*", step_metric="harness/step")

    # coverage_at_20 은 나중에 추가된 지표라 예전 결과 json 에는 없다. 없으면 그 계열만 뺀다.
    p, q, c = res["prior_all_samples"], res["posterior"], res.get("coverage_at_20") or {}
    pre = f"harness/{weights}"
    payload = {
        "harness/step": step,
        f"{pre}/prior_valid": p["valid"], f"{pre}/prior_strict": p["strict"],
        f"{pre}/prior_consistent": p["consistent"], f"{pre}/queens_mean": p["queen_count_mean"],
        f"{pre}/posterior_valid": q["valid"],
        f"{pre}/majority_at_N": res["majority_at_N_valid"],
        f"{pre}/verifier_sel_at_N": res["verifier_selected_at_N_valid"],
        f"{pre}/distinct_at_N": res["distinct_at_N_mean"],
        f"{pre}/puzzles_with_ge1_valid": res["puzzles_with_ge1_valid"],
    }
    payload.update({f"{pre}/pass_at_{k}": v for k, v in res["pass_at"].items()})
    if c.get("coverage") is not None:
        payload[f"{pre}/coverage_at_N"] = c["coverage"]
        for b in c["by_solution_bucket"]:
            if b["coverage"] is not None:
                payload[f"{pre}/coverage_by_sol/{b['bucket']}"] = b["coverage"]
    wrun.log(payload)
    wrun.finish()
    print(f"  [wandb] {project}/{harness_run_name(train_run, weights, seed)} <- step {step}")


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
    # Table 3b ablation: --mu-scale 0 = "stochasticity only", --sigma-scale 0 = "guide only".
    ap.add_argument("--mu-scale", type=float, default=1.0)
    ap.add_argument("--sidecar", default="data/nqueens-8/sidecar_8.json",
                    help="coverage 의 total(x) 출처. per-x 전체 해 개수를 담고 있다.")
    ap.add_argument("--wandb", action="store_true",
                    help="측정 직후 결과를 wandb 곡선에 얹는다. 기본 off = 기존 동작 그대로.")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net, _ = load_model(args.ckpt_dir, args.step)
    inputs = np.load(f"{args.data}/all__inputs.npy").astype(np.int64)
    labels = np.load(f"{args.data}/all__labels.npy").astype(np.int64)
    B = len(inputs)

    pri = rollout(net, inputs, labels, K=args.N, sigma_scale_=args.sigma_scale,
                  mode="prior", seed=args.seed, verbose=False, mu_scale_=args.mu_scale)
    post = rollout(net, inputs, labels, K=1, sigma_scale_=args.sigma_scale,
                   mode="posterior", seed=args.seed, verbose=False, mu_scale_=args.mu_scale)

    cand, v = pri["preds"], pri["v"]                       # [N, B, L], [N, B]
    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "n": B, "N": args.N,
           "seed": args.seed, "sigma_scale": args.sigma_scale, "mu_scale": args.mu_scale,
           "tag": args.tag}

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

    # coverage: 위 지표들과 다른 축(정답해 집합의 발견 비율). 기존 키는 건드리지 않는다.
    totals = solution_counts(inputs, args.sidecar)
    res["coverage_at_20"] = coverage_at_N(cand, inputs, args.n, totals)
    res["metric_defs"] = METRIC_DEFS
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
    c = res["coverage_at_20"]
    print(f"  coverage@{args.N}={c['coverage']:.4f} (found {c['found_mean']:.2f} / total {c['total_mean']:.2f}, "
          f"scored {c['n_puzzles_scored']}, total없음 {c['n_puzzles_no_total']}, found>total {c['n_found_gt_total']})")
    for b in c["by_solution_bucket"]:
        if b["n_puzzles"]:
            print(f"    해{b['bucket']:>5}: n={b['n_puzzles']:4d}  coverage={b['coverage']:.4f}  "
                  f"found {b['found_mean']:.2f}/{b['total_mean']:.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  [out] {args.out}")

    if args.wandb:
        log_to_wandb(res, args.seed)


if __name__ == "__main__":
    main()
