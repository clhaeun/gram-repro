"""N-Queens harness: prior N=20 샘플을 **valid 기준** 논문 지표 4종으로 집계.

논문 지표 (② Conflict 는 GC 전용이라 gc_harness.py):

  ① Accuracy (Table 1 "single-sample")   = (1/(N·B)) Σ_x Σ_k valid(ŷ_x^k)
     valid = 퀸 정확히 n개 ∧ 행·열·양대각(row−col, row+col) 중복 없음.
     consistent(입력 힌트 보존)는 포함하지 않는다. pass@1 의 추정량.
     -> flat_stats() = prior_all_samples.valid
  ③ Coverage@20 (Table 1)                = (1/B) Σ_x found(x)/total(x)
     -> coverage_at_N(). found 판정은 valid ∧ consistent (아래 docstring).
  ④ majority@20 (§2.3 selection 방법 1)   -> majority_row() -> majority_at_N_valid
  ⑤ verifier@20 = LPRM best-of-N (§2.3 방법 2, Appendix A.2 Eq.16)
     LPRM = value head v_ψ(z_t), 최종 출력 정확도 r 을 회귀하도록 joint 학습
     (L_LPRM = Σ_t (v_ψ(z_t) − r)²). 추론 시 v_ψ 최고점 1개를 valid 채점. 정답 미사용.
     -> main() 의 rows[argmax(v[:,b])] -> verifier_selected_at_N_valid

진단 지표(pass@k, distinct, prior_single_k0, posterior 등)는 2026-08-03 minimal
정리에서 삭제 -- 필요하면 legacy/sweep_harness_legacy.py.
--sigma-scale / --mu-scale 은 논문 Table 3 ablation 용이라 유지.

experiments/eval_harness.py 를 쓰지 않는 이유: 그쪽 NQueensTask.verify 는
`strict`(valid AND consistent)를 정답 기준으로 반환한다.  논문 축은 valid 라
별도로 집계한다.  샘플링 RNG 스트림은 gram_harness.rollout 그대로 (seed*100003+k).

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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.gram_harness import load_model, rollout   # noqa: E402
from evaluators.nqueens import verify_boards               # noqa: E402


# ---------------------------------------------------------------------------
# 논문 지표 (Table 1 / §2.3).  번호는 모듈 docstring 의 ①~⑤.
# ---------------------------------------------------------------------------

def flat_stats(preds, inputs, n):
    """preds [K, B, L] -> ① Accuracy: 전 샘플 각각 채점한 valid 평균 (K=N).

    verify_boards 는 consistent/strict 도 반환하지만 출력 표에는 valid 만 낸다 --
    consistent 는 ③ coverage_at_N 내부의 found 판정(valid ∧ consistent)에 쓰인다.
    """
    K, B, L = preds.shape
    r = verify_boards(preds.reshape(K * B, L), np.tile(inputs, (K, 1)), n)
    return dict(valid=float(r["valid"].mean()))


def majority_row(rows):
    """④ majority@20 의 selection: 최빈 보드, 동률은 첫 등장 (eval_harness.py:255-258 과 동일)."""
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
    """③ 논문 Table 1 의 coverage = "unique valid solutions discovered with 20 samples".

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
METRIC_DEFS = {
    "prior_all_samples.valid": "① 논문 Table 1 single-sample accuracy. 퍼즐당 N 샘플 각각 채점한 평균, "
                               "valid 기준(퀸 n개 + 행/열/양대각 중복 없음). 입력 힌트 보존(consistent)은 포함하지 않음.",
    "majority_at_N_valid": "④ 논문 §2.3 selection 방법 1. N 샘플의 최빈 보드 1개(동률은 첫 등장)를 valid 채점.",
    "verifier_selected_at_N_valid": "⑤ 논문 §2.3 방법 2 = LPRM best-of-N (Appendix A.2 Eq.16). "
                                    "v_ψ(z_t) 점수 최고 샘플 1개를 valid 채점. 정답 미사용.",
    "coverage_at_20": "③ 논문 Table 1 coverage. 퍼즐마다 found/total 을 구해 전체 평균. "
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
    p, c = res["prior_all_samples"], res.get("coverage_at_20") or {}
    pre = f"harness/{weights}"
    payload = {
        "harness/step": step,
        f"{pre}/prior_valid": p["valid"],
        f"{pre}/majority_at_N": res["majority_at_N_valid"],
        f"{pre}/verifier_sel_at_N": res["verifier_selected_at_N_valid"],
    }
    if c.get("coverage") is not None:
        payload[f"{pre}/coverage_at_N"] = c["coverage"]
        for b in c["by_solution_bucket"]:
            if b["coverage"] is not None:
                payload[f"{pre}/coverage_by_sol/{b['bucket']}"] = b["coverage"]
    wrun.log(payload)
    wrun.finish()
    print(f"  [wandb] {project}/{harness_run_name(train_run, weights, seed)} <- step {step}")


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

    cand, v = pri["preds"], pri["v"]                       # [N, B, L], [N, B]
    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "n": B, "N": args.N,
           "seed": args.seed, "sigma_scale": args.sigma_scale, "mu_scale": args.mu_scale,
           "tag": args.tag}

    res["prior_all_samples"] = flat_stats(cand, inputs, args.n)     # ① Accuracy

    # ④ majority / ⑤ verifier (LPRM best-of-N)
    maj, sel = [], []
    for b in range(B):
        rows = cand[:, b, :]
        maj.append(majority_row(rows))
        sel.append(rows[int(np.argmax(v[:, b]))])       # ⑤ v = lrpm_head 점수 [N, B]
    maj, sel = np.stack(maj), np.stack(sel)
    res["majority_at_N_valid"] = float(verify_boards(maj, inputs, args.n)["valid"].mean())
    res["verifier_selected_at_N_valid"] = float(verify_boards(sel, inputs, args.n)["valid"].mean())

    # ③ coverage: 위 지표들과 다른 축(정답해 집합의 발견 비율)
    totals = solution_counts(inputs, args.sidecar)
    res["coverage_at_20"] = coverage_at_N(cand, inputs, args.n, totals)
    res["metric_defs"] = METRIC_DEFS

    p = res["prior_all_samples"]
    print(f"{res['checkpoint']}")
    print(f"  prior  valid(all {args.N})={p['valid']:.4f}")
    print(f"  majority@{args.N}={res['majority_at_N_valid']:.4f} "
          f"verifier_sel@{args.N}={res['verifier_selected_at_N_valid']:.4f}")
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
