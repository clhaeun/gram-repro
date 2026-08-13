"""N-Queens harness -- 논문 지표 ① Accuracy ③ Coverage@20 ④ majority@20 ⑤ verifier@20.

샘플링 RNG 는 gram_harness.rollout (seed*100003+k).
--sigma-scale / --mu-scale 은 추론 시 스케일링이라 학습 시 ablation 인 Table 3b 와
직접 대응하지 않는다.

사용: python experiments/sweep_harness.py --ckpt-dir <dir> --step step_N --out <json>
"""
import argparse, hashlib, json, os, sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.gram_harness import load_model, rollout   # noqa: E402
from evaluators.nqueens import queen_key, verify_boards    # noqa: E402


def flat_stats(preds, inputs, n):
    """① Accuracy (Table 1 single-sample): 전 샘플 각각 채점한 valid 평균.

    queen_count_* 는 같은 K*B 전샘플 축의 통계다.  학습 로그의
    `nqueens/queen_count_mean` 은 in-loop eval 의 단일 샘플 축이라 눈금이 다르다 --
    직접 비교 금지 (사수 지시 2026-08-06 로 harness 축을 추가 산출).
    """
    K, B, L = preds.shape
    r = verify_boards(preds.reshape(K * B, L), np.tile(inputs, (K, 1)), n)
    qc = r["queen_count"]
    return dict(valid=float(r["valid"].mean()),
                queen_count_mean=float(qc.mean()),
                queen_count_min=int(qc.min()), queen_count_max=int(qc.max()))


def majority_row(rows):
    """④ majority@20 (§2.3 방법 1): 최빈 보드 1개, 동률은 첫 등장."""
    return np.array(Counter(tuple(r) for r in rows).most_common(1)[0][0], dtype=np.int64)


def solution_counts(inputs, sidecar_path):
    """total(x) = sidecar 의 per-x 전체 해 개수 (라벨 개수 아님).  키 sha1[:16], 실패 -1."""
    tbl = json.load(open(sidecar_path))["inputs"]
    out = np.full(len(inputs), -1, dtype=np.int64)
    for i, row in enumerate(inputs):
        e = tbl.get(hashlib.sha1(row.astype(np.uint8).tobytes()).hexdigest()[:16])
        if e is not None:
            out[i] = len(e["solutions"])
    return out


SOL_BUCKETS = [("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21+", 21, 10 ** 9)]


def coverage_at_N(cand, inputs, n, totals):
    """③ Coverage@20 (Table 1): mean_x(found/total).  found = valid∧consistent unique
    (힌트 어긴 valid 보드는 해집합 밖이라 consistent 필수 -- n_found_gt_total=0 이 검산).

    유일화 키는 nqueens.queen_key (퀸 칸 인덱스) -- 토큰 벡터 유일화가 아니다.
    """
    N, B, L = cand.shape
    r = verify_boards(cand.reshape(N * B, L), np.tile(inputs, (N, 1)), n)
    ok = (r["valid"] & r["consistent"]).reshape(N, B)
    found = np.zeros(B, dtype=np.int64)
    for b in range(B):
        found[b] = len({queen_key(cand[k, b]) for k in range(N) if ok[k, b]})
    have = totals > 0
    cov = np.zeros(B, dtype=np.float64)
    cov[have] = found[have] / totals[have]
    res = dict(
        coverage=float(cov[have].mean()) if have.any() else None,
        n_puzzles_scored=int(have.sum()), n_puzzles_no_total=int((~have).sum()),
        found_mean=float(found[have].mean()) if have.any() else None,
        total_mean=float(totals[have].mean()) if have.any() else None,
        n_found_gt_total=int((found[have] > totals[have]).sum()))
    res["by_solution_bucket"] = [                       # 논문 Fig 15 대응 (JSON 전용)
        dict(bucket=name, n_puzzles=int(m.sum()),
             coverage=(float(cov[m].mean()) if m.any() else None),
             found_mean=(float(found[m].mean()) if m.any() else None),
             total_mean=(float(totals[m].mean()) if m.any() else None))
        for name, lo, hi in SOL_BUCKETS for m in [have & (totals >= lo) & (totals <= hi)]]
    # pair 확장 test 에서 입력(퍼즐) 축 재집계용 행 원자료
    ihash = [hashlib.sha1(v.astype(np.uint8).tobytes()).hexdigest()[:16] for v in inputs]
    res["per_row"] = {
        "ihash": ihash, "totals": totals.tolist(), "found": found.tolist()}
    # coverage 축 3종은 눈금이 다르다 -- 인용 시 어느 축인지 밝힐 것.
    #   coverage                 행 축 macro
    #   found_mean/total_mean    micro (합계비)
    #   coverage_macro_by_input  퍼즐 축 macro
    agg = {}
    for h, t, f in zip(ihash, totals.tolist(), found.tolist()):
        if t > 0:
            a = agg.setdefault(h, [0, 0]); a[0] += f; a[1] += t
    res["coverage_macro_by_input"] = (
        float(np.mean([min(f, t) / t for f, t in agg.values()])) if agg else None)
    res["n_inputs_scored"] = len(agg)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--data", default="data/nqueens-8/test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    # 추론 시 스케일링.  학습 시 ablation 인 논문 Table 3b 와 직접 대응하지 않는다.
    ap.add_argument("--sigma-scale", type=float, default=1.0)  # 0 = guide only
    ap.add_argument("--mu-scale", type=float, default=1.0)     # 0 = stochasticity only
    ap.add_argument("--sidecar", default="data/nqueens-8/sidecar_8.json")
    # N_sup 외삽용. None = 오버라이드 없음 (체크포인트 학습값 그대로).
    # eval 에선 gram.py 의 halted = (new_steps >= halt_max_steps) 만 남으므로 N_sup = 이 값.
    ap.add_argument("--halt-max-steps", type=int, default=None)
    # 측정 전용.  transition 별 KL/μ/σ 를 <out>.diagkl.json 에 따로 쓴다.
    # --out 본 JSON 은 이 플래그 유무와 무관하게 같아야 한다 (궤적 불변 검증).
    ap.add_argument("--diag-kl", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    net, _ = load_model(args.ckpt_dir, args.step)
    if args.halt_max_steps is not None:
        net.model.config.halt_max_steps = args.halt_max_steps   # inner 와 공유 객체
    if args.diag_kl:
        net.model.config.diag_kl_per_transition = True          # inner 와 공유 객체
        net.model.inner.diag_kl_reset()
    inputs = np.load(f"{args.data}/all__inputs.npy").astype(np.int64)
    labels = np.load(f"{args.data}/all__labels.npy").astype(np.int64)
    B = len(inputs)

    pri = rollout(net, inputs, labels, K=args.N, sigma_scale_=args.sigma_scale,
                  mode="prior", seed=args.seed, verbose=False, mu_scale_=args.mu_scale)
    cand, v = pri["preds"], pri["v"]                           # [N, B, L], [N, B]

    res = {"checkpoint": os.path.join(args.ckpt_dir, args.step), "n": B, "N": args.N,
           "seed": args.seed, "sigma_scale": args.sigma_scale, "mu_scale": args.mu_scale,
           "halt_max_steps": net.model.config.halt_max_steps}
    res["prior_all_samples"] = flat_stats(cand, inputs, args.n)                  # ①
    maj = np.stack([majority_row(cand[:, b, :]) for b in range(B)])              # ④
    sel = np.stack([cand[int(np.argmax(v[:, b])), b, :] for b in range(B)])      # ⑤ LPRM best-of-N
    res["majority_at_N_valid"] = float(verify_boards(maj, inputs, args.n)["valid"].mean())
    res["verifier_selected_at_N_valid"] = float(verify_boards(sel, inputs, args.n)["valid"].mean())
    res["coverage_at_20"] = coverage_at_N(cand, inputs, args.n,
                                          solution_counts(inputs, args.sidecar))  # ③

    c = res["coverage_at_20"]
    if c["n_found_gt_total"]:
        print(f"WARNING: n_found_gt_total={c['n_found_gt_total']} (0 이어야 정상)")
    cov_micro = c["found_mean"] / c["total_mean"]        # 합계비 = micro
    print(f"{res['checkpoint']}  hms={res['halt_max_steps']} "
          f"acc={res['prior_all_samples']['valid']:.4f} "
          f"cov_row_macro@{args.N}={c['coverage']:.4f} "
          f"cov_micro@{args.N}={cov_micro:.4f} "
          f"cov_puzzle_macro@{args.N}={c['coverage_macro_by_input']:.4f} "
          f"maj={res['majority_at_N_valid']:.4f} "
          f"vsel={res['verifier_selected_at_N_valid']:.4f} "
          f"qc={res['prior_all_samples']['queen_count_mean']:.4f}  [out] {args.out}")
    # 쓰기가 실패해도 stdout 으로 전량 덤프해 로그에서 복구할 수 있게 한다.
    try:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    except OSError as e:
        print(f"!!! WRITE FAILED {args.out}: {e}\n"
              f"!!! 아래 RESULT_JSON 한 줄이 전량 산출물이다 (복구용)", flush=True)
        print("RESULT_JSON " + json.dumps(res), flush=True)

    # 진단은 반드시 별도 파일 -- 본 JSON 에 섞으면 on/off md5 대조가 깨진다.
    if args.diag_kl:
        dk = net.model.inner.diag_kl_result()
        dk_out = os.path.splitext(args.out)[0] + ".diagkl.json"
        with open(dk_out, "w") as f:
            json.dump({"checkpoint": res["checkpoint"], "H_cycles": net.model.config.H_cycles,
                       "n_forwards": net.model.inner.diag_kl_n, "per_transition": dk}, f, indent=2)
        print(f"  [diag-kl] {dk_out}  " +
              " ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in dk.items() if "/kl_t" in k))


if __name__ == "__main__":
    main()
