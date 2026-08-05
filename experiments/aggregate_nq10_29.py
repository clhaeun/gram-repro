"""29번 nq10 최종 harness 집계 (crash 로 125940 step 조기 종료된 2 run).

축 규약 (사수 지시 2026-08-04 정정판):
  - 헤드라인 = 최종 step x EMA.  live 병기.
    근거: 논문 B.2 가 EMA decay 0.9999 를 명시하고 Brock et al. / Song & Ermon 을 인용
    (두 논문 모두 평가에 EMA 사용).  TRM 원본 pretrain.py 는 save_train_state 호출부가
    한 곳뿐이고 거기 넘기는 것이 EMA 사본이며 raw 저장 경로가 아예 없다.
  - coverage 헤드라인 = micro(행 가중) = Sigma found / Sigma total = found_mean/total_mean
    (분모 행 집합이 동일하므로 정확).  macro(퍼즐 가중) = coverage 필드 = mean_x(found/total).
    50번 aggregate_final.py:35-36 의 cov_pair/cov_puzzle 정의와 동일.
  - run 별 값 = harness seed 0/1 평균, hs 편차 병기.
  - queen_count_mean 은 학습 로그에 이미 찍힌 값을 그대로 읽는다 (새 산출 코드 없음):
    EMA = "[EVAL @ step N] ... nqueens/queen_count_mean=",
    live = "[EVAL RAW @ step N] ... nqueens/queen_count_mean_raw=".

사용: python experiments/aggregate_nq10_29.py
"""
import glob, json, os, re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
LOGS = os.path.join(ROOT, "logs")

RUNS = [("nq10_repro_s2", "baseline (add)", [(125940, "최종"), (104950, "EMA 정점"), (88158, "raw 정점")]),
        ("nq10_fp_a3p_s1", "FPRM A3'", [(125940, "최종"), (121742, "EMA 정점"), (113346, "raw 정점")])]

FOOTNOTE = ("nq10_repro_s2 / nq10_fp_a3p_s1 은 서버 crash 로 138,548 중 125,940 step "
            "(91.6~93.9%)에서 종료. lr_min_ratio=1.0 이라 남은 구간에 anneal 없음.")


def queen_counts(run):
    """학습 로그에서 step -> (EMA qc, live qc). 로그에 찍힌 값을 그대로 읽는다."""
    out = {}
    txt = open(os.path.join(LOGS, f"{run}.log"), "rb").read().decode("utf8", "replace")
    for tag, key in (("EVAL", "queen_count_mean"), ("EVAL RAW", "queen_count_mean_raw")):
        for m in re.finditer(rf"\[{tag} @ step (\d+)\][^\n\r]*?nqueens/{key}=([0-9.]+)", txt):
            step, v = int(m.group(1)), float(m.group(2))
            e = out.setdefault(step, {})
            e["ema" if tag == "EVAL" else "live"] = v
    return out


def load(run, step, weight):
    """harness seed 0/1 -> 지표 dict 리스트."""
    out = []
    for hs in (0, 1):
        f = f"{RES}/fin_{run}_step{step}_{weight}_s{hs}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        c = d["coverage_at_20"]
        pr = c.get("per_row")
        micro_pr = None                      # per_row 로 micro 독립 재계산 -> 검산용
        if pr:
            t = np.array(pr["totals"]); fnd = np.array(pr["found"]); m = t > 0
            micro_pr = float(fnd[m].sum() / t[m].sum())
        out.append(dict(
            hs=hs,
            acc=d["prior_all_samples"]["valid"],          # 단일시도 (valid 기준)
            maj=d["majority_at_N_valid"],
            verif=d["verifier_selected_at_N_valid"],
            cov_micro=c["found_mean"] / c["total_mean"],  # 행 가중 (헤드라인)
            cov_macro=c["coverage"],                      # 퍼즐 가중 (병기)
            cov_micro_perrow=micro_pr,
            n_rows=c["n_puzzles_scored"], n_no_total=c["n_puzzles_no_total"],
            n_found_gt_total=c["n_found_gt_total"],
            buckets=c["by_solution_bucket"],
        ))
    return out


def agg(lst, k):
    v = [r[k] for r in lst if r.get(k) is not None]
    return (float(np.mean(v)), float(max(v) - min(v))) if v else (None, None)


def main():
    print("데이터축: data/nqueens-10/test (pair 확장), N=20, sigma/mu_scale=1.0")
    print("헤드라인 = 최종 step x EMA / live 병기 · coverage micro 헤드라인 / macro 병기")
    print(f"각주: {FOOTNOTE}\n")

    hdr = (f"{'run':16s} {'데이터축':10s} {'step':>7s} {'가중치':7s} "
           f"{'단일시도':>9s} {'cov micro':>10s} {'cov macro':>10s} "
           f"{'maj@20':>8s} {'verif@20':>9s} {'qcount':>7s}")
    gaps = {}
    for run, label, steps in RUNS:
        qc = queen_counts(run)
        print(f"===== {run}  ({label}) =====")
        print(hdr)
        last = None
        for step, what in steps:
            for weight in ("ema", "live"):
                lst = load(run, step, weight)
                if not lst:
                    print(f"{'':16s} {'pair(행)':10s} {step:>7d} {weight:7s}  없음")
                    continue
                last = lst
                acc, dacc = agg(lst, "acc")
                cmi, _ = agg(lst, "cov_micro")
                cma, _ = agg(lst, "cov_macro")
                maj, _ = agg(lst, "maj")
                vf, _ = agg(lst, "verif")
                gaps.setdefault((run, step), {})[weight] = dict(acc=acc, cov=cmi, what=what)
                q = qc.get(step, {}).get(weight)
                qs = f"{q:7.4f}" if q is not None else "     없음"
                mark = "★" if weight == "ema" else " "
                head = f"{mark}{run[:15]}" if weight == "ema" else ""
                print(f"{head:16s} {'pair(행)':10s} {step:>7d} {weight:7s} "
                      f"{acc*100:>8.2f}% {cmi*100:>9.2f}% {cma*100:>9.2f}% "
                      f"{maj*100:>7.2f}% {vf*100:>8.2f}% {qs}"
                      f"   [{what}, hs {len(lst)}개 평균, hs편차 {dacc*100:.3f}pp]")
                for r in lst:
                    if r["cov_micro_perrow"] is not None and abs(r["cov_micro_perrow"] - r["cov_micro"]) > 1e-9:
                        print(f"      WARNING micro 불일치 hs{r['hs']}: "
                              f"per_row={r['cov_micro_perrow']:.9f} vs found/total={r['cov_micro']:.9f}")
                    if r["n_found_gt_total"]:
                        print(f"      WARNING n_found_gt_total={r['n_found_gt_total']} (0 이어야 정상)")
        if last:
            print(f"  (분모: 채점행 {last[0]['n_rows']:,} / total 없음 {last[0]['n_no_total']:,})")
        print()

    print("===== EMA - live 격차 (EMA 가 live 대비 낮은 지점 표시) =====")
    for (run, step), d in sorted(gaps.items()):
        if "ema" not in d or "live" not in d:
            continue
        da = (d["ema"]["acc"] - d["live"]["acc"]) * 100
        dc = (d["ema"]["cov"] - d["live"]["cov"]) * 100
        flag = "  <- EMA 가 live 보다 낮음" if da < 0 else ""
        print(f"  {run:16s} step {step:>7d} ({d['ema']['what']:8s}) "
              f"단일시도 EMA {d['ema']['acc']*100:6.2f}% / live {d['live']['acc']*100:6.2f}% ({da:+6.2f}pp) | "
              f"cov EMA {d['ema']['cov']*100:6.2f}% / live {d['live']['cov']*100:6.2f}% ({dc:+6.2f}pp){flag}")

    print("\n===== 해 개수 버킷별 coverage (헤드라인: 최종 step 125940 x EMA, hs0) =====")
    for run, label, _ in RUNS:
        lst = load(run, 125940, "ema")
        if not lst:
            continue
        print(f"  {run} ({label})")
        for b in lst[0]["buckets"]:
            cov = f"{b['coverage']*100:6.2f}%" if b["coverage"] is not None else "   n/a"
            print(f"    {b['bucket']:>5s}  n={b['n_puzzles']:>6,}  cov={cov}  "
                  f"found={b['found_mean']:.3f}/{b['total_mean']:.3f}")


if __name__ == "__main__":
    main()
