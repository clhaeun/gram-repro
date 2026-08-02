"""pair final 라운드 집계 (fin_*.json -> 논문 대조 표).

축 규약 (사수 지시):
  - 헤드라인 = 최종 step x EMA (선택 효과 없음, 논문 축)
  - run 별 값 = harness seed 0/1 평균, mean±std 는 seed(run) 3개에 대해
  - coverage 는 두 축: 퍼즐 가중 = mean_x(found/total) (harness 의 coverage 필드),
    pair 가중 = Σfound/Σtotal = found_mean/total_mean (동일 퍼즐셋이라 정확)
  - 논문 병기: nq8 99.7±0.3 / 90.3±1.9, nq10 89.7±2.7 / 57.5±3.4

사용: python experiments/aggregate_final.py
"""
import glob, json, re, os
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "results")
PAPER = {("nq8", "acc"): (99.7, 0.3), ("nq8", "cov"): (90.3, 1.9),
         ("nq10", "acc"): (89.7, 2.7), ("nq10", "cov"): (57.5, 3.4)}
FINAL_STEP = {"nq8": 67428, "nq10": 58772}


def collect(task, arm, weight):
    """run(seed) -> dict(metric -> hs평균값). 최종 step 만."""
    runs = {}
    pat = f"{RES}/fin_{task}_final_{arm}_s*_step{FINAL_STEP[task]}_{weight}_s*.json"
    for f in sorted(glob.glob(pat)):
        m = re.search(rf"final_{arm}_s(\d)_step\d+_{weight}_s(\d)", f)
        seed, hs = int(m.group(1)), int(m.group(2))
        d = json.load(open(f))
        c = d["coverage_at_20"]
        runs.setdefault(seed, []).append(dict(
            strict=d["prior_all_samples"]["strict"],
            pass20=d["pass_at"]["20"],
            maj=d["majority_at_N_valid"],
            verif=d["verifier_selected_at_N_valid"],
            cov_puzzle=c["coverage"],
            cov_pair=c["found_mean"] / c["total_mean"],
            n=c["n_puzzles_scored"],
        ))
    out = {}
    for seed, lst in runs.items():
        out[seed] = {k: float(np.mean([r[k] for r in lst])) for k in lst[0]}
        out[seed]["hs_spread_strict"] = float(max(r["strict"] for r in lst) - min(r["strict"] for r in lst)) if len(lst) > 1 else 0.0
        out[seed]["n_hs"] = len(lst)
    return out


def fmt(vals, pct=True):
    a = np.array(vals) * (100 if pct else 1)
    if len(a) == 1:
        return f"{a[0]:.1f} (1-seed)"
    return f"{a.mean():.1f}±{a.std(ddof=1):.1f}"


def main():
    for task in ["nq8", "nq10"]:
        print(f"\n===== {task} (최종 step {FINAL_STEP[task]}) =====")
        print(f"논문: accuracy {PAPER[(task,'acc')][0]}±{PAPER[(task,'acc')][1]}, "
              f"coverage {PAPER[(task,'cov')][0]}±{PAPER[(task,'cov')][1]}")
        for arm in ["add", "catnd"]:
            for weight in ["ema", "live"]:
                runs = collect(task, arm, weight)
                if not runs:
                    print(f"  {arm:6s} {weight:4s}: 없음")
                    continue
                seeds = sorted(runs)
                tag = "★헤드라인" if weight == "ema" else "  (병기)  "
                print(f"  {arm:6s} {weight:4s} {tag} seeds={seeds} "
                      f"(hs {runs[seeds[0]]['n_hs']}개 평균, n={runs[seeds[0]]['n']:,})")
                for k, label in [("strict", "단일시도 strict"), ("cov_puzzle", "coverage(퍼즐)"),
                                 ("cov_pair", "coverage(pair)"), ("pass20", "pass@20"),
                                 ("maj", "majority"), ("verif", "verifier")]:
                    vals = [runs[s][k] for s in seeds]
                    per = "  ".join(f"s{s}={runs[s][k]*100:.2f}" for s in seeds)
                    print(f"      {label:16s} {fmt(vals):>12s}   [{per}]")
                sp = max(runs[s]["hs_spread_strict"] for s in seeds)
                print(f"      (harness seed 간 strict 최대 편차 {sp*100:.3f}pp)")


if __name__ == "__main__":
    main()
