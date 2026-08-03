"""coverage 를 퍼즐 가중 / pair 가중 두 축으로 재집계한다 (GPU 불필요).

배경: 사수님 데이터는 (입력, 해) 쌍 단위로 저장돼 있어 해가 k 개인 입력이 k 번 등장한다.
우리는 같은 입력을 1 행으로 저장해서 통계 축이 다르다.  coverage 격차가 정의가 아니라
집계 단위 차이인지 보려면 두 축을 나란히 놔야 한다.

    퍼즐 가중 (현행) = mean_x( found(x) / total(x) )
    pair 가중 (신규) = Σ_x found(x) / Σ_x total(x)

pair 가중은 total(x) 를 가중치로 쓴 가중평균과 같다:
    Σ_x total(x)·(found(x)/total(x)) / Σ_x total(x) = Σ found / Σ total

상한(모델이 완벽해도 못 넘는 값)도 같은 두 축으로:
    퍼즐 가중 상한 = mean_x( min(N,total)/total )
    pair 가중 상한 = Σ_x min(N,total) / Σ_x total

입력은 experiments/results/covcurve*.json 의 per_puzzle(totals, found_1/5/10/20).

    python experiments/coverage_pair_weight.py
"""

import glob
import json
import os
import sys

import numpy as np

PAPER = {8: (90.3, 1.9), 10: (57.5, 3.4)}
NS = [1, 5, 10, 20]


def load(path):
    d = json.load(open(path))
    pp = d["per_puzzle"]
    t = np.array(pp["totals"], dtype=np.int64)
    have = t > 0
    found = {N: np.array(pp[f"found_{N}"], dtype=np.int64)[have] for N in NS}
    return d["n"], t[have], found, int((~have).sum())


def axes(t, f):
    """(퍼즐 가중, pair 가중) 을 비율로."""
    return float((f / t).mean()), float(f.sum() / t.sum())


def bounds(t):
    return {N: (float((np.minimum(N, t) / t).mean()), float(np.minimum(N, t).sum() / t.sum()))
            for N in NS}


def dist_table(t, label):
    """입력을 해 개수만큼 복제하면 나오는 pair 축 분포."""
    tot_pairs = int(t.sum())
    print(f"\n### {label} — 입력 축 vs pair 축 분포")
    print(f"입력 {len(t)}개 -> pair {tot_pairs}행")
    print(f"\n| 해 개수 k | 입력 수 | 입력 축 비중 | pair 행 수 (=k×입력수) | pair 축 비중 |")
    print("|---|---|---|---|---|")
    ks = sorted(set(int(x) for x in t))
    shown = 0
    for k in ks:
        c = int((t == k).sum())
        if k > 20 and shown > 12:
            continue
        shown += 1
        print(f"| {k} | {c} | {c/len(t)*100:.2f}% | {c*k} | {c*k/tot_pairs*100:.2f}% |")
    for lo, hi, name in [(1, 1, "해 1개"), (2, 5, "해 2-5"), (6, 20, "해 6-20"), (21, 10**9, "해 21+")]:
        m = (t >= lo) & (t <= hi)
        c = int(m.sum()); p = int(t[m].sum())
        print(f"| **{name}** | **{c}** | **{c/len(t)*100:.2f}%** | **{p}** | **{p/tot_pairs*100:.2f}%** |")


def main():
    files = sorted(glob.glob("experiments/results/covcurve*.json"))
    by_n = {}
    for p in files:
        n, t, f, nz = load(p)
        by_n.setdefault(n, []).append((os.path.basename(p), t, f, nz))

    for n in sorted(by_n):
        t0 = by_n[n][0][1]
        for name, t, _, _ in by_n[n]:
            assert np.array_equal(t, t0), f"{name}: totals 가 다르다 -- 같은 test 셋이 아니다"
        b = bounds(t0)
        mu, sd = PAPER[n]

        print(f"\n\n{'='*100}")
        print(f"## {n}x{n}   입력 {len(t0)}  Σtotal(=pair 행 수) {int(t0.sum())}   논문 {mu}±{sd}")
        print(f"{'='*100}")
        print("\n### 상한 (모델 무관, 데이터만으로 결정)")
        print("| N | 퍼즐 가중 상한 | pair 가중 상한 |")
        print("|---|---|---|")
        for N in NS:
            print(f"| {N} | {b[N][0]*100:.2f}% | **{b[N][1]*100:.2f}%** |")

        print("\n### coverage 실측 — 두 축 병기")
        print("| run | N | 퍼즐 가중 | pair 가중 | pair/상한 | 논문 대비(pair) |")
        print("|---|---|---|---|---|---|")
        for name, t, f, nz in by_n[n]:
            short = name.replace("covcurve8_", "").replace("covcurve_", "").replace(".json", "")
            for N in NS:
                pw, pr = axes(t, f[N])
                print(f"| {short} | {N} | {pw*100:.2f}% | **{pr*100:.2f}%** | "
                      f"{pr/b[N][1]*100:.1f}% | {pr*100-mu:+.1f}%p |")

    # [2] 8x8 도 사수님 형식으로 역산
    for n in sorted(by_n):
        dist_table(by_n[n][0][1], f"{n}x{n} test")

    # 검산
    print("\n\n### 검산 (지시된 기대값과 대조)")
    if 10 in by_n:
        b10 = bounds(by_n[10][0][1])
        for N, exp in [(1, 0.339), (20, 0.955)]:
            got = b10[N][1]
            ok = abs(got - exp) < 0.001
            print(f"10x10 pair 상한 N={N:<2}: 기대 {exp:.3f}  실측 {got:.4f}  -> {'일치' if ok else '★불일치'}")


if __name__ == "__main__":
    sys.exit(main())
