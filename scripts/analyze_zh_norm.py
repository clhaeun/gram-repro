#!/usr/bin/env python3
"""parse_wandb_offline.py 가 뽑은 CSV 로 z_H/z_L norm 구간 통계와 EMA 대조표를 만든다.

용법:
    python3 scripts/analyze_zh_norm.py reports/data/cellP_series.csv --label P \
        --bin 20000 --fine 80000:140000:5000
"""
import argparse
import csv
import collections
import numpy as np


def load(path):
    d = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            d[r["key"]].append((int(r["step"]), float(r["value"])))
    return {k: (np.array([s for s, _ in v]), np.array([x for _, x in v])) for k, v in d.items()}


def stats(vals):
    if len(vals) == 0:
        return None
    finite = vals[np.isfinite(vals)]
    n_bad = len(vals) - len(finite)
    if len(finite) == 0:
        return dict(n=len(vals), n_bad=n_bad, min=float("nan"), max=float("nan"),
                    mean=float("nan"), p99=float("nan"))
    return dict(n=len(vals), n_bad=n_bad, min=finite.min(), max=finite.max(),
                mean=finite.mean(), p99=np.percentile(finite, 99))


def bins_from(lo, hi, width):
    return [(a, min(a + width, hi)) for a in range(lo, hi, width)]


def table(data, key, edges, title):
    if key not in data:
        print(f"\n### {title}\n\n확인 못 함 — CSV 에 `{key}` 없음.")
        return
    steps, vals = data[key]
    print(f"\n### {title}\n")
    print("| step 구간 | n | min | mean | p99 | max | 비유한 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for a, b in edges:
        m = (steps >= a) & (steps < b)
        st = stats(vals[m])
        if st is None or st["n"] == 0:
            print(f"| {a//1000}k–{b//1000}k | 0 | — | — | — | — | — |")
            continue
        print(f"| {a//1000}k–{b//1000}k | {st['n']} | {st['min']:.3f} | {st['mean']:.3f} "
              f"| {st['p99']:.3f} | {st['max']:.3f} | {st['n_bad']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--label", default="?")
    ap.add_argument("--bin", type=int, default=20000)
    ap.add_argument("--fine", default=None, help="lo:hi:width 형식, 예 80000:140000:5000")
    args = ap.parse_args()

    data = load(args.csv)
    steps_all = data["train/diag/z_h_norm"][0]
    last = int(steps_all.max())
    print(f"# 셀 {args.label} — z_H / z_L norm 구간 통계\n")
    print(f"원본: `{args.csv}`  ·  step 1..{last}  ·  기록 {len(steps_all)} 건")

    edges = bins_from(0, ((last // args.bin) + 1) * args.bin, args.bin)
    table(data, "train/diag/z_h_norm", edges, f"z_H norm — {args.bin//1000}k 구간")
    table(data, "train/diag/z_l_norm", edges, f"z_L norm — {args.bin//1000}k 구간")

    if args.fine:
        lo, hi, w = (int(x) for x in args.fine.split(":"))
        fe = bins_from(lo, hi, w)
        table(data, "train/diag/z_h_norm", fe, f"z_H norm — 세분 {lo//1000}k–{hi//1000}k ({w//1000}k)")
        table(data, "train/diag/z_l_norm", fe, f"z_L norm — 세분 {lo//1000}k–{hi//1000}k ({w//1000}k)")

    # ---- EMA / live 대조표
    print("\n### EMA·live accuracy 와 z_H 대조\n")
    ema = data.get("all.accuracy")
    raw = data.get("all/accuracy_raw")
    if ema is None:
        print("확인 못 함 — `all.accuracy` 없음.")
        return
    zs, zv = data["train/diag/z_h_norm"]
    ls, lv = data.get("train/diag/z_l_norm", (None, None))
    print("| eval step | EMA acc | live acc | z_H@step | z_H mean(±500) | z_H max(±500) | z_L@step |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    raw_map = dict(zip(raw[0], raw[1])) if raw is not None else {}
    for s, a in zip(*ema):
        s = int(s)
        i = int(np.argmin(np.abs(zs - s)))
        w = (zs >= s - 500) & (zs <= s + 500)
        seg = zv[w][np.isfinite(zv[w])]
        zl = "—"
        if ls is not None:
            j = int(np.argmin(np.abs(ls - s)))
            zl = f"{lv[j]:.3f}"
        print(f"| {s} | {a:.4f} | {raw_map.get(s, float('nan')):.4f} | {zv[i]:.3f} "
              f"| {seg.mean() if len(seg) else float('nan'):.3f} "
              f"| {seg.max() if len(seg) else float('nan'):.3f} | {zl} |")


if __name__ == "__main__":
    main()
