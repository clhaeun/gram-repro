"""[50] per-puzzle 덤프 정합성 확인.  값만 찍는다.

① N=20 덤프의 found 가 기존 산출물과 일치하는가
     nq8 / nq8_s1 : experiments/results/fin_*_step67428_ema_s0.json 의 coverage_at_20
     nq10         : experiments/results/rp_nq10_repro_s0_step138534_ema_s0.json 의
                    coverage_at_20.per_row  +  results/figdata/nq10_perpuzzle.csv (unique 축)
② coverage = sum(found)/sum(n_solutions) 와 기존 coverage_at_20.coverage 대조
③ n_found_gt_total 집계
"""
import csv, json, os, sys

import numpy as np

FIG = "results/figdata"
REF = {
    "nq8": "experiments/results/fin_nq8_final_add_s0_step67428_ema_s0.json",
    "nq8_s1": "experiments/results/fin_nq8_final_add_s1_step67428_ema_s0.json",
    "nq10": "experiments/results/rp_nq10_repro_s0_step138534_ema_s0.json",
}
NS = [1, 5, 10, 20]


def load(tag, N):
    p = f"{FIG}/{tag}_perpuzzle_N{N}.csv"
    rows = list(csv.DictReader(open(p)))
    g = lambda k: np.array([int(r[k]) for r in rows])          # noqa: E731
    return dict(path=p, n_rows=len(rows), ihash=[r["ihash"] for r in rows],
                found=g("found"), total=g("n_solutions"), fv=g("first_valid"),
                nval=g("n_valid_samples"), nok=g("n_valid_consistent"))


def main():
    tags = sys.argv[1:] or ["nq8", "nq8_s1", "nq10"]
    for tag in tags:
        missing = [N for N in NS if not os.path.exists(f"{FIG}/{tag}_perpuzzle_N{N}.csv")]
        if missing:
            print(f"\n================ {tag} ================\n  미완료 N={missing}")
            continue
        print(f"\n================ {tag} ================")
        d = {N: load(tag, N) for N in NS}
        t = d[20]["total"]; have = t > 0
        for N in NS:
            f = d[N]["found"]
            print(f"  N={N:<2} rows={d[N]['n_rows']:<6} sum_found={f[have].sum():<7} "
                  f"sum_n_solutions={t[have].sum():<7} coverage={f[have].sum()/t[have].sum():.6f}  "
                  f"gt_total={int((f[have] > t[have]).sum())}  "
                  f"first_valid={d[N]['fv'].mean():.6f}  n_valid_mean={d[N]['nval'].mean():.4f}")
        # N 간 정합
        for a, b in zip(NS, NS[1:]):
            bad = int((d[a]["found"] > d[b]["found"]).sum())
            print(f"  monotone found_N{a} <= found_N{b}: 위반 {bad} 행")
        same_fv = all((d[N]["fv"] == d[1]["fv"]).all() for N in NS)
        print(f"  first_valid 가 N 전체에서 동일: {same_fv}   (rollout 이 표본 k 마다 재시드)")
        print(f"  found_N1 == n_valid_consistent_N1: {bool((d[1]['found'] == d[1]['nok']).all())}")

        ref = json.load(open(REF[tag])); c = ref["coverage_at_20"]
        f20 = d[20]["found"]
        print(f"  -- 기존 {REF[tag]}")
        print(f"     coverage(row macro) ref={c['coverage']:.10f}  csv={(f20[have]/t[have]).mean():.10f}  "
              f"delta={abs(c['coverage'] - (f20[have]/t[have]).mean()):.3e}")
        print(f"     micro(sum/sum)      ref={c['found_mean']/c['total_mean']:.10f}  "
              f"csv={f20[have].sum()/t[have].sum():.10f}")
        print(f"     found_mean ref={c['found_mean']:.10f} csv={f20[have].mean():.10f}   "
              f"total_mean ref={c['total_mean']:.10f} csv={t[have].mean():.10f}")
        print(f"     n_found_gt_total ref={c['n_found_gt_total']} csv={int((f20[have] > t[have]).sum())}")
        if "per_row" in c:
            pr = np.array(c["per_row"]["found"]); pt = np.array(c["per_row"]["totals"])
            print(f"     per_row found 완전일치={bool((pr == f20).all())}  "
                  f"불일치행={int((pr != f20).sum())}   totals 완전일치={bool((pt == t).all())}")

        old = f"{FIG}/{tag}_perpuzzle.csv"
        if os.path.exists(old):
            o = list(csv.DictReader(open(old)))
            first = {}
            for i, h in enumerate(d[20]["ihash"]):
                first.setdefault(h, i)
            idx = np.array([first[h] for h in dict.fromkeys(d[20]["ihash"])])
            idx = idx[t[idx] > 0]
            print(f"  -- 기존 {old}  rows={len(o)}  새 덤프 unique 대표행={len(idx)}")
            for N in NS:
                col = f"found_n{N}"
                if col not in o[0]:
                    continue
                ov = np.array([int(r[col]) for r in o])
                nv = d[N]["found"][idx]
                print(f"     {col}: 일치={bool((ov == nv).all())}  불일치={int((ov != nv).sum())}  "
                      f"sum old={ov.sum()} new={nv.sum()}")
            ovs = np.array([int(r["n_solutions"]) for r in o])
            print(f"     n_solutions 일치={bool((ovs == t[idx]).all())}")
            ov1 = np.array([int(r["valid_n1"]) for r in o])
            print(f"     valid_n1 vs first_valid 일치={bool((ov1 == d[1]['fv'][idx]).all())}  "
                  f"불일치={int((ov1 != d[1]['fv'][idx]).sum())}")


if __name__ == "__main__":
    sys.exit(main())
