"""[50] ⑥ depth utilization 곡선 데이터.  reports/sudoku50 의 merged JSON -> figdata CSV.

effective layers 는 FPRM 4.4.2 의 축(Sudoku 640 vs 1040)과 맞추기 위해
supervision step 하나가 소비하는 transformer layer 수로 환산한다:

    per_step = H_cycles * (L_cycles + 1) * L_layers
    effective_layers = per_step * halt_max_steps

sudoku_paper_34 는 H_cycles=3, L_cycles=6, L_layers=2 이므로 per_step = 42
(hms 16 -> 672, 128 -> 5376, 320 -> 13440).

    python experiments/depth_util_summary.py
"""
import csv
import glob
import json
import os
import re
import sys

import yaml

CK = "checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34"
OUT = "results/figdata/depth_util_sudoku.csv"
COLS = ["hms", "effective_layers", "seed", "n", "prior_exact", "pass_at_N_exact",
        "majority_exact", "vsel_exact", "prior_valid", "prior_violations_mean", "source"]


def per_step_layers(arch):
    return int(arch["H_cycles"]) * (int(arch["L_cycles"]) + 1) * int(arch["L_layers"])


def rows_from(pattern, source):
    out = []
    for p in sorted(glob.glob(pattern)):
        d = json.load(open(p))
        m = re.search(r"_h(\d+)_", os.path.basename(p))
        out.append({
            "hms": d["halt_max_steps"], "seed": d["seed"] if "seed" in d else int(m.group(1)),
            "n": d["n"],
            "prior_exact": d["prior_all_samples"]["exact"],
            "pass_at_N_exact": d["pass_at_N"]["exact"],
            "majority_exact": d["majority_at_N"]["exact"],
            "vsel_exact": d["verifier_selected_at_N"]["exact"],
            "prior_valid": d["prior_all_samples"]["valid"],
            "prior_violations_mean": d["prior_all_samples"]["violations_mean"],
            "source": source, "_path": p,
        })
    return out


def main():
    arch = yaml.safe_load(open(os.path.join(CK, "all_config.yaml")))["arch"]
    ps = per_step_layers(arch)

    rows = (rows_from("reports/sudoku50/n20000_hms*_N20_h*_merged.json", "50 (4샤드 기준선)")
            + rows_from("reports/sudoku50/n20000_hms*_N20_h*_merged8.json", "50 (8샤드)"))
    rows.sort(key=lambda r: (r["hms"], r["seed"]))
    for r in rows:
        r["effective_layers"] = ps * int(r["hms"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    meta = {
        "ckpt_dir": CK, "step": "step_65100",
        "weight": "EMA (bare step file; step_65100.live 가 live)",
        "step_md5": "b050e90165c998abf79f55b3e27193c6 (sudoku_paper_34_eval50 사본과 동일)",
        "data": "data/sudoku-extreme-1k-aug-1000/test", "limit": 20000,
        "N": 20, "sigma_scale": 1.0, "mu_scale": 1.0,
        "per_step_layers": ps,
        "per_step_formula": "H_cycles(%d) * (L_cycles(%d)+1) * L_layers(%d)" % (
            arch["H_cycles"], arch["L_cycles"], arch["L_layers"]),
        "effective_layers": "per_step_layers * halt_max_steps",
        "fprm_reference": "FPRM 4.4.2 Sudoku: post-norm 640 / pre-norm 1040 effective layers",
        "hms_present": sorted({int(r["hms"]) for r in rows}),
        "hms_missing": "8, 32, 64 는 62번 격자에 있다 (같은 ckpt/데이터/N/limit 조건). "
                       "곡선을 합칠 때 그쪽 merged JSON 을 같은 스키마로 붙일 것.",
        "sources": {r["_path"]: r["source"] for r in rows},
    }
    with open(OUT.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[out] {OUT}  ({len(rows)} rows, per_step_layers={ps})")
    for r in rows:
        print(f"  hms={r['hms']:>4} eff={r['effective_layers']:>6} h{r['seed']} n={r['n']:>5} "
              f"prior={r['prior_exact']:.4f} pass@20={r['pass_at_N_exact']:.4f} "
              f"maj={r['majority_exact']:.4f} vsel={r['vsel_exact']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
