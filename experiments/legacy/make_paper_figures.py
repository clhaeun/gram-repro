"""논문 Table 1 / Figure 10 / 12 / 15 를 논문 레이아웃 그대로 재현해 PNG 로 만든다.

원칙 (2026-07-30 지시):
  * wandb 커스텀 차트를 쓰지 않는다 -- 축 범위·범례 통제가 안 된다. matplotlib PNG 만 만들고
    reports/figures/ 에 커밋한다. wandb 에는 이미지로만 올린다.
  * 논문에 없는 요소는 넣지 않는다 (예: coverage 천장선 -> Figure 15 에서 제외).
  * 논문 축 상한을 우리 데이터가 넘으면 막대를 자르지 않고 y 를 늘리되, 논문 상한 위치에
    가로 점선 + "paper y-max" 라벨을 둔다.

    GRAM_PYLIBS=<matplotlib 경로> python experiments/make_paper_figures.py
    GRAM_PYLIBS=... python experiments/make_paper_figures.py --wandb   # 이미지 업로드까지

주의: 논문 PDF 가 이 박스에 없다. 캡션 문구는 지시받은 형식을 따랐고 **논문 원문과 글자
단위로 일치하는지는 확인하지 못했다**. 축 눈금/범위는 지시받은 값을 그대로 썼다.
"""

import argparse
import collections
import json
import os
import sys

OUT = "reports/figures"

# --- 지시받은 논문 축 ---
FIG10_XTICKS = {"nq8": [3, 6, 9, 12, 15, 18], "nq10": [0, 20, 40, 60, 80]}
FIG10_PAPER_YMAX = {"nq8": 3000, "nq10": 15000}
FIG12_XTICKS = [3, 6, 9, 12, 15, 18]
FIG12_PAPER_YMAX = 1500
FIG15_XLIM = {"nq8": (2, 18), "nq10": (0, 90)}

# Figure 15 / coverage 곡선의 checkpoint (캡션에 명시한다)
FIG15_CKPT = {
    "nq8": "experiments/results/covcurve8_nq8_v2_base_b500_s0.json",   # 정점 step_37587.live
    "nq10": "experiments/results/covcurve_nq10_on_b520_s1.json",       # 정점 step_37466.live
}

SIDECARS = {
    "nq8": ("data/nqueens-8/sidecar_8.json", "inputs", "solutions"),
    "nq10": ("data/nqueens-10/sidecar_10.json", "inputs", "solutions"),
    "gc8": ("data/graph-coloring-8/sidecar_gc_8.json", "graphs", "colorings"),
}

# 팔레트: validate_palette.js 통과 (light surface #fcfcfb).
# N 별 고정 색이고 순환시키지 않는다.
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
N_COLORS = {1: "#2a78d6", 5: "#eb6834", 10: "#1baf7a", 20: "#eda100"}
BAR = "#2a78d6"
HILITE = "#e8f0fb"          # Table 1 의 GRAM(ours) 행 배경


# ---------------------------------------------------------------- 데이터

def solution_counts(task, split=None):
    """sidecar 에서 (해 개수 -> 입력 수). split=None 이면 train+test 전체."""
    path, key, sub = SIDECARS[task]
    items = (json.load(open(path)) or {}).get(key, {})
    cnt = collections.Counter()
    for v in items.values():
        if split and v.get("split") != split:
            continue
        cnt[len(v.get(sub, []))] += 1
    return dict(cnt)


def coverage_by_total(path):
    """covcurve JSON -> (checkpoint, {해 개수: {N: coverage}}, {해 개수: 퍼즐 수})."""
    d = json.load(open(path))
    cov, npz = {}, {}
    for k, v in d["by_total"].items():
        k = int(k)
        cov[k] = {int(n): c for n, c in v["coverage"].items()}
        npz[k] = v["n_puzzles"]
    return d["checkpoint"], cov, npz


# ---------------------------------------------------------------- 스타일

def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color="#e4e3df", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5d4cf")
    ax.tick_params(colors=INK2, labelsize=9)


def _caption(fig, title, sub="(ours, reproduced)"):
    fig.suptitle(title, color=INK, fontsize=11, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.925, sub, color=INK2, fontsize=9, ha="left")


# ---------------------------------------------------------------- Figure 10

def figure10(plt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax, task, label in ((axes[0], "nq8", "N-Queens 8x8"),
                            (axes[1], "nq10", "N-Queens 10x10")):
        cnt = solution_counts(task)
        ks = sorted(cnt)
        ax.bar(ks, [cnt[k] for k in ks], width=0.9 if task == "nq8" else 1.0,
               color=BAR, zorder=3)
        _style(ax)
        ax.set_xticks(FIG10_XTICKS[task])
        ax.set_xlabel("number of valid solutions", color=INK2, fontsize=9.5)
        ax.set_ylabel("number of puzzles", color=INK2, fontsize=9.5)
        top = max(cnt.values())
        pm = FIG10_PAPER_YMAX[task]
        ax.set_ylim(0, max(top * 1.12, pm * 1.05))
        if top > pm:
            # 논문 y 상한을 넘는다 -> 막대를 자르지 않고 상한 위치를 표시한다
            ax.axhline(pm, color=INK2, lw=1.2, ls="--", zorder=4)
            ax.annotate(f"paper y-max ({pm:,})", (ax.get_xlim()[1], pm),
                        textcoords="offset points", xytext=(-6, 5), ha="right",
                        fontsize=8.5, color=INK2)
            ax.annotate(f"{top:,}", (ks[0], top), textcoords="offset points",
                        xytext=(3, 3), ha="left", fontsize=8.5, color=INK)
        n = sum(cnt.values())
        ax.set_title(f"{label}  (train+test, {n:,} puzzles)",
                     color=INK, fontsize=10, loc="left")
    _caption(fig, "Figure 10: Distribution of the number of valid solutions per puzzle")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


# ---------------------------------------------------------------- Figure 12

def figure12(plt):
    fig, ax = plt.subplots(figsize=(7.4, 4.5), facecolor=SURFACE)
    cnt = solution_counts("gc8")
    ks = sorted(cnt)
    ax.bar(ks, [cnt[k] for k in ks], width=0.9, color=BAR, zorder=3)
    _style(ax)
    ax.set_xticks(FIG12_XTICKS)
    ax.set_xlim(0, max(FIG12_XTICKS) + 1.5)
    ax.set_ylim(0, FIG12_PAPER_YMAX)
    ax.set_xlabel("number of valid colorings", color=INK2, fontsize=9.5)
    ax.set_ylabel("number of graphs", color=INK2, fontsize=9.5)
    shown = sum(v for k, v in cnt.items() if k <= max(FIG12_XTICKS) + 1)
    n = sum(cnt.values())
    ax.set_title(f"Graph Coloring 8-vertex  (train+test, {n:,} graphs)",
                 color=INK, fontsize=10, loc="left")
    # x 범위가 논문 눈금(<=19)이라 꼬리가 잘린다 -- 잘린 양을 명시한다
    ax.annotate(f"{n - shown} graphs with >19 colorings are outside the x-range "
                f"(max {max(cnt)})", (0.98, 0.94), xycoords="axes fraction",
                ha="right", fontsize=8.5, color=INK2)
    _caption(fig, "Figure 12: Distribution of the number of valid colorings per graph "
                  "(8-vertex)")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


# ---------------------------------------------------------------- Figure 15

def figure15(plt):
    # 패널 순서는 논문과 같이 좌=8x8, 우=10x10 로 고정한다 (알파벳 정렬이면 뒤바뀐다)
    order = ["nq8", "nq10"]
    avail = [(t, FIG15_CKPT[t]) for t in order
             if t in FIG15_CKPT and os.path.exists(FIG15_CKPT[t])]
    fig, axes = plt.subplots(1, len(avail), figsize=(5.8 * len(avail), 4.9),
                             facecolor=SURFACE, squeeze=False)
    caps = []
    for ax, (task, path) in zip(axes[0], avail):
        ck, cov, npz = coverage_by_total(path)
        lo, hi = FIG15_XLIM[task]
        ks = [k for k in sorted(cov) if lo <= k <= hi]
        for N in (1, 5, 10, 20):
            ax.plot(ks, [cov[k].get(N) for k in ks], "-o", color=N_COLORS[N],
                    lw=2, ms=4.5, label=f"GRAM (N={N})", zorder=3)
        _style(ax)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("number of valid solutions", color=INK2, fontsize=9.5)
        ax.set_ylabel("Coverage", color=INK2, fontsize=9.5)
        label = "N-Queens 8x8" if task == "nq8" else "N-Queens 10x10"
        ax.set_title(f"{label}  ({ck.split('/')[-2]} / {ck.split('/')[-1]})",
                     color=INK, fontsize=9.5, loc="left")
        caps.append(f"{label}: {ck}")
    # 범례는 곡선을 가리지 않게 그림 하단으로 뺀다
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, labelcolor=INK2, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))
    fig.text(0.985, 0.035, "(HRM/TRM: not reproduced)", color=INK2, fontsize=8.5,
             ha="right")
    _caption(fig, "Figure 15: Coverage as a function of the number of valid solutions")
    fig.text(0.012, 0.895, " | ".join(caps), color=INK2, fontsize=7.5, ha="left")
    fig.tight_layout(rect=[0, 0.075, 1, 0.885])
    return fig


# ---------------------------------------------------------------- Table 1

TABLE1_ROWS = [
    # (라벨, nq8 acc, nq8 cov, nq10 acc, nq10 cov, gc8 conf, gc8 cov, gc10 conf, gc10 cov)
    ("Direct Pred (8L)",) + ("—",) * 8,
    ("Direct Pred (32L)",) + ("—",) * 8,
    ("Looped TF",) + ("—",) * 8,
    ("HRM",) + ("—",) * 8,
    ("TRM",) + ("—",) * 8,
    ("AR",) + ("—",) * 8,
    ("MDLM",) + ("—",) * 8,
    ("GRAM (paper)", "99.7 ± 0.3", "0.903 ± 0.019", "89.7 ± 2.7", "0.575 ± 0.034",
     "2.7 ± 2.1", "0.858 ± 0.005", "—", "—"),
    ("GRAM (ours)", "99.77 †", "0.9964", "85.6 ± 5.9", "0.9592 ‡",
     "0.925 ± 0.502 §", "0.842 ± 0.026", "—", "—"),
]
TABLE1_NOTES = [
    "† nq8 Accuracy: `nq8_v2_base_b500_s0` 정점(step_37587.live) 단일 seed 값이다. "
    "**단일 seed, 정점 선택**이라 논문의 seed 간 mean±std 와 축이 다르다. 같은 run 의 "
    "최종 step 은 97.53 이고, 격차 2.24%p 가 정점 선택 효과다. "
    "v5_add_b0 3-seed 최종 step harness 가 나오면 그 값으로 교체한다.",
    "‡ Coverage 는 전부 **N=20**(cov@20) 이다. 논문 coverage 의 N 은 확인되지 않았고 "
    "**N=1 가설은 기각됐다**(8×8 의 N=1 천장 0.7909 < 논문 하단 0.884). "
    "N 을 낮추면 두 task 모두 논문 아래다 -- REPORT_FINAL §1.4.",
    "§ gc8 Conflict 는 20샘플 **합** 축을 가정한 값이다(퍼즐당 평균 0.046 × 20). "
    "**논문 집계 축이 미확정**이라 곱이 맞는지 확인되지 않았다. 퍼즐당 평균 기준이면 "
    "0.046 ± 0.025 다.",
    "nq10 Accuracy 85.6 ± 5.9 는 `nq10_on_b520_s{0,1,2}` 의 92.28 / 83.61 / 81.03 "
    "mean ± std 다(seed 3개). `nq10_on_b450_s0`(66.68)은 β 가 달라 제외했다.",
    "GC 10-vertex 는 미실시다.",
    "**baseline 7행과 GC 10-vertex 논문값은 이 박스에 없다** -- 논문 PDF 가 로컬에 없고 "
    "리포트에도 기록이 없다(GRAM 행만 있다). 값을 받으면 채운다.",
]
TABLE1_HEADER = [
    ["", "N-Queens 8x8", "", "N-Queens 10x10", "", "GC 8-vertex", "", "GC 10-vertex", ""],
    ["Model", "Accuracy", "Coverage", "Accuracy", "Coverage",
     "Conflict ↓", "Coverage", "Conflict ↓", "Coverage"],
]


def table1_markdown():
    out = ["# Table 1 — 논문 대비 (레이아웃 동일)", "",
           "| Model | 8×8 Accuracy | 8×8 Coverage | 10×10 Accuracy | 10×10 Coverage "
           "| GC-8 Conflict ↓ | GC-8 Coverage | GC-10 Conflict ↓ | GC-10 Coverage |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in TABLE1_ROWS:
        cells = list(r)
        if cells[0] == "GRAM (ours)":
            cells = [f"**{c}**" for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    out += ["", "## 각주", ""] + [f"- {n}" for n in TABLE1_NOTES]
    return "\n".join(out) + "\n"


def table1_png(plt):
    nrow, ncol = len(TABLE1_ROWS) + 2, 9
    fig, ax = plt.subplots(figsize=(13.5, 0.36 * nrow + 1.0), facecolor=SURFACE)
    ax.axis("off")
    cells = TABLE1_HEADER + [list(r) for r in TABLE1_ROWS]
    # bbox 로 축을 꽉 채운다 -- loc="upper center" 만 쓰면 아래에 빈 공간이 남는다
    t = ax.table(cellText=cells, cellLoc="center", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#d5d4cf")
        c.set_linewidth(0.6)
        if i < 2:                                  # 헤더 2줄
            c.set_text_props(color=INK, weight="bold")
            c.set_facecolor("#f2f1ed")
        else:
            label = cells[i][0]
            if label == "GRAM (ours)":             # 우리 행만 강조
                c.set_facecolor(HILITE)
                c.set_text_props(color=INK, weight="bold")
            else:
                c.set_text_props(color=INK if label.startswith("GRAM") else INK2)
        if j == 0:
            c.set_text_props(ha="left")
    ax.set_title("Table 1: Accuracy / Coverage / Conflict on N-Queens and Graph Coloring",
                 color=INK, fontsize=11, loc="left", pad=16)
    fig.text(0.012, 0.028,
             "(ours, reproduced)  |  Only the GRAM (ours) row is our measurement. "
             "Baseline rows and GC 10-vertex paper values are unavailable (see notes). "
             "All Coverage values are at N=20.",
             color=INK2, fontsize=8.5, ha="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pylibs", default=os.environ.get("GRAM_PYLIBS"))
    ap.add_argument("--wandb", action="store_true", help="PNG 를 wandb 에 이미지로 올린다")
    ap.add_argument("--project", default="GRAM-repro-summary")
    args = ap.parse_args()

    if args.pylibs and os.path.isdir(args.pylibs) and args.pylibs not in sys.path:
        sys.path.insert(0, args.pylibs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    figs = {"fig10_solution_distribution": figure10(plt),
            "fig12_gc_coloring_distribution": figure12(plt),
            "fig15_coverage_by_solutions": figure15(plt),
            "table1_paper_vs_ours": table1_png(plt)}
    made = []
    for name, fig in figs.items():
        p = os.path.join(OUT, f"{name}.png")
        fig.savefig(p, dpi=170, facecolor=SURFACE)
        made.append(p)
        print(f"  {p}")
    md = os.path.join(OUT, "table1_paper_vs_ours.md")
    open(md, "w").write(table1_markdown())
    made.append(md)
    print(f"  {md}")

    if args.wandb:
        os.environ.setdefault("WANDB_SILENT", "true")
        import wandb
        run = wandb.init(project=args.project, name="paper-figures",
                         job_type="figures", reinit=True)
        run.log({f"figures/{n}": wandb.Image(os.path.join(OUT, f"{n}.png"))
                 for n in figs})
        run.finish()
        print(f"  wandb 업로드 완료 -> {args.project}/paper-figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
