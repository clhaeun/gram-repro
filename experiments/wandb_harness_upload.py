"""harness/scan 결과(experiments/results/*.json)를 wandb 로 올린다.

sweep_harness.py 는 오프라인 스크립트라 결과가 JSON 으로만 남는다. 학습 곡선은
wandb 에 있는데 판정 근거인 harness 수치는 없어서 둘을 한 화면에서 못 본다.
이 스크립트가 그 간극을 메운다. 세 가지 모드가 있고 서로 독립이다.

  summary  각 학습 run 의 wandb summary 에 `harness/*` 키를 써넣는다.
           -> runs 테이블에 컬럼으로 떠서 β 별 정렬/필터/scatter 가 된다.
           step 과 무관하므로 순서 문제가 없다. **기존 키를 덮지 않는다.**

  table    306개 전 결과를 wandb.Table 하나로 올린다(+ Artifact).
           -> UI 에서 정렬/필터되는 전체 비교표.

  curves   학습 run 하나당 harness run 하나를 만들어 `.live` 스캔 곡선을 그린다.
           group=<학습 run_name>, job_type=harness 라 UI 에서 나란히 붙는다.
           x 축은 커스텀 metric `harness/step` 이다 -- 끝난 run 에 resume 해서
           과거 step 에 log 하면 wandb 가 _step 단조성 때문에 버리기 때문이다.

안전장치:
  * 기본이 --dry-run 이다. 실제 업로드는 --commit 을 줘야 한다.
  * state=="running" 인 wandb run 은 **건드리지 않는다**(학습 중 오염 방지).
  * summary 는 `harness/` 네임스페이스만 쓴다.

사용법:
    python experiments/wandb_harness_upload.py --mode summary            # dry-run
    python experiments/wandb_harness_upload.py --mode summary --commit
    python experiments/wandb_harness_upload.py --mode all --commit
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT_RE = re.compile(r"checkpoints/([^/]+)/([^/]+)/step_(\d+)(\.live)?$")
RESULTS = "experiments/results/*.json"


# ---------------------------------------------------------------------------
# 곡선 run 의 이름/id 규칙 -- sweep_harness.py --wandb 와 공유한다.
# id 를 결정적으로 만들어야 측정 시점 업로드와 사후 backfill 이 같은 run 에 모인다.
# ---------------------------------------------------------------------------

def harness_run_name(train_run, weights, seed):
    base = f"{train_run}__{weights}_scan"
    return base if seed in (0, None) else f"{base}_seed{seed}"


def harness_run_id(project, train_run, weights, seed):
    import hashlib
    key = f"{project}/{train_run}/{weights}/{seed}"
    return "hs" + hashlib.sha1(key.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# 결과 JSON 색인
# ---------------------------------------------------------------------------

def load_results(pattern=RESULTS):
    """sweep_harness 스키마인 것만 골라 평평한 레코드로."""
    out, skipped = [], 0
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            skipped += 1
            continue
        m = CKPT_RE.match(str(d.get("checkpoint", "")))
        if not m or "prior_all_samples" not in d:
            skipped += 1          # diag_*/contradiction_* 등 다른 스키마
            continue
        project, run, step, live = m.group(1), m.group(2), int(m.group(3)), bool(m.group(4))
        p, q = d["prior_all_samples"], d["posterior"]
        out.append(dict(
            file=f, family=os.path.basename(f).split("_")[0],
            project=project, run=run, step=step,
            weights="live" if live else "step_N",
            seed=d.get("seed"), N=d.get("N"), sigma_scale=d.get("sigma_scale"),
            prior_valid=p["valid"], prior_consistent=p["consistent"], prior_strict=p["strict"],
            queens_mean=p["queen_count_mean"], queens_min=p["queen_count_min"],
            queens_max=p["queen_count_max"],
            posterior_valid=q["valid"], posterior_strict=q["strict"],
            pass_at_1=d["pass_at"].get("1"), pass_at_20=d["pass_at"].get("20"),
            majority_at_N=d["majority_at_N_valid"],
            verifier_sel_at_N=d["verifier_selected_at_N_valid"],
            distinct_at_N=d["distinct_at_N_mean"],
            puzzles_ge1_valid=d.get("puzzles_with_ge1_valid"),
        ))
    return out, skipped


def run_meta(project, run):
    """학습 run 의 all_config.yaml 에서 β/ema/one-hot 을 읽는다. 없으면 빈 dict."""
    p = f"checkpoints/{project}/{run}/all_config.yaml"
    if not os.path.exists(p):
        return {}
    try:
        c = yaml.safe_load(open(p))
    except Exception:
        return {}
    a = c.get("arch", {}) or {}
    return dict(beta=a.get("kl_coef"), y_onehot=a.get("y_onehot"),
                ema=c.get("ema"), ema_rate=c.get("ema_rate"),
                epochs=c.get("epochs"), train_seed=c.get("seed"))


def weight_label(rec, meta):
    """step_N 이 EMA 사본인지 raw 인지. ema=False run 은 step_N 이 곧 raw 다."""
    if rec["weights"] == "live":
        return "live"
    return "ema" if meta.get("ema") else "raw"


# ---------------------------------------------------------------------------
# wandb run 매핑
# ---------------------------------------------------------------------------

def wandb_index(api, projects):
    """config.run_name -> wandb run. 동명이 여럿이면 finished 우선, 그다음 step 최대."""
    idx = defaultdict(list)
    for proj in projects:
        try:
            for r in api.runs(f"{api.default_entity}/{proj}", per_page=500):
                name = (r.config or {}).get("run_name")
                if name:
                    idx[(proj, name)].append(r)
        except Exception as e:
            print(f"  [warn] {proj} 조회 실패: {type(e).__name__}: {e}")
    picked, ambiguous = {}, []
    for k, rs in idx.items():
        if len(rs) > 1:
            ambiguous.append((k, [(r.id, r.state) for r in rs]))
        rs.sort(key=lambda r: (r.state == "finished", (r.summary or {}).get("_step") or 0))
        picked[k] = rs[-1]
    return picked, ambiguous


# ---------------------------------------------------------------------------
# mode: summary
# ---------------------------------------------------------------------------

def build_summary(recs, meta):
    """한 학습 run 의 harness 결과 -> summary 에 쓸 dict.

    peak 은 `.live` 스캔(seed 0)에서 고른다 -- REPORT_audit_measure.md 4-2 규약.
    포인트가 3개 미만이면 곡선으로 보지 않고 peak 을 만들지 않는다.
    """
    out = {"harness/n_results": len(recs)}
    live0 = sorted([r for r in recs if r["weights"] == "live" and r["seed"] == 0],
                   key=lambda r: r["step"])
    if len(live0) >= 3:
        best = max(live0, key=lambda r: r["prior_valid"])
        last = live0[-1]
        out.update({
            "harness/live_peak_valid": best["prior_valid"],
            "harness/live_peak_step": best["step"],
            "harness/live_peak_is_boundary": best["step"] == last["step"],
            "harness/live_final_valid": last["prior_valid"],
            "harness/live_scan_points": len(live0),
            "harness/live_peak_pass_at_20": best["pass_at_20"],
            "harness/live_peak_posterior_valid": best["posterior_valid"],
            "harness/live_peak_majority_at_20": best["majority_at_N"],
            "harness/live_peak_distinct_at_20": best["distinct_at_N"],
            "harness/live_peak_queens_mean": best["queens_mean"],
        })
        # 같은 step 의 EMA/raw 대조가 있으면 함께
        same = [r for r in recs if r["weights"] == "step_N" and r["step"] == best["step"]]
        if same:
            lbl = weight_label(same[0], meta)
            out[f"harness/{lbl}_at_live_peak_valid"] = sum(r["prior_valid"] for r in same) / len(same)
    else:
        best = max(recs, key=lambda r: r["prior_valid"])
        out.update({"harness/best_valid": best["prior_valid"],
                    "harness/best_step": best["step"],
                    "harness/best_weights": weight_label(best, meta)})
    for k, v in meta.items():
        if v is not None:
            out[f"harness/cfg_{k}"] = v
    return out


# ---------------------------------------------------------------------------
# mode: unified  --  세 task(nq8/nq10/gc8) x 두 서버(29/50) 를 한 wandb.Table 로
#
# 기존 --mode summary/table 과 목적이 다르다:
#   summary : 학습 run 각각의 wandb summary 에 harness 요약을 써넣는다
#   table   : 한 project 안에 원시 결과 인덱스를 올린다
#   unified : 세 task 를 한 표에 담아 별도 통합 project 로 올린다 (+ 패널)
# 두 서버가 같은 스크립트로 자기 JSON 을 같은 표에 올리는 것이 목적이므로
# server 열을 로컬 자산으로 판정한다.
# ---------------------------------------------------------------------------

UNIFIED_PROJECT = "GRAM-repro-summary"

TASK_OF = {"Nqueens-8-ACT-torch": "nq8", "Nqueens-10-ACT-torch": "nq10",
           "Graph-coloring-8-ACT-torch": "gc8"}
# L = seq_len + puzzle_emb_len.  beta_sum = beta_mean / L
L_OF = {"nq8": 80, "nq10": 116, "gc8": 44}

# run_configs/hinton/README.md 가 명시한 검증용 run (1~102 step). 집계에서 뺀다.
PROBE_RUNS = {"nd_off_cat_m4", "nd_off_cat_m4_b", "nd_yproj_on", "nd_both_on"}


def is_probe(run):
    return run in PROBE_RUNS or run.startswith("p100_") or run.startswith("probe_")


def cfg_and_server(project, run):
    """(config dict, server) -- 로컬 checkpoints 에 있으면 50, hinton 사본에만 있으면 29."""
    local = f"checkpoints/{project}/{run}/all_config.yaml"
    hinton = f"run_configs/hinton/{project}/{run}.yaml"
    path, server = (local, "50") if os.path.exists(local) else (hinton, "29")
    if not os.path.exists(path):
        return {}, "?"
    try:
        c = yaml.safe_load(open(path)) or {}
    except Exception:
        return {}, server
    a = c.get("arch", {}) or {}
    return dict(beta=a.get("kl_coef"), reduction=a.get("kl_reduction"),
                y_inject=a.get("y_inject"), logvar_bias=a.get("logvar_bias_init"),
                nodecay_y_proj=c.get("nodecay_y_proj"), y_onehot=a.get("y_onehot"),
                seed=c.get("seed"), epochs=c.get("epochs"), ema=c.get("ema")), server


def beta_pair(cfg, task):
    """(beta_sum, beta_mean). reduction 키가 없는 구코드 run 은 눈금 불명 -> None."""
    b, red, Lv = cfg.get("beta"), cfg.get("reduction"), L_OF.get(task)
    if b is None or Lv is None or red not in ("sum", "mean"):
        return (None, None)
    return (b, b * Lv) if red == "sum" else (b / Lv, b)


UNIFIED_COLS = ["server", "task", "run", "step", "step_kind", "weights", "metric_type",
                "y_inject", "logvar_bias", "nodecay_y_proj", "beta_sum", "beta_mean",
                "seed", "epochs", "harness_seed", "single_valid", "coverage_at20",
                "pass_at20", "majority_at20", "majority_canon_at20", "verifier_at20",
                "distinct_at20", "conflict_mean", "conflict_std",
                "single_valid_std", "coverage_at20_std", "source"]

# 논문값. ± 를 담으려면 *_std 열이 필요해 지시 스펙에 2열을 더했다.
PAPER_ROWS = [
    dict(task="nq8", single_valid=0.997, single_valid_std=0.003,
         coverage_at20=0.903, coverage_at20_std=0.019),
    dict(task="nq10", single_valid=0.897, single_valid_std=0.027,
         coverage_at20=0.575, coverage_at20_std=0.034),
    dict(task="gc8", conflict_mean=2.7, conflict_std=2.1,
         coverage_at20=0.858, coverage_at20_std=0.005),
]


def load_unified(pattern=RESULTS):
    """결과 JSON -> unified 행. harness 행만 만든다(in-loop 는 add_inloop_rows)."""
    raw, skipped = [], 0
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            skipped += 1
            continue
        m = CKPT_RE.match(str(d.get("checkpoint", "")))
        if not m or "prior_all_samples" not in d:
            skipped += 1
            continue
        project, run, step, live = m.group(1), m.group(2), int(m.group(3)), bool(m.group(4))
        task = TASK_OF.get(project)
        if task is None or is_probe(run):
            skipped += 1
            continue
        cfg, server = cfg_and_server(project, run)
        p = d["prior_all_samples"]
        cov = d.get("coverage_at_20") or {}
        bs, bm = beta_pair(cfg, task)
        raw.append(dict(
            server=server, task=task, run=run, step=step,
            weights="live" if live else ("EMA" if cfg.get("ema") else "raw"),
            metric_type="harness",
            y_inject=cfg.get("y_inject"), logvar_bias=cfg.get("logvar_bias"),
            nodecay_y_proj=cfg.get("nodecay_y_proj"),
            beta_sum=bs, beta_mean=bm, seed=cfg.get("seed"), epochs=cfg.get("epochs"),
            harness_seed=d.get("seed"),
            single_valid=p.get("valid"), coverage_at20=cov.get("coverage"),
            pass_at20=(d.get("pass_at") or {}).get("20"),
            majority_at20=d.get("majority_at_N_valid"),
            majority_canon_at20=d.get("majority_at_N_valid_canonical"),
            verifier_at20=d.get("verifier_selected_at_N_valid"),
            distinct_at20=d.get("distinct_at_N_mean"),
            conflict_mean=p.get("conflicts_mean"), conflict_std=p.get("conflicts_std"),
            single_valid_std=None, coverage_at20_std=None, source="ours",
        ))
    return raw, skipped


def add_inloop_rows(rows):
    """로그의 in-loop 곡선 정점을 metric_type=inloop 행으로 추가.

    harness 미측정 run(v4 등)이 표에서 사라지지 않게 하려는 것이고, harness 행과
    축이 다르므로(단일 배치 accuracy, coverage 없음) 섞어 읽지 말아야 한다.
    """
    import gzip
    EVAL = re.compile(r"^\[EVAL @ step (\d+)\]")
    RAWE = re.compile(r"^\[EVAL RAW @ step (\d+)\]")
    out = []
    for project, task in TASK_OF.items():
        d = f"checkpoints/{project}"
        if not os.path.isdir(d):
            continue
        for run in sorted(os.listdir(d)):
            if is_probe(run) or not os.path.isdir(f"{d}/{run}"):
                continue
            log = next((c for c in (f"logs/{run}.log.gz", f"logs/{run}.log")
                        if os.path.exists(c)), None)
            if log is None:
                continue
            key = {"nq8": "nqueens/accuracy", "nq10": "nqueens/accuracy",
                   "gc8": "gc/accuracy"}[task]
            best = {"EMA": (None, None), "live": (None, None)}
            last = {"EMA": None, "live": None}
            op = gzip.open if log.endswith(".gz") else open
            with op(log, "rt", errors="replace") as fh:
                for line in fh:
                    for pat, w, k in ((EVAL, "EMA", key), (RAWE, "live", key + "_raw")):
                        m = pat.match(line)
                        if not m:
                            continue
                        v = re.search(re.escape(k) + r"=([0-9.eE+-]+)", line)
                        if not v:
                            continue
                        s, f = int(m.group(1)), float(v.group(1))
                        last[w] = s
                        if best[w][0] is None or f > best[w][0]:
                            best[w] = (f, s)
            cfg, server = cfg_and_server(project, run)
            bs, bm = beta_pair(cfg, task)
            for w, (val, step) in best.items():
                if val is None:
                    continue
                kind = "peak+final" if step == last[w] else "peak"
                out.append(dict(
                    server=server, task=task, run=run, step=step, step_kind=kind,
                    weights=w, metric_type="inloop",
                    y_inject=cfg.get("y_inject"), logvar_bias=cfg.get("logvar_bias"),
                    nodecay_y_proj=cfg.get("nodecay_y_proj"), beta_sum=bs, beta_mean=bm,
                    seed=cfg.get("seed"), epochs=cfg.get("epochs"), harness_seed=None,
                    single_valid=val, coverage_at20=None, pass_at20=None,
                    majority_at20=None, majority_canon_at20=None, verifier_at20=None,
                    distinct_at20=None, conflict_mean=None, conflict_std=None,
                    single_valid_std=None, coverage_at20_std=None, source="ours"))
    return rows + out


def mark_step_kind(rows):
    """(server, task, run, weights) 안에서 peak / final 을 데이터로 정한다.

    peak  = single_valid 최대 step,  final = step 최대.  둘이 같으면 peak+final.
    파일명·통보에 의존하지 않는다.  정점은 test set 으로 checkpoint 를 고른 것이므로
    선택 효과를 포함하고, 그 크기가 peak-final 격차다.
    """
    groups = defaultdict(list)
    for r in rows:
        if r["metric_type"] == "harness":
            groups[(r["server"], r["task"], r["run"], r["weights"])].append(r)
    for g in groups.values():
        scored = [r for r in g if r["single_valid"] is not None]
        peak = max(scored, key=lambda r: r["single_valid"])["step"] if scored else None
        final = max(r["step"] for r in g)
        for r in g:
            if r["step"] == peak and r["step"] == final:
                r["step_kind"] = "peak+final"
            elif r["step"] == peak:
                r["step_kind"] = "peak"
            elif r["step"] == final:
                r["step_kind"] = "final"
            else:
                r["step_kind"] = "mid"
    return rows


def build_unified(pattern=RESULTS):
    rows, skipped = load_unified(pattern)
    rows = mark_step_kind(rows)
    rows = add_inloop_rows(rows)
    for p in PAPER_ROWS:
        rows.append({**{c: None for c in UNIFIED_COLS}, **p,
                     "server": "-", "run": "paper", "step_kind": "-",
                     "weights": "-", "metric_type": "paper", "source": "paper"})
    data = [[r.get(c) for c in UNIFIED_COLS]
            for r in sorted(rows, key=lambda r: (r["source"] != "paper", r["task"],
                                                 r["server"], r["run"], r["step"] or 0,
                                                 r["weights"] or "", r["harness_seed"] or -1))]
    return UNIFIED_COLS, data, skipped


# ---------------------------------------------------------------------------
# 패널 (Panel A / F / G)
#
# 축 규약을 캡션에 박는다 -- 가중치(EMA|live) x 집계(단일시도=pass@1) x step_kind(peak|final).
# 이중 y축은 쓰지 않는다: 스케일이 다른 지표는 별도 subplot 으로 분리한다
# (accuracy 0~1 vs conflict 0~2.7, valid 0~1 vs distinct 1~20).
# ---------------------------------------------------------------------------

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"      # 검증 통과 (validate_palette.js)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color="#e4e3df", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5d4cf")
    ax.tick_params(colors=INK2, labelsize=9)


def _bars(ax, groups, series, fmt="{:.3f}", ymax=None, log=False):
    """groups: [label], series: [(name, [vals], color)]. 값은 직접 라벨한다."""
    import numpy as np
    n, w = len(series), 0.8 / max(len(series), 1)
    x = np.arange(len(groups))
    for i, (name, vals, col) in enumerate(series):
        off = (i - (n - 1) / 2) * w
        vs = [0 if v is None else v for v in vals]
        ax.bar(x + off, vs, w * 0.92, label=name, color=col, zorder=3,
               edgecolor=SURFACE, linewidth=2)          # 2px surface gap
        for xi, v, raw in zip(x + off, vs, vals):
            if raw is None:
                continue
            ax.annotate(fmt.format(raw), (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, color=INK)
    if log:
        ax.set_yscale("log")
    elif ymax:
        ax.set_ylim(0, ymax)


def build_panels(cols, data):
    """{name: matplotlib Figure}. 데이터는 unified 표에서만 읽는다.

    matplotlib 이 없으면 RuntimeError. 공용 conda env 를 건드리지 않으려면 별도
    디렉토리에 설치하고 GRAM_PYLIBS 로 알려준다 (학습이 도는 박스에서 numpy 가
    갈리는 것을 막기 위한 조치다):

        pip install --target ~/pylibs matplotlib
        GRAM_PYLIBS=~/pylibs python experiments/wandb_harness_upload.py --mode unified --panels
    """
    extra = os.environ.get("GRAM_PYLIBS")
    if extra and os.path.isdir(extra) and extra not in sys.path:
        sys.path.insert(0, extra)
    try:
        import matplotlib
    except ImportError as e:
        raise RuntimeError(
            "matplotlib 없음 -- 표만 올리려면 --panels 를 빼거나, GRAM_PYLIBS 로 "
            "별도 설치 경로를 지정하라 (docstring 참고)") from e
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ci = {c: i for i, c in enumerate(cols)}

    def pick(task, source="ours", metric_type="harness", kind=None, weights="live"):
        out = [r for r in data if r[ci["task"]] == task and r[ci["source"]] == source
               and r[ci["metric_type"]] == metric_type
               and (weights is None or r[ci["weights"]] == weights)
               and (kind is None or (r[ci["step_kind"]] or "").startswith(kind))]
        return out

    def best(rows, col):
        vs = [r[ci[col]] for r in rows if r[ci[col]] is not None]
        return max(vs) if vs else None

    def paper(task, col):
        rows = [r for r in data if r[ci["task"]] == task and r[ci["source"]] == "paper"]
        return rows[0][ci[col]] if rows else None

    figs = {}

    # ---- Panel A : ours(peak) vs ours(final) vs paper.  conflict 는 A2 로 분리 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [2.4, 1]})
    tasks = ["nq8", "nq10", "gc8"]
    for ax, col, ttl in ((ax1, "single_valid", "single-attempt valid (= pass@1)"),):
        pk = [best(pick(t, kind="peak"), col) for t in tasks]
        fn = [best(pick(t, kind="final"), col) for t in tasks]
        pp = [paper(t, col) for t in tasks]
        _bars(ax, tasks, [("ours - peak", pk, C1), ("ours - final", fn, C2),
                          ("paper", pp, C3)], ymax=1.15)
        _style(ax)
        ax.set_title(ttl, color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=3,
                  loc="upper center", bbox_to_anchor=(0.5, -0.08))
    # A2: GC conflict (낮을수록 좋음, 스케일이 달라 별도 축)
    gc_pk = min([r[ci["conflict_mean"]] for r in pick("gc8", kind="peak")
                 if r[ci["conflict_mean"]] is not None] or [None])
    _bars(ax2, ["gc8"], [("ours (lowest run)", [gc_pk], C1),
                         ("paper", [paper("gc8", "conflict_mean")], C3)],
          fmt="{:.3f}", log=True)
    _style(ax2)
    ax2.set_title("GC conflict (lower is better, log scale)", color=INK, fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Panel A - ours vs paper  |  weights=live, aggregation=single-attempt (pass@1), "
                 "peak includes test-set selection bias", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    figs["A_paper_vs_ours"] = fig

    # ---- Panel F : distinct@20 과 valid.  스케일이 달라 small multiples ----
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), facecolor=SURFACE)
    for ax, col, ttl, ymax in ((axes[0], "distinct_at20", "distinct@20 (diversity, 1-20)", 21),
                               (axes[1], "single_valid", "single-attempt valid", 1.15)):
        lo, hi = [], []
        for t in tasks:
            vs = [r[ci[col]] for r in pick(t, weights=None) if r[ci[col]] is not None]
            lo.append(min(vs) if vs else None)
            hi.append(max(vs) if vs else None)
        _bars(ax, tasks, [("min", lo, C1), ("max", hi, C2)],
              fmt="{:.2f}" if col == "distinct_at20" else "{:.3f}", ymax=ymax)
        _style(ax)
        ax.set_title(ttl, color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=2)
    fig.suptitle("Panel F - diversity and accuracy are different axes (split, never dual-y)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    figs["F_distinct_vs_valid"] = fig

    # ---- Panel G : 실패 3종 prior vs posterior (같은 0~1 축) ----
    import json as _json
    modes = [("lr decay", "nq8_on_b500_decay"), ("forcehalt (b_mean 1.0)", "nq8_off_b100_forcehalt"),
             ("low beta (b_sum 0.0009)", "gram_nq8_fix_b")]
    pri, pos, labels = [], [], []
    for label, run in modes:
        # `.live` 를 우선 쓰고, 없으면 step_N 으로 떨어진다. EMA off 로 돌린 run 은
        # step_N 이 곧 raw 라 `.live` 파일이 애초에 없다 (gram_nq8_fix_b).
        got = {}
        for f in glob.glob(RESULTS):
            try:
                d = _json.load(open(f))
            except Exception:
                continue
            ck = str(d.get("checkpoint", ""))
            if f"/{run}/" not in ck or "prior_all_samples" not in d:
                continue
            got.setdefault("live" if ".live" in ck else "step_N", []).append(d)
        picked = got.get("live") or got.get("step_N") or []
        vs_p = [d["prior_all_samples"]["valid"] for d in picked]
        vs_q = [(d.get("posterior") or {}).get("valid") for d in picked]
        if not vs_p:
            continue
        labels.append(label)
        pri.append(sum(vs_p) / len(vs_p))
        qq = [v for v in vs_q if v is not None]
        pos.append(sum(qq) / len(qq) if qq else None)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    _bars(ax, labels, [("prior", pri, C1), ("posterior", pos, C2)], ymax=1.15)
    _style(ax)
    ax.set_title("Panel G - three failure modes: posterior survives, only prior dies",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=2)
    fig.tight_layout()
    figs["G_prior_vs_posterior"] = fig
    return figs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["summary", "table", "curves", "unified", "all"],
                    default="summary")
    ap.add_argument("--summary-project", default=UNIFIED_PROJECT,
                    help="unified 모드가 쓸 통합 project")
    ap.add_argument("--panels", action="store_true",
                    help="unified 모드에서 Panel A/F/G figure 도 함께 올린다")
    ap.add_argument("--csv", default=None, help="unified 표를 CSV 로도 저장")
    ap.add_argument("--panel-dir", default="panels", help="dry-run 에서 패널 PNG 를 저장할 곳")
    ap.add_argument("--commit", action="store_true", help="실제 업로드 (기본은 dry-run)")
    ap.add_argument("--pattern", default=RESULTS)
    ap.add_argument("--table-run-name", default="harness-index")
    ap.add_argument("--only-run", default=None, help="특정 학습 run 만")
    args = ap.parse_args()

    import wandb
    os.environ.setdefault("WANDB_SILENT", "true")

    # ---------- unified ----------
    # wandb.Api 를 쓰지 않는다 -- 오프라인/무자격에서도 표를 만들 수 있어야 한다.
    if args.mode == "unified":
        cols, data, skipped = build_unified(args.pattern)
        print(f"모드=unified  {'COMMIT' if args.commit else 'DRY-RUN (업로드 안 함)'}")
        print(f"{len(data)}행 x {len(cols)}열  (스키마 불일치·probe 제외 {skipped}건)")
        ci = {c: i for i, c in enumerate(cols)}
        cnt = defaultdict(int)
        for r in data:
            cnt[(r[ci['source']], r[ci['server']], r[ci['task']], r[ci['metric_type']])] += 1
        print("\n  source  server  task   metric_type   행")
        for k, v in sorted(cnt.items()):
            print(f"  {k[0]:<7} {k[1]:<7} {k[2]:<6} {k[3]:<13} {v:>4}")
        if args.csv:
            import csv as _csv
            with open(args.csv, "w", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(cols)
                w.writerows(data)
            print(f"\n  CSV -> {args.csv}")
        panels = build_panels(cols, data) if args.panels else {}
        if panels and not args.commit:
            os.makedirs(args.panel_dir, exist_ok=True)
            for name, fig in panels.items():
                p = os.path.join(args.panel_dir, f"panel_{name}.png")
                fig.savefig(p, dpi=150, facecolor=SURFACE)
                print(f"  panel -> {p}")
        if args.commit:
            run = wandb.init(project=args.summary_project, name="unified-summary",
                             job_type="summary", reinit=True)
            run.log({"summary/all": wandb.Table(columns=cols, data=data)})
            for name, fig in panels.items():
                run.log({f"panel/{name}": wandb.Image(fig)})
            art = wandb.Artifact("gram-unified-summary", type="summary")
            art.add(wandb.Table(columns=cols, data=data), "all")
            run.log_artifact(art)
            run.finish()
            print(f"  업로드 완료 -> project '{args.summary_project}'")
        return

    recs, skipped = load_results(args.pattern)
    if args.only_run:
        recs = [r for r in recs if r["run"] == args.only_run]
    by_run = defaultdict(list)
    for r in recs:
        by_run[(r["project"], r["run"])].append(r)
    print(f"결과 {len(recs)}건 / 스키마 불일치로 건너뜀 {skipped}건 / 학습 run {len(by_run)}개")
    print(f"모드={args.mode}  {'COMMIT' if args.commit else 'DRY-RUN (업로드 안 함)'}\n")

    api = wandb.Api(timeout=60)
    projects = sorted({p for p, _ in by_run})
    picked, ambiguous = wandb_index(api, projects)
    if ambiguous:
        print(f"[note] 동명 wandb run {len(ambiguous)}건 -> finished/최대 step 우선 선택")

    # ---------- summary ----------
    if args.mode in ("summary", "all"):
        wrote = skip_running = missing = 0
        for (proj, run), rs in sorted(by_run.items()):
            wr = picked.get((proj, run))
            if wr is None:
                print(f"  [miss ] {run}: wandb run 없음 (결과 {len(rs)}건)")
                missing += 1
                continue
            if wr.state == "running":
                print(f"  [skip ] {run}: 학습 진행 중 -- 건드리지 않음")
                skip_running += 1
                continue
            s = build_summary(rs, run_meta(proj, run))
            if "harness/live_peak_valid" in s:
                kind, val, step = ".live peak", s["harness/live_peak_valid"], s["harness/live_peak_step"]
                mark = " (경계)" if s["harness/live_peak_is_boundary"] else ""
            else:
                # .live 곡선이 3점 미만이라 규약상 peak 이라 부를 수 없다
                kind, val, step = f"best/{s['harness/best_weights']}", s["harness/best_valid"], s["harness/best_step"]
                mark = ""
            print(f"  [{'write' if args.commit else 'plan '}] {run:<28} {wr.id}  "
                  f"{kind:>11}={val*100:6.2f}% @ {step:<6}{mark:<7} keys={len(s)}")
            if args.commit:
                wr.summary.update(s)
                wr.update()
                wrote += 1
        print(f"\nsummary: 기록 {wrote} / 진행중 건너뜀 {skip_running} / 매칭실패 {missing}")

    # ---------- table ----------
    if args.mode in ("table", "all"):
        cols = ["family", "project", "run", "beta", "y_onehot", "ema_run", "step",
                "weights", "seed", "prior_valid", "posterior_valid", "pass@1", "pass@20",
                "majority@20", "verifier_sel@20", "distinct@20", "queens_mean",
                "queens_min", "queens_max", "prior_consistent", "prior_strict", "file"]
        rows = []
        for r in sorted(recs, key=lambda r: (r["run"], r["step"], r["weights"], r["seed"] or 0)):
            m = run_meta(r["project"], r["run"])
            rows.append([r["family"], r["project"], r["run"], m.get("beta"), m.get("y_onehot"),
                         m.get("ema"), r["step"], weight_label(r, m), r["seed"],
                         r["prior_valid"], r["posterior_valid"], r["pass_at_1"], r["pass_at_20"],
                         r["majority_at_N"], r["verifier_sel_at_N"], r["distinct_at_N"],
                         r["queens_mean"], r["queens_min"], r["queens_max"],
                         r["prior_consistent"], r["prior_strict"], r["file"]])
        print(f"\ntable: {len(rows)}행 x {len(cols)}열 -> run '{args.table_run_name}'")
        if args.commit:
            run = wandb.init(project=projects[0], name=args.table_run_name,
                             job_type="harness-index", reinit=True)
            t = wandb.Table(columns=cols, data=rows)
            run.log({"harness/all_results": t})
            art = wandb.Artifact("harness-results", type="harness")
            art.add(t, "all_results")
            run.log_artifact(art)
            run.finish()
            print("  업로드 완료")

    # ---------- curves ----------
    if args.mode in ("curves", "all"):
        made = 0
        for (proj, run), rs in sorted(by_run.items()):
            live0 = sorted([r for r in rs if r["weights"] == "live" and r["seed"] == 0],
                           key=lambda r: r["step"])
            if len(live0) < 3:
                continue
            print(f"  [{'make ' if args.commit else 'plan '}] {run:<28} .live {len(live0)}포인트")
            made += 1
            if not args.commit:
                continue
            wrun = wandb.init(project=proj, group=run, job_type="harness",
                              name=harness_run_name(run, "live", 0),
                              id=harness_run_id(proj, run, "live", 0),
                              resume="allow", reinit=True,
                              config=dict(source_run=run, **run_meta(proj, run)))
            wrun.define_metric("harness/step")
            wrun.define_metric("harness/*", step_metric="harness/step")
            for r in live0:
                wrun.log({"harness/step": r["step"],
                          "harness/live/prior_valid": r["prior_valid"],
                          "harness/live/posterior_valid": r["posterior_valid"],
                          "harness/live/pass_at_20": r["pass_at_20"],
                          "harness/live/majority_at_20": r["majority_at_N"],
                          "harness/live/distinct_at_20": r["distinct_at_N"],
                          "harness/live/queens_mean": r["queens_mean"]})
            wrun.finish()
        print(f"\ncurves: {made}개 run 대상")


if __name__ == "__main__":
    main()
