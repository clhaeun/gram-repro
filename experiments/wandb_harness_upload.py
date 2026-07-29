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
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["summary", "table", "curves", "all"], default="summary")
    ap.add_argument("--commit", action="store_true", help="실제 업로드 (기본은 dry-run)")
    ap.add_argument("--pattern", default=RESULTS)
    ap.add_argument("--table-run-name", default="harness-index")
    ap.add_argument("--only-run", default=None, help="특정 학습 run 만")
    args = ap.parse_args()

    import wandb
    os.environ.setdefault("WANDB_SILENT", "true")

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
