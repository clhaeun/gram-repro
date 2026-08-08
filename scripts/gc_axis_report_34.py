#!/usr/bin/env python3
"""34번 GC/Sudoku run 의 eval 지표를 ★raw / EMA 축으로 분리★해 값만 출력한다.

축 이름 규약 (pretrain.py 실측):
  · EMA 축   = `[EVAL @ step N]`      키 접미사 없음 (all.*, gc/*)
               pretrain.py:698 -- ema_helper 로 교체한 가중치로 평가.
  · raw 축   = `[EVAL RAW @ step N]`  키 접미사 `_raw` (all/*_raw, gc/*_raw)
               pretrain.py:729 -- 학습 중인 live 가중치로 같은 eval 을 한 번 더.
  ema=True 일 때만 raw 축이 함께 나온다 (dual eval).

train 축은 두 지표를 구분한다:
  · train/accuracy       = 토큰 단위 정확도
  · train/exact_accuracy = 시퀀스 전부 맞은 비율

wandb history 에서 EMA 축 키는 nested_key 로 들어와 `all.accuracy` 처럼 `.` 로,
raw 축 키는 flat 이라 `all/accuracy_raw` 처럼 `/` 로 나온다. 둘 다 그대로 표기한다.

용법: python3 scripts/gc_axis_report_34.py [--run wandb/run-XXXX]
"""
import argparse
import os
from collections import defaultdict

from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb


def read_history(run_dir):
    cand = [f for f in os.listdir(run_dir) if f.endswith(".wandb")]
    if not cand:
        raise SystemExit(f"no .wandb in {run_dir}")
    out = defaultdict(dict)
    ds = datastore.DataStore()
    ds.open_for_scan(os.path.join(run_dir, cand[0]))
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        step, vals = None, {}
        for it in rec.history.item:
            key = it.key or ".".join(it.nested_key)
            if key == "_step":
                try:
                    step = int(float(it.value_json))
                except ValueError:
                    pass
                continue
            try:
                vals[key] = float(it.value_json)
            except ValueError:
                pass
        if step is None:
            continue
        for k, v in vals.items():
            out[k][step] = v
    return out


def at_or_below(series, step):
    ks = [s for s in series if s <= step]
    return series[max(ks)] if ks else None


def f(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def table(title, steps, cols, hist, nd=4, lookup=None):
    """cols = [(표시이름, wandb키)]. lookup=None 이면 정확히 그 step 값만."""
    widths = [max(len(c[0]), 10) for c in cols]
    print(f"\n### {title}")
    print("  " + f"{'step':>7} | " + " | ".join(f"{c[0]:>{w}}" for c, w in zip(cols, widths)))
    print("  " + "-" * (10 + sum(w + 3 for w in widths)))
    for st in steps:
        cells = []
        for (_, key), w in zip(cols, widths):
            series = hist.get(key, {})
            v = at_or_below(series, st) if lookup == "below" else series.get(st)
            cells.append(f"{f(v, nd):>{w}}")
        print("  " + f"{st:>7} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()

    runs = []
    if args.run:
        runs = [(os.path.basename(args.run), args.run)]
    else:
        for d in sorted(os.listdir("wandb")):
            p = os.path.join("wandb", d)
            if not os.path.isdir(p) or os.path.islink(p):
                continue
            meta = os.path.join(p, "files", "wandb-metadata.json")
            if not os.path.exists(meta):
                continue
            txt = open(meta, errors="ignore").read()
            for name in ("gc8_paper_34", "gc10_paper_34", "sudoku_paper_34"):
                if name in txt:
                    runs.append((name, p))
                    break

    for name, run_dir in runs:
        hist = read_history(run_dir)
        # eval step 축: EMA 쪽 all.accuracy 를 기준으로 잡는다
        anchor = "all.accuracy" if "all.accuracy" in hist else "all/accuracy_raw"
        steps = sorted(hist.get(anchor, {}))
        print("\n" + "=" * 100)
        print(f"# {name}   ({run_dir})   eval 시점 {len(steps)}개")
        print("=" * 100)

        # --- evaluator 출력 키 (gc/*) 를 전부 축별로 ---
        gc_ema = sorted(k for k in hist if k.startswith("gc/") and not k.endswith("_raw"))
        gc_raw = sorted(k for k in hist if k.startswith("gc/") and k.endswith("_raw"))
        if gc_ema:
            table("gc/* — EMA 축 (`[EVAL @ step]`, 접미사 없음)",
                  steps, [(k.replace("gc/", ""), k) for k in gc_ema], hist)
        if gc_raw:
            table("gc/* — raw 축 (`[EVAL RAW @ step]`, 접미사 `_raw`)",
                  steps, [(k.replace("gc/", ""), k) for k in gc_raw], hist)
        if not gc_ema and not gc_raw:
            print("\n### gc/* 키 없음 (evaluator 미지정 run)")

        # --- all/* 손실·정확도 축별 ---
        all_ema = sorted(k for k in hist if k.startswith("all."))
        all_raw = sorted(k for k in hist if k.startswith("all/") and k.endswith("_raw"))
        if all_ema:
            table("all.* — EMA 축", steps,
                  [(k.replace("all.", ""), k) for k in all_ema], hist)
        if all_raw:
            table("all/*_raw — raw 축", steps,
                  [(k.replace("all/", ""), k) for k in all_raw], hist)

        # --- train 축: accuracy 와 exact_accuracy 구분 ---
        table("train 축 (eval step 이하 최근값)", steps, [
            ("accuracy", "train/accuracy"),
            ("exact_accuracy", "train/exact_accuracy"),
            ("kl_loss", "train/kl_loss"),
            ("lm_loss", "train/lm_loss"),
            ("lprm_loss", "train/lprm_loss"),
            ("z_h_norm", "train/diag/z_h_norm"),
            ("z_l_norm", "train/diag/z_l_norm"),
            ("eps_mu_norm", "train/diag/eps_mu_norm"),
            ("eps_std_norm", "train/diag/eps_std_norm"),
        ], hist, lookup="below")


if __name__ == "__main__":
    main()
