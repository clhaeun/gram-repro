#!/usr/bin/env python3
"""34번 Sudoku/GC 재현 run 의 eval 시점 진행표를 값만 출력한다.

축: run | step | train acc | eval acc | kl_loss | grad_norm
    | diag/z_h_norm | diag/eps_std_norm | diag/eps_mu_norm

★ grad_norm 은 기록되지 않는다. pretrain.py:364 가
  `torch.nn.utils.clip_grad_norm_(..., 1.0)` 로 클립만 하고 반환값(= clip 이전
  전체 grad norm)을 metrics 에 넣지 않는다. wandb 키 목록에도 없다. 따라서 이
  열은 항상 `n/a` 다 -- 값을 만들어내지 않는다.

★ diag/* 는 stdout 로그에 안 나온다. pretrain.py:671 이 프린트 대상을
  'loss' 또는 'accuracy' 포함 키로만 거른다. wandb 에는 전부 실려 있으므로
  로컬 .wandb 파일에서 읽는다 (scripts/parse_wandb_offline.py 와 같은 경로).

train 축 값은 eval step 과 정확히 같은 step 의 기록이 없을 수 있어(로깅 간격)
eval step 이하의 가장 가까운 step 값을 쓰고, 그 step 을 괄호로 병기한다.

용법:
    python3 scripts/progress_report_34.py
    python3 scripts/progress_report_34.py --runs wandb/run-XXXX wandb/run-YYYY
"""
import argparse
import os
import sys
from collections import defaultdict

from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb

# run 이름 -> eval accuracy 로 쓸 키 (평가기별로 다르다)
EVAL_ACC_KEY = {
    "gc": "gc/accuracy",          # graph_coloring evaluator
    "sudoku": "all.exact_accuracy",  # evaluator 없음 -> all/* 만 나온다
}

TRAIN_KEYS = ["train/accuracy", "train/kl_loss", "train/diag/z_h_norm",
              "train/diag/eps_std_norm", "train/diag/eps_mu_norm"]


def read_history(run_dir):
    """{key: {step: value}} 를 돌려준다."""
    cand = [f for f in os.listdir(run_dir) if f.endswith(".wandb")]
    if not cand:
        raise SystemExit(f"no .wandb in {run_dir}")
    path = os.path.join(run_dir, cand[0])

    out = defaultdict(dict)
    ds = datastore.DataStore()
    ds.open_for_scan(path)
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
        step = None
        vals = {}
        for it in rec.history.item:
            key = it.key or ".".join(it.nested_key)
            v = it.value_json
            if key == "_step":
                try:
                    step = int(float(v))
                except ValueError:
                    pass
                continue
            try:
                vals[key] = float(v)
            except ValueError:
                pass
        if step is None:
            continue
        for k, v in vals.items():
            out[k][step] = v
    return out


def nearest_at_or_below(series, step):
    ks = [s for s in series if s <= step]
    if not ks:
        return None, None
    s = max(ks)
    return series[s], s


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def report(label, run_dir, eval_acc_key):
    hist = read_history(run_dir)
    if eval_acc_key not in hist:
        # evaluator 가 없는 run 은 all.* 만 있다
        alt = [k for k in hist if k.startswith("all.") and "accuracy" in k]
        print(f"[{label}] eval acc 키 {eval_acc_key} 없음. 사용 가능: {sorted(alt)}")
        if not alt:
            return
        eval_acc_key = sorted(alt)[0]

    eval_steps = sorted(hist[eval_acc_key])
    print(f"\n## {label}   (run_dir {os.path.basename(run_dir)}, eval acc key = {eval_acc_key})")
    hdr = (f"{'step':>7} | {'train acc':>9} | {'eval acc':>9} | {'kl_loss':>10} | "
           f"{'grad_norm':>9} | {'z_h_norm':>10} | {'eps_std_norm':>12} | {'eps_mu_norm':>11} | train@")
    print(hdr)
    print("-" * len(hdr))
    for st in eval_steps:
        tr_acc, s1 = nearest_at_or_below(hist.get("train/accuracy", {}), st)
        kl, _ = nearest_at_or_below(hist.get("train/kl_loss", {}), st)
        zh, _ = nearest_at_or_below(hist.get("train/diag/z_h_norm", {}), st)
        es, _ = nearest_at_or_below(hist.get("train/diag/eps_std_norm", {}), st)
        em, _ = nearest_at_or_below(hist.get("train/diag/eps_mu_norm", {}), st)
        ev = hist[eval_acc_key][st]
        print(f"{st:>7} | {fmt(tr_acc):>9} | {fmt(ev):>9} | {fmt(kl):>10} | "
              f"{'n/a':>9} | {fmt(zh,3):>10} | {fmt(es,4):>12} | {fmt(em,4):>11} | {s1}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()

    if args.runs:
        pairs = [(os.path.basename(r), r, "gc/accuracy") for r in args.runs]
    else:
        pairs = []
        base = "wandb"
        for d in sorted(os.listdir(base)):
            p = os.path.join(base, d)
            cfg = os.path.join(p, "files", "wandb-metadata.json")
            # `latest-run` 은 심볼릭 링크라 같은 run 이 두 번 잡힌다.
            if not os.path.isdir(p) or os.path.islink(p):
                continue
            name = None
            if os.path.exists(cfg):
                for line in open(cfg, errors="ignore"):
                    if "gc8_paper_34" in line:
                        name = "gc8_paper_34"
                    elif "gc10_paper_34" in line:
                        name = "gc10_paper_34"
                    elif "sudoku_paper_34" in line:
                        name = "sudoku_paper_34"
                    if name:
                        break
            if name:
                key = "gc/accuracy" if name.startswith("gc") else "all.exact_accuracy"
                pairs.append((name, p, key))

    if not pairs:
        raise SystemExit("34번 run 을 wandb/ 에서 못 찾았다. --runs 로 지정하라.")

    print("★ grad_norm 은 기록되지 않는다 (pretrain.py:364 가 clip 만 하고 값을 "
          "metrics 에 넣지 않는다). 전 행 n/a.")
    for label, p, key in pairs:
        try:
            report(label, p, key)
        except Exception as e:
            print(f"[{label}] 읽기 실패: {e}")


if __name__ == "__main__":
    main()
