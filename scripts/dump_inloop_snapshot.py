#!/usr/bin/env python3
"""진행 중 학습의 in-loop eval 곡선을 텍스트로 덤프한다.

checkpoint 가 날아가도 곡선은 남긴다는 것이 목적. logs/<run>.log 를 파싱해
reports/snapshot_<서버>_<시각>.md 를 만든다.

  python scripts/dump_inloop_snapshot.py                      # logs/ 에서 활성 run 자동 탐지
  python scripts/dump_inloop_snapshot.py nq10_A3p_ii_a7525_lr3e4 ...

파싱 대상 (pretrain.py 표준 출력):
  [EVAL @ step N]     ...  -> EMA 가중치 (ema=True 인 run)
  [EVAL RAW @ step N] ...  -> raw/live 가중치
진행률은 tqdm 마지막 조각에서 step / it/s / ETA 를 뽑는다.
"""
import os
import re
import sys
import glob
import socket
import subprocess
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "logs")
OUT = os.path.join(REPO, "reports")

EVAL_RE = re.compile(r"\[EVAL(?P<raw> RAW)? @ step (?P<step>\d+)\](?P<body>[^\n]*)")
KV_RE = re.compile(r"([A-Za-z0-9_/]+)=(-?[\d.]+(?:e[-+]?\d+)?)")
# tqdm:  " 25%|##  | 34100/138548 [2:08:21<5:51:40,  4.95it/s]"
TQDM_RE = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)(it/s|s/it)\]")

# 스냅샷에 남길 지표. (표 헤더, EMA 키, raw 키)
COLS = [
    ("strict_acc", "nqueens/strict_accuracy", "nqueens/strict_accuracy_raw"),
    ("queen_cnt", "nqueens/queen_count_mean", "nqueens/queen_count_mean_raw"),
    ("kl_loss", "all/kl_loss", "all/kl_loss_raw"),
    ("lm_loss", "all/lm_loss", "all/lm_loss_raw"),
    ("exact_acc", "all/exact_accuracy", "all/exact_accuracy_raw"),
]


def parse_log(path):
    """(evals, progress) 를 돌려준다. evals[step] = {"ema": {...}, "raw": {...}}"""
    with open(path, "r", errors="replace") as f:
        text = f.read()

    evals = {}
    for m in EVAL_RE.finditer(text):
        step = int(m.group("step"))
        kind = "raw" if m.group("raw") else "ema"
        kv = {k: float(v) for k, v in KV_RE.findall(m.group("body"))}
        evals.setdefault(step, {})[kind] = kv

    prog = None
    for m in TQDM_RE.finditer(text[-200000:]):
        cur, tot, elapsed, eta, rate, unit = m.groups()
        rate = float(rate)
        prog = {
            "step": int(cur),
            "total": int(tot),
            "elapsed": elapsed,
            "eta": eta,
            "it_s": rate if unit == "it/s" else (1.0 / rate if rate else 0.0),
        }
    return evals, prog


def train_tail(path):
    """마지막 [step N] train/... 줄의 kv."""
    with open(path, "r", errors="replace") as f:
        text = f.read()[-200000:]
    hits = re.findall(r"\[step (\d+)\]((?: [A-Za-z0-9_/]+=[-\d.e+]+)+)", text)
    if not hits:
        return None, {}
    step, body = hits[-1]
    return int(step), {k: float(v) for k, v in KV_RE.findall(body)}


def fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def run_cmdline(run):
    """ps 에서 해당 run 의 pretrain.py 커맨드라인을 한 줄 가져온다 (없으면 None)."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "cmd"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if f"+run_name={run}" in line and "pretrain.py" in line and "torchrun" not in line:
            return " ".join(line.split())
    return None


def active_runs():
    """지금 도는 pretrain.py 의 +run_name 목록 (로그 파일이 있는 것만)."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "cmd"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return []
    names = []
    for m in re.finditer(r"\+run_name=(\S+)", out):
        n = m.group(1)
        if n not in names and os.path.exists(os.path.join(LOGS, n + ".log")):
            names.append(n)
    return names


def main():
    runs = sys.argv[1:] or active_runs()
    if not runs:
        print("활성 run 없음 — logs/ 에서 최근 수정된 .log 를 지정하라.", file=sys.stderr)
        return 1

    host = socket.gethostname()
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M")
    out_path = os.path.join(OUT, f"snapshot_{host}_{stamp}.md")

    L = []
    L.append(f"# in-loop eval 스냅샷 — {host} @ {now.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")
    L.append("checkpoint 소실 대비 텍스트 보존본. `logs/<run>.log` 의")
    L.append("`[EVAL @ step]`(EMA) / `[EVAL RAW @ step]`(raw) 을 그대로 옮긴 것이다.")
    L.append("")

    for run in runs:
        path = os.path.join(LOGS, run + ".log")
        if not os.path.exists(path):
            L.append(f"## {run}\n\n로그 없음: `{path}`\n")
            continue
        evals, prog = parse_log(path)
        tstep, tkv = train_tail(path)

        L.append(f"## {run}")
        L.append("")
        if prog:
            pct = 100.0 * prog["step"] / prog["total"] if prog["total"] else 0.0
            L.append(
                f"- 진행: **step {prog['step']} / {prog['total']}** ({pct:.1f}%) · "
                f"**{prog['it_s']:.2f} it/s** · elapsed {prog['elapsed']} · **ETA {prog['eta']}**"
            )
        L.append(f"- 로그: `logs/{run}.log` (mtime {datetime.fromtimestamp(os.path.getmtime(path)):%Y-%m-%d %H:%M:%S})")
        L.append(f"- eval 지점: {len(evals)}개")
        if tstep is not None:
            tr = " ".join(f"{k}={v:.4f}" for k, v in tkv.items())
            L.append(f"- 최근 train @ step {tstep}: `{tr}`")
        cmd = run_cmdline(run)
        if cmd:
            L.append("")
            L.append("설정 (실행 커맨드라인):")
            L.append("")
            L.append("```")
            L.append(cmd)
            L.append("```")
        L.append("")

        hdr = ["step"]
        for name, _, _ in COLS:
            hdr += [f"EMA {name}", f"raw {name}"]
        L.append("| " + " | ".join(hdr) + " |")
        L.append("|" + "---|" * len(hdr))
        for step in sorted(evals):
            e = evals[step].get("ema", {})
            r = evals[step].get("raw", {})
            row = [str(step)]
            for _, ke, kr in COLS:
                row.append(fmt(e.get(ke)))
                row.append(fmt(r.get(kr)))
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    os.makedirs(OUT, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
