#!/usr/bin/env python3
"""진행 중 학습의 in-loop eval 곡선을 텍스트로 덤프한다.

checkpoint 가 날아가도 곡선은 남긴다는 것이 목적. logs/<run>.log 를 파싱해
reports/snapshot_<서버>_<시각>.md 를 만든다.

  python scripts/dump_inloop_snapshot.py                      # logs/ 에서 활성 run 자동 탐지
  python scripts/dump_inloop_snapshot.py --all                # 끝난 run 까지 logs/ 전부
  python scripts/dump_inloop_snapshot.py nq10_A3p_ii_a7525_lr3e4 ...

파싱 대상 (pretrain.py 표준 출력):
  [EVAL @ step N]     ...  -> EMA 가중치 (ema=True 인 run)
  [EVAL RAW @ step N] ...  -> raw/live 가중치
  [step N] train/...  ...  -> kl_loss / lm_loss
진행률은 tqdm 마지막 조각에서 step / it/s / ETA 를 뽑는다.

★ kl_loss / lm_loss 는 eval 줄이 아니라 train 줄에서 가져온다.
  eval 줄의 all/kl_loss 는 EMA·raw 양쪽 다 항상 0.0000 이다 (eval 경로에서
  KL 을 계산하지 않는다). 그 값을 표에 넣으면 0 만 늘어선 열이 된다.
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

# eval 줄에서 뽑는 지표. (표 헤더, EMA 키, raw 키)
COLS = [
    ("strict_acc", "nqueens/strict_accuracy", "nqueens/strict_accuracy_raw"),
    ("queen_cnt", "nqueens/queen_count_mean", "nqueens/queen_count_mean_raw"),
    ("exact_acc", "all/exact_accuracy", "all/exact_accuracy_raw"),
]
# train 줄에서 뽑는 지표. (표 헤더, train 키)
TRAIN_COLS = [
    ("kl_loss", "train/kl_loss"),
    ("lm_loss", "train/lm_loss"),
]
TRAIN_TOL = 64  # train 로그는 100 step 배수라 eval step 과 정확히 안 맞는다.


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


TRAIN_RE = re.compile(r"\[step (\d+)\]((?: [A-Za-z0-9_/]+=[-\d.e+]+)+)")


def parse_train(path):
    """전체 [step N] train/... 줄을 {step: kv} 로."""
    with open(path, "r", errors="replace") as f:
        text = f.read()
    return {int(m.group(1)): {k: float(v) for k, v in KV_RE.findall(m.group(2))}
            for m in TRAIN_RE.finditer(text)}


def train_near(train, step, tol=TRAIN_TOL):
    """eval step 에 가장 가까운 train 줄 (tol step 이내, 없으면 빈 dict)."""
    if not train:
        return {}
    k = min(train, key=lambda s: abs(s - step))
    return train[k] if abs(k - step) <= tol else {}


def train_tail(train):
    """마지막 [step N] train/... 줄의 (step, kv)."""
    if not train:
        return None, {}
    k = max(train)
    return k, train[k]


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


def all_runs():
    """logs/ 의 모든 run (끝난 것 포함). `_` 로 시작하는 boot/tee 사본은 뺀다."""
    names = []
    for p in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
        n = os.path.basename(p)[:-4]
        if not n.startswith("_") and os.path.getsize(p) >= 1024:
            names.append(n)
    return names


def main():
    argv = [a for a in sys.argv[1:] if a != "--all"]
    want_all = "--all" in sys.argv[1:]
    runs = argv or (all_runs() if want_all else active_runs())
    if not runs:
        print("활성 run 없음 — logs/ 에서 최근 수정된 .log 를 지정하라.", file=sys.stderr)
        return 1

    live = set(active_runs())
    # 컨테이너 박스는 hostname 이 해시라 사람이 읽는 이름을 env 로 덮어쓴다 (예: lecun).
    host = os.environ.get("SNAPSHOT_HOST") or socket.gethostname()
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M")
    out_path = os.path.join(OUT, f"snapshot_{host}_{stamp}.md")

    L = []
    L.append(f"# in-loop eval 스냅샷 — {host} @ {now.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")
    L.append("checkpoint 소실 대비 텍스트 보존본. `logs/<run>.log` 의")
    L.append("`[EVAL @ step]`(EMA) / `[EVAL RAW @ step]`(raw) 을 그대로 옮긴 것이다.")
    L.append("`kl_loss` / `lm_loss` 는 eval 줄이 아니라 eval step 에 가장 가까운")
    L.append(f"`[step N] train/...` 값이다 (±{TRAIN_TOL} step). eval 줄의 kl 은 항상 0 이다.")
    L.append("")
    if live:
        L.append("지금 도는 run: " + ", ".join(f"`{r}`" for r in sorted(live)))
        L.append("")

    for run in runs:
        path = os.path.join(LOGS, run + ".log")
        if not os.path.exists(path):
            L.append(f"## {run}\n\n로그 없음: `{path}`\n")
            continue
        evals, prog = parse_log(path)
        train = parse_train(path)
        tstep, tkv = train_tail(train)

        # eval 이 한 번도 없는 로그는 학습 로그가 아니다 (harness / 부팅 로그).
        # 단, 지금 도는 run 은 첫 eval 전이어도 진행 상황을 남긴다.
        if not evals and run not in live:
            continue

        L.append(f"## {run}" + ("  (RUNNING)" if run in live else ""))
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

        if not evals:
            L.append("eval 미도달 (eval_interval 첫 지점 전).")
            L.append("")
            continue

        hdr = ["step"]
        for name, _, _ in COLS:
            hdr += [f"EMA {name}", f"raw {name}"]
        hdr += [name for name, _ in TRAIN_COLS]
        L.append("| " + " | ".join(hdr) + " |")
        L.append("|" + "---|" * len(hdr))
        for step in sorted(evals):
            e = evals[step].get("ema", {})
            r = evals[step].get("raw", {})
            t = train_near(train, step)
            row = [str(step)]
            for _, ke, kr in COLS:
                row.append(fmt(e.get(ke)))
                row.append(fmt(r.get(kr)))
            for _, kt in TRAIN_COLS:
                row.append(fmt(t.get(kt)))
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    os.makedirs(OUT, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
