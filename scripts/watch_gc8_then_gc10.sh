#!/bin/bash
# gc8_paper_34 완주를 감시하다가 gc10 을 GPU 4-7 에 기동한다.
#
# ★ 완주하지 않은 종료(크래시/OOM/kill)에는 기동하지 않는다. 조건 3개를 전부
#   만족해야만 launch 하고, 아니면 사유만 남기고 끝낸다:
#     (1) pretrain.py + run_name=gc8_paper_34 프로세스가 사라졌다
#     (2) logs/gc8_paper_34.log 의 최대 step 이 GC8_TOTAL-100 이상 (완주 근사)
#     (3) GPU 4-7 이 전부 1000MiB 미만 (유휴)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GC8_TOTAL=45546
LOG=logs/gc8_paper_34.log
WLOG=logs/_watch_gc8_then_gc10.log

say() { echo "[$(date '+%F %T')] $*" >> "$WLOG"; }

say "watcher 시작 (gc8 총 step ${GC8_TOTAL})"

while pgrep -f "pretrain.py.*run_name=gc8_paper_34" > /dev/null; do
  sleep 300
done
say "(1) gc8 프로세스 종료 감지"

maxstep=$(grep -oE '\[step [0-9]+\]' "$LOG" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
maxstep=${maxstep:-0}
say "(2) 로그 최대 step = ${maxstep}"
if [ "$maxstep" -lt $((GC8_TOTAL - 100)) ]; then
  say "중단: 완주 미달 (${maxstep} < $((GC8_TOTAL - 100))). gc10 기동하지 않는다."
  exit 1
fi

for g in 4 5 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  if [ "$used" -gt 1000 ]; then
    say "중단: GPU $g 사용 중 (${used}MiB). gc10 기동하지 않는다."
    exit 1
  fi
done
say "(3) GPU 4-7 유휴 확인 -- gc10 launch"

tmux new-session -d -s gc10_paper_34 -c "$PWD" 2>/dev/null
tmux send-keys -t gc10_paper_34 \
  "setsid nohup env GPUS=4,5,6,7 RUN=gc10_paper_34 PORT=29813 bash scripts/train_gc10_v2.sh 2>&1 | tee -a logs/_boot_gc10_paper_34.log" C-m
say "gc10_paper_34 기동 요청 완료 (총 step 87,637 = 5000*13461/768, kl_coef 0.45)"
