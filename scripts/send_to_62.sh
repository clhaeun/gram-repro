#!/bin/bash
# [50 -> 62] Sudoku 데이터 + sudoku_paper_34 모델 전송.
#
#   bash scripts/send_to_62.sh          # 전부 (모델 -> 데이터 -> 검증)
#   bash scripts/send_to_62.sh model    # 모델 111MB 만
#   bash scripts/send_to_62.sh data     # 데이터 1.28GiB 만
#   bash scripts/send_to_62.sh verify   # 62번 md5 출력만
#
# 키 등록 전이면 단계마다 비밀번호를 묻는다.  한 번만 묻게 하려면 먼저:
#   as-chaeun ssh-copy-id -i /workspace/home/chaeun/.ssh/id_ed25519.pub chaeun@143.248.135.62
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEST=chaeun@143.248.135.62
CKDIR=checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch
DATA=data/sudoku-extreme-1k-aug-1000
WHAT="${1:-all}"

do_mkdir() {
  echo "== [0/3] 62번 디렉토리 생성"
  ssh "$DEST" "mkdir -p gram-repro/data gram-repro/$CKDIR" || return 1
}

do_model() {
  echo "== [1/3] 모델 111MB -> gram-repro/$CKDIR/sudoku_paper_34"
  scp -rp "$CKDIR/sudoku_paper_34" "$DEST:gram-repro/$CKDIR/" || return 1
}

do_data() {
  echo "== [2/3] 데이터 1.28GiB -> gram-repro/$DATA"
  scp -rp "$DATA" "$DEST:gram-repro/data/" || return 1
}

do_verify() {
  echo "== [3/3] 62번 md5 (50번 기준값은 reports/TRANSFER_50_to_62.md)"
  ssh "$DEST" 'cd ~/gram-repro && python3 - <<PY
import hashlib, os
import numpy as np
def fmd5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8<<20), b""): h.update(c)
    return h.hexdigest()
for root in ["data/sudoku-extreme-1k-aug-1000",
             "checkpoints/Sudoku-extreme-1k-aug-1000-test10k-v2-ACT-torch/sudoku_paper_34"]:
    print("=====", root)
    if not os.path.isdir(root):
        print("  (없음)"); continue
    for dp, _, ns in sorted(os.walk(root)):
        for n in sorted(ns):
            p = os.path.join(dp, n); rel = os.path.relpath(p, root)
            if n.endswith(".npy"):
                a = np.load(p)
                print(f"  {rel:38s} {os.path.getsize(p):>12,}  file={fmd5(p)}  tobytes12={hashlib.md5(a.tobytes()).hexdigest()[:12]}  shape={a.shape} {a.dtype}")
            else:
                print(f"  {rel:38s} {os.path.getsize(p):>12,}  file={fmd5(p)}")
PY'
}

rc=0
case "$WHAT" in
  model)  do_mkdir && do_model || rc=1 ;;
  data)   do_mkdir && do_data  || rc=1 ;;
  verify) do_verify || rc=1 ;;
  all)    do_mkdir && do_model && do_data && do_verify || rc=1 ;;
  *)      echo "usage: bash scripts/send_to_62.sh [all|model|data|verify]"; rc=2 ;;
esac
echo "== 종료 rc=$rc"
exit $rc
