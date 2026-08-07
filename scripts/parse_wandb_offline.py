#!/usr/bin/env python3
"""wandb offline run(.wandb) 에서 history 시계열을 뽑아 CSV 로 덤프한다.

offline run 은 서버에 올라간 적이 없으므로 public API 로 못 읽는다.
.wandb 파일은 wandb 내부 leveldb 유사 포맷이라, wandb SDK 의 DataStore 로
레코드를 순차 스캔해 Record protobuf 를 직접 파싱하는 것이 유일한 경로다.

용법:
    python3 scripts/parse_wandb_offline.py wandb/offline-run-XXXX \
        --keys train/diag/z_h_norm train/diag/z_l_norm \
        --out reports/data/zh_norm_P.csv

--keys 를 생략하면 모든 history 키를 덤프한다.
--list-keys 는 키 목록과 각 키의 기록 횟수만 출력하고 끝낸다.

출력 CSV 스키마: step,key,value
  step  = wandb 의 _step. pretrain.py:628 이 step=train_state.step 으로 log 하므로
          이 값은 곧 학습 step 이다 (eval 도 같은 step 축에 실린다).
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb


def resolve_wandb_file(path: str) -> str:
    """디렉토리를 주면 그 안의 단일 .wandb 파일을 찾아준다."""
    if os.path.isfile(path):
        return path
    cands = [f for f in os.listdir(path) if f.endswith(".wandb")]
    if len(cands) != 1:
        raise SystemExit(f"{path}: .wandb 파일이 정확히 1개가 아니다 -> {cands}")
    return os.path.join(path, cands[0])


def item_key(it) -> str:
    # HistoryItem 은 평문 key 또는 nested_key(리스트) 중 하나를 쓴다.
    return it.key if it.key else ".".join(it.nested_key)


def scan(path: str, wanted=None, progress_every=50000):
    """(step, key, value) 를 순차 yield. wanted=None 이면 전부."""
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    n_rec = 0
    while True:
        try:
            data = ds.scan_data()
        except Exception as e:  # 실행 중인 run 은 꼬리가 잘려 있을 수 있다
            print(f"  [warn] step {n_rec} 부근에서 스캔 중단: {e}", file=sys.stderr)
            break
        if data is None:
            break
        n_rec += 1
        if progress_every and n_rec % progress_every == 0:
            print(f"  ... {n_rec} records", file=sys.stderr)
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        items = {item_key(it): it.value_json for it in rec.history.item}
        step_s = items.get("_step")
        if step_s is None:
            continue
        try:
            step = int(float(step_s))
        except ValueError:
            continue
        for k, v in items.items():
            if k.startswith("_"):
                continue
            if wanted is not None and k not in wanted:
                continue
            try:
                val = json.loads(v)
            except Exception:
                continue
            if isinstance(val, (int, float)):
                yield step, k, float(val)
    print(f"  총 {n_rec} records 스캔", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="offline-run 디렉토리 또는 .wandb 파일")
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--list-keys", action="store_true")
    args = ap.parse_args()

    path = resolve_wandb_file(args.run)
    print(f"[src] {path}", file=sys.stderr)

    if args.list_keys:
        c = Counter()
        for _step, k, _v in scan(path, None):
            c[k] += 1
        for k, n in sorted(c.items()):
            print(f"{n:9d}  {k}")
        return

    wanted = set(args.keys) if args.keys else None
    rows = list(scan(path, wanted))
    if not rows:
        print("기록 없음 -- 확인 못 함", file=sys.stderr)

    out = args.out
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "key", "value"])
            w.writerows(rows)
        print(f"[out] {out}  ({len(rows)} rows)", file=sys.stderr)
    else:
        w = csv.writer(sys.stdout)
        w.writerow(["step", "key", "value"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
