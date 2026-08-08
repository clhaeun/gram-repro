# 사고 기록 — 산출 디렉토리 소유권으로 nq8 σ 스윕 7셀 소실 (2026-08-07, 50번)

## 무슨 일

nq8 σ 반사실 스윕 7셀(σ = 0.25 / 0.5 / 0.75 / 1.0 / 1.25 / 1.5 / 2.0)이 계산은
전부 끝났으나 JSON 을 하나도 못 남기고 종료했다.

```
File "/workspace/TinyRecursiveModels/experiments/sweep_harness.py", line 154, in main
    with open(args.out, "w") as f:
PermissionError: [Errno 13] Permission denied:
    '/workspace/home/chaeun/tmp/gram50/sigma/nq8_A0_s0_ema_sigma2.0.json'
```

## 원인

이 박스는 컨테이너 root 로 들어오지만, GPU 작업은 호스트 `gpustat`/`nvidia-smi` 에
소유자가 보이도록 `/usr/local/bin/as-chaeun` (setpriv --reuid=1048) 으로 낮춰 돌린다.

스윕 러너 스크립트가

```bash
mkdir -p "$OUT"                       # <- root 셸이 실행 -> drwxr-xr-x root root
...
as-chaeun env CUDA_VISIBLE_DEVICES=$GPU python3 experiments/sweep_harness.py ... --out "$OUT/..."
```

구조여서 **디렉토리는 root 가 만들고 쓰기는 chaeun 이 한다**. 755 라 chaeun 은
그 안에 파일을 만들 수 없다. harness 는 전 계산을 마친 뒤 마지막 줄에서 죽으므로
GPU 시간만 쓰고 산출물은 0 이 된다.

nq10 6셀은 같은 디렉토리를 썼으나 21:28 에 소유권을 고친 시점이 그 셀들의 쓰기
시점(22:42~22:55)보다 앞서 손실이 없었다.

## 복구

nq8 7셀 값은 **로그에서 전량 복구됐다** — `sweep_harness.main()` 이 JSON 을 쓰기
*전에* 요약 한 줄을 stdout 으로 print 하기 때문이다
(`experiments/sweep_harness.py` 의 `print(f"{res['checkpoint']} hms=... acc=...")`).
복구 출처: `/workspace/home/chaeun/tmp/gram50/sigma/nq8_gpu7.log`.
단 print 에 없는 축(`found_mean`, `total_mean`, `per_row`)은 복구 불가이며
`cov_micro x total_mean` 로 역산할 때 print 의 4자리 정밀도가 상한이다.

## 재발 방지 (적용됨)

1. `experiments/sweep_harness.py` — 쓰기를 try/except 로 감싸고 실패 시
   `RESULT_JSON <전량 json.dumps>` 를 stdout 에 뱉는다. 로그만 있으면 완전 복구된다.
2. 산출 스크립트는 `mkdir` 직후 `chown -R chaeun:chaeun "$OUT"; chmod -R 775 "$OUT"`
   를 실행한다 (또는 `mkdir` 자체를 `as-chaeun` 으로 한다).

## 이 박스에서 GPU 작업을 돌릴 때의 규칙

- `as-chaeun <command>` 로 감싼다. root 로 돌리면 호스트에서 점유자 식별이 안 된다.
- 환경변수는 `as-chaeun env KEY=VAL ... <command>` 형태로 넘긴다.
- **산출 경로는 chaeun 이 쓸 수 있어야 한다.** Claude 세션 scratchpad
  (`/tmp/claude-0/...`)는 root 전용(drwx------)이라 chaeun 이 못 쓴다.
  산출물은 `/workspace/home/chaeun/` 하위나 repo 안에 쓰고, 새로 만든 디렉토리는
  소유권을 확인한다.
- root 로 무언가 실행한 뒤에는 `find . -user root` 로 repo 안에 root 소유 파일이
  생겼는지 확인한다 (hydra 의 `outputs/` 가 root 소유가 되면 이후 chaeun 학습이
  `PermissionError` 로 즉사한다 — 실제로 발생했다).
