# [C] 1단계 — `fixed` + lr 3e-4 smoke 300 step 완료. **대조 대기 중 (34번 기준값 미수령)**

실행: 2026-08-06, 62번(jurgen)
상태: **smoke 통과(코드 경로 정상). 셀 S·T 는 기동하지 않았다.**

## 왜 멈췄나

지시서: *"34번 GPU5 의 fixed+3e-4 로그와 같은 step 구간의 `train/lm_loss`, `train/kl_loss` 가
같은 대역인지 대조. (34번 로그 값은 내가 따로 전달) 대역이 다르면 중단하고 보고."*

**34번 기준값을 아직 못 받았다.** 대조 없이 "통과" 를 선언하고 2셀(4 GPU)을 기동하는 것은
지시서의 게이트를 건너뛰는 것이므로, 우리 쪽 수치만 확보하고 정지했다.

## smoke 구성

```bash
RUN=smoke_fixed_a7525_lr3e4_s0 RESIDUAL_SCALE=fixed ALPHA1=0.75 ALPHA2=0.25 \
NODECAY_MARKED=False LR=3e-4 WARMUP=500 SEED=0 EPOCHS=30 EVALINT=30 GPUS=1,2 PORT=29985 \
WANDB_MODE=offline bash scripts/train_nqueens10_stageA.sh
```

| 항목 | 값 |
|---|---|
| 로그 | `logs/smoke_fixed_a7525_lr3e4_s0.log`, `logs/smoke_launch.out` |
| wandb | `wandb/offline-run-20260806_120051-0ijxy270` |
| GPU | 1,2 (2 GPU) · PORT 29985 |
| 속도 | 5.41 it/s |
| 종료 | step ~850 에서 수동 kill (300 step 확보 후 GPU 반납) |

### 지시서 주의사항 이행

- ✅ `fixed` + `NODECAY_MARKED=False` — ValueError 즉사 없이 기동. `[nodecay]` 라인 미출력 = 마킹 0개, 예상대로.
- ✅ `setsid nohup` 으로 기동.
- ✅ launch 1분 뒤 `nvidia-smi` 확인 — GPU 1,2 각 10,415 MiB / 97~100%.
- ✅ `WANDB_MODE=offline`.
- ✅ 셀 R(GPU 5) 무간섭 확인. kill 후 R 프로세스 9개 생존.
- ✅ 완주 전 정점 확정 없음 — 아래 표는 대조용 원자료일 뿐이다.

## 우리 쪽 실측 (seed 0, fixed, α₁=0.75 α₂=0.25, lr 3e-4, warmup 500, 2 GPU)

| step | `train/lm_loss` | `train/kl_loss` | `train/lprm_loss` | lr |
|---:|---:|---:|---:|---:|
| 50 | 0.9042 | 773.5478 | 3.2315 | 3.00e-05 |
| 100 | 0.4642 | 134.6476 | 2.5801 | 6.00e-05 |
| 150 | 0.3248 | 49.0578 | 2.2818 | 9.00e-05 |
| 200 | 0.2762 | 29.5071 | 1.8410 | 1.20e-04 |
| 250 | 0.2522 | 24.1199 | 0.8423 | 1.50e-04 |
| **300** | **0.2464** | **17.6830** | 0.2053 | 1.80e-04 |

참고로 kill 직전까지 (게이트와 무관, 추세 확인용):

| step | lm_loss | kl_loss | 비고 |
|---:|---:|---:|---|
| 400 | 0.2476 | 13.9690 | `train/accuracy=0.9264` 첫 관측 |
| 500 | 0.2169 | 12.0058 | warmup 종료, lr=3.00e-04 도달 |
| 650 | 0.1935 | 3.9324 | |
| 800 | 0.0787 | 2.1278 | `train/accuracy=0.9775`, `exact_accuracy=0.1263` |
| 850 | 0.2169 | 1.6245 | kill |

## 코드 경로 판정 (34번 대조와 무관하게 성립하는 부분)

- **NaN 없음.** 셀 Q(`outer_only`)가 step 50 에서 전 손실 NaN 이었던 것과 대비된다 (`RESULT_outer_only_nan_62.md`).
- **warmup 500 정상.** lr 이 3.00e-05 → 3.00e-04 로 선형 상승 후 고정.
- **KL 하강 정상.** 773 → 17.7 (step 300) → 1.62 (step 850).
- **학습 신호 생성됨.** step 800 에서 `train/accuracy=0.9775`.

→ **`fixed` 경로 자체는 굴러간다.** 남은 것은 "34번과 같은 대역인가" 뿐이다.

## ★ 다음 (62번 조치 대기)

1. **34번 GPU5 fixed+3e-4 의 step 50~300 `lm_loss`/`kl_loss` 를 전달해 주십시오.**
   위 표와 대조 후 통과/중단을 판정합니다.
2. 통과 시 즉시 기동할 2셀 (조립 검증 완료, GPU 1~4 유휴 확인됨):

```bash
# 셀 S — 34번 GPU5(seed 0) in-loop 0.8323 의 seed 복제
RUN=nq10_S_fixed_a7525_lr3e4_s1 RESIDUAL_SCALE=fixed ALPHA1=0.75 ALPHA2=0.25 \
NODECAY_MARKED=False LR=3e-4 WARMUP=500 SEED=1 EPOCHS=990 EVALINT=30 GPUS=1,2 PORT=29986 \
WANDB_MODE=offline setsid nohup bash scripts/train_nqueens10_stageA.sh &

# 셀 T — α₂ 를 낮춰 K 루프 중첩 감쇠(α₂^K) 완화
RUN=nq10_T_fixed_a7510_lr3e4_s0 RESIDUAL_SCALE=fixed ALPHA1=0.75 ALPHA2=0.10 \
NODECAY_MARKED=False LR=3e-4 WARMUP=500 SEED=0 EPOCHS=990 EVALINT=30 GPUS=3,4 PORT=29987 \
WANDB_MODE=offline setsid nohup bash scripts/train_nqueens10_stageA.sh &
```

## 단서 — 대조 시 감안할 것

- 이 smoke 는 **2 GPU (world_size=2)** 다. 34번 GPU5 가 1 GPU 였다면 `global_batch_size=768` 은
  같아도 **GRAM 의 노이즈 샘플링 RNG 소비 패턴이 달라** step 단위 비트 일치는 기대할 수 없다.
  "같은 대역" 판정은 자릿수·추세 기준으로 봐야 한다.
- `train/accuracy` 가 대부분 step 에서 0.0000 인 것은 halt 된 시퀀스에서만 집계되기 때문이다.
  이상 신호가 아니다.

## 확인 못 함

- **34번 GPU5 와의 실제 대역 일치 여부** — 기준값 미수령.
- **in-loop 0.8323 재현 여부** — 셀 S 미기동.
