# REPORT_off_line_29.md — OFF 라인 β=2.4 / 2.6 판정 (29번 hinton, GPU 6~7)

- 일시: 2026-07-28
- 서버: **hinton**, user `chaeun`, repo `/home/chaeun/gram-repro`
- 사용 GPU: **6, 7만**. 0~3(ON 연장 6k 학습)·4~5는 미접촉 — 아래 4절에 점유 기록.
- 대상: `nq8_off_b240_ema`(β=2.4), `nq8_off_b260_ema`(β=2.6) — 완주했으나 판정이 안 됐던 2개
- 상태: **Task 1 완료**. `b220_ema6k` 2개는 6절 참조 (본 작업 범위 밖).

**요약**: `.live`(raw) 기준으로 두 run 모두 **곡선 내부에 정점이 잡혔다**.
β=2.4 는 **45.52% @ step 14807**, β=2.6 은 **40.28% @ step 6834**.
β=2.2 의 `.live` 48.66%(경계값, 정점 미관측)와 나란히 놓으면 **2.2 → 2.4 → 2.6 으로
48.66% → 45.52% → 40.28% 단조 하강**이고, 정점이 곡선 내부로 들어온 것도 2.4 부터다.
→ **OFF 라인의 β 최적점은 2.4 보다 왼쪽에 있다.**

in-loop(EMA) 곡선은 두 run 다 마지막 step 이 최댓값이라 정점이 안 보였다. EMA 곡선으로
peak 을 골랐다면 β=2.6 은 실제 `.live` 정점(6834, 40.28%)이 아니라 최저점 부근
(17085, 9.58%)을 집었을 것이다 — **4.2 배 차이**.

---

## 1. 대상 설정 확인

`all_config.yaml` 에서 직접 읽었다.

| run | `arch.kl_coef` | `ema` | `ema_rate` | `y_onehot` | `epochs` | `seed` | checkpoint |
|---|---|---|---|---|---|---|---|
| `nq8_off_b240_ema` | **2.4** | True | 0.9999 | False | 3000 | 0 | `step_*` 15 + `.live` 15 |
| `nq8_off_b260_ema` | **2.6** | True | 0.9999 | False | 3000 | 0 | `step_*` 15 + `.live` 15 |

두 run 모두 전곡선(eval 15회분)이 남아 있어 스캔이 가능했다.

## 2. 방법

`REPORT_audit_measure.md` 4-2 규약대로 **peak step 은 `.live` 곡선에서** 골랐다.
in-loop 곡선은 EMA on 이면 EMA 사본의 곡선이라 peak 선택에 쓸 수 없다.

1. `.live` 15포인트 전곡선 스캔 — `sweep_harness.py`, harness seed 0, N=20,
   test 772 전수, `--sigma-scale 1.0`, run 당 GPU 1장
2. `.live` prior valid 최댓값 step 을 raw peak 으로 확정
3. 그 step 에서 **EMA 가중치(`step_N`)와 `.live` 둘 다** × **harness seed 0/1**

전 job `experiments/sweep_harness.py` (판정 기준 = **valid**). 1 job 실측 3분 49초.

결과 json:
- 스캔: `experiments/results/live_<run>_step_<N>_live_seed0.json` (30개)
- peak: `experiments/results/off29_<run>_step_<N>[_live]_seed<S>.json` (8개)

## 3. `.live` 전곡선 (15포인트, harness seed 0) — prior valid

| step | β=2.4 | β=2.6 |
|---|---|---|
| 1139 | 0.00% | 0.00% |
| 2278 | 0.00% | 0.00% |
| 3417 | 0.00% | 0.01% |
| 4556 | 0.47% | 0.27% |
| 5695 | 1.29% | 2.58% |
| 6834 | 29.79% | **40.28%** |
| 7973 | 22.69% | 29.99% |
| 9112 | 23.92% | 19.26% |
| 10251 | 13.64% | 31.84% |
| 11390 | 12.89% | 11.04% |
| 12529 | 26.34% | 19.57% |
| 13668 | 35.44% | 20.21% |
| 14807 | **45.52%** | 14.98% |
| 15946 | 36.81% | 10.91% |
| 17085 | 42.50% | 9.58% |
| **`.live` peak** | **45.52% @ 14807 (내부 정점)** | **40.28% @ 6834 (내부 정점)** |

β=2.6 은 6834 이후 9.58% 까지 떨어져 **하강 구간이 뚜렷하다**. β=2.4 는 14807 에서 꺾이지만
마지막 17085(42.50%)가 15946(36.81%)보다 높아 곡선이 평평하고 진동한다.

### 3-1. in-loop(EMA) 곡선과의 불일치

`logs/<run>.log` 의 `nqueens/accuracy`(=valid):

| step | β=2.4 EMA | β=2.6 EMA |
|---|---|---|
| 13668 | 17.10% | 27.46% |
| 14807 | 24.48% | 33.68% |
| 15946 | 31.87% | 31.87% |
| 17085 | **32.90%** | **34.72%** |

두 run 다 **마지막 step 이 최댓값** = EMA 기준으로는 정점 미관측. `.live` 정점(14807 / 6834)과
어긋난다. 규약이 실제로 값을 잡은 사례다.

## 4. peak step harness — EMA vs `.live`, seed 0 / 1

N=20, test 772 전수, `--sigma-scale 1.0`.

| β | step | 가중치 | prior valid s0 / s1 | 평균 | posterior valid | pass@20 | majority@20 | verifier_sel@20 | distinct@20 | queens mean (min–max) |
|---|---|---|---|---|---|---|---|---|---|---|
| **2.4** | 14807 | **`.live`** | 45.52 / 44.68% | **45.10%** | 100.00 / 100.00% | 98.96 / 98.96% | 71.76 / 70.34% | 49.35 / 47.41% | 8.26 / 8.36 | 7.98 (5–11) |
| 2.4 | 14807 | EMA | 24.22 / 23.82% | 24.02% | 93.78 / 93.78% | 72.67 / 73.70% | 35.10 / 34.33% | 19.69 / 25.00% | 8.46 / 8.50 | 7.97 (6–10) |
| **2.6** | 6834 | **`.live`** | 40.28 / 41.58% | **40.93%** | 99.61 / 99.22% | 98.70 / 98.32% | 71.50 / 74.61% | 38.99 / 39.51% | 9.35 / 9.21 | 7.88 (6–10) |
| 2.6 | 6834 | EMA | 0.00 / 0.00% | **0.00%** | 0.00 / 0.00% | 0.00 / 0.00% | 0.00 / 0.00% | 0.00 / 0.00% | 1.00 / 1.00 | 2.78 (1–3) |

**harness seed 편차**: `.live` 는 0.84 %p(2.4) / 1.30 %p(2.6). EMA 는 0.40 %p(2.4) / 0(2.6).
2 seed 뿐이라 편차 폭으로만 적는다.

**재현성 대조**: peak step 의 `.live` seed 0 을 스캔과 별도 job 으로 다시 쟀고
**45.52% / 40.28% 로 스캔값과 정확히 일치**했다(부동소수 완전 일치).

### 4-1. β=2.6 의 EMA 는 step 6834 에서 붕괴해 있다

prior/posterior 모두 valid 0.00%, `distinct@20 = 1.00`(20 샘플 전부 동일한 보드),
queen 수 2.78 (1–3). 같은 step 의 in-loop EMA 로그가 `nqueens/accuracy=0.0000`,
`nqueens/queen_count_mean=2.7772` 로 **harness 측정과 일치**한다 — 측정 오류가 아니라
그 step 의 EMA 가중치 자체가 퇴화 상태다.

산술적 사실 하나: `ema_rate=0.9999` 에서 `mu^t` 는 step 6834 에 **0.5049** 다. 즉 shadow 의
50.5% 가 아직 init 가중치이고, `models/ema.py` 의 bias correction 은 분모 0.4951 로 나눈다.
step 14807 은 0.2275 / 0.7725, 17085 는 0.1811 / 0.8189.
→ β=2.6 의 `.live` 정점이 초기 step 에 있다는 사실과, 그 step 의 EMA 가 쓸 수 없다는 사실이
겹쳤다. **다만 이 산술이 붕괴의 원인이라고 단정하지 않는다** — 인과는 측정하지 않았다.
(`REPORT_audit_measure.md` 5절도 EMA/`.live` 괴리의 원인은 미규명으로 남겨두었다.)

### 4-2. EMA vs `.live` 의 우열이 또 뒤집힌다

| 조건 | EMA | `.live` | 방향 |
|---|---|---|---|
| 2.2 OFF @ 17085 (기존, `REPORT_ema_seed_29.md` 4절) | **63.12%** | 48.66% | EMA 우세 |
| 2.4 OFF @ 14807 (이번) | 24.02% | **45.10%** | `.live` 가 1.9 배 |
| 2.6 OFF @ 6834 (이번) | 0.00% | **40.93%** | `.live` 만 유효 |

`REPORT_ema_seed_29.md` 1절의 "조건에 따라 반대로 뒤집힌다" 가 OFF 라인 안에서도 재현됐다.

## 5. 판정

`.live`(raw) 축 기준. 비교군 수치는 각 리포트에서 그대로 인용했다.

| β | run | 가중치 | prior valid | peak step | 정점 위치 |
|---|---|---|---|---|---|
| 2.2 | `nq8_off_b220` (EMA off) | raw(`step_N`) | **56.15%** | 17085 | 경계 — 미관측 |
| 2.2 | `nq8_off_b220_ema_s0` | `.live` | 48.66% | 17085 | 경계 — 미관측 |
| **2.4** | `nq8_off_b240_ema` | `.live` | **45.10%** (s0/s1 평균) | **14807** | **내부** |
| **2.6** | `nq8_off_b260_ema` | `.live` | **40.93%** (s0/s1 평균) | **6834** | **내부** |

1. **β=2.4, 2.6 은 둘 다 β=2.2 보다 낮다.** 48.66% → 45.10% → 40.93% 단조 하강.
   (β=2.2 는 EMA on run 의 `.live`, 즉 같은 축이다. EMA off run 의 raw 56.15% 와는 run 이 달라
   직접 비교하지 않는다.)
2. **정점이 곡선 내부로 들어온 것이 β=2.4 부터다.** β≤2.2 는 전부 마지막 step 이 최댓값이었고
   (`REPORT_sweep_29.md` 요약), 2.4/2.6 은 내부 정점 + 하강 구간을 갖는다.
   즉 **β 를 올리면 정점이 앞당겨지면서 값도 낮아진다.**
3. → **OFF 라인의 최적 β 는 2.4 미만이다.** 이 두 점은 상한 쪽 경계를 확정해주지만
   최적점 자체는 감싸지 못한다. 최적점 특정에는 **β < 2.2 구간**(=왼쪽 대역)의 측정이 필요하다.
4. **posterior 는 β 와 무관하게 거의 포화**다(`.live` 기준 99.2~100%). 갈리는 것은 prior 뿐이다.

## 6. `nq8_off_b220_ema6k` 2개 — 본 작업 범위 밖

지시상 "완주 후 추가" 대상이다. 작업 중 상태 변화가 있어 사실만 기록한다.

- 두 run 다 **완주**했다: `step_*` 30개(6000 epoch), wandb sync 완료, 진행바 34170/34187.
- 작업 도중 **GPU 4·5 에서 다른 프로세스가** 이 두 run 의 `sweep_harness.py` 스캔을
  시작했다(pid 3706676 / 3706740, `--ckpt-dir ...b220_ema6k_s0 / _s1`). 내가 띄운 job 이 아니다.
  `experiments/results/live_nq8_off_b220_ema6k_s*_step_*_live_seed0.json` 이 쌓이는 중이다.
- 지시대로 GPU 6~7 만 쓰므로 **이 스캔에 개입하지 않았고 결과도 넣지 않았다.**
  해당 스캔이 끝나면 3~5절 표에 같은 형식으로 추가하면 된다.

## 7. GPU 점유 기록

| GPU | 내용 | 내 작업? |
|---|---|---|
| 0~3 | `pretrain.py` 4 프로세스, 각 14166 MiB (ON 연장 6k) | ❌ 조회만 |
| 4~5 | 작업 시작 시 14166 MiB(OFF 2.2 6k 학습) → 완주 후 타 프로세스의 b220_ema6k harness | ❌ 조회만 |
| **6** | `nq8_off_b240_ema` 스캔 15 + peak 4 job | ✅ |
| **7** | `nq8_off_b260_ema` 스캔 15 + peak 4 job | ✅ |

launch 직전 `nvidia-smi` 로 6·7 이 각 **1 MiB / 0%** 임을 확인하고 시작했다.

## 8. 하지 않은 것 / 확인 못 한 것

1. **왼쪽 대역(β<2.2) 측정을 하지 않았다.** 지시 제목에 "OFF 왼쪽 대역"이 있으나
   본문에 Task 1 만 명세돼 있었다. 어떤 β 를 몇 개, 어느 GPU 로 돌릴지 지정이 없어
   임의로 정하지 않았다. **지시 대기.**
2. **EMA 붕괴(4-1)의 인과를 규명하지 않았다.** `mu^t` 산술은 사실이지만 그것이 원인이라는
   검증은 하지 않았다.
3. **harness seed 2개뿐**이다. 편차 폭만 적었고 분산 추정은 하지 않았다.
4. **run 반복 1회씩**이다. β=2.4 / 2.6 각각 단일 run 이라 run 간 변동을 모른다
   (β=1.8 ON 에서 seed 간 0.38% ~ 29.71% 로 갈린 전례가 있다 — `REPORT_ema_seed_29.md`).
5. **`.live` 스캔은 harness seed 0 만**이다(15포인트 × 2 seed × 2 run = 60 job 은 과하다고 판단).
   peak step 에서만 seed 0/1 을 쟀다.
6. **`sweep_harness.py` 기타 파라미터는 기본값** (`--sigma-scale 1.0`, `--n 8`,
   `--data data/nqueens-8/test`, `--N 20`).

## 9. 재현 커맨드

```bash
# .live 전곡선 스캔 (run 당 GPU 1장)
for s in 1139 2278 3417 4556 5695 6834 7973 9112 10251 11390 12529 13668 14807 15946 17085; do
  CUDA_VISIBLE_DEVICES=6 python experiments/sweep_harness.py \
    --ckpt-dir checkpoints/Nqueens-8-ACT-torch/nq8_off_b240_ema --step step_${s}.live \
    --seed 0 --out experiments/results/live_nq8_off_b240_ema_step_${s}_live_seed0.json
done   # β=2.6 은 GPU 7, run 이름만 교체

# peak step harness (EMA + .live, seed 0/1)
for W in "" ".live"; do for S in 0 1; do
  SUF=$([ -z "$W" ] && echo "" || echo "_live")
  CUDA_VISIBLE_DEVICES=6 python experiments/sweep_harness.py \
    --ckpt-dir checkpoints/Nqueens-8-ACT-torch/nq8_off_b240_ema --step step_14807${W} \
    --seed $S --out experiments/results/off29_nq8_off_b240_ema_step_14807${SUF}_seed${S}.json
done; done   # β=2.6 은 step 6834, GPU 7
```
