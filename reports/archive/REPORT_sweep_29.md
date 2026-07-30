# REPORT_sweep_29.md — N-Queens 8×8 β sweep, one-hot **OFF** (29번 hinton, GPU 0~3)

- 일시: 2026-07-28
- 서버: **hinton** (`hostname` = `hinton`), user `chaeun`, repo `/home/chaeun/gram-repro`
- env: conda `gram` (`~/miniconda3/bin/activate gram`)
- 사용 GPU: **0, 1, 2, 3** (4~7은 조회만, 미사용)
- 목적: one-hot OFF 조건에서 **순수 β 효과**로 prior valid의 peak β를 촘촘히 특정
- 상태: **완료** (Task 0 / 1 / 2 / 3 전부 완료, 4-run 완주 + peak checkpoint harness 완료)

**요약**: prior valid 는 측정한 4점(β=1.0/1.4/1.8/2.2)에서 **단조 증가**했고
(0.61% → 0.73% → 26.48% → 56.15%), **최댓값이 경계인 β=2.2 에 있다**.
β=1.8/2.2 는 in-loop 곡선의 peak도 마지막 step(17085)이라 **하강 구간이 관측되지 않았다**.
→ 이 sweep은 peak β를 감싸지 못했다. peak 특정에는 β>2.2 구간이 필요하다.

---

## 1. Task 0 — 환경 확인

지시한 4개 항목 전부 확인.

| 항목 | 커맨드 | 결과 | 기대 | 판정 |
|---|---|---|---|---|
| CUDA 장치 수 | `python -c "import torch;print(torch.cuda.device_count())"` | `8` | 8 | ✅ |
| 데이터 | `ls data/nqueens-8/train/` | 아래 6개 파일 존재 | 존재 | ✅ (재생성 불필요) |
| config one-hot 필드 | `grep -c y_onehot config/arch/gram_nqueens8.yaml` | `3` | 3 | ✅ |
| GPU 유휴 | `nvidia-smi` | 0~3 전부 `1 MiB` | >1000MiB면 skip | ✅ skip 대상 없음 |

### 데이터
```
data/nqueens-8/train/  →  all__inputs.npy  all__labels.npy  all__puzzle_identifiers.npy
                          all__puzzle_indices.npy  all__group_indices.npy  dataset.json
```
`dataset.json`: `vocab_size=3, seq_len=64, num_puzzle_identifiers=1, total_groups=4376,
total_puzzles=7194, mean_puzzle_examples=1.0`. `inputs`/`labels` 둘 다 `(7194, 64)`.
→ 데이터 존재하므로 `build_nqueens_dataset.py` 재생성은 하지 않았다.

### config one-hot 필드 (3개)
```
config/arch/gram_nqueens8.yaml:39: y_onehot: False
config/arch/gram_nqueens8.yaml:40: y_onehot_proj: zeropad
config/arch/gram_nqueens8.yaml:41: y_onehot_scale: 1.0
```

### GPU (launch 직전)
```
0, 1 MiB    1, 1 MiB    2, 1 MiB    3, 1 MiB     <- 전부 유휴, 4-run 그대로 사용
```
4~7번도 유휴였으나 **지시대로 0~3만 사용**했다.

### launch 전 추가 점검 (지시 항목 외, launch 리스크 제거용)

- **run_name 중복 없음**: 이 박스에는 `checkpoints/` 디렉터리 자체가 없었다(이 repo에서의 첫 학습).
  `nq8_off_b100/140/180/220` 4개 모두 신규.
- **로그 `tee` (-a 아님)**: `train_nqueens8_fix.sh` 마지막 줄이 `| tee logs/${RUN}.log`. append 아님. ✅
- **conda 활성**: sh 상단은 `source ~/venv-gram/bin/activate 2>/dev/null || true` (bengio venv, 이 박스엔 없음 → 무시됨).
  그래서 지시대로 매 실행 앞에 `source ~/miniconda3/bin/activate gram &&` 를 붙여 launch했다.
- **GPU preflight**: sh L53-56 이 이미 해당 GPU >1000MiB면 `FATAL` 로 중단한다. 4-run 모두 preflight 통과.
- **dry-run 조립 확인** (`DRYRUN=1`, run1):
  ```
  arch.kl_coef=1.0  arch.kl_reduction=mean  arch.noise_head_expansion=1.5
  arch.y_onehot=False  arch.y_onehot_proj=zeropad  arch.y_onehot_scale=0.573
  ema=False  ema_rate=0.9999  lr_warmup_steps=500  +run_name=nq8_off_b100
  epochs=3000  eval_interval=200  global_batch_size=768  evaluators=[{name: nqueens@NQueens}]
  ```
- **peak checkpoint 확보 가능 여부** (Task 3 전제): `config/cfg_pretrain.yaml:22` 가
  `checkpoint_every_eval: True` 이다(`pretrain.py:79` 의 dataclass 기본값 `False` 를 덮어씀).
  따라서 eval 15회마다 checkpoint가 남고 **peak checkpoint에 harness를 걸 수 있다**. 확인 완료.
- **wandb**: sh가 `WANDB_MODE=online` 을 강제한다. 이 박스에 자격증명이 있는지 미리 확인했고
  (`/home/chaeun/.netrc`) 정상 동작한다. `save_code_and_config()` 는 `wandb.run is None` 이면
  `all_config.yaml` 을 안 남기므로(`pretrain.py:516-517`), 이게 죽으면 Task 3 harness가 불가능했다.
  → 정상이므로 그대로 online 유지.

---

## 2. Task 1 — launch

`tmux` 세션 4개로 분리(창 분리), 각 run 독립 GPU / 독립 포트.

| run_name | β (`arch.kl_coef`) | GPU | PORT | tmux 세션 |
|---|---|---|---|---|
| `nq8_off_b100` | 1.0 | 0 | 29701 | `nq8_b100` |
| `nq8_off_b140` | 1.4 | 1 | 29702 | `nq8_b140` |
| `nq8_off_b180` | 1.8 | 2 | 29703 | `nq8_b180` |
| `nq8_off_b220` | 2.2 | 3 | 29704 | `nq8_b220` |

실제 커맨드(예: run1):
```bash
tmux new-session -d -s nq8_b100 \
  "source ~/miniconda3/bin/activate gram && \
   RUN=nq8_off_b100 YONEHOT=False BETA=1.0 EMA=False GPUS=0 PORT=29701 bash train_nqueens8_fix.sh"
```

- 4개 run 모두 preflight 통과 후 학습 시작, `[Rank 0, World Size 1]: Epoch 0` 진입 확인.
- 총 step 수 `17093` (= 3000 epoch × 4376 group / 768 batch), eval 15회 (200 epoch마다 ≈ 1139 step).
- 학습 시작 직후 실측 속도 ≈ 2.2 it/s → **run당 ETA ≈ 2h07m** (지시의 ~2h10m과 일치).
- 4-run 동시 실행 시 GPU당 14,070 MiB / util 98~100%.

---

## 3. Task 2 — launch 후 검증

### 3-1. config dump (`checkpoints/Nqueens-8-ACT-torch/<run>/all_config.yaml`)

지시한 4개 확인 항목 전부 일치.

| 항목 | b100 | b140 | b180 | b220 | 기대 | 판정 |
|---|---|---|---|---|---|---|
| `arch.y_onehot` | `false` | `false` | `false` | `false` | false | ✅ |
| `arch.kl_coef` | **1.0** | **1.4** | **1.8** | **2.2** | 1.0/1.4/1.8/2.2 | ✅ |
| `ema` | `false` | `false` | `false` | `false` | false | ✅ |
| `num_params` | **9966085** | **9966085** | **9966085** | **9966085** | 9966085 | ✅ |

동일하게 확인한 공통 파라미터 (4-run 전부 같은 값):
```
arch.kl_reduction: mean          arch.noise_head_expansion: 1.5
arch.H_cycles: 3                 arch.L_cycles: 4
arch.L_layers: 2                 arch.halt_max_steps: 16
arch.y_onehot_proj: zeropad      arch.y_onehot_scale: 0.573
ema_rate: 0.9999                 lr_warmup_steps: 500
epochs: 3000                     eval_interval: 200
global_batch_size: 768           seed: 0
checkpoint_every_eval: true
```

`arch.y_onehot_scale=0.573` 은 dump에 남지만 **이번 arm에서는 코드상 쓰이지 않는다**
(`gram.py` 의 scale 곱이 `y_onehot` 분기 안에만 있음, `y_onehot=False` 이므로 미진입).

`num_params` 산출: `pretrain.py:624` 가 이 값을 wandb에만 기록하고 stdout엔 안 찍는다.
그래서 각 run의 `all_config.yaml` 을 그대로 로드해 모델을 직접 생성하고
`sum(x.numel() for x in model.parameters())` 로 셌다 (4-run 전부 9,966,085 — 기대치와 정확히 일치).

### 3-2. 학습 상태 검증 (NaN / nqueens accuracy / eval kl_loss)

첫 eval은 **step 1139** 이다. sh의 `eval_interval=200` 은 epoch 단위이고
4376 group / 768 batch ≈ 5.7 step/epoch 이므로 200 epoch ≈ 1139 step.
(지시의 "200 step 검증"을 첫 eval 지점 검증으로 읽고 그 시점에 확인했다.)

| 검증 항목 | b100 | b140 | b180 | b220 | 판정 |
|---|---|---|---|---|---|
| NaN / inf 출현 | 0건 | 0건 | 0건 | 0건 | ✅ 없음 |
| `nqueens/accuracy` 로그 출력 | 찍힘 | 찍힘 | 찍힘 | 찍힘 | ✅ |
| eval `kl_loss` | **0.0000** | **0.0000** | **0.0000** | **0.0000** | ✅ = 0.0 |
| Traceback / OOM / FATAL | 없음 | 없음 | 없음 | 없음 | ✅ |
| checkpoint 저장 | `step_1139` | `step_1139` | `step_1139` | `step_1139` | ✅ |

→ **4-run 모두 정상. kill 대상 없음.**

step 1139 eval 원문:
```
b100 (β=1.0)  all/accuracy=0.8325 all/kl_loss=0.0000 all/lm_loss=0.5331 all/q_halt_accuracy=1.0000 all/steps=16.0000
              nqueens/accuracy=0.0000 nqueens/consistent_rate=1.0000 nqueens/queen_count_mean=10.5933 nqueens/strict_accuracy=0.0000 n_eval=772
b140 (β=1.4)  all/accuracy=0.9184 all/kl_loss=0.0000 all/lm_loss=0.2663 all/q_halt_accuracy=1.0000 all/steps=16.0000
              nqueens/accuracy=0.0000 nqueens/consistent_rate=1.0000 nqueens/queen_count_mean=2.7772 nqueens/strict_accuracy=0.0000 n_eval=772
b180 (β=1.8)  all/accuracy=0.9184 all/kl_loss=0.0000 all/lm_loss=0.3001 all/q_halt_accuracy=1.0000 all/steps=16.0000
              nqueens/accuracy=0.0000 nqueens/consistent_rate=1.0000 nqueens/queen_count_mean=2.7772 nqueens/strict_accuracy=0.0000 n_eval=772
b220 (β=2.2)  all/accuracy=0.8750 all/kl_loss=0.0000 all/lm_loss=0.3826 all/q_halt_accuracy=1.0000 all/steps=16.0000
              nqueens/accuracy=0.0000 nqueens/consistent_rate=0.0000 nqueens/queen_count_mean=0.0000 nqueens/strict_accuracy=0.0000 n_eval=772
```
step 1139 은 전체 17093 step 중 6.7% 지점(15회 eval 중 1회차)이라 `nqueens/accuracy` 가
4-run 전부 0.0000 인 것은 이 시점에서 판단 근거가 되지 않는다. 여기서 본 것은
"지표가 정상적으로 계산·출력되는가"이고, 곡선 해석은 4절에서 한다.

---

## 4. Task 3 — 완주 후 결과

4-run 전부 완주. 각 15회 eval / 15개 checkpoint, NaN·Traceback·OOM 0건, GPU 0~3 반납 확인.

### 4-1. in-loop `nqueens/accuracy` 곡선 (eval 15포인트)

| step | β=1.0 | β=1.4 | β=1.8 | β=2.2 |
|---|---|---|---|---|
| 1139 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2278 | 0.0065 | 0.0039 | 0.0013 | 0.0013 |
| 3417 | **0.0091** | **0.0078** | 0.0000 | 0.0272 |
| 4556 | 0.0091 | 0.0013 | 0.0544 | 0.1710 |
| 5695 | 0.0000 | 0.0026 | 0.1477 | 0.3368 |
| 6834 | 0.0000 | 0.0000 | 0.0622 | 0.0984 |
| 7973 | 0.0078 | 0.0000 | 0.0725 | 0.1580 |
| 9112 | 0.0091 | 0.0000 | 0.1477 | 0.3005 |
| 10251 | 0.0065 | 0.0000 | 0.0440 | 0.2474 |
| 11390 | 0.0026 | 0.0013 | 0.1101 | 0.2720 |
| 12529 | 0.0065 | 0.0000 | 0.0570 | 0.5026 |
| 13668 | 0.0000 | 0.0000 | 0.0829 | 0.3368 |
| 14807 | 0.0065 | 0.0026 | 0.1451 | 0.2863 |
| 15946 | 0.0013 | 0.0000 | 0.1412 | 0.1528 |
| 17085 | 0.0013 | 0.0000 | **0.2604** | **0.5479** |

**하강 시작점 (early stop용): 관측되지 않았다.**

- β=1.8, β=2.2 는 **peak가 마지막 step(17085)** 이다. 곡선 진폭이 크지만(β=2.2 는
  12529의 0.5026 다음 13668에서 0.3368, 15946에서 0.1528로 내려갔다가 17085에서 0.5479)
  추세가 끝까지 상승이고 **지속적인 하강 구간이 없다**. 이 4개 run에서는 early stop 근거가 안 나온다.
- β=1.0, β=1.4 는 전 구간 ≤0.0091 / ≤0.0078 로 사실상 바닥이다. 이 두 run의 "peak step"
  (둘 다 3417)은 **바닥 노이즈 중 최댓값**이지 의미 있는 peak가 아니다.
  β=1.0 은 0.0091 이 step 3417·4556·9112 세 곳에서 동률이라 그중 가장 이른 3417을 잡았다.
- **이 sweep은 peak β를 위에서 감싸지 못했다.** β=2.2 가 상한이 아니라 아직 오르는 중이다.

### 4-2. peak checkpoint harness

`experiments/sweep_harness.py` (valid 기준, N=20, seed=0, sigma_scale=1.0, test set 772 퍼즐).
결과 json: `experiments/results/sweep_nq8_off_<run>_<step>.json`

| β | peak step | prior valid (N=20 전체) | prior single k=0 | consistent | strict | **posterior valid** | queens mean (min–max) |
|---|---|---|---|---|---|---|---|
| 1.0 | 3417 | 0.61% | 0.91% | 98.85% | 0.61% | 60.62% | 7.71 (6–11) |
| 1.4 | 3417 | 0.73% | 1.04% | 98.78% | 0.73% | 75.52% | 7.84 (5–11) |
| 1.8 | 17085 | **26.48%** | 27.07% | 60.46% | 19.31% | **100.00%** | 8.01 (6–10) |
| 2.2 | 17085 | **56.15%** | 58.29% | 98.43% | 55.69% | **100.00%** | 7.92 (4–14) |

selection 지표 (전부 valid 기준):

| β | peak step | pass@1 | pass@5 | pass@10 | **pass@20** | majority@20 | verifier_sel@20 | **distinct@20** |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 3417 | 0.6% | 2.9% | 5.5% | 10.1% | 0.8% | 0.9% | 18.18 |
| 1.4 | 3417 | 0.7% | 3.5% | 6.7% | 12.2% | 1.9% | 1.0% | 19.33 |
| 1.8 | 17085 | 26.5% | 75.0% | 92.3% | **98.83%** | 53.1% | 26.8% | 14.11 |
| 2.2 | 17085 | 56.2% | 93.3% | 98.3% | **99.87%** | 88.5% | 57.1% | 7.65 |

posterior 경로는 β=1.8/2.2에서 valid=consistent=strict=100.00%, queens 8.00 (8–8) 로 완전히 붙었다.
β=1.0/1.4 는 posterior도 60.62% / 75.52% 로 낮다.

---

## 5. β-정확도 관계 (수치)

**prior valid (peak checkpoint, harness N=20 전체 평균)**
```
β=1.0  →   0.61%
β=1.4  →   0.73%
β=1.8  →  26.48%
β=2.2  →  56.15%
```
측정한 4점 전 구간에서 **β에 대해 단조 증가**. 1.0→1.4 는 +0.12%p 로 거의 평평하고,
1.4→1.8 에서 +25.75%p 로 급격히 꺾인 뒤 1.8→2.2 에서 다시 +29.67%p 오른다.
**측정 범위 안에 최댓값이 없다** (경계인 β=2.2 가 최대).

**in-loop peak accuracy**: 0.0091 → 0.0078 → 0.2604 → 0.5479. 같은 형태.
(harness single k=0 값 0.91% / 1.04% / 27.07% / 58.29% 와 자릿수 일치. 두 지표는
샘플링 RNG 스트림이 달라 정확히 같지는 않다.)

**pass@20**: 10.1% → 12.2% → 98.8% → 99.9%. β=1.8에서 이미 포화에 가깝다.

**distinct@20**: 18.18 → 19.33 → 14.11 → 7.65. prior valid와 반대로 β=1.4 이후 **감소**한다.
β=2.2 는 20샘플 중 서로 다른 보드가 평균 7.65개다.

**posterior valid**: 60.62% → 75.52% → 100.00% → 100.00%. β와 함께 증가하며 1.8에서 포화.

### 5-1. 브리핑 기준값과의 비교 가능성 (설정 차이, 사실만)

브리핑의 "β=1.5에서 0.29"와 이번 축은 **같은 눈금이 아니다**. sh를 직접 확인한 차이:

| 스크립트 | `arch.kl_reduction` | `ema` | 비고 |
|---|---|---|---|
| `gram_nq8_sweep.sh` (REPORT_klsweep, β=1.5 → prior valid 30.29% 출처) | **sum** | **True** | |
| `train_nqueens8_1.5.sh`, `train_nqueens8.sh` | mean | **True** | |
| `train_nqueens8_fix.sh` (**이번 sweep**, 지시대로) | mean | **False** | |

`kl_reduction` 이 sum이냐 mean이냐에 따라 같은 `kl_coef` 숫자의 실효 β가 달라진다.
이번 β=1.0/1.4 가 바닥(≤0.9%)인 것과 klsweep β=1.5 의 30.29% 가 충돌하는 것처럼 보이는 데에는
이 축 차이와 EMA 유무가 함께 걸려 있고, **이 4개 run만으로는 둘 중 무엇이 원인인지 분리되지 않는다.**
(50번의 one-hot ON 짝 비교도 `train_nqueens8_fix.sh` 기준이면 이번과 같은 축이다.)

### 5-2. 다음 sweep 제안 (수치 근거)

peak가 측정 범위 밖이므로 **β > 2.2 구간**을 열어야 peak를 특정할 수 있다.
β=1.8/2.2 는 17085 step에서도 상승 중이라 **epoch 연장 시 더 오를 여지**도 남아 있다.
seed 노이즈가 커서(β=2.2 곡선이 15946에서 0.1528 → 17085에서 0.5479) 단일 seed 곡선의
step별 값은 신뢰구간이 넓다. 단정하지 않는다.
