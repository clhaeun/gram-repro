# REPORT_final_round_29.md — 최종 라운드(pair 데이터) 29번 몫

- 작업일: 2026-08-02 ~ 08-03, 29번(hinton).
- 데이터: **pair 확장본** `data/nqueens-8/` (train 7,194행 / test **1,270행** / 772 unique 입력).
  `-uniq` 아님. 빌드 근거는 [REPORT_pair_expand_29.md](REPORT_pair_expand_29.md).
- 이 라운드가 **최종 보고용**이다. 중간 설정 변경 없음.

---

## 1. 학습 — 29번 4건

### 1-1. 공통 config (`all_config.yaml` 실측)

`kl_coef: 0.07` / `kl_reduction: sum` / `logvar_bias_init: 0.0` / `y_onehot: true` /
`ema: true` / `ema_rate: 0.9999` / `dual_eval: true` / `lr_min_ratio: 1.0` /
`nodecay_logvar_bias: false` / `epochs: 7200` / `eval_interval: 200` /
`global_batch_size: 768` / `lr_warmup_steps: 500`

### 1-2. ★ EPOCHS 를 7299 → 7200 으로 고쳤다

`pretrain.py:657` 에 하드 assert 가 있다:
```python
assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."
```
`eval_interval=200` 이 `train_nqueens8_v2.sh:90` 에 하드코딩이라 7299 는 **즉시 죽는다**
(7299 ÷ 200 = 36.495). 200 의 배수 중 최근접이 7200 이고, 이건 `catnd_s2` 도 나머지 3건과
같은 값이어야 3-seed mean±std 가 성립하기 때문에 선택의 여지가 없었다.

| | EPOCHS | eval | **step** | 기존 uniq 68,375 대비 |
|---|---|---|---|---|
| 채택 | **7200** | 36 | **67,443** (최종 eval 67,428) | −932 (−1.4%) |
| 지시값 | 7299 | 36.49 | — | assert 실패로 실행 불가 |
| 대안 | 7400 | 37 | 69,317 | +942 (더 멂) |

### 1-3. launch 기록

launch 전 `nvidia-smi` 점유 확인 필수 규율을 지켰다. 1차 3건은 2026-08-02 13:29:06 에
GPU 5/6/7 각 **1 MiB** 확인 후, `catnd_s2` 는 08-03 00:16:53 에 GPU 4 **1 MiB** 확인 후 launch.
tmux 세션 `final29`, 창 분리, 포트 중복 없음, `tee`(`-a` 없음), run_name 중복 0건.

| run | GPU | PORT | SEED | y_inject | nodecay_y_proj | 소요 | eval | 상태 |
|---|---|---|---|---|---|---|---|---|
| `nq8_final_add_s2` | 5 | 29631 | 2 | add | false | 8:33:02 | 36/36 | 완주 |
| `nq8_final_catnd_s0` | 6 | 29632 | 0 | concat | **true** | 9:27:06 | 36/36 | 완주 |
| `nq8_final_catnd_s1` | 7 | 29633 | 1 | concat | **true** | 8:43:39 | 36/36 | 완주 |
| `nq8_final_catnd_s2` | 4 | 29931 | 2 | concat | **true** | 진행중 | — | 08-03 00:17 launch |

3건 모두 `wandb: Find logs at:` 로 정상 종료, checkpoint 72개(36 eval × {EMA, .live}),
최종 `step_67428` + `step_67428.live` 존재. 에러·traceback **0건**.

**첫 eval 이 `nqueens/accuracy=0.0000` 인 것은 정상이다.** `all/accuracy=0.8750` 은 정확히
56/64 = 빈칸 비율이고 `queen_count_mean=0.0000` 이라 전량 빈 보드를 내놓는 초기 퇴화 상태다.
과거 run 도 같았고(`nq8_v5_add_b0_b500_s0` 는 step 5,695 까지 0.0000 이었고 최종 99.29%),
late-riser 대역이라 조기 kill 하지 않는다.

**★ `nqueens/n_eval=1270.0000`** — pair 확장 test 가 실제로 물렸다는 실측 증거.
uniq run 의 772 와 축이 다르므로 기존 수치와 직접 비교 불가.

---

## 2. ★ 관측 기록 — concat arm 의 최종 step 불안정 (in-loop, 판정 아님)

**아직 판정이 아니다.** 모든 판정은 `.live` 기준·harness 축이고 그건 §3 이다. 여기 적는 것은
harness 에서 재현되는지 대조하기 위한 관측 원자료다.

| run | 최종 EMA | 최종 raw(.live) | EMA 정점 | raw 정점 |
|---|---|---|---|---|
| `add_s2` | **98.90%** | 97.95% | 98.90% @ **67428 (=최종)** | 97.95% @ **67428 (=최종)** |
| `catnd_s0` | **72.20%** | 92.28% | 94.33% @ 50571 | 94.80% @ 44952 |
| `catnd_s1` | **97.72%** | 58.43% | 98.90% @ 58063 | 98.82% @ 63682 |

**두 concat seed 의 붕괴 축이 서로 반대다.**
- `catnd_s0`: **EMA 가 무너진다** — 94.33% (50571) → 72.20% (67428), −22.1%p.
  같은 구간에서 raw 는 92.28% 로 버틴다.
- `catnd_s1`: **raw 가 무너진다** — 98.82% (63682) → 58.43% (67428), −40.4%p.
  같은 구간에서 EMA 는 97.72% 로 버틴다.
- `add_s2`: EMA·raw 둘 다 정점 == 최종. 붕괴 없음.

즉 concat 은 "EMA 가 나쁘다" 도 "raw 가 나쁘다" 도 아니고, **어느 쪽이 무너질지가 seed 마다
다르다**. 이게 harness 축에서도 재현되면 "concat 은 최종 step 이 불안정하다" 가 arm 판정의
근거가 된다. `catnd_s2` 완주 후 3-seed 로 다시 본다.

원문 (`logs/nq8_final_catnd_s1.log` 최종 eval):
```
[EVAL @ step 67428] ... nqueens/accuracy=0.9772 nqueens/consistent_rate=1.0000 nqueens/n_eval=1270.0000 nqueens/queen_count_mean=7.9843 nqueens/strict_accuracy=0.9772
[EVAL RAW @ step 67428] ... nqueens/accuracy_raw=0.5843 nqueens/consistent_rate_raw=0.9575 nqueens/n_eval_raw=1270.0000 nqueens/queen_count_mean_raw=7.8787 nqueens/strict_accuracy_raw=0.5835
```

---

## 3. harness (진행중)

`experiments/run_final_harness.sh`, 28건, GPU 5/6/7 shard 3분할, 08-03 00:19 시작.
데이터는 pair test(`data/nqueens-8/test`, 1,270행), sidecar 는 `sidecar_8.json`
(uniq 시절과 **byte 동일** — `def8aac8f11ac931e7004cac505d509d`).

| run | step | 비고 |
|---|---|---|
| `add_s2` | 67428 | 정점 == 최종이라 1개 |
| `catnd_s0` | 67428 / 50571(EMA정점) / 44952(raw정점) | |
| `catnd_s1` | 67428 / 58063(EMA정점) / 63682(raw정점) | |

각 (run, step) × {EMA, .live} × harness seed {0,1}. 헤드라인은 **최종 step × EMA**
(선택 효과 없음, 논문 축). 결과 채워지는 대로 갱신.

### 3-1. coverage 두 축 — `sweep_harness.py` 에 행 단위 원자료를 추가했다

pair 확장으로 test 에 같은 입력이 k 번 들어가면서 `coverage_at_20.coverage` 한 값이 어느
축인지 애매해졌다. 계산해 보면

    mean_row(found/total) = (1/R)·Σ_x k·(found_x / k) = Σ_x found_x / R  =  **pair 축**

이라 기존 `coverage` 키가 그대로 **pair 가중**이다(키 이름 안 바꿨다 — 개명 금지).
입력(퍼즐) 축 `mean_x(found_x/total_x)` 은 같은 x 의 k 개 행을 먼저 묶어야 나오는데
집계값만으로는 복원이 안 된다. 그래서 `res["per_row"] = {ihash, totals, found}` 를
**순수 추가**했다(`experiments/sweep_harness.py`). 기존 키는 하나도 건드리지 않았으므로
50번 파서에 영향 없다.

주의: 같은 x 의 k 개 행은 각자 독립적으로 20샘플을 뽑는다(합쳐서 20k 샘플). 입력당 20샘플을
모아 세는 uniq 시절 정의와 표본 구성이 다르므로 기존 covcurve 수치와 직접 비교 불가.

---

## 4. 남은 것 / 하지 않은 것

- **50번 8건 미실행** — 지시문 미전달. nq10 `EVALINT` 결정 대기(§ 아래).
- **nq10 `EVALINT` 미결**: `EPOCHS=420 / EVALINT=30` 이면 eval 42회 → **14회**
  (간격 1,442 → 4,198 step)로 줄어 "EMA 정점" 해상도가 3배 나빠진다.
  `EVALINT=10` 이면 step 예산 변동 없이 42회(간격 1,399) 복원. 임의로 안 바꿨다.
- **논문 병기 대상값**: nq8 99.7±0.3 / 90.3±1.9, nq10 89.7±2.7 / 57.5±3.4.
- **기존 v5h/v6h 32건은 "-uniq 조건"** 이다. 이 라운드 수치와 섞어 읽지 말 것.
- git commit 하지 않았다.
