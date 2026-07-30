# REPORT_FINAL.md — GRAM 재현 종합 (초안)

작성 2026-07-30. 근거 리포트 29건은 삭제하지 않았고 [REPORT_INDEX.md](REPORT_INDEX.md) 에 색인돼 있다.
이 문서는 종합만 하고, 수치마다 출처를 링크한다.

**이 문서는 50번(lecun docker)이 정본으로 관리한다.** 29번은 데이터(결과 JSON)만 올리고,
종합·서술은 여기서만 갱신한다.

> **초안 표시 이유 (먼저 읽을 것)**
> 1. **10×10 은 이 서버에 측정치가 없다.** 29번에 harness 결과가 있고 JSON 수령 대기 중이다.
>    §1.2 에 29번이 통보한 값을 옮겨 적었으나 **JSON 원본 대조 전이라 잠정값**이다.
> 2. **8×8 최고 in-loop 수치(v4, 0.9974~1.0000)는 harness 미측정이다.** 표본 1개 추정이라
>    확정값이 아니므로, §1.1 확정 수치는 그보다 낮은 b500 대역(0.9882)으로 잡았다.
> 3. **§3 발견 2·3 의 수치는 실측으로 교체했다.** `logvar_bias` 는 지시문의 −0.134 와 실측
>    −0.0603 이 **둘 다 맞고 학습 길이가 다르다**는 것이 확인돼 병기했다(§3-2). EMA 바닥
>    구간은 실측치(31.9k~33.0k, 회복 36~38k)로 확정했다(§3-3).

---

## 1. 논문 대비 결과표

### 1.1 N-Queens 8×8

전부 **harness 경로**(`experiments/sweep_harness.py`), **N=20 samples**, **test 772 전수**,
**harness seed 0/1 평균**, **`.live`(raw SGD) 가중치**, 정점 checkpoint. 축 정의는 §4.

| 지표 | 우리 최고 | 논문 (Table 1) | 차이 |
|---|---|---|---|
| **Accuracy** (단일시도 valid ≡ pass@1) | **0.9882** | **0.997 ± 0.003** | −0.9%p (논문 구간 밖, 근접) |
| **Coverage@20** | **0.9937** | **0.903 ± 0.019** | **+9.1%p (논문 초과)** |
| Consistent (입력 힌트 보존) | 0.9997 | 미기재 | — |
| pass@20 | 1.0000 | 미기재 | — |

- 출처: `nq8_on_b500_ema6k` / `step_31892.live`,
  `experiments/results/band_nq8_on_b500_ema6k_live_s{0,1}.json`, [REPORT_b45_band_29.md](reports/archive/REPORT_b45_band_29.md)
- **coverage 초과는 자랑거리가 아니다.** test 셋의 64%가 해 1개짜리라 coverage 상한이 1.0 에
  붙어 있다([REPORT_b45_band_29.md](reports/archive/REPORT_b45_band_29.md) §4). 논문과 같은 축인지 미확인 — §5 질문 1.

**harness 미측정인 더 높은 in-loop 수치** (`nqueens/accuracy`, prior 경로, in-loop eval):

| run | EMA 정점 | raw 정점 |
|---|---|---|
| `nq8_v4_catnd_b007_s2` | **0.9974** @ 63784 | **0.9974** @ 67201 |
| `nq8_v4_add_b007_s3` | 0.9883 @ 59228 | 0.9741 @ 39865 |
| `nq8_v4_add_b007_s2` | 0.9858 @ 62645 | **1.0000** @ 52394 |
| `nq8_v4_catnd_b007_s3` | 0.9819 @ 68340 | 0.9767 @ 51255 |

in-loop 는 axis 가 다르다(단일 배치 accuracy, coverage 없음). **논문값과 나란히 놓으려면
harness 재측정이 필요하다** — 4 run × 2 step × 2 seed = 16 job. 이 표가 §1.1 을 대체하지 않는다.

### 1.2 N-Queens 10×10 — **29번 측정, JSON 수령 대기 (잠정값)**

측정은 **29번(hinton)** 에서 이뤄졌다. 아래는 29번이 통보한 값이고, **결과 JSON 원본을 아직
받지 못해 대조 전이다.** JSON 수령 후 이 표를 확정하고 잠정 표시를 뗀다.

| run | Accuracy (단일시도 valid) | Coverage@20 |
|---|---|---|
| `nq10_on_b520_s0` | **0.9228** | 0.9513 ~ 0.9592 (4 run 범위) |
| `nq10_on_b520_s1` | 0.8361 | 〃 |
| `nq10_on_b520_s2` | 0.8103 | 〃 |
| `nq10_on_b450_s0` | **미통보** | 〃 |
| **논문 (Table 1)** | 미확보 (논문 파일이 로컬에 없음) | 미확보 |

- **seed 편차가 또 크다**: 같은 β=5.2 를 seed 만 바꿔 0.9228 / 0.8361 / 0.8103 — 폭 0.113.
  8×8 에서 관측한 seed 지배 현상(§5.1 항목 3)이 10×10 에서도 재현된다.
- **coverage 는 run 별 값이 아니라 4 run 범위로만 통보받았다.** run 별 분해는 JSON 수령 후 채운다.
- 논문값을 아직 확보하지 못해 **"논문 대비" 판정은 불가**하다. β=0.045, epochs 1K / 8 GPU / 3h
  (Table 7) 만 알고 있다.

이 서버(50번)의 10×10 자산 상태 — 재현이 필요할 때의 출발점:

| 항목 | 상태 |
|---|---|
| 데이터 | **있음.** `data/nqueens-10`, k∈{7,8,9}, 43,420 입력 / 724 해, 무결성 통과 ([REPORT_expand_prep.md](reports/archive/REPORT_expand_prep.md)) |
| checkpoint | **없음.** `checkpoints/` 에 Nqueens-10 디렉토리 자체가 없다 |
| 결과 JSON | **없음.** 29번에서 받아야 한다 |
| 로그 | `logs/failed_nq10_2129/`, `logs/{wait,watch}_nq10_g{4,5,6,7}.log.gz` — 이 서버의 실패 기록 |

### 1.3 Graph Coloring 8-vertex

논문 축(**run 간 mean ± std**)으로 집계. 정점 `.live`, harness seed 0/1 평균.
출처 [REPORT_gc8_50.md](reports/archive/REPORT_gc8_50.md) §3, `experiments/results/gc5_*.json` 24건.

| 지표 | bias 0 3-run | 4-run 전체 | 논문 (8-vertex) |
|---|---|---|---|
| **Conflict** ↓ | **0.046 ± 0.025** | 0.064 ± 0.041 | **2.7 ± 2.1** |
| **Coverage@20** | **0.842 ± 0.026** | 0.836 ± 0.024 | **0.858 ± 0.005** |
| Accuracy (valid) | 0.957 / 0.981 (개별) | — | 미기재 |

- **conflict 는 run 평균 기준 논문의 약 1/58.** "1/23~1/120" 서술은 과장이었고
  [REPORT_gc8_50.md](reports/archive/REPORT_gc8_50.md) §3-1 에서 정정됐다.
- **coverage 는 논문에 0.016 미달**이고, 우리 std 가 논문의 5배(3.1% vs 0.6%)다.
  ± 방향이 지표마다 엇갈려 "더 안정적/덜 안정적" 어느 쪽으로도 요약 불가.
- **축 한계**: 우리 3-run 은 순수 seed 반복이 아니라 arm(concat s0/s1 + add s0)이 섞였다.
  논문의 "over runs" 와 같은 축이 되려면 단일 config seed 반복이 필요하다.

---

## 2. 확정 config 와 논문 대비 차이

기준: `checkpoints/Nqueens-8-ACT-torch/nq8_v4_catnd_b007_s2/all_config.yaml` (최신 8×8 arm).

| 항목 | 우리 값 | 논문 | 판정 |
|---|---|---|---|
| `kl_coef` (β) | **0.07** | 0.07 (8×8), 0.045 (10×10), 0.5 (GC-8) | **눈금 문제 — 아래** |
| `kl_reduction` | **sum** | 표기만 있고 눈금 미명시 | **[이탈 가능]** |
| `logvar_bias_init` | **0.0** | 미기재 | **[미기재]** |
| `nodecay_logvar_bias` | false | 미기재 | [미기재] |
| `weight_decay` | 1.0 | 1.0 | [일치] |
| `epochs` | **12,000** | **3,000** (8×8, Table 7) | **[이탈] 4배** |
| `ema` / `ema_rate` | true / 0.9999 | 0.9999 ("For stability") | [일치] |
| EMA bias correction | **적용** | 미기재 | **[이탈]** ([REPORT_audit.md](reports/archive/REPORT_audit.md)) |
| `y_onehot` / `y_inject` | true / concat | 미기재 (우리 처방) | [미기재] |
| `hidden_size` / `H,L_cycles` / `halt_max_steps` | 512 / 3,4 / 16 | 동일 | [일치] |

### 2.1 β 눈금 — sum 이라 논문값과 직접 비교가 안 된다

`kl_reduction: sum` 은 KL 을 L 축으로 **합**한다. L = seq_len + puzzle_emb_len = 64 + 16 = **80**
(8×8), 28 + 16 = **44** (GC-8). 따라서 **sum β 1 ≡ mean β 80**이고, 논문의 0.07 이 어느 눈금인지에
따라 우리 값의 의미가 80배 달라진다.

- 논문 0.07 이 **sum** 이라면 우리 0.07 은 논문값 그대로다.
- 논문 0.07 이 **mean** 이라면 등가점은 **sum 0.07/80 = 0.000875** 이고, 우리가 정점을 찾은
  대역(sum 1.6~5.0)은 논문의 **1,800~5,700배**다.
- 역방향 검증도 했다: 논문 0.07 을 sum 으로 보고 mean 환산한 β=5.6 대역을 실측했더니
  **작동하지 않았다**(0.00%, [REPORT_grid_upper_29.md](reports/archive/REPORT_grid_upper_29.md)).

→ **논문 눈금 미확정이 남은 최대 리스크다** (§5 질문 2). GC 는 논문이 8-node 0.5 를 명시해
우리가 그 값을 그대로 썼고, 결과도 논문 수준이다 — **GC 는 눈금이 sum 쪽임을 시사하는 간접 증거다.**

### 2.2 epochs — 4배 더 돌렸다

논문 Table 7 은 8×8 을 3,000 epoch / 8 GPU / 1h 로 적는다. 우리는 **12,000 epoch / RTX 3090 1장**
이다. §3 발견 3 때문에 늘렸다(3,000 epoch 지점은 EMA 곡선의 바닥 구간이다). **논문과 같은 예산에서
같은 수치가 나오는지는 확인하지 않았다.**

---

## 3. 핵심 발견 5개

### 3-1. β 는 L-sum 눈금이다 (mean 환산 ×L)

`kl_reduction` 키가 없던 시절 `losses.py` 의 주석 토글로 L-mean 이 선택돼 있었고, 그것이 초기
전멸의 원인이었다([REPORT_night2.md](reports/archive/REPORT_night2.md)). 명시 키로 고친 뒤 sum 눈금에서
정점이 잡힌다. 환산 배율은 L=80(8×8) / 44(GC-8) 이고 **puzzle_emb 16칸을 포함**한다.
**단 reduction 을 고쳐도 N-Queens 는 바로 살지 않았다** — 실패 양상만 바뀌었다.

### 3-2. σ inflation 은 wd 아티팩트다 — **σ 궤적은 학습 길이의 함수다**

`logvar_bias` 를 −4 로 초기화한 arm 은 wd 가 bias 를 0 쪽으로 끌어올리고, σ 는
**σ ≈ exp(bias/2)** 로 따라 팽창한다. **핵심은 이 팽창이 끝나는 지점이 아니라, 학습 길이에
따라 σ 가 어디까지 가 있느냐다.** 같은 init(−4)에서 예산만 다르게 잡으면 다른 값이 나온다:

| 측정 | 예산 | step | bias | σ = exp(bias/2) | 출처 |
|---|---|---|---|---|---|
| 8-run 기준 | **6,000 epochs** | ~34k | **−0.134** | **0.935** | 29번 통보 |
| `gc8_v5_add_bm4_s0` | **12,000 epochs** | 60,192 | **−0.0603** | **0.970** | 최종 checkpoint 실측 |
| (init 0 대조) `gc8_v5_add_b0_s0` | 12,000 | 60,192 | −0.0013 | 0.9993 | 최종 checkpoint 실측 |
| (init 0 대조) `nq8_v4_catnd_b007_s2` | 12,000 | 68,340 | −0.0031 | 0.9984 | 최종 checkpoint 실측 |

실측 방법: `prior_logvar_bias` / `post_logvar_bias` 를 CPU 로 로드해 직접 읽었다. 두 파라미터는
소수점 4자리까지 같은 값으로 수렴한다(−0.0603 / −0.0604).

- **−0.134(34k) → −0.0603(68k) 은 단조 진행이다.** wd 가 계속 끌어올리므로 **학습이 길수록 0 에
  가까워지고, σ 는 1 에 가까워진다.** 초기값 −4(σ=0.135)와 0(σ=1.0)의 차이는 학습이 길어지면
  사라진다 — 12k epoch 지점에서 −4 arm(0.970)과 0 arm(0.999)은 이미 같은 대역이다.
- 따라서 **`logvar_bias_init` 은 최종 σ 를 정하는 손잡이가 아니라, σ 가 목표치에 도달하는 시각을
  늦추는 손잡이다.** 짧은 예산에서만 init 이 결과를 지배한다.
- **인과 방향 주의**: [REPORT_gc8_50.md](reports/archive/REPORT_gc8_50.md) §4-1 은 "wd 가 bias 를
  올리는 속도가 KL 붕괴보다 훨씬 느리다 → 순서상 KL 붕괴가 먼저" 라고 적었다(step 1,881 에서
  −3.387, 3,762 에서 −2.805). 즉 **wd 는 σ 를 팽창시키지만, bias −4 arm 이 24.5k step 단색에
  갇힌 원인은 조기 posterior collapse 쪽에 가깝다.** "σ inflation = wd 아티팩트" 는 현상 기술로는
  맞고 실패 원인 지목으로는 과하다.
- **미대조**: −0.134 는 29번 통보값이고 이 서버에서 checkpoint 로 확인하지 못했다(해당 6k run 의
  step 34k checkpoint 가 이 박스에 없다). 8-run 각각의 값인지 평균인지도 미확인.

### 3-3. EMA 는 붕괴가 아니라 늦게 수렴한다 (바닥·회복 구간 = 실측 확정치)

`nq8_v4_catnd_b007_s2` 의 in-loop EMA `nqueens/accuracy` 곡선 실측:

| step | 25.1k | 29.6k | **31.9k** | **33.0k** | 35.3k | 36.4k | 37.6k | 39.9k | 41.0k | 43.3k | 44.4k | … | 63.8k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EMA acc | 0.021 | 0.062 | **0.027** | **0.022** | 0.061 | 0.096 | 0.207 | 0.409 | 0.531 | 0.670 | 0.745 | | **0.997** |

- **바닥은 31.9k~33.0k**(0.022~0.027), **회복 시작은 36~38k**, 0.5 돌파가 41k, 0.99 대 도달이 60k 이후다.
  (초기 서술 "33k~40k 바닥, 48k 이후 회복" 은 이 실측으로 대체됐다 — 바닥이 더 좁고 회복이 더 이르다.
  40k 시점에 이미 0.41 로 상승 중이므로 48k 는 회복 시작점이 아니다.)
- **§3-2 와 같은 이야기다**: σ 궤적이 학습 길이의 함수인 것처럼, EMA 곡선도 예산을 짧게 잡으면
  바닥 구간에서 끊긴다. 두 발견 모두 **"짧은 예산에서의 관측을 모델 성질로 오독하는" 같은 함정**을 가리킨다.
- **결론은 유지된다**: 논문 예산(3,000 epoch ≈ 17k step)에서 끊으면 이 곡선은 바닥 이전이거나
  바닥 구간이고, "붕괴"로 오독된다. 실제로 [REPORT_offup_50.md](reports/archive/REPORT_offup_50.md) 는 EMA in-loop
  15 eval 전부 0.0000 인 4 run 을 "전멸"로 판정하려다 `.live` 로 살아 있음을 확인했다.

### 3-4. 저β 실패 기전 = posterior shortcut (forcehalt 로 판별)

`nq8_off_b100_forcehalt` (β=1.0, one-hot OFF, `halt_exploration_prob=1.0` 으로 halting 강제),
`step_2278.live`, harness seed 0:

| 경로 | valid | consistent | strict | queen_count (mean / min–max) |
|---|---|---|---|---|
| **posterior** (정답 y 를 봄) | **0.9534** | **1.0000** | 0.9534 | 7.97 / 7–9 |
| **prior** | **0.0083** | 0.9907 | 0.0083 | 7.79 / 5–11 |

- **115배 격차.** posterior 는 거의 완벽하고 prior 만 죽는다. `consistent` 는 두 경로 모두 99% 이상이라
  **입력 힌트는 잘 보존한다** — 못 하는 것은 "나머지를 스스로 채우기"다.
- 이것이 [REPORT_klsweep.md](reports/archive/REPORT_klsweep.md) 의 "posterior valid 100% 는 라벨 복사" 와
  [REPORT_contradiction.md](reports/archive/REPORT_contradiction.md) 의 "KL≈0 인데 posterior 100%" 를 같은 기전으로 설명한다:
  **β 가 낮으면 KL 이 posterior 를 prior 로 끌어당기지 못해, posterior 가 y 를 우회 복사하는 지름길을
  학습하고 prior 는 아무것도 배우지 못한다.**
- **한계**: forcehalt run 은 1개(seed 1개, β 1점)뿐이고 step 2278 이 전부다. β 축으로 이 격차가
  어떻게 닫히는지는 미측정.

### 3-5. lr decay 기각 — prior 만 선택적으로 죽는다

같은 β=5.0 / one-hot ON / 6,000 epoch 에서 **`lr_min_ratio` 만** 바꾼 대조:

| run | `lr_min_ratio` | prior valid (`.live`) | consistent | coverage@20 | EMA valid |
|---|---|---|---|---|---|
| `nq8_on_b500_s1` | **1.0** (상수) | **0.9331** | 0.9999 | 0.9875 | 0.0000 |
| `nq8_on_b500_s2` | **1.0** (상수) | **0.9075** | 0.9997 | 0.9911 | 0.0076 |
| `nq8_on_b500_decay` | **0.1** (코사인 감쇠) | **0.0327** | 0.9948 | 0.3172 | 0.0000 |

- **28배 하락.** seed 편차(0.9331 vs 0.9075, 1.03배)로 설명되지 않는 크기다.
- **"선택적 사망" 근거**: decay run 도 `consistent` 는 **0.9948** 로 멀쩡하다. 즉 힌트 보존 능력과
  백본은 살아 있고 **prior 의 생성 능력만** 무너진다. coverage 도 0.99 → 0.32 로 같이 떨어진다.
- 출처: `experiments/results/b500_nq8_on_b500_{s1,s2,decay}_step_3*_live_s{0,1}.json` (seed 0/1 평균).

---

## 4. 측정 방식 명세

`metric_defs` 키가 결과 JSON 132건에 박혀 있다(원문이 정본). 요약:

| 축 | 정의 | JSON 키 |
|---|---|---|
| **valid** | 퀸 n개 + 행/열/양대각 중복 없음. **입력 힌트 보존은 보지 않는다** | `prior_all_samples.valid` |
| **consistent** | 입력에 주어진 퀸을 전부 보존했는가 | `prior_all_samples.consistent` |
| **strict** | valid AND consistent | `prior_all_samples.strict` |
| **conflict** (GC) | 위반 edge 수 (낮을수록 좋음). GC 논문 헤드라인 | `conflicts_mean` |
| **coverage@20** | 퍼즐별 (발견한 valid∧consistent 서로 다른 해 수) / (sidecar 의 per-x 전체 해 수) 평균 | `coverage_at_20.coverage` |
| 단일시도 | N 샘플 각각 채점한 평균 ≡ pass@1 | `prior_all_samples.*` |
| pass@k | N 샘플 중 k 개 뽑아 1개 이상 valid 일 확률(불편추정량) | `pass_at` |
| majority@N | 최빈 보드 1개를 채점 (동률은 최초 출현) | `majority_at_N_valid` |
| verifier@N | 모델 자신의 `lprm_head` 최고점 1개를 채점 (**정답 미사용**) | `verifier_selected_at_N_valid` |
| distinct@N | 서로 다른 보드 수 (**다양성이지 정확도가 아니다**) | `distinct_at_N_mean` |

**5축을 항상 명시해야 한다**: 가중치(`.live` / EMA) × 경로(in-loop / harness) × 표본(1 / 20) ×
기준(valid / strict) × 집계(단일시도 / pass@k / selection / coverage).
[REPORT_b45_band_29.md](reports/archive/REPORT_b45_band_29.md) §0 이 이 규약의 정본이다.

### 4.1 논문 축과 **일치하는** 것

- **coverage@20** — 논문 Table 1 정의(found/total 퍼즐 평균)와 계산이 같다. 버킷 분해도
  논문 Fig 15 축({1, 2–5, 6–20, 21+})에 맞춰 뽑았다.
- **conflict** — GC 논문 헤드라인 지표. 집계 축(run 간 mean±std)까지 맞췄다([REPORT_gc8_50.md](reports/archive/REPORT_gc8_50.md) §3-1).
- **가중치** — 논문이 EMA 0.9999 를 명시하므로 EMA 사본이 원칙상 논문 축이다.

### 4.2 논문 축과 **불일치하거나 미확인**인 것

1. **우리 헤드라인은 `.live`(raw) 다.** EMA 로 판정하면 여러 run 이 "전멸"로 오분류되므로
   raw 를 썼다([REPORT_offup_50.md](reports/archive/REPORT_offup_50.md), [REPORT_grid_upper_29.md](reports/archive/REPORT_grid_upper_29.md)).
   **논문과 다른 가중치 축이다.**
2. **EMA bias correction 은 논문 미기재다.** 우리는 적용한다([REPORT_audit.md](reports/archive/REPORT_audit.md) 판정 [이탈]).
3. **peak step 선택 규칙이 논문에 없다.** 우리는 raw 곡선 정점을 쓴다. EMA 곡선으로 고르면
   같은 run 이 4.2배까지 달라진다([REPORT_off_line_29.md](reports/archive/REPORT_off_line_29.md) §4-1).
4. **GC 의 "over runs" 축이 불완전하다** — arm 이 섞인 3-run 이라 순수 seed 반복이 아니다.
5. **coverage 상한 포화** — 8×8 test 셋의 64%가 해 1개짜리라 우리 coverage 는 상한에 붙는다.
   논문이 같은 test 분포를 썼는지 미확인.
6. **Table 3b ablation 은 조건이 다르다** — 논문의 50.27/0.00 은 μ=0 / σ=0 으로 **학습까지 한**
   모델이고, 우리 82.90/23.58 은 학습된 모델을 **추론 시점에** 끈 것이다([REPORT_guide_diag.md](reports/archive/REPORT_guide_diag.md)).

---

## 5. 미해결 + 사수 질문

### 5.1 미해결 (측정으로 닫을 수 있는 것)

| # | 항목 | 필요 작업 |
|---|---|---|
| 1 | **v4 4-run harness 미측정** — 8×8 최고 수치가 in-loop 뿐 | 4 run × 2 step × 2 seed = **16 job** (§1.1) |
| 2 | **10×10 JSON 미수령** — 29번 측정치를 통보로만 받았다 | 29번이 `experiments/results/` 에 JSON push → 받아서 §1.2 확정. **10×10 논문값도 미확보** |
| 3 | **seed 편차가 β 효과보다 크다** — 같은 설정 3회가 0.7869/0.6267/0.3146 | 단일 config seed 5개 반복. 그 전엔 β 순위 무의미 ([REPORT_finesweep_50.md](reports/archive/REPORT_finesweep_50.md)) |
| 4 | **GC 를 논문 축에 정확히 올리기** | `concat+nodecay/bias0/β0.5` seed 2/3/4 추가 → seed 5개 |
| 5 | **재현성 자체가 안 된다** — 같은 seed 가 같은 궤적을 안 만든다(L2 55%) | `use_deterministic_algorithms` 등 결정성 설정 도입 검토 ([REPORT_audit_measure.md](reports/archive/REPORT_audit_measure.md)) |
| 6 | forcehalt 격차의 β 의존성 (§3-4 한계) | β 축 3~4점 × forcehalt |

### 5.2 사수 질문

1. **coverage 비교가 성립하는가?** 우리 8×8 coverage 0.9937 이 논문 0.903±0.019 를 넘는데,
   우리 test 셋은 64%가 해 1개짜리다. 논문이 같은 생성 규칙(k∈{5,6,7} 제거)과 같은 test split 을
   썼는가? 아니면 이 비교를 접어야 하는가?
2. **논문 β=0.07 은 sum 눈금인가 mean 눈금인가?** 이것으로 §2.1 의 80배가 갈린다. GC 는 논문
   명시값 0.5(sum 가정)로 논문 수준 결과가 나왔는데, 8×8 은 mean 환산 대역(5.6)이 죽었다.
3. ~~**`logvar_bias` −0.134 의 출처는?**~~ **해결됨** — 6,000 epochs(~34k step) 8-run 기준값이고,
   −0.0603 은 12,000 epochs(68k step) 기준이다. 둘 다 맞고 예산이 다르다. §3-2 에 병기했다.
   **남는 질문**: −0.134 가 8-run 각각의 값인가 평균인가? 그리고 그 6k run 의 step 34k
   checkpoint 를 이 박스로 받을 수 있는가(직접 대조용)?
4. **논문 예산(3,000 epoch)에서의 수치를 요구하는가?** 우리는 12,000 을 돌렸고 3,000 지점은
   EMA 바닥 구간이다(§3-3). 논문 예산으로 맞추면 수치가 크게 떨어질 것이다.
5. **헤드라인을 `.live` 로 유지하는가?** 논문 축은 EMA 지만, EMA 로 판정하면 오분류가 반복 관측됐다.
   두 값을 항상 병기하는 현재 방식을 최종 보고에서도 유지할지.
6. **EMA bias correction 을 끄고 논문과 맞출 것인가?** 논문 미기재 항목이고 우리 판정은 [이탈]이다.
