# REPORT_FINAL.md — GRAM 재현 종합 (초안)

작성 2026-07-30. 근거 리포트 29건은 삭제하지 않았고 [REPORT_INDEX.md](REPORT_INDEX.md) 에 색인돼 있다.
이 문서는 종합만 하고, 수치마다 출처를 링크한다.

**이 문서는 50번(lecun docker)이 정본으로 관리한다.** 29번은 데이터(결과 JSON)만 올리고,
종합·서술은 여기서만 갱신한다.

> **초안 표시 이유 (먼저 읽을 것)**
> 1. **10×10 은 29번 측정치로 확정됐다**(JSON 대조 완료, §1.2). 50번 자체 4-run 은 학습 중이다.
>    통보값과 JSON 이 s0↔s1 뒤바뀐 곳이 있어 **JSON 기준으로 고쳤다.**
> 2. **8×8 헤드라인은 50번 자산 0.9843 이다.** 29번 수치(0.9882 / 0.9977)는 이 박스에 가중치가
>    없어 재현·재측정이 불가능하므로 §1.1 에서 별도 행으로 분리했다. v4 in-loop 0.9974~1.0000 은
>    표본 1개 추정이라 확정값이 아니다(harness 측정 대기).
> 3. **§3 발견 2·3 의 수치는 실측으로 교체했다.** `logvar_bias` 는 지시문의 −0.134 와 실측
>    −0.0603 이 **둘 다 맞고 학습 길이가 다르다**는 것이 확인돼 병기했다(§3-2). EMA 바닥
>    구간은 실측치(31.9k~33.0k, 회복 36~38k)로 확정했다(§3-3).
> 4. **모든 결과표에 `server` 열이 있다(29 | 50).** 두 서버 결과를 섞어 판단한 사례가 여러 건
>    있었다 — 자산 소재는 **§0** 을 먼저 볼 것.

---

## 0. 자산 목록 — 어느 run 이 어느 서버에 있는가

**두 서버 결과를 섞어 판단한 사례가 여러 건 있었다.** 이 절이 그 방지책이고, 아래 모든 결과표에는
`server` 열이 붙는다. 섞어 읽지 말 것.

**판정 방법**: 학습은 자기 박스에 `logs/<run>.log` 를 남긴다. 50번에서
`logs/<run>.log{,.gz}` 가 있으면 여기서 돌았고, 로그도 checkpoint 도 없는데 결과 JSON 만
있으면 **29번이 측정해 repo 에 push 한 것**이다(`experiments/results/` 는 git 추적 대상).

| 구분 | run | 50번 로그 | 50번 ckpt | 결과 JSON |
|---|---|---|---|---|
| **50번 — 8×8 v2/v4** | `nq8_v2_sum007_{add,catnd}_s{0,1}` | 있음 | 있음 | 24건 |
| | `nq8_v4_{add,catnd}_b007_s{2,3}` | 있음 | 있음 | **0건 (harness 미측정)** |
| **50번 — 8×8 β sweep** | `nq8_{on,off}_b*` 중 `b210_ema6k`, `b220_ema6k_s2`, `b230_ema6k`, `b160_ema6k`, `b280~b340_ema`, `b500_{s1,s2,decay}`, `b100_forcehalt`, `oh_*`, `ctrl_b15` 등 **28 run** | 있음 | 있음 | 있음 |
| **50번 — GC** | `gc8_v5_{catnd_s0,catnd_s1,add_b0_s0,add_bm4_s0}` | 있음 | 있음 | 32건 |
| | `gc8_v5_catnd_s{2,3,4}` (seed 반복) | 있음 | 학습 중 | 미측정 |
| **50번 — 10×10** | `nq10_v5_{catnd_s0,catnd_s1,catnd_s2,add_bm4_s0}` | 있음 | 학습 중 | 미측정 |
| **29번 — β 4.0~5.6 대역** | `nq8_on_b{400,420,450,480,500,520,560}_ema6k`, `nq8_on_b560_s{1,2}`, `nq8_off_b500_ema6k` | **없음** | **없음** | 있음 |
| **29번 — β 격자/EMA** | `nq8_off_b220_ema6k_s{0,1}`, `b240_ema`, `b260_ema`, `nq8_on_b{140,160,170,180_s3,190,200,200_ema,450_ema,560_ema}` 등 **26 run** | **없음** | **없음** | 있음 |
| **29번 — 10×10** | `nq10_on_b520_s{0,1,2}`, `nq10_on_b450_s0` | **없음** | **없음** | 있음 (16건 + covcurve 4건) |
| **29번 — v2 base** | `nq8_v2_base_b500_s0`, `nq8_v2_cat_b500_s0` 등 | **없음** | **없음** | 있음 (band + covcurve8) |

**주의 사항 3가지**

1. **29번 run 은 이 박스에서 재측정할 수 없다.** 가중치가 없다. 결과 JSON 에 적힌 값이 전부이고,
   `.live` 곡선 재스캔이나 다른 N 으로의 재측정이 불가능하다.
2. **결과 JSON 561건 중 309건이 이미 없는 checkpoint 를 참조한다.** 대부분 위 29번 run 이다
   (2026-07-30 checkpoint 정리 이전부터 없었다 — `reports/cleanup/delete_candidates.txt` 에 없다).
3. **`nq8_v2_base_b500_s0` 는 29번이 `fc7edd4` 로 JSON 을 올려 확정됐다** (정점 0.9977).
   가중치는 여전히 없어 재측정은 불가하고, `run_configs/hinton/` 의 config 로 손잡이만 대조된다.

---

## 1. 논문 대비 결과표

### 1.1 N-Queens 8×8

전부 **harness 경로**(`experiments/sweep_harness.py`), **N=20 samples**, **test 772 전수**,
**harness seed 0/1 평균**, **`.live`(raw SGD) 가중치**, 정점 checkpoint. 축 정의는 §4.

**헤드라인은 50번 자산 기준이다.** 29번 수치는 이 박스에서 재현·재측정이 불가능하므로(§0 주의 1)
별도 행으로 분리하고 출처를 명시한다.

| server | run / checkpoint | step_kind | Accuracy (단일시도 valid) | Coverage@20 | pass@20 | maj@20 | vsel@20 | dist@20 |
|---|---|---|---|---|---|---|---|---|
| **29** | **`nq8_v2_base_b500_s0` / `step_37587.live`** | **peak** | **0.9977** | **0.9964** | 1.0000 | 1.0000 | 0.9987 | 1.65 |
| 29 | `nq8_v2_base_b500_s0` / `step_39865.live` | **final** | **0.9753** | 0.9804 | 1.0000 | 0.9929 | 0.9838 | 1.82 |
| 29 | `nq8_on_b500_ema6k` / `step_31892.live` | peak | 0.9882 | 0.9937 | 1.0000 | — | — | — |
| 29 | `nq8_on_b560_ema6k` / 정점 `.live` | peak | 0.9856 | 0.9844 | 1.0000 | — | — | — |
| **50** | **`nq8_v2_sum007_catnd_s0` / `step_39865.live`** | peak | **0.9843** | **0.9847** | 1.0000 | — | — | — |
| 50 | `nq8_v2_sum007_add_s0` / `step_29614.live` | peak | 0.9612 | 0.9835 | 1.0000 | — | — | — |
| 50 | `nq8_on_b500_s1` / `step_33031.live` | peak | 0.9331 | 0.9875 | 1.0000 | — | — | — |
| — | **논문 (Table 1)** | 미기재 | **0.997 ± 0.003** | **0.903 ± 0.019** | 미기재 | 미기재 | 미기재 | 미기재 |

- **8×8 최고 = 29번 `nq8_v2_base_b500_s0` 정점 0.9977 이고, 논문 구간(99.4~100.0) 안에 들어간다.**
  근거 `experiments/results/band_nq8_v2_base_b500_s0_live_s{0,1}.json` (harness seed 0/1),
  checkpoint `step_37587.live`. 이전 판(2026-07-30 오전)에서 "미검증"으로 적었던 것은 그 시점에
  결과 JSON 이 repo 에 없었기 때문이고, 29번이 `fc7edd4` 로 올려 확정됐다.
- **정점과 최종을 반드시 병기한다 — 격차 2.24%p 가 선택 효과의 크기다.** 정점 `step_37587` 은
  **test set 으로 checkpoint 를 고른 결과**이므로 낙관 편향을 포함한다. 최종 `step_39865` 는
  그 편향이 없고 0.9753 이다. 논문이 어느 쪽 규칙을 썼는지는 미기재 — §5 질문 4.
- **이 run 의 EMA 는 두 지점 다 0.0000 이다.** 39,885 step 에서 종료돼 EMA 가 수렴 전이다
  (`consistent` 는 0.43 / 0.62 로 남아 있어 "죽은" 게 아니라 미수렴이다). §3-3 의 EMA 늦은 수렴이
  같은 현상이다.
- **50번 최고 = 0.9843** 으로 논문 대비 **−1.3%p**(구간 밖).
  출처 `v2_nq8_v2_sum007_catnd_s0_step_39865_live_s{0,1}.json`.
- **29번 run 은 이 박스에서 재측정할 수 없다** — 가중치가 없다(§0 주의 1). 다만 `run_configs/hinton/`
  으로 **config 대조는 가능해졌다**(§2).
- **coverage 를 논문과 비교하는 것 자체가 아직 성립하지 않는다 — §1.4 를 볼 것.**
  우리 0.9964 는 **N=20** 값이고, N 을 낮추면 논문 아래로 내려간다(N=1 에서 0.7883).

**harness 미측정인 더 높은 in-loop 수치** (`nqueens/accuracy`, prior 경로, in-loop eval, 전부 50번):

| server | run | EMA 정점 | raw 정점 |
|---|---|---|---|
| 50 | `nq8_v4_catnd_b007_s2` | **0.9974** @ 63784 | **0.9974** @ 67201 |
| 50 | `nq8_v4_add_b007_s3` | 0.9883 @ 59228 | 0.9741 @ 39865 |
| 50 | `nq8_v4_add_b007_s2` | 0.9858 @ 62645 | **1.0000** @ 52394 |
| 50 | `nq8_v4_catnd_b007_s3` | 0.9819 @ 68340 | 0.9767 @ 51255 |

in-loop 는 axis 가 다르다(단일 배치 accuracy, coverage 없음)**이고 표본 1개 추정이라 확정값이
아니다.** 논문값과 나란히 놓으려면 harness 측정이 필요하다. **이 표가 위 표를 대체하지 않는다.**

> **⚠️ v4 의 raw 정점 checkpoint 4개는 2026-07-30 정리에서 삭제됐다 — 그 step 은 영구히
> 측정 불가하다.**
>
> 보존 규칙이 **EMA 곡선 정점**(`[EVAL @ step N]`)을 기준으로 삼았고, v4 는 harness 측정 이력이
> 없어(규칙 ①에 안 걸림) **raw 정점만 세 규칙 전부를 빠져나갔다.** raw 곡선이 있는 21 run 중
> v4 4개만 해당한다 — 나머지 17개는 raw 정점이 harness 측정 이력·EMA 정점·마지막 step 과
> 겹쳐 살아남았다. `queue_023.sh` 단계 ③(v4 harness)이 이 때문에 `FileNotFoundError` 로 실패했다.
>
> 대체 가능한 생존 step (EMA / raw in-loop 값 병기, 전부 `step_N` + `step_N.live` 보유):
>
> | server | run | 삭제된 raw 정점 | 생존 최선 대체 | 그 step 의 raw / EMA |
> |---|---|---|---|---|
> | 50 | `nq8_v4_catnd_b007_s2` | 67201 (raw 0.9974) | **63784** | 0.9961 / 0.9974 |
> | 50 | `nq8_v4_add_b007_s3` | 39865 (raw 0.9741) | **68340** | 0.9676 / 0.9858 |
> | 50 | `nq8_v4_catnd_b007_s3` | 51255 (raw 0.9767) | **68340** | 0.9275 / 0.9819 |
> | 50 | `nq8_v4_add_b007_s2` | **52394 (raw 1.0000)** | 68340 | 0.8433 / 0.7850 |
>
> `catnd_s2` 는 손실이 거의 없다(63784 의 raw 0.9961 ≈ 삭제된 정점 0.9974). **`add_s2` 는
> 손실이 크다** — raw 1.0000 이던 정점이 사라지고 생존 최선이 0.8433 이다.
> **교훈**: 이 프로젝트의 판정축은 `.live`(raw) 인데(§4.2) 보존 규칙은 EMA 축이었다.
> 다음 정리에서는 **raw 정점도 보존 규칙에 넣어야 한다.**

### 1.2 N-Queens 10×10 — 29번 측정 (JSON 대조 완료)

29번(hinton)이 측정하고 `fc7edd4` 로 결과 JSON 16건을 올렸다. 아래는 **JSON 실측치**다
(`.live`, harness seed 0/1 평균, test 6,513 전수).

| server | run / checkpoint | Accuracy (단일시도 valid) | Coverage@20 | pass@20 | Consistent |
|---|---|---|---|---|---|
| 29 | `nq10_on_b520_s1` / `step_37466.live` | **0.9228** | 0.9592 | 1.0000 | 1.0000 |
| 29 | `nq10_on_b520_s0` / `step_28820.live` | 0.8361 | 0.9513 | 0.9990 | 0.9982 |
| 29 | `nq10_on_b520_s2` / `step_41789.live` | 0.8103 | 0.9558 | 0.9988 | 0.9996 |
| 29 | `nq10_on_b450_s0` / `step_31702.live` | 0.6668 | 0.9512 | 0.9975 | 0.9970 |
| 50 | `nq10_v5_{catnd_s0,catnd_s1,catnd_s2,add_bm4_s0}` | **학습 중** (2026-07-30 04:34 launch, 60,550 step, ~10h50m) | 미측정 | | |
| — | **논문 (Table 1)** | **0.897 ± 0.027** | **0.575 ± 0.034** | 미기재 | 미기재 |

- **통보값과 JSON 이 어긋난 곳이 있었다**: 0.9228 은 `s0` 이 아니라 **`s1`** 이고, `s0` 은 0.8361
  이다(통보는 s0↔s1 이 뒤바뀜). `nq10_on_b450_s0` 는 "미통보"였는데 JSON 에 0.6668 로 있다.
  이 표가 JSON 기준 정본이다.
- **Accuracy 는 논문 구간(0.870~0.924) 안에 최고 run 하나가 들어간다** — `s1` 0.9228.
  나머지는 0.667~0.836 으로 아래다.
- **coverage 0.951~0.959 는 논문 0.575 를 크게 넘는데, 이것은 축이 다르기 때문이다 — §1.4.**
  N=1 로 내리면 0.359~0.533 으로 논문 아래가 된다.
- **seed 편차가 또 크다**: 같은 β=5.2 를 seed 만 바꿔 0.9228 / 0.8361 / 0.8103 — 폭 0.113.
  8×8 에서 관측한 seed 지배 현상(§5.1 항목 3)이 10×10 에서도 재현된다.
- **EMA 는 4 run 전부 0.0000 이다**(`consistent` 는 0.87~0.99). 6,000 epoch 급에서 끊긴
  run 들이라 §3-3 의 EMA 늦은 수렴 구간에 걸려 있다.

- **seed 편차가 또 크다**: 같은 β=5.2 를 seed 만 바꿔 0.9228 / 0.8361 / 0.8103 — 폭 0.113.
  8×8 에서 관측한 seed 지배 현상(§5.1 항목 3)이 10×10 에서도 재현된다.
- 논문 10×10 예산은 1K epochs / 8 GPU / 3h (Table 7), β=0.045 다.

**50번은 2026-07-30 04:34 에 자체 10×10 4-run 을 launch 했다** — β **0.045 sum(논문 10×10
명시값)**, `y_onehot=True`, `y_inject=concat`(3 seed) + `add`/`logvar_bias_init=−4.0`(1개),
epochs 1,260, `eval_interval=30`, `dual_eval=True`. **29번의 β=5.2 계열과 β 가 달라 비교축이
다르다** — 29번은 대역 탐색, 50번은 논문값 검증이다. 섞어 읽지 말 것.

| 항목 | 상태 |
|---|---|
| 데이터 | **있음.** `data/nqueens-10`, k∈{7,8,9}, 43,420 입력 / 724 해, 무결성 통과 ([REPORT_expand_prep.md](reports/archive/REPORT_expand_prep.md)) |
| checkpoint | **학습 중** (`checkpoints/Nqueens-10-ACT-torch/nq10_v5_*`, 04:34 생성) |
| 29번 결과 JSON | **없음.** 29번에서 받아야 한다 |
| 과거 실패 기록 | `logs/failed_nq10_2129/`, `logs/{wait,watch}_nq10_g{4,5,6,7}.log.gz` |

### 1.3 Graph Coloring 8-vertex

논문 축(**run 간 mean ± std**)으로 집계. 정점 `.live`, harness seed 0/1 평균.
출처 [REPORT_gc8_50.md](reports/archive/REPORT_gc8_50.md) §3, `experiments/results/gc5_*.json` 24건.

**GC 는 전부 50번 자산이다** — 29번에는 GC run 이 없다. 이 절만은 서버 혼용 위험이 없다.

| server | 지표 | bias 0 3-run | 4-run 전체 | 논문 (8-vertex) |
|---|---|---|---|---|
| 50 | **Conflict** ↓ | **0.046 ± 0.025** | 0.064 ± 0.041 | **2.7 ± 2.1** |
| 50 | **Coverage@20** | **0.842 ± 0.026** | 0.836 ± 0.024 | **0.858 ± 0.005** |
| 50 | Accuracy (valid) | 0.957 / 0.981 (개별) | — | 미기재 |

- **conflict 는 run 평균 기준 논문의 약 1/58.** "1/23~1/120" 서술은 과장이었고
  [REPORT_gc8_50.md](reports/archive/REPORT_gc8_50.md) §3-1 에서 정정됐다.
- **coverage 는 논문에 0.016 미달**이고, 우리 std 가 논문의 5배(3.1% vs 0.6%)다.
  ± 방향이 지표마다 엇갈려 "더 안정적/덜 안정적" 어느 쪽으로도 요약 불가.
- **축 한계**: 우리 3-run 은 순수 seed 반복이 아니라 arm(concat s0/s1 + add s0)이 섞였다.
  논문의 "over runs" 와 같은 축이 되려면 단일 config seed 반복이 필요하다.

### 1.4 coverage 의 N 판정 — **"논문은 N=1" 가설은 기각된다**

coverage 는 N(표본 수)에 강하게 의존한다. 그래서 "우리가 논문을 넘었다"는 서술이 축을 맞춘
비교인지 검증했다. `experiments/coverage_curve.py` (29번 제공), harness seed 0, `.live`, test 전수.

**8×8** (n_puzzles=772). 상한 = 퍼즐마다 `min(N, total)/total` 을 구해 평균한 값(도달 가능 최댓값):

| server | run / checkpoint | N=1 | N=5 | N=10 | N=20 |
|---|---|---|---|---|---|
| 29 | `nq8_v2_base_b500_s0` / `step_37587.live` (peak) | **0.7883** | 0.9523 | 0.9850 | 0.9958 |
| 29 | `nq8_v2_base_b500_s0` / `step_39865.live` (final) | 0.7644 | 0.9307 | 0.9652 | 0.9824 |
| 29 | `nq8_on_b500_ema6k` / `step_31892.live` | 0.7767 | 0.9508 | 0.9840 | 0.9942 |
| 29 | `nq8_on_b560_ema6k` / `step_34170.live` | 0.7754 | 0.9293 | 0.9631 | 0.9821 |
| 29 | `nq8_v2_cat_b500_s0` / `step_31892.live` | 0.7661 | 0.9512 | 0.9862 | 0.9953 |
| — | **상한 (per-puzzle)** | **0.7909** | 0.9938 | 0.9978 | 1.0000 |
| — | 달성률 (peak run / 상한) | **0.997** | 0.958 | 0.987 | 0.996 |
| — | **논문 0.903 ± 0.019** (구간 0.884~0.922) | 구간 **밖(아래)** | 구간 **밖(위)** | 밖(위) | 밖(위) |

**8×8 에서 논문값은 N=1 로 나올 수 없다 — 원리적으로 불가능하다.**
우리 test 셋의 N=1 **상한이 0.7909** 이고 논문 구간 하단이 0.884 다. 어떤 모델도 N=1 에서
0.884 를 넘을 수 없다. 상한이 낮은 이유는 해 개수 분포다: 772 퍼즐 중 494개(64%)가 해 1개,
172개가 2개, 69개가 3개다 — 해가 k 개인 퍼즐의 N=1 상한은 1/k 다.

**10×10** (n_puzzles=6,513):

| server | run / checkpoint | N=1 | N=5 | N=10 | N=20 |
|---|---|---|---|---|---|
| 29 | `nq10_on_b520_s1` / `step_37466.live` | **0.5328** | 0.8364 | 0.9137 | 0.9598 |
| 29 | `nq10_on_b520_s0` / `step_28820.live` | 0.4654 | 0.8100 | 0.8991 | 0.9512 |
| 29 | `nq10_on_b520_s2` / `step_41789.live` | 0.4479 | 0.8117 | 0.9046 | 0.9545 |
| 29 | `nq10_on_b450_s0` / `step_31702.live` | 0.3588 | 0.7628 | 0.8849 | 0.9508 |
| — | **상한 (per-puzzle)** | **0.5930** | 0.9589 | 0.9902 | 0.9979 |
| — | **논문 0.575 ± 0.034** (구간 0.541~0.609) | 구간 **밖(아래, 최고 run 도)** | 밖(위) | 밖(위) | 밖(위) |

**10×10 에서도 겹치지 않는다.** 최고 run 의 N=1 이 0.5328 로 구간 하단 0.541 보다
**0.008 낮다.** 나머지 3 run 은 0.359~0.465 로 한참 아래다. "겹쳤다"로 보였던 것은
**4 run 중 최고 하나만 본 것**이고, 그마저 구간 밖이다.

**두 task 의 차이는 상한이 설명한다.** 10×10 의 N=1 상한(0.5930)은 논문 구간(0.541~0.609)
**안에** 있어서 N=1 가설이 성립할 여지가 있어 보였다. 8×8 의 N=1 상한(0.7909)은 논문 구간보다
**낮아서** 여지가 없다. 즉 **"Table 1 coverage 는 N=1" 가설은 8×8 이 반증한다.**

**그러면 무엇이 남는가**

1. **우리 coverage 가 논문을 넘었다는 서술은 N=20 에서만 성립한다.** 두 task 모두 N=1 에서는
   논문 아래이고, N=5 부터 위다. 논문값은 양쪽 다 **N=1 과 N=5 사이**에 떨어진다
   (8×8 은 보간하면 N≈3, 10×10 은 N≈1.3). **정수 N 어느 하나로도 재현되지 않는다.**
2. 따라서 **coverage 는 지금 상태로 논문과 비교할 수 없다.** 우열 주장을 넣지 말아야 한다.
   논문의 N 이 확인되기 전까지는 N 곡선 전체를 병기하는 것이 유일하게 정직한 서술이다.
3. **달성률(상한 대비)이 더 정보가 많다.** 8×8 은 N=1 에서 0.997 로 사실상 천장이다 —
   모델이 아니라 **test 셋의 해 개수 분포가 coverage 를 결정하고 있다.** 10×10 은 N=1 에서
   0.605~0.898 로 여유가 있어, 여기서는 coverage 가 모델 성능을 실제로 반영한다.
4. **논문 test 셋이 우리와 같은 분포인지 확인해야 한다** — 같은 생성 규칙(k∈{5,6,7} 제거)이라도
   split 이 다르면 해 개수 분포가 달라지고 상한이 움직인다. §5 질문 1 이 이것이다.


---

## 2. 확정 config 와 논문 대비 차이

기준: `checkpoints/Nqueens-8-ACT-torch/nq8_v4_catnd_b007_s2/all_config.yaml` (**server 50**,
최신 8×8 arm).

**29번 config 대조 구멍은 닫혔다** — 29번이 `run_configs/hinton/` 에 64개 `all_config.yaml`
사본을 올렸다(`c195a36`). β≥4.0 대역 + `base_b500` 계열 **29 run** 을 대조한 결과:

| 항목 | 29번 대역 | 50번 기준 | 판정 |
|---|---|---|---|
| `kl_reduction` | **mean (전 run)** | **sum** | **다르다 — 환산하면 같은 영역(§2.1)** |
| `y_onehot` | True (OFF 계열 2개 제외) | True | 일치 |
| `y_inject` | v2/v5 계열만 명시(add/concat), 그 외 **키 없음**(구코드) | concat | **부분 일치** |
| `logvar_bias_init` | v2/v5 계열만 명시(0.0 / −4.0), 그 외 키 없음 | 0.0 | **부분 일치** |
| `nodecay_y_proj` / `nodecay_logvar_bias` | v5 계열만 False/False, 그 외 **키 없음** | True / False | **불일치** |
| `epochs` | 3,000 / 6,000 / 7,000 / 12,000 혼재 | 12,000 | **다르다 — §2.1 의 예산 함정** |
| `lr_min_ratio` / `weight_decay` | 1.0 / 1.0 | 1.0 / 1.0 | 일치 |
| `ema` / `dual_eval` | True / True (`_ema6k` 이후) | True / True | 일치 |
| `hidden_size` / `H,L_cycles` / `halt_max_steps` | 512 / 3,4 / 16 | 동일 | 일치 |

- **최고 수치 run `nq8_v2_base_b500_s0` 의 손잡이**: β 5.0 mean(= sum 0.0625), `y_inject=add`,
  **`logvar_bias_init=−4.0`**, `y_onehot=True`, epochs 7,000, seed 0, EMA+dual_eval.
  **50번 최고(`v2_sum007_catnd_s0`)와 `y_inject`·`logvar_bias_init` 이 정반대다**
  (concat/0.0 vs add/−4.0). 즉 8×8 상위권은 이 두 손잡이에 **둔감**하다 — 둘 다 0.98 이상.
- **`nodecay_*` 키가 29번 구코드 run 에는 없다.** 키 부재는 코드 기본값으로 돈 것이고 기본값이
  버전에 따라 달랐을 수 있어, **29번 구코드 run 과 50번 v4 를 손잡이 단위로 1:1 비교하는 것은
  여전히 불가능하다.** 대역 비교(β)까지만 유효하다.
- **집계에서 제외**: `run_configs/hinton/README.md` 가 명시한 검증용 probe 4개
  (`nd_off_cat_m4`, `nd_off_cat_m4_b`, `nd_yproj_on`, `nd_both_on`, 1~102 step) 와
  `p100_*` / `probe_*` (18 step / 1 step) 는 학습 실험이 아니다.

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

환산식: `KL_sum = L × KL_mean` 이므로 **β_sum = β_mean / L**.

**두 서버의 눈금이 달랐고, 환산하면 같은 영역이었다** (`run_configs/hinton/` 64개 config 대조로
2026-07-30 확인). 이것이 지금까지 β 판단을 어지럽힌 원인이다:

| server | 표기 눈금 | 표기 β | **β_sum 환산** | β_mean 환산 | 대표 run |
|---|---|---|---|---|---|
| 29 | **mean** | 4.0 ~ 5.6 | **0.050 ~ 0.070** | 4.0 ~ 5.6 | `nq8_on_b{400..560}_ema6k`, `nq8_v2_base_b500_s0` |
| 50 | **mean** | 5.0 | **0.0625** | 5.00 | `nq8_on_b500_{s1,s2,decay}` |
| 50 | **sum** | 0.07 | **0.0700** | 5.60 | `nq8_v2_sum007_*`, `nq8_v4_*` |
| — | 논문 | 0.07 | 0.07 (sum 해석) | 5.60 | — |

- **두 서버가 독립적으로 β_sum 0.05~0.07 에 도달했다.** 8×8 상위 수치 전부가 이 구간이다:
  0.9977(29, mean 5.0 = sum 0.0625), 0.9882(29, mean 5.0), 0.9856(29, mean 5.6 = sum 0.070),
  0.9843(50, sum 0.070), 0.9331(50, mean 5.0).
- **즉 논문 0.07 을 sum 으로 읽는 것이 실측과 맞는다.** GC 도 논문 명시값 0.5 를 sum 으로 써서
  논문 수준 결과가 나왔다(§1.3). 10×10 도 같다 — 50번 sum 0.045, 29번 mean 5.2 인데
  0.045 × 116 = 5.22 로 **같은 점**이다.
- **"mean 환산 대역 5.6 이 죽었다"는 과거 결론은 β 문제가 아니라 예산 문제였다.**
  [REPORT_grid_upper_29.md](reports/archive/REPORT_grid_upper_29.md) 가 0.00% 로 보고한
  `nq8_on_b{450,560}_ema` 는 **3,000 epochs** run 이고, 같은 β 를 **6,000 epochs** 로 돌린
  `_ema6k` 는 0.9527 / 0.9856 이다:

| β (mean) | 3,000 epochs | 6,000 epochs | 배율 |
|---|---|---|---|
| 4.5 | `nq8_on_b450_ema` **0.0007** | `nq8_on_b450_ema6k` **0.9527** | **1,360배** |
| 5.6 | `nq8_on_b560_ema` **0.0000** | `nq8_on_b560_ema6k` **0.9856** | — |

  **§3-3(EMA 늦은 수렴) 과 같은 함정이다.** 논문 예산 3,000 epochs 에서 끊으면 정답 β 조차
  0% 로 보인다 — §2.2 와 §5 질문 4 가 여기에 걸린다.

→ **눈금 리스크는 해소됐다.** 남은 것은 "논문이 3,000 epochs 로 0.997 을 냈다면 우리와 무엇이
다른가" 이고, 이것이 이제 최대 미해결이다.

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

| server | 측정 | 예산 | step | bias | σ = exp(bias/2) | 근거 |
|---|---|---|---|---|---|---|
| **29** | 8-run 기준 | **6,000 epochs** | ~34k | **−0.134** | **0.935** | 29번 통보 (이 박스에 ckpt 없음) |
| **50** | `gc8_v5_add_bm4_s0` | **12,000 epochs** | 60,192 | **−0.0603** | **0.970** | 최종 checkpoint 실측 |
| 50 | (init 0 대조) `gc8_v5_add_b0_s0` | 12,000 | 60,192 | −0.0013 | 0.9993 | 최종 checkpoint 실측 |
| 50 | (init 0 대조) `nq8_v4_catnd_b007_s2` | 12,000 | 68,340 | −0.0031 | 0.9984 | 최종 checkpoint 실측 |

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

`nq8_v4_catnd_b007_s2` (**server 50**) 의 in-loop EMA `nqueens/accuracy` 곡선 실측
(`logs/nq8_v4_catnd_b007_s2.log.gz`):

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

| server | 경로 | valid | consistent | strict | queen_count (mean / min–max) |
|---|---|---|---|---|---|
| 50 | **posterior** (정답 y 를 봄) | **0.9534** | **1.0000** | 0.9534 | 7.97 / 7–9 |
| 50 | **prior** | **0.0083** | 0.9907 | 0.0083 | 7.79 / 5–11 |

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

| server | run | `lr_min_ratio` | prior valid (`.live`) | consistent | coverage@20 | EMA valid |
|---|---|---|---|---|---|---|
| 50 | `nq8_on_b500_s1` | **1.0** (상수) | **0.9331** | 0.9999 | 0.9875 | 0.0000 |
| 50 | `nq8_on_b500_s2` | **1.0** (상수) | **0.9075** | 0.9997 | 0.9911 | 0.0076 |
| 50 | `nq8_on_b500_decay` | **0.1** (코사인 감쇠) | **0.0327** | 0.9948 | 0.3172 | 0.0000 |

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
5. **coverage 의 N 이 논문과 다르다(확정)** — §1.4. 우리 헤드라인은 N=20 이고, N=1 로 내리면
   두 task 모두 논문 아래다. 논문값은 정수 N 어느 하나로도 재현되지 않는다.
   8×8 은 N=1 달성률이 0.997 로 천장에 붙어 **모델이 아니라 test 셋 해 개수 분포가
   coverage 를 결정한다.**
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

1. **논문 coverage 의 N 은 몇인가?** §1.4 에서 **"N=1" 가설은 기각됐다** — 8×8 의 N=1 상한
   (0.7909)이 논문 구간 하단(0.884)보다 낮아 원리적으로 불가능하다. 논문값은 두 task 모두
   **N=1 과 N=5 사이**에 떨어지고 정수 N 으로 재현되지 않는다. 논문의 N 을 확인해 주셔야
   coverage 비교가 성립한다. **함께 확인 필요**: 논문 test split 의 해 개수 분포(우리는 8×8 이
   64% 해-1개, 10×10 이 38%)가 같은가 — 다르면 상한이 움직여 비교가 또 깨진다.
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
7. **`nq8_v2_base_b500_s0` (0.9977) 의 결과 JSON 을 받을 수 있는가?** 이 박스에 흔적이 전무해
   검증하지 못했다(§0 주의 3). **논문값 0.997 을 넘는 유일한 수치이므로 최종 보고의 헤드라인이
   될 수 있는데, 지금 상태로는 인용할 수 없다.** JSON + `all_config.yaml` 이 필요하다.
8. **29번 자산을 50번으로 옮길 것인가?** 29번 run 26+개는 결과 JSON 만 있고 가중치가 없어
   재측정·재현이 불가능하다(§0). 최종 보고에 쓸 run 만이라도 checkpoint 를 받아둘지.

---

## 6. 패널 캡션 (wandb `GRAM-repro-summary`)

`python experiments/wandb_harness_upload.py --mode unified --panels --commit` 로 올린다.
표는 `summary/all` (646행 × 27열), 패널은 `panel/*`. **모든 캡션에 축을 박는다.**

**공통 축 규약** — 가중치 `live`(raw SGD) / `EMA`, 집계 **단일시도 = pass@1**,
`step_kind` **peak**(=test set 으로 고른 step, **선택 효과 포함**) / **final**(마지막 step,
선택 효과 없음) / peak+final(둘이 같은 step).

**Panel A — ours vs paper.** 왼쪽은 단일시도 valid, `weights=live`. nq8 은
peak 0.998 / final 0.976 / 논문 0.997 로, **논문값이 우리 peak 과 final 사이에 있다**
— peak 만 인용하면 논문을 넘고, final 만 인용하면 못 넘는다. 격차 2.24%p 가 선택 효과의 크기다.
GC conflict 는 스케일이 달라(0.02 vs 2.7) **오른쪽 별도 축 + log scale** 로 분리했다 —
이중 y축은 쓰지 않았다. nq10 에 final 막대가 없는 것은 29번 측정이 peak+final 한 지점뿐이기 때문이다.

**Panel F — 다양성과 정확도.** distinct@20(1~20)과 단일시도 valid(0~1)은 **다른 축이고,
distinct 는 정확도가 아니다.** 이중 y축 대신 좌우 subplot 으로 분리했다(지시는 "겹치기"였으나
스케일이 달라 겹치면 오독을 만든다 — 분리로 바꿨다). task 별 min/max 를 보여준다:
nq8 은 distinct 가 1.65~2.79 로 거의 결정론인데 valid 가 0.98 이고, gc8 은 distinct 20.00
(20 샘플 전부 다름)인데 valid 0.98 이다 — **다양성과 정확도가 독립임을 보여주는 대비다.**

**Panel G — 실패 3종.** prior vs posterior, 같은 0~1 축, `.live` 우선(없으면 `step_N`).
lr decay 0.033/0.859, forcehalt(β_mean 1.0) 0.008/0.951, 저β(β_sum 0.0009) 0.000/0.997.
**세 경우 모두 posterior 는 살아 있고 prior 만 죽는다** — §3-4 의 posterior shortcut 이
실패 기전이라는 근거다. (저β run 은 EMA off 라 `step_N` 이 곧 raw 다.)

**GC conflict 의 ± 는 run 간 std(논문 축)이다** — 예측 5,100개(255 퍼즐 × 20 샘플) 사이의
std(±0.169)가 아니다. 축을 혼동한 이력이 있다([REPORT_gc8_50.md](reports/archive/REPORT_gc8_50.md) §3-1).
**그리고 우리 GC 3-run 은 순수 seed 반복이 아니라 arm 이 섞인 근사다**(concat s0/s1 + add s0).
논문의 "over runs" 와 같은 축이 되려면 단일 config seed 반복이 필요하다 — seed 2/3 학습 중,
seed 4 대기(§5.1 항목 4).

### 6.1 표의 한계

- **`server=?` 8행** (`gram_nq8_fix_b`, `nq8_b{03,07,15,30}`) — `all_config.yaml` 이 두 박스
  어디에도 없어 서버를 판정하지 못했다. β·손잡이 열도 비어 있다.
- **`metric_type=inloop` 71행은 harness 와 섞어 읽으면 안 된다** — 단일 배치 accuracy 이고
  coverage 가 없다. harness 미측정 run(v4 등)이 표에서 사라지지 않게 넣은 것이다.
- **`beta_sum`/`beta_mean` 은 `kl_reduction` 키가 있는 run 만 채워진다.** 29번 구코드 run 은
  키가 없어 눈금 불명이고 두 열이 비어 있다(§2).
- probe run 제외: `nd_off_cat_m4`, `nd_off_cat_m4_b`, `nd_yproj_on`, `nd_both_on`, `p100_*`, `probe_*`.
