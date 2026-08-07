# REPORT_final_pair_50 — pair 확장 데이터 최종 라운드 집계 (2026-08-02 23:31 마감)

데이터: pair 확장 (`data/nqueens-{8,10}`, test = nq8 1,270행 / nq10 19,220행).
축 규약: **헤드라인 = 최종 step × EMA** (nq8 67,428 / nq10 138,534 — §2 정정 참조,
구 기준 58,772 는 §2a), run 값 = harness
seed 0/1 평균, mean±std 는 학습 seed 3개(또는 가용 수)에 대해. 결과 40건 전부
`fin_*.json` 으로 즉시 커밋됨. 이전 v5h/v6h 32건은 **-uniq 조건**(test 입력당 1행)이라
직접 비교 금지 — 별도 보관.

## 1. 8×8 (add arm; catnd·add_s2 는 29번 몫이라 이 표엔 없음)

| 지표 | EMA ★헤드라인 | .live 병기 | 논문 |
|---|---|---|---|
| 단일시도 strict | **99.4±0.5** [99.72/98.99] | 96.8±2.2 | **99.7±0.3** |
| coverage@20 — 행 평균비 (구표기 "퍼즐" ← ★오표기였음) | 96.5±0.1 | 95.4±1.3 | — |
| coverage@20 — 행 합계비 (구표기 "pair") | **86.8±0.2** | 84.5±2.8 | **90.3±1.9**† |
| coverage@20 — 퍼즐 축 (unique 772 재집계) | 미측정‡ | 미측정‡ | — |
| pass@20 | 100.0±0.0 | 100.0±0.0 | — |
| majority | 100.0±0.0 | 99.8±0.2 | — |
| verifier | 99.4±0.3 | 97.7±2.3 | — |
| queen_count_mean (전샘플) | 8.000±0.000 | 8.000±0.011 | — |

- 2-seed (s0/s1). 29번 add_s2 합류 시 3-seed 로 갱신.
- ★ **coverage 축 확정 (2026-08-06)**: 86.8±0.2 는 `fin_nq8_final_add_{s0,s1}_step67428_ema_s{0,1}.json`
  의 found_mean/total_mean **행 합계비** (s0 86.95/86.39, s1 86.81/87.00), 96.5±0.1 은 같은 파일의
  `coverage` 필드 = **행 단위 평균비** (행 1,270개 축 — unique 퍼즐 772개 축이 아니므로 구표기
  "(퍼즐)" 은 오표기). 두 축 격차 9.7pp (nq10 은 34pp).
- † 논문 Table 1 의 90.3±1.9 는 **집계 축 미기재** — 우리 어느 축과 대응하는지 확정 불가, 사실 병기.
- ‡ 진짜 퍼즐 축(772 unique input 등가중)은 fin_ JSON 에 per_row 원자료가 없어 재집계 불가.
  신판 sweep_harness 재실행(GPU ~10분) 필요 — GPU 금지 지시 하에 미실행, 승인 시 채움.
- EMA 축 정상성: queen_count_mean EMA 7.9998~8.0003 (8 근처, 퇴화 아님 → live 재측정 불요).
- coverage 상한(두 축 모두 100% — test 최대 해 18개 < N=20)이므로 행 합계비
  86.8 vs 논문 90.3 의 3.5pp 는 상한이 아니라 실제 미회수분.
- accuracy 는 논문 재현권(99.4±0.5 vs 99.7±0.3, ±겹침).

## 2. 10×10 — ★ A0 기준값 정정 (사수 지시 2026-08-04): `nq10_repro_s0/s1`

A0 기준값을 논문 예산 재현 run(990ep, 최종 step **138,534**)으로 교체한다.
harness 는 2차 최소화 신판 `sweep_harness.py` (지표 4종, N=20, hseed 0/1 평균 —
hseed 편차 최대 0.5pp). **구판 fin_* 지표(단일시도 strict, pair coverage 등)와
눈금이 달라 §2a 구 기준값과 직접 비교 금지.**

최종 step 138,534 × pair 축 × hseed 0/1 평균, 2-seed(s0/s1) mean±std:

| 지표 | EMA ★헤드라인 | live 병기 | 논문 |
|---|---|---|---|
| prior 전샘플 valid | 74.2±31.1‡ | 83.8±0.5 | **89.7±2.7** |
| coverage@20 micro(행) | 80.9±7.9 | 83.9±0.1 | **57.5±3.4** |
| coverage@20 macro(퍼즐) | 93.8±4.3 | 95.5±0.5 | — |
| majority@20 valid | 96.9±4.3 | 98.5±1.5 | — |
| verifier@20 valid | 81.3±24.5 | 95.2±4.5 | — |
| nqueens/queen_count_mean | 9.60±0.40 | 9.61±0.18 | — |

run × step × 가중치 전개 (단일시도 / cov micro / cov macro / majority / verifier / queen_cnt):

| run | step | 가중치 | 데이터축 | 단일시도 | cov micro | cov macro | maj | vsel | queen_cnt |
|---|---|---|---|---|---|---|---|---|---|
| s0 | 138,534 최종=EMA정점 | EMA | pair | 96.22 | 86.52 | 96.86 | 99.97 | 98.58 | 9.881 |
| s0 | 138,534 | live | pair | 84.12 | 83.96 | 95.79 | 99.56 | 98.39 | 9.483 |
| s0 | 134,336 raw정점 | EMA | pair | 96.01 | 86.49 | 96.80 | 99.94 | 98.58 | 9.882 |
| s0 | 134,336 | live | pair | 93.14 | 85.19 | 96.03 | 99.63 | 98.36 | 9.814 |
| s1 | 138,534 최종=raw정점 | EMA‡ | pair | 52.17 | 75.30 | 90.72 | 93.82 | 63.99 | 9.311 |
| s1 | 138,534 | live | pair | 83.44 | 83.76 | 95.15 | 97.51 | 92.04 | 9.730 |
| s1 | 67,168 EMA정점 | EMA | pair | 88.55 | 85.68 | 96.53 | 99.30 | 91.05 | 9.817 |
| s1 | 67,168 | live | pair | 49.63 | 74.90 | 89.47 | 88.26 | 77.80 | 9.804 |

- ‡ **s1 은 최종 step 에서 EMA 가 live 대비 비정상적으로 낮다** (52.2 vs 83.4 —
  in-loop 곡선상 88k~126k EMA≈0 붕괴 후 회복 중 상태). EMA ★ 열의 ±31.1 은 이
  한 run 이 지배 — 사실 표기만, 별도 탐지 없음.
- queen_count_mean 은 in-loop eval 로그(`[EVAL @ step]`/`[EVAL RAW @ step]`)에서
  추출 (harness 재계산 아님).
- 정점 병기: s0 은 정점=최종 (raw정점 134,336 도 동급). s1 EMA 정점 67,168 은
  88.6 으로 최종(52.2)보다 훨씬 높다 — 정점 이후 EMA 축이 내려앉는 패턴
  (REPORT_ema_collapse.md 계열).
- 검산: n_found_gt_total 은 16건 중 14건에서 0, live 축 2건에서 미소 초과
  (s0@134,336 live 2행, s1@67,168 live 14행 / 38,440 행-평가) — sidecar 해 수
  과소 등재 가능성, coverage 에 유의미한 영향 없음.
- coverage micro 는 논문 57.5±3.4 를 크게 상회 (83.9±0.1 live) — 논문 coverage 의
  측정 절차(축/절단) 차이 가능성 있음, 사실 병기.
- 각주: 학습(s0/s1)은 구 ema.py(bias correction 포함), 평가는 신 ema.py(ed6e3e2,
  TRM 원본). 138k step 예산에서 bias correction 효과는 잔존 init 0.0001% 로 무시
  가능 (사수 확인 2026-08-04).
- **참고 행: `nq10_repro_s2` — crash 미완주 (마지막 harness 실측 step 125,940),
  기준값에서 제외.** harness 실측 8건 = `fin_nq10_repro_s2_step{104950,125940}_{ema,live}_s{0,1}.json`
  (92b2981). hseed 0/1 평균 (prior / maj / vsel / cov micro):
  125,940 EMA 89.1/99.6/95.0/86.4 · live 51.0/86.5/87.1/69.1;
  104,950 EMA 95.1/99.9/94.5/88.0 · **live 0.0/0.0/0.0/0.0 (live 축 붕괴 —
  EMA 는 정상, §2 s1 의 EMA 붕괴와 거울상)**.
  seed 1개 방침으로 재학습 중단 (사수 지시 2026-08-04). 50번 재시도 사본
  `nq10_repro_s2_crashed0804/`(58,772 에서 08-04 05:40 리부트로 중단)은 별개 —
  위 실측과 혼동 금지.
- `run_harness_nq10_29.sh` 는 24건을 계획했으나 결과는 8건. `nq10_fp_a3p_s1` 12건과
  raw 정점 88158 은 미측정. checkpoint 는 남아 있어 재측정 가능.

## 2a. 구 기준 기록 — `nq10_final_add` 3-seed (420ep, 최종 58,772; §2 로 대체됨)

catnd arm 은 최종 설정(y_inject=add 하드코딩)에 없는 arm 이라 논문 대비 표에서
제외 — §2b 탐색 라운드 기록으로 이동 (사수 지시 2026-08-04).

| 지표 | add EMA ★ | add live | 논문 |
|---|---|---|---|
| 단일시도 strict | 62.2±13.3 | 70.5±14.3 | **89.7±2.7** |
| coverage (퍼즐) | 81.7±3.6 | 82.1±4.3 | — |
| coverage (pair) | 43.9±5.1 | 45.7±4.8 | **57.5±3.4** |
| pass@20 | 99.9±0.1 | 99.9±0.1 | — |
| majority | 93.8±6.2 | 97.6±2.3 | — |
| verifier | 46.0±19.5 | 80.4±17.3 | — |

seed 별 단일시도 strict (EMA / live):
- add: s0 50.61/54.77, s1 76.75/82.56, s2 59.28/74.20

coverage 상한: 퍼즐 99.79% / pair 95.14% (해 21개 이상 입력의 N=20 절단) — pair 축
43.9~45.7 vs 논문 57.5±3.4 의 격차는 상한(95.1)이 아니라 실제 미달.

## 2b. 탐색 라운드 기록 — catnd arm (논문 대비 제외)

y_inject=concat(no-decay) 탐색 arm. 최종 설정에 채택되지 않았고 minimal 이후
코드로는 로드 불가 (legacy/gram_legacy.py 필요) — 기록 보존용.

| 지표 | catnd EMA | catnd live |
|---|---|---|
| 단일시도 strict | 42.3±36.8† | 72.9±7.0 |
| coverage (퍼즐) | 54.5±47.1† | 82.8±3.0 |
| coverage (pair) | 30.2±26.2† | 46.7±3.5 |
| pass@20 | 66.7±57.3† | 99.9±0.1 |
| majority | 61.6±53.6† | 97.5±0.8 |
| verifier | 37.8±33.4† | 87.0±4.4 |

- seed 별 단일시도 strict (EMA / live): s0 **0.03**/75.69, s1 67.59/64.91, s2 59.15/77.99
- **† catnd EMA 통계는 s0 의 EMA 축 0.03% (같은 checkpoint live 는 75.7%)가 지배해
  mean±std 가 무의미하다.** † 원인은 이후 **EMA puzzle_emb 누락**으로 확인됨
  (EMA shadow 병리가 아니라 catnd 경로의 EMA 스냅샷 결손 — 사수 확인 2026-08-04).

## 3. 관찰 (요약 없이 사실 병기)

1. **nq8 accuracy 논문 재현**: 99.4±0.5 (2-seed) vs 99.7±0.3. coverage pair 축은
   86.8±0.2 로 논문 90.3±1.9 대비 −3.5pp (상한 100%).
2. **nq10 은 EMA < live**: add 62.2 vs 70.5. 58,772 step / 420 epochs 에서
   EMA(mu 0.9999)가 아직 raw 를 못 따라간 lag 로 읽힌다. 사수 규약의
   "최종 step × EMA" 헤드라인이 nq10 에선 live 대비 체계적으로 낮게 나온다 — 사실 병기.
3. catnd_s0 EMA 이상치는 §2b 로 이동 — 원인은 EMA puzzle_emb 누락으로 확인
   (당초 "EMA shadow 병리 재발" 해석은 철회).
4. **verifier 선택은 EMA 에서 특히 불안정** (add EMA 46.0±19.5 vs live 80.4±17.3).
5. **harness seed 편차는 무시 가능** (strict 최대 0.2pp) — 편차 축은 학습 seed.
6. 학습 seed 편차가 nq10 에서 큼 (add ±13~14pp). 논문 ±2.7 과 대비.

## 5. GC-8 논문 예산 5K 재현 — `gc8_repro_s0/s1/s2` (Table 7 대조, 2026-08-04)

학습: EPOCHS=5000 EVALINT=250, 총 step **9,400** (지시문 45,540 은 실측과 상이 —
groups 1,445 × 5,000 / 768 = 9,407 진행바, 저장 눈금 9,400). s0 은 사수측 launch
(root 소유), s1/s2 는 50번 launch. **셋 다 신 ema.py(ed6e3e2, bias correction 없음)
로 학습.** harness = 신판 gc_harness (canonical, selftest 4종 PASS), 최종 step ×
{EMA, live} × hseed 0/1 평균, 3-seed mean±std:

| 지표 | EMA ★헤드라인 | live 병기 | 논문 Table 7 |
|---|---|---|---|
| conflict (mean) | 8.32±2.35§ | 2.31±0.23 | **2.7±2.1** |
| coverage@20 (canonical) | 0.05±0.04§ | 15.38±5.79 | **85.8±0.5** |
| majority@20 canonical | 0.00±0.00§ | 7.12±3.32 | — |
| verifier@20 | 0.00±0.00§ | 8.37±2.72 | — |

- § **EMA 축은 3-seed 전부 붕괴 — run 병리가 아니라 산술적 필연.** 9,400 step ×
  mu 0.9999 ⇒ shadow 에 init 이 mu^9400 = **39.1%** 잔존. 신 ema.py 에는 이 항을
  제거하던 bias correction 이 없다 (구 ema.py 였다면 (shadow−mu^t·w0)/(1−mu^t) 로
  제거). nq10 138k step(잔존 1e-6)과 달리 GC 5K 예산에서는 "최종 step × EMA"
  헤드라인이 성립하지 않는다 — 사실 병기.
- 교차검산: in-loop `[EVAL @ 9400]` s1 conflicts_mean 8.22 ↔ harness EMA 8.13 (정합).
- live 축: conflict 2.31±0.23 은 논문 2.7±2.1 재현권. coverage 15.4±5.8 은 논문
  85.8±0.5 대비 대미달.
- **별도 표기 — 논문 예산 6.4배** (`gc8_v5_add_b0_s0`, 32,000ep = 60,192 step,
  구 ema.py 학습, 신판 gc_harness 재실측 hseed 0/1 평균):
  EMA conflict 0.034 / cov **90.2** / maj 99.8 / vsel 98.4;
  live conflict 0.179 / cov 79.2 / maj 95.5 / vsel 93.7.
  6.4배 예산에서는 coverage 가 논문 수준(85.8)을 상회 — 5K 미달이 예산 축 문제임을
  시사 (논문은 5K 로 85.8 주장 → 절차 차이 여지, 사실 병기).

## 5b. G3 게이트 — μ/σ 스케일 그리드 (2026-08-05, `reports/g3/`)

추론 시 prior eps = mu_scale·μ + sigma_scale·σ·noise 로 스케일 (학습 없음,
`sweep_harness --mu-scale/--sigma-scale`). 논문 Table 3b 는 학습 시 ablation 이라
직접 대응은 아니다 (13cad12 주석). coverage 는 달성률 평균 / 합계 비 병기.

**표 C — nq10 A0** (`nq10_repro_s0/step_138534`, EMA; 기준 (1,1) acc 96.21 /
cov 86.52 / 합계비 51.96):

| (μ,σ) | acc | cov 달성률 | cov 합계비 |
|---|---|---|---|
| (0,1) | 23.04 | 59.13 | 22.71 |
| (1,0) | 85.49 | 27.21 | 8.70 |

★ **nq10 에서 A0 는 μ 의존이 실재한다 — μ 제거 시 acc 96.21 → 23.04.
논문 Table 3b 의 방향(99.69 → 50.27)과 일치.
nq8 에서 μ 무의존으로 보였던 것은 backbone 포화(99.4%)에 의한 착시다.
문제가 어려워지면 μ 가 실제 정보를 나른다.**
("우리 GRAM 은 latent 를 안 쓴다"는 우려가 nq8 특유 현상으로 해소됨.)

**표 A — nq8 A0** (`nq8_final_add_s0/step_67428`, EMA; 기준 (1,1) acc 99.73 /
cov 96.55 / 합계비 86.95): acc 는 σ=2 붕괴 외 μ/σ 무의존 — (1,0) 99.53,
(0,0) 98.19, (0,1) 99.23 (논문 σ=0→0.00, μ=0→50.27 모두 미재현 = 위 착시).
coverage 만 σ 의존: σ=0 에서 60.55/33.67 로 하락, (0,1) 96.10/85.27.
σ=2 는 전멸 (0.00). 전개값은 `reports/g3/nq8_final_add_s0_mu*_sg*.json` 8건
+ `reports/g3/A0_nq8_from_log.md`.

**표 B — nq8 A3′** (`fp_a3p_nq8_s0/step_67428.live`): ★ EMA 축은 폐기하고
live 로 재측정했다. 기준 (1,1) 이 EMA acc 3.19 vs live 68.16 으로 EMA 가중치
자체가 퇴화 — 퇴화 축 위의 그리드라 μ/σ 해석 불능이었다 (σ 민감도가 μ 효과처럼
보이는 오독 발생, 정정됨). live 재측정은 기준 (1,1) 을 첫 셀로 실측해 기존
harness live 값(68.16/90.40/75.44)과 일치함을 선확인했다 (CHECKLIST §7).
EMA 폐기표 원본은 `reports/g3/fp_a3p_nq8_s0_mu*_sg*.json` 7건에 남긴다.

| (μ,σ) live | acc | cov 달성률 | cov 합계비 |
|---|---|---|---|
| (1,1) 기준 | 68.16 | 90.40 | 75.44 |
| (1,0) | 32.68 | 20.55 | 11.05 |
| (1,0.5) | 49.80 | 69.56 | 48.75 |
| (0,0) | 25.20 | 15.20 | 8.52 |
| (0,0.5) | 30.67 | 53.65 | 36.92 |
| (0,1) | 27.38 | 55.91 | 39.16 |
| (1,2)·(0,2) | 0.00 | 0.00 | 0.00 |

A3′ 는 nq8 에서도 latent 의존이 뚜렷하다: μ 제거 시 acc 68.16 → 27.38 (−40.8pp,
논문 방향), σ 제거 시 acc 68.16 → 32.68 + cov 90.40 → 20.55. A0(표 A) 와의
대조 = "backbone 이 포화한 A0 만 latent 잉여, FPRM arm 은 latent 를 쓴다".

★★ **σ=0 대비가 이 게이트의 핵심이다 — A3′ 는 노이즈가 성능의 본체다.**

| σ=0 (노이즈 완전 제거) | acc | cov 달성률 |
|---|---|---|
| **A0** `nq8_final_add_s0` (1,0) | **99.53** | 60.55 |
| **A0** (0,0) | **98.19** | 59.69 |
| **A3′** `fp_a3p_nq8_s0` live (1,0) | **32.68** | 20.55 |
| **A3′** live (0,0) | **25.20** | 15.20 |

같은 8×8 문제에서 노이즈를 끄면 A0 는 99.53 → 98.19 로 거의 안 움직이는데
A3′ 는 68.16 → 32.68 로 절반 이하가 된다. σ=0 셀은 20 샘플이 전부 같은 답이 되어
acc = maj@20 = vsel@20 이 정확히 일치하는데(다양성 0), **그 상태에서도 A0 는 98%
유효해를 낸다.** 즉 A0 에게 노이즈는 coverage 를 넓히는 다양성 장치일 뿐이고
정답 능력 자체는 backbone 이 이미 갖고 있다. A3′ 는 그 반대로, 노이즈를 빼면
정답 능력 자체가 무너진다 — **FPRM arm 에서 σ 는 탐색 폭이 아니라 계산의 본체다.**

곁들여, 양쪽 다 σ=2 에서 전멸(0.00)한다. 즉 학습된 σ 눈금이 정점이고 위아래
어느 쪽으로 틀어도 손해다 — inference-time 에 σ 를 손볼 여지는 없다.

## 5c. nq10 Stage A — lr 축 판정 (2026-08-05, 29번)

`residual_scale=input-independent`, α₁/α₂ init 0.75/0.25, `nodecay_marked=True`,
990ep / EVALINT=30 (총 138,548 step). seed 0, 2셀:
`nq10_A3p_ii_a7525_lr3e4` (lr 3e-4, warmup 500) / `nq10_A3p_ii_a7525_w2000`
(lr 1e-4, warmup 2000). 둘 다 8시간 20분 내외.

> **lr 3e-4 는 수렴을 3배 앞당기지만 nq10 138k 예산을 완주하지 못한다.**
> 100k 이후 KL 스파이크가 재발해 128k 클러스터 → 130k eval 붕괴 →
> 137,100 NaN 으로 이어진다. 같은 예산에서 lr 1e-4(w2000)는 warmup 이후
> 스파이크 0건으로 완주했다. **이는 lr schedule 문제가 아니라 lr 크기 자체의
> 문제다** — 두 논문 모두 constant LR 이 명시 설정이다.

근거 1 — 수렴 속도 (live strict_accuracy, in-loop):

| step | lr3e4 | w2000 | A0(`nq10_repro_s2`) |
|---|---|---|---|
| 8,396 | **0.4071** | 0.0006 | 0.0000 |
| 25,188 | 0.6504 | 0.0029 | 0.5725 |
| 83,960 | 0.8410 (정점) | 0.4684 | — |
| 113,346 | 0.6210 | **0.7671** (정점) | — |

lr3e4 는 8,396 step 에서 이미 0.41 로, A0 가 같은 값에 도달하는 25k 대비 3배 빠르다.
w2000 은 이륙이 67k 늦었으나 113,346 에서 0.7671 정점을 찍고 138,534 까지 완주했다.

근거 2 — `train/kl_loss > 5` 스파이크 (50 step 간격 로그, 20k 구간별):

| 구간 | lr3e4 max / 건수 | w2000 max / 건수 |
|---|---|---|
| 0~20k (warmup) | 768.60 / 19 | 3056.44 / 35 |
| 20k~60k | 34.29 / 3 | 3.19 / **0** |
| 60k~100k | 4.37 / **0** | 2.19 / **0** |
| **100k~120k** | **14.20 / 3** | 1.72 / **0** |
| **120k~140k** | **48.86 / 6** | 2.30 / **0** |

w2000 은 warmup 이후 138k 까지 스파이크 0건. lr3e4 는 60k~100k 를 조용히 통과한 뒤
후기에 불안정이 재발했다 — 초기 과도기 폭주가 아니라 **장기 학습 중 축적된 불안정**이다.

근거 3 — 사건 순서:

```
112,750 / 118,100 / 118,250 / 123,150   산발 스파이크 재발
128,100~128,350                          5연속 클러스터 (최대 48.86, 250 step 내)
   130,138 eval                          양축 동시 붕괴 (EMA 0.0003 / live 0.0104)
   134,336 eval                          live 0.6332 회복, EMA 0.0000 · qc 17.41
      137,100                            NaN 발산 (전 지표 동시)
   138,534 eval                          전 지표 0, 가중치 26/37 텐서 NaN
```

- **유보**: 로그가 50 step 간격이라 137,051~137,099 의 트리거는 관측 불가.
  직전 700 step(136,400~137,050)은 train acc 0.93~0.98 로 정상이었다 — 국소적으로는
  전조가 없다. 전조는 9,000 step 규모의 장기 축에서만 보인다.
- **schedule 각주**: 두 셀 다 warmup 이후 lr 상수(`lr_min_ratio=1.0`). 이는 FPRM 부록
  전 태스크(Sudoku p29 / Maze p30 "constant LR after 2,000-step warm-up",
  state-tracking p31 "no LR warm-up", ARC p32 "constant LR with no warm-up")와
  정합적인 명시 설정이며 우리 임의 선택이 아니다. GRAM 논문은 schedule 을 언급하지 않는다.
  **FPRM 부록의 "stepsize-decay 0.997" 은 learning rate 가 아니라 fixed-point iteration
  의 damping η (Algorithm 1) 다** — 추론 시 수렴용이고 우리가 이식하지 않은 기여 2 에
  속한다. 혼동 금지.
- α 학습은 두 셀 모두 확인됨 (wd 붕괴 아님). live 축 sigmoid, init 0.75/0.25 →
  최종 lr3e4 0.6312/0.3527 (gap 0.279, std 0.477/0.722) ·
  w2000 0.6707/0.2463 (gap 0.424, std 0.090/0.173). 평균이 모이면서 원소별 std 가
  커지는 형태라 wd 붕괴(전 원소 logit 0 수렴)와 반대다.

## 5d. 노이즈 head 상태 — state_dict 직접 실측 (2026-08-07, 50번)

사실만 기록한다. 해석은 붙이지 않는다.

측정 대상 (둘 다 완주분):
- nq8 : `checkpoints/Nqueens-8-ACT-torch/nq8_final_add_s0/step_67428`
- nq10: `checkpoints/Nqueens-10-ACT-torch/nq10_repro_s0/step_138534`

**★ β 가 다르다 — 두 task 의 KL 값을 직접 비교할 때 반드시 명기할 것.**
`all_config.yaml` 실측: nq8 `arch.kl_coef=0.07`, nq10 `arch.kl_coef=0.045`.
`kl_balance` 는 둘 다 0.8. kl_reduction 은 sum 고정 (`losses.py:90-95`, 분기 없음).

logvar bias (init 값은 `gram.py:214-215` 의 `nn.Parameter(torch.tensor(0.0))`):

| | nq8 | nq10 |
|---|---|---|
| `prior_logvar_bias` | −0.00429853 (exp(0.5v)=0.99785304) | −0.00319413 (exp(0.5v)=0.99840421) |
| `post_logvar_bias` | −0.00430220 (exp(0.5v)=0.99785121) | −0.00318404 (exp(0.5v)=0.99840925) |

head weight L2 norm (SwiGLU `down_proj.weight`+`gate_up_proj.weight`, numel 786,432):

| head | nq8 L2 | nq10 L2 |
|---|---|---|
| `prior_mu` | 3.1643 | 2.8651 |
| `prior_logvar` | **3.9893** | **3.5515** |
| `post_mu` | 3.2843 | 3.1373 |
| `post_logvar` | **3.9645** | **3.5380** |

★ logvar head 의 L2 가 mu head 보다 크다 (nq8 3.99 vs 3.16, nq10 3.55 vs 2.87).
  즉 logvar head 가 weight decay 로 소거된 상태는 아니다.

★ σ 의 학습 중 궤적은 두 run 모두 **존재하지 않는다**. `logs/nq8_final_add_s0.log`
  (2026-08-02) 와 `logs/nq10_repro_s0.log` (2026-08-03) 는 문자열 `diag` 출현 0 회이며,
  두 run 은 `_latent_diag`(`gram.py:290-303`) 도입 이전이다. 런처가
  `WANDB_MODE=online` 이라 `wandb/` 하위 offline run 에도 없다.

transition 별 KL/μ/σ 는 `sweep_harness.py --diag-kl` (커밋 8b7a5f1) 로 잰다 —
산출물은 `<out>.diagkl.json`.

### 5d-1. transition 별 KL/μ/σ 실측 (2026-08-07)

산출 JSON (둘 다 `--seed 0`, hseed 0, EMA 가중치):
- nq8 : `--N 20`, `n_forwards=960`   (β=0.07)
- nq10: `--N 4`,  `n_forwards=2432`  (β=0.045)
★ N 과 β 가 다르다 -- 두 task 의 `kl_t` 를 직접 비교할 때 반드시 병기할 것.

| task | t | `kl_t` | `mu.abs().mean()` | `mu.norm(dim=-1).mean()` | `std.mean()` | `std.norm(dim=-1).mean()` | `std.std(dim=-1).mean()` |
|---|---|---|---|---|---|---|---|
| nq8 | 1 | 0.6029170103681584 | 0.04204483263505002 | 1.1911979079246522 | 0.9890838551024596 | 22.386782733599343 | 0.0025737155711491747 |
| nq8 | 2 | 0.5339916805426280 | 0.0425424203296036 | 1.2080980962763230 | 0.9893979130312800 | 22.394055936733880 | 0.0024753337636260160 |
| nq8 | 3 | 0.5025137939723209 | 0.0433026498144802 | 1.2316266297052303 | 0.9894596561168630 | 22.395473478237786 | 0.0024468890323381250 |
| nq10 | 1 | 0.7833457380337151 | 0.0490491197484324 | 1.3833711289212500 | 0.9894356484720973 | 22.394562700861380 | 0.0032848917306086820 |
| nq10 | 2 | 0.7553918048077705 | 0.0473762411152404 | 1.3478293804647892 | 0.9894367023126075 | 22.394915220768826 | 0.0032520941947854895 |
| nq10 | 3 | 0.7566729099857376 | 0.0486650494046778 | 1.3855957292804593 | 0.9896416487417331 | 22.399604192689846 | 0.0032147157686076276 |

계산식 (`models/recursive_reasoning/gram.py`), `std_q = torch.exp(0.5*logvar_q.float())` (`:343`):
`kl_t` = `kl.sum(-1).sum(-1).mean()` (`:344`, D축 합 -> L축 합 -> batch 평균) ·
`eps_mu_norm` = `mu_q.float().norm(dim=-1).mean()` (`:348`) ·
`eps_std_norm` = `std_q.norm(dim=-1).mean()` (`:349`) ·
`eps_mu_absmean` = `mu_q.float().abs().mean()` (`:350`) ·
`eps_std_mean` = `std_q.mean()` (`:351`) ·
`eps_std_dimstd` = `std_q.std(dim=-1).mean()` (`:355`).

파생값 (t1 기준, D=512 / √D=22.627417):

- `std.std(dim=-1) / std.mean()` = nq8 **0.26%** (0.0025737/0.98908) ·
  nq10 **0.33%** (0.0032849/0.98944)
- logvar_bias 만으로 예측되는 std = nq10 **0.99840** (`exp(0.5*-0.00319413)`),
  실측 `std.mean()` = **0.98944**, 차이 **−0.0090**
- `mu.norm/√D` = nq8 **0.0527** / nq10 **0.0612** (task 간 다름) ·
  `std.mean()` = nq8 **0.9891** / nq10 **0.9894** (task 간 같음)
- `kl_t1/kl_t3` = nq8 **1.200** (0.6029170/0.5025138) ·
  nq10 **1.035** (0.7833457/0.7566729)

## 6. 남은 것

- 29번: nq8_final_add_s2, nq8_final_catnd_s0/s1/s2 → 합류 시 nq8 3-seed / catnd 표 갱신.
- FPRM arm (A3′ → A1+A2, 1차 판정 diag/z_h_norm boundedness) 은 이 표 마감 후
  별도 착수 — 이 표에 섞지 않는다 (사수 [1]/[3]).
- 집계 재현: `python experiments/aggregate_final.py` (fin_*.json 만 있으면 됨).
