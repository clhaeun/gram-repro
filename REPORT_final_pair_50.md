# REPORT_final_pair_50 — pair 확장 데이터 최종 라운드 집계 (2026-08-02 23:31 마감)

데이터: pair 확장 (`data/nqueens-{8,10}`, test = nq8 1,270행 / nq10 19,220행).
축 규약: **헤드라인 = 최종 step × EMA** (nq8 67,428 / nq10 58,772), run 값 = harness
seed 0/1 평균, mean±std 는 학습 seed 3개(또는 가용 수)에 대해. 결과 40건 전부
`fin_*.json` 으로 즉시 커밋됨. 이전 v5h/v6h 32건은 **-uniq 조건**(test 입력당 1행)이라
직접 비교 금지 — 별도 보관.

## 1. 8×8 (add arm; catnd·add_s2 는 29번 몫이라 이 표엔 없음)

| 지표 | EMA ★헤드라인 | .live 병기 | 논문 |
|---|---|---|---|
| 단일시도 strict | **99.4±0.5** [99.72/98.99] | 96.8±2.2 | **99.7±0.3** |
| coverage@20 (퍼즐) | 96.5±0.1 | 95.4±1.3 | — |
| coverage@20 (pair) | **86.8±0.2** | 84.5±2.8 | **90.3±1.9** |
| pass@20 | 100.0±0.0 | 100.0±0.0 | — |
| majority | 100.0±0.0 | 99.8±0.2 | — |
| verifier | 99.4±0.3 | 97.7±2.3 | — |

- 2-seed (s0/s1). 29번 add_s2 합류 시 3-seed 로 갱신.
- coverage 상한(퍼즐/pair 모두 100% — test 최대 해 18개 < N=20)이므로 pair 축
  86.8 vs 논문 90.3 의 3.5pp 는 상한이 아니라 실제 미회수분.
- accuracy 는 논문 재현권(99.4±0.5 vs 99.7±0.3, ±겹침).

## 2. 10×10 (add arm, 3-seed 완비)

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

## 4. 남은 것

- 29번: nq8_final_add_s2, nq8_final_catnd_s0/s1/s2 → 합류 시 nq8 3-seed / catnd 표 갱신.
- FPRM arm (A3′ → A1+A2, 1차 판정 diag/z_h_norm boundedness) 은 이 표 마감 후
  별도 착수 — 이 표에 섞지 않는다 (사수 [1]/[3]).
- 집계 재현: `python experiments/aggregate_final.py` (fin_*.json 만 있으면 됨).
