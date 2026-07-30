# 29번(hinton) in-loop 곡선 요약 — v2 base + v5 4-run

50번 REPORT_FINAL 의 "(미검증)" 행에 **어느 step 을 harness 로 재야 하는지**를 채우기 위한 표다.
harness 는 GPU 가 필요해 대기 중이고, 아래는 GPU 없이 로그에서 뽑은 값이다.

## ★ 지표 축 (혼동 금지)

    metric_type = inloop      경로: pretrain.py 의 eval 루프 (harness 아님)
    표본        = 1           퍼즐당 예측 1개.  evaluators/nqueens.py:99-118 이
                              preds["preds"] 한 장만 누적한다.  pass@k 도 majority 도 아니다.
    기준        = valid        제약 만족만.  input 보존(strict)은 별도 키.
    집계        = 단일시도     nqueens/accuracy = valid.sum() / 본 퍼즐 수
    raw         = .live        비EMA(원 SGD) 가중치.  로그 키 nqueens/accuracy_raw
    EMA         = step_N       편향보정 EMA 사본.    로그 키 nqueens/accuracy

**in-loop 에는 coverage 지표가 없다** (`evaluators/nqueens.py:19` 주석: "in-loop eval은 input당
1샘플이다. coverage는 experiments/eval_harness.py 담당"). coverage 를 인용할 때는 반드시
harness JSON 출처를 붙여야 한다.

## 표

출처: `logs/<run>.log` 의 `[EVAL RAW @ step N]` / `[EVAL @ step N]` 라인.
공통 설정: β=5.0 (kl_reduction=mean), y_onehot=True, ema_rate=0.9999, dual_eval=True,
lr_min_ratio=1.0, nodecay 두 플래그 False.

| run | 라운드 | y_inject | logvar_bias | epochs | evals | 정점 step (raw) | raw 정점 | 정점 step (EMA) | EMA 정점 | 최종 step | 최종 raw | 최종 EMA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nq8_v2_base_b500_s0 | v2 | add | −4.0 | 7,000 | 35 | **37587** | **99.87%** | 1139 | 0.00% | 39865 | 97.28% | 0.00% |
| nq8_v5_add_b0_b500_s0 | v5 | add | 0.0 | 12,000 | 60 | **54672** | **100.00%** | 56950 | 99.61% | 68340 | 98.19% | 99.35% |
| nq8_v5_add_b0_b500_s1 | v5 | add | 0.0 | 12,000 | 60 | **43282** | **100.00%** | 66062 | 100.00% | 68340 | 98.58% | 100.00% |
| nq8_v5_add_b0_b500_s2 | v5 | add | 0.0 | 12,000 | 60 | **66062** | **100.00%** | 68340 | 99.87% | 68340 | 98.19% | 99.87% |
| nq8_v5_add_bm4_b500_s1 | v5 | add | −4.0 | 12,000 | 60 | **53533** | **100.00%** | 66062 | 100.00% | 68340 | 99.48% | 99.61% |

## harness 대기열 (우선순위)

1. `nq8_v2_base_b500_s0` — **정점 37587 은 측정 완료**
   (`band_nq8_v2_base_b500_s0_{live,ema}_s{0,1}.json`, harness seed 0/1 평균
    단일시도 valid 99.77% / coverage@20 99.64%).
   최종 39865 도 측정 완료 (`..._step39865_...json`, 97.53% / 98.04%).
2. `nq8_v5_add_b0_b500_s0/s1/s2` — **미측정**. 정점(raw) + 최종 두 지점 권장.
   EMA 정점이 raw 정점과 다른 step 이므로 EMA 쪽은 최종 step 이 대표값에 가깝다.
3. `nq8_v5_add_bm4_b500_s1` — 미측정. 위와 동일.

## 정점 한 점으로 대표하면 안 되는 이유

raw 곡선의 후반 진동이 크다.  `nq8_v5_add_b0_b500_s0` 은 54672 에서 100.00% 인데
58089 에서 59.20%, 60367 에서 30.57% 로 떨어진다.  같은 설정 seed 3 개 중 s0 만
이렇게 흔들리고 s1/s2 는 98% 이상을 유지한다.

반면 EMA 는 4 개 v5 run 전부 60k 이후 96~100% 로 안정적이다.  구간 평균:

| run | 40–50k raw / EMA | 50–60k raw / EMA | 60k+ raw / EMA |
|---|---|---|---|
| v5_add_b0_s0 | 87.79% / 28.87% | 88.50% / **98.70%** | 84.88% / **96.15%** |
| v5_add_b0_s1 | 96.71% / 97.41% | 98.20% / 98.49% | 99.06% / **99.76%** |
| v5_add_b0_s2 | 98.38% / 94.14% | 99.41% / 98.09% | 99.27% / **99.56%** |
| v5_add_bm4_s1 | 97.52% / 16.10% | 98.96% / **96.24%** | 93.42% / **99.48%** |

## EMA 붕괴는 붕괴가 아니라 늦은 수렴이다

v2 이전 라운드가 34k~40k 에서 끝나 EMA 가 0% 로 보였다.  v5(68,375 step)에서
`bias 0` 은 27k~49k, `bias −4` 는 51k 에서 EMA 가 90% 를 넘긴다.

| run | EMA≥90% 최초 step | EMA>raw 최초 step |
|---|---|---|
| v5_add_b0_s0 | 48,977 | 48,977 |
| v5_add_b0_s1 | 27,336 | 34,170 |
| v5_add_b0_s2 | 34,170 | 14,807 |
| v5_add_bm4_s1 | 51,255 | 56,950 |

`v5_add_b0_s0` 은 22,780 에서 47.3% 까지 올랐다가 33,031~39,865 구간에서 0.0~0.5% 로
무너진 뒤 48,977 이후 98~99.6% 로 회복한다.  v2/v3 에서 관측한 "EMA 15~94% 요동" 이
이 중간 붕괴 구간이었다.
