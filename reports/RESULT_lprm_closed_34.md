# LPRM(verifier) head 점수 분포 probe — 중단 (34번)

**상태**: 중단. 저자 확인 결과 LPRM(verifier)은 중요도가 낮고 Sudoku/GC 재현이 우선이다.

## 중단 시점 산출물

- 계획 12건 (`nq10_A3p_fx_a7525_lr3e4` step117544/138534 × {ema,live} × seed{0,1},
  `nq10_A3p_ii_a7525` step79762 × {ema,live} × seed{0,1})
- 완료 JSON **0건**. 8건 실행 중(GPU 0–7, 경과 ~1h02m), 4건 대기 상태에서 종료.
- `reports/lprm/*.log` 8건만 잔존 (내용: `torch.load` FutureWarning 1줄).
- probe 스크립트는 세션 scratchpad에만 존재했고 repo에 커밋되지 않았다.
  `experiments/sweep_harness.py` 는 수정하지 않았다.

## 유일한 실측값 (64퍼즐 스모크)

fx_a7525_lr3e4 step138534 EMA, 64퍼즐 스모크: top1 0.297 < random 0.418,
LPRM valid 0.8481±0.051 / invalid 0.8846±0.052, gap −0.0365. 전수 미측정.

## 참고: 전수 stageA harness 의 verifier@20 (별도 산출, 보존됨)

| checkpoint | prior(전 샘플) | majority@20 | verifier@20 |
|---|---|---|---|
| `nq10_A3p_fx_a7525_lr3e4` step138534 ema | 0.4025 | 0.8720 | 0.1713 |
| `nq10_A3p_fx_a7525_lr3e4` step117544 live | 0.8539 | 0.9948 | 0.9697 |
| `nq10_A3p_ii_a7525` step79762 ema | 0.5104 | 0.8941 | 0.5977 |

출처: `reports/stageA/nq10_A3p_*_h0.json` (`experiments/sweep_harness.py`).
