# N-Queens 진단 2: SEQ_LEN 수정 + prior 실패 분해 (REPORT_diag2.md)

작업: 2026-07-24. **학습 launch 없음**(진단만). GPU 0~3. 판정·해석은 사용자. 수치+원본 경로만.

**한 줄 요약**: SEQ_LEN 버그는 실재하나 `eval_harness.py`에는 영향이 없었고(seq_len을 데이터에서 받음)
`gram_harness.rollout`(e1~e4)에만 영향. depth×sigma 4셀 어디서도 prior valid이 회복되지 않는다(전부 0%).
실패 사유는 전 셀에서 **queen 개수**(배치 충돌이 아님, `right_count_but_invalid=0`). L-sum run은 q_halt
발화율 0.0(depth 16), L-mean run은 1.0(depth 1) — 실패 모드 2종이 분리된다.

---

## Task 1 — SEQ_LEN 수정 + 회귀 앵커

### 수정 diff (`experiments/gram_harness.py`, 백업 `*.bak_0724_*`)
3개 사용처를 모듈 상수 → `inputs.shape[-1]`로 교체:
```python
# _batch():
-    inp = np.zeros((batch_size, SEQ_LEN), dtype=np.int32)
+    seq_len = inputs.shape[-1]          # was hardcoded SEQ_LEN=81 (sudoku); breaks N-Queens (64)
+    inp = np.zeros((batch_size, seq_len), dtype=np.int32)
# rollout():
-    preds = np.zeros((K, n, SEQ_LEN), dtype=np.int64)   ->  seq_len = inputs.shape[-1]; np.zeros((K, n, seq_len)...)
-    p_final = torch.zeros(batch_size, SEQ_LEN, ...)      ->  torch.zeros(batch_size, seq_len, ...)
```
모듈 상수 `SEQ_LEN=81`은 주석만 달아 남겨둠(다른 참조 없음, 삭제 시 import 깨질 여지 회피). `ast.parse` 통과.

### ⚠️ 지시서 전제 정정 (중요)
지시서: "eval_harness.py도 같은 rollout을 쓴다". **사실이 아니다.** `experiments/eval_harness.py`는
`gram_harness.rollout`/`_batch`를 쓰지 않고 자체 `sample_dataset`을 가지며, **seq_len을 데이터에서 받는다**
(`eval_harness.py:227` `n, seq_len = inputs.shape`). 즉 **이전 N-Queens eval_harness 수치는 SEQ_LEN 버그에
오염되지 않았다.** 버그의 실제 영향 범위는 `gram_harness.rollout`을 직접 쓰는 스크립트(e1~e4)뿐이다.
(내 이전 세션 `diag_posterior_prior.py`도 자체 seq_len을 써서 무관했다.)

### 회귀 앵커 — **통과** (`experiments/results/eval_harness_regression_postfix.json`)
`gram_sudoku_halt_only/step_65100`, n=1000, N=20, seed=0:
| 지표 | 기대 | 실측(수정 후) |
|---|---|---|
| single | ≈70.5% | **70.40%** |
| majority@20 | ≈88.9% | **89.20%** |
| pass@20 | ≈92.5% | **92.60%** |
| majority@5 / pass@5 | — | 79.50% / 85.50% |
수정 전(`eval_harness_regression.json`)과 **완전히 동일** → eval_harness는 이 버그에 무영향임이 앵커로도 확인.

---

## Task 2 — 재측정 (valid vs strict, prior, N=20, test 772, seed 0)

`experiments/diag_task2_remeasure.py` (한 번 샘플링해 valid·strict 동시 채점).

| checkpoint | 헤드라인 | single | pass@20 | maj@20 | distinct@20 |
|---|---|---|---|---|---|
| mean1/step_17085 | **valid** | 0.00% | 0.00% | 0.00% | 18.51 |
| | strict | 0.00% | 0.00% | 0.00% | 18.51 |
| nh512/step_17085 | **valid** | 0.00% | 0.00% | 0.00% | 20.00 |
| | strict | 0.00% | 0.00% | 0.00% | 20.00 |

→ `experiments/results/task2_{mean1,nh512}_s17085.json`.
prior 보드는 valid도 아니고 clue 보존(consistent)도 안 하므로 **valid=strict=0**(둘이 같은 이유: valid이 0이면 strict도 0).

### 버그 전/후 비교 (효과 크기 = 0)
| checkpoint | 이전 distinct@20 | 이전 측정 step | 재측정 distinct@20 |
|---|---|---|---|
| mean1 | 18.48 (`nq8_mean1_final.json`) | step_17085 | **18.51** (Δ0.03, RNG/bf16) |
| nh512 | 1.00 (`nh512_peak.json`) | **step_3417** | 20.00 (step_17085) |

- mean1은 같은 step에서 18.48→18.51로 **사실상 불변** → eval_harness가 오염되지 않았다는 직접 증거.
- nh512의 이전 "distinct=1.00"은 **step_3417**(EMA-지연 all-empty 지점) 수치였다. step_17085(최종)에선 20.00(전부 다른 범람 보드).
  즉 그 차이는 버그가 아니라 **측정 step 차이**다.

---

## Task 3 — prior 실패 4셀 분해 (깊이 × 노이즈)

`experiments/diag_depth_sigma.py`. prior, K=1, test 772. `observed_steps`로 실제 깊이 확인.
→ `experiments/results/nq8_depth_sigma_grid_{mean1,nh512}.json`.

### mean1/step_17085 (노이즈 head 16.26M) — queen 부족형
| 셀 | steps | valid | queen_count mean[min,max] | 실패사유(invalid=772) |
|---|---|---|---|---|
| depth1 · σ=0.0 | 1 | 0.00% | 0.00 [0,0] | count=772 (전부 빈 보드) |
| depth1 · σ=1.0 | 1 | 0.00% | 0.69 [0,4] | count=772 (row12 col11 d1·7 d2·8) |
| depth16 · σ=0.0 | 16 | 0.00% | 0.00 [0,0] | count=772 (전부 빈 보드) |
| depth16 · σ=1.0 | 16 | 0.00% | 1.56 [0,5] | count=772 (row65 col68 d1·32 d2·48) |

### nh512/step_17085 (노이즈 head 10M) — queen 범람형
| 셀 | steps | valid | queen_count mean[min,max] | 실패사유(invalid=772) |
|---|---|---|---|---|
| depth1 · σ=0.0 | 1 | 0.00% | 21.37 [15,28] | count·row·col·d1·d2 = 772 전부 |
| depth1 · σ=1.0 | 1 | 0.00% | 29.01 [19,38] | count·row·col·d1·d2 = 772 전부 |
| depth16 · σ=0.0 | 16 | 0.00% | 30.16 [24,36] | count·row·col·d1·d2 = 772 전부 |
| depth16 · σ=1.0 | 16 | 0.00% | 32.12 [25,39] | count·row·col·d1·d2 = 772 전부 |

**수치 사실 (판정 아님):**
- **어느 셀에서도 valid > 0%가 되지 않는다.** depth를 16→1로 줄여도, σ를 1→0으로 꺼도 회복 없음.
- σ=0(노이즈 완전 제거)에서 mean1은 **정확히 빈 보드**(qc=0), nh512는 여전히 21~30 queen. → 노이즈가 아니라
  μ_p(deterministic mean) 자체가 이미 빈 보드/범람을 가리킨다.
- depth 증가는 실패를 **악화**시키는 방향(qc가 mean1 0→1.56, nh512 21→32로 증가)이지 개선이 아니다.
- 두 모델이 **정반대**로 실패: mean1=queen 부족(count 미달), nh512=queen 과다(count 초과 + 전 제약 충돌).

참고용 판정 기준(사용자용): "깊이1 회복→깊이 불일치, σ0 회복→노이즈 누적, 둘 다 무효→μ_p 자체가 틀렸다".
실측은 **둘 다 무효**(어느 축도 valid 회복 없음).

---

## Task 4 — q_halt 발화율 (posterior, 512샘플, step 1~3)

`experiments/diag_qhalt.py`. 각 step에서 `(q_halt_logits > 0)` 비율(gram.py:458의 halt 조건).
→ `experiments/results/qhalt_*.json`.

| run | reduction | data | fire rate [s1,s2,s3] | q_halt logit mean [s1,s2,s3] |
|---|---|---|---|---|
| gram_nq8_mean1 | L-mean | nqueens | **1.000, 1.000, 1.000** | 5.52, 5.83, 5.84 |
| gram_nq8_mean_nh512 | L-mean | nqueens | **1.000, 1.000, 1.000** | 5.52, 7.14, 7.21 |
| **gram_nq8_b007_v2** | **L-sum** | nqueens | **0.000, 0.000, 0.000** | **−17.20, −17.18, −17.18** |
| gram_sudoku_halt_only | L-sum | sudoku | 0.225, 0.537, 0.617 | −3.24, −0.30, 0.29 |

**수치 사실 (판정 아님):**
- L-mean 두 run은 step 1에서 발화율 **1.0** → 학습 시 supervision step 1에서 즉시 halt(깊이≈1).
- **L-sum run(b007_v2)은 발화율 0.0**, logit ≈ −17 → q_halt가 **전혀** 발화 안 함 → 항상 max_steps(16)까지.
  즉 L-sum은 깊이 16을 겪고도 all-empty 실패 → **L-sum 실패는 깊이 문제가 아님**(Task 4의 분리 목적 달성).
- 정상 sudoku halt_only는 0.225→0.617로 점진 발화(정상 ACT 곡선). nqueens는 극단(0.0 또는 1.0).
→ **실패 모드 2종 분리**: L-mean=조기 halt(깊이 1), L-sum=무한 halt 억제(깊이 16). 병리가 서로 다르다.

---

## Task 5 — 실패 사유 분해

`evaluators/nqueens.py`의 `verify_boards` 반환 dict에 `count_ok/row_ok/col_ok/d1_ok/d2_ok` 추가(백업 `*.bak_0724_*`).
Task 3 각 셀에서 집계(위 표에 병기). 핵심 수치:

- **모든 셀·두 모델에서 `right_count_but_invalid = 0`**. 즉 "queen 8개를 맞췄는데 충돌해서 invalid"인 보드가 **하나도 없다**.
  실패는 전적으로 **개수(count_ok=772/772)**에서 발생한다.
- mean1: 개수 **미달**(0~1.5개). nh512: 개수 **초과**(21~32개, 그 결과 row/col/대각 전부 충돌).
→ 병리는 "배치는 하는데 규칙 위반"이 아니라 "**애초에 올바른 개수의 queen을 못 놓음**".

---

## 발견한 이상

1. **`experiments/gram_harness.py:30 SEQ_LEN=81`** — 수정 완료. 단 영향 범위는 `rollout` 사용 스크립트(e1~e4)뿐이고
   eval_harness/diag_posterior_prior는 애초에 무관했다(지시서 전제와 다름, 위 정정).
2. **b007_v2(L-sum) q_halt logit ≈ −17로 포화** → 이 run의 이전 wandb `train/steps=0`은 count=0 아티팩트가 아니라
   "q_halt 무발화로 항상 16 step, 그런데 그 logged step에 halted&valid 샘플이 0"이었을 가능성. 발화율 측정(0.0)이 더 신뢰가능.
3. **eval halting이 논문 A.1과 이탈**(gram.py:452 `if self.training` 게이트)은 REPORT_diag에서 이미 보고. 이번 Task 3에서
   depth를 강제로 1로 낮춰도 valid 회복이 없으므로, 이 이탈이 실패의 **충분조건은 아님**이 수치로 확인됨(사실만).

---

## 임의로 정한 것 (전부 명시)

1. **Task 2에 valid·strict 둘 다 계산하는 별도 스크립트(`diag_task2_remeasure.py`)를 새로 만들었다.**
   eval_harness의 nqueens 경로는 `strict`만 헤드라인으로 내므로(`NQueensTask.verify`=strict), 논문 헤드라인 `valid`를
   얻으려면 자체 채점이 필요했다. 샘플링은 eval_harness.sample_dataset(동일 prior 경로, seed 0)을 재사용.
2. **깊이 override를 config 필드 직접 수정으로 구현**(`inner.config.halt_max_steps = depth`). inner.config와 model.config는
   동일 객체라 wrapper의 halt 조건에 반영됨. `observed_steps`로 실제 1/16 step을 확인(로그에 출력).
3. **Task 5의 실패사유를 "invalid 보드 중 각 제약 위반 수"로 집계**(중복 카운트 허용, 한 보드가 여러 제약 위반 가능).
   추가로 `right_count_but_invalid`(개수는 맞고 충돌)를 별도 집계 — 이게 병리 구분의 핵심이라 판단.
4. **q_halt 발화율은 posterior 경로·EMA checkpoint·512샘플**로 측정. 학습은 live 가중치였으나(디스크에 없음),
   logit이 ±17 수준으로 포화라 EMA/live 차이에 강건하다고 판단.
5. **b007_v2를 L-sum 대표로 선택**(지시서 명시). 다른 sum run(b0045_v2 등)도 동일 경향으로 추정하나 측정은 b007_v2만.
6. 신규 스크립트: `diag_depth_sigma.py`, `diag_task2_remeasure.py`, `diag_qhalt.py`. 수정: `gram_harness.py`(SEQ_LEN),
   `evaluators/nqueens.py`(실패사유 필드 추가). **`pretrain.py`·`gram.py`·`losses.py` 등 학습/모델 코드는 일절 미변경.**

## 수치 → 원본 경로 색인
- Task1: `experiments/results/eval_harness_regression_postfix.json`, diff는 `gram_harness.py.bak_0724_*` 대비
- Task2: `experiments/results/task2_{mean1,nh512}_s17085.json`; 이전값 `nq8_mean1_final.json`(step_17085), `nh512_peak.json`(step_3417)
- Task3: `experiments/results/nq8_depth_sigma_grid_{mean1,nh512}.json`
- Task4: `experiments/results/qhalt_{gram_nq8_mean1,gram_nq8_mean_nh512,gram_nq8_b007_v2,halt_only}.json`
- Task5: Task3 json의 `fail_*`/`right_count_but_invalid` 필드
- 코드 근거: `experiments/gram_harness.py`, `experiments/eval_harness.py:227`, `models/recursive_reasoning/gram.py:452,458`, `evaluators/nqueens.py`

### 재현 커맨드
```bash
cd /workspace/TinyRecursiveModels
CUDA_VISIBLE_DEVICES=0 python experiments/diag_depth_sigma.py \
  --ckpt-dir checkpoints/Nqueens-8-ACT-torch/gram_nq8_mean_nh512 --step step_17085 \
  --out experiments/results/nq8_depth_sigma_grid_nh512.json
CUDA_VISIBLE_DEVICES=0 python experiments/diag_qhalt.py \
  --ckpt-dir checkpoints/Nqueens-8-ACT-torch/gram_nq8_b007_v2 --step step_17085
```
