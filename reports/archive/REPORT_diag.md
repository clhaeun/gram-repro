# N-Queens 실패 원인 진단 보고 (REPORT_diag.md)

작업: 2026-07-23 ~ 24. **학습 launch 없음** (지시대로 검사·진단만). GPU 0~3 사용.
지시대로 판정하지 않고 수치와 근거만 제시한다. 해석은 사용자가.

**핵심 산출물(B1)을 먼저**: **백본은 유효 보드를 만들 줄 안다.** 최종 checkpoint에서
posterior 경로 valid rate가 mean1(16.26M) **100.00%**, nh512(10M) **99.87%**다.
실패하는 것은 **prior 경로**뿐이다(양쪽 다 valid 0.00%). 즉 문제는 KL/σ(prior) 층이지
백본이나 순수 암기가 아니다. 단, 이 판정은 **EMA 사본 기준**이라는 중요한 단서가 붙는다(§A7, §B1-주의).

---

## Task A — 구성 검사 (코드 근거)

모든 인용은 현재 작업트리 `models/recursive_reasoning/gram.py` / `models/losses.py`.

### A1. q_head gradient 격리 — **논문과 일치**
`gram.py:386`
```python
q_logits = self.q_head(z_H[:, 0].detach()).to(torch.float32)
```
입력 `z_H[:, 0]`가 `.detach()`됨 → q_head gradient가 recursive core로 흐르지 않는다.
논문 A.1 "does not propagate gradients into the recursive core"와 일치.

### A2. LPRM gradient — **논문과 일치(stop-grad 없음)**
- head: `gram.py:192` `self.lrpm_head = CastedLinear(D, 1, bias=True)`
- 입력 수집: `gram.py:350` (H_cycles−1 루프, **`torch.no_grad()` 안** → grad 없음),
  `gram.py:379` (마지막 grad-carrying transition, `z_H = z_H + eps` 직후 → **grad 있음**).
- `gram.py:388` `lrpm_logits = self.lrpm_head(torch.stack(lrpm_z, dim=1))...` — `lrpm_z`에 detach 없음.
  → **마지막 원소만 recursive core로 grad를 전달**한다(앞 원소들은 no_grad 블록 소산).
- target은 detach: `losses.py` `r = (...).detach()`, 손실 `losses.py:113`
  `((outputs["lrpm_logits"] - r.unsqueeze(-1))**2).sum(-1).sum()`.
- 계수: `losses.py:143` `lprm_coef = getattr(self.model.config, "lprm_coef", 1.0)`, 모든 nq8 config에서 **1.0**.
논문 A.2(Eq 16)는 stop-gradient를 언급하지 않는다 → 현재 코드(입력 stop-grad 없음, target만 stop-grad)와 일치.

### A3. KL 위치 — **논문과 일치(Eq 14)**
`kl`이 loss로 반환되는 곳은 **마지막 transition 한 곳뿐**: `gram.py:366` `kl = self.kl_loss(...)`.
H_cycles−1 루프 안의 `gram.py:344` `kl_t = self.kl_loss(...)`는 **`trace_kl` 계측 전용**(loss 미투입,
`self._kl_trace`에만 append). prior 경로는 `gram.py:373` `kl = None`. → supervision step당 마지막 transition에만 KL.

### A4. eval halting — **논문과 불일치(사실 확인)**
`gram.py:449` `halted = is_last_step` → 기본은 최종 step에서만 정지.
`gram.py:452` `if self.training and (self.config.halt_max_steps > 1):` 안에서만 `q_halt_logits > 0`(`gram.py:458`) 조기 정지가 적용됨.
→ **eval에서는 항상 `halt_max_steps`(=16)를 돈다.** σ(q_halt)>0.5 조기 정지는 eval에서 쓰이지 않는다(TRM 상속).
논문 A.1은 추론 시 σ(q_halt)>0.5 정지를 명시 → **불일치**. (다만 max_steps를 다 도는 것이 조기 정지보다
불리하지 않으므로 이번 실패의 원인일 가능성은 낮다 — 사실만 보고.)

### A5. train 실제 도달 깊이 (`train/steps`)
| run | 최종 `train/steps` | 출처 |
|---|---|---|
| gram_nq8_b0045_v2 (sum) | **1.144** | wandb offline `ddrungec`* |
| gram_nq8_b0045 (sum, v1) | 1.126 | wandb offline |
| gram_nq8_b007 / b010 (sum, v1) | 1.103 / 1.111 | wandb offline |
| gram_nq8_b007_v2 / b010_v2 | 0.0 (해당 로그 step에서 count=0) | wandb offline |
| **sudoku halt_only (정상)** | **16.0** | `logs/gram_sudoku_halt_only.log` (`steps=16.0000`) |

*mean_nh512/mean1은 wandb 기록이 없어 `train/steps` 미확보(로그에도 미출력). 
**정상 Sudoku는 16 step 내내 도는데, nqueens는 ~1.1 step에서 정지한다** — q_halt가 거의 즉시 halt를 켠다.
`train/steps`는 posterior+train 모드 기준 halted 샘플 평균 정지 step이다.

### A6. σ 실측 (최종 EMA checkpoint, nqueens test 256샘플; `experiments/sigma_probe.py`)
| checkpoint | prior σ std_mean | post σ std_mean | post KL/sample |
|---|---|---|---|
| **sudoku halt_only (정상, 참조)** | **0.955** | — | — (e4_sigma json) |
| gram_nq8_mean_nh512 / step_17085 | **0.4285** | 0.3810 | 256.2 |
| gram_nq8_mean_nh512 / step_3417 | 0.1537 | 0.1633 | **192,380** |
| gram_nq8_b0045_v2 (sum) / 최종 | 0.668 | 0.668 | (wandb kl_raw≈0.0007) |
| gram_nq8_b007_v2 / b010_v2 (sum) | 0.661 / 0.661 | 0.661 / 0.661 | kl_raw≈0.002 / 0.0003 |

→ 파일: `experiments/results/sigma_nh512_s17085.json`, `sigma_nh512_s3417.json`.
정상 Sudoku(0.955) 대비 nqueens는 σ가 작다(0.43~0.67). step_3417 EMA 사본은 KL/sample이
192,380으로 폭발 상태(= 아래 §B1 step_3417 all-empty와 정합).

### A7. EMA — **저장 checkpoint = EMA 사본. live 가중치는 디스크에 없음.**
- `pretrain.py:686` `save_train_state(config, train_state_eval)` 저장. 
- `pretrain.py:656-657` ema=True일 때 `train_state_eval.model = ema_helper.ema_copy(...)`.
- `models/ema.py` `ema_copy` = deepcopy 후 shadow(EMA 평균) 주입.
→ **저장된 모든 nq8 checkpoint는 EMA 사본**이고, SGD live 가중치는 메모리에만 있다가 run 종료 시 소멸.
`train exact_accuracy=1.0`은 **live 가중치**에서 측정된 값이고, in-loop `[EVAL]`과 이번 진단은 전부 **EMA 사본**이다.
모든 nq8 config가 `ema_rate: 0.9999`.

---

## Task B — 결정적 진단

방법: `experiments/diag_posterior_prior.py`(신규). gram_harness의 SEQ_LEN=81 하드코딩을 우회하려
seq_len을 데이터(64)에서 받는다. test 772문항 전수, K=1, halting까지, `evaluators.nqueens.verify_boards`로 채점.

### B1. posterior vs prior 수치표

| checkpoint | 경로 | valid | consistent | strict | queen_count mean [min,max] |
|---|---|---|---|---|---|
| **mean1 (16.26M) / step_17085 (최종)** | **posterior** | **100.00%** | 100.00% | 100.00% | **8.00 [8,8]** |
| | prior | 0.00% | 0.00% | 0.00% | 1.56 [0,5] |
| **nh512 (10M) / step_17085 (최종)** | **posterior** | **99.87%** | 100.00% | 99.87% | 8.00 [8,9] |
| | prior | 0.00% | 0.00% | 0.00% | **32.12 [25,39]** |
| nh512 / step_3417 ("peak") | posterior | 0.00% | 0.00% | 0.00% | 0.00 [0,0] |
| | prior | 0.00% | 0.00% | 0.00% | 0.00 [0,0] |
| mean1 / step_3417 | posterior | 0.00% | 0.00% | 0.00% | 0.00 [0,0] |
| | prior | 0.00% | 0.00% | 0.00% | 0.00 [0,0] |

파일: `experiments/results/diag_pp_{mean1_s17085,nh512_s17085,nh512_s3417,mean1_s3417}.json`.

**두 판정 기준 중 어느 쪽인가 (최종 checkpoint 기준):**
> **"posterior valid 높고 prior 0 → 백본은 유효 보드를 만들 줄 안다. prior가 못 따라오는 것이 문제(KL/σ 층)."**

최종 checkpoint 2개 모두 이 경우다. posterior는 label을 보고 valid 보드를 **99.87~100%** 만들고,
입력 clue까지 100% 보존한다(consistent=100%). prior는 0%.
흥미롭게 **두 모델의 prior 실패 방향이 반대**다:
- mean1(노이즈 head 큼): prior → 평균 1.56 queen (거의 all-empty)
- nh512(노이즈 head 작음): prior → 평균 32 queen (보드 범람)
둘 다 "prior가 posterior를 근사 못 함"의 두 극단이다.

**⚠️ 주의 (EMA 단서, §A7):** 이 판정은 **EMA 사본** 기준이다. `train exact=1.0`은 live 가중치에서
측정됐고 디스크에 없어 직접 대조 불가하다. 따라서 "백본이 유효 보드를 만들 줄 안다"는 **EMA 사본에서 확증**됐고
(오히려 EMA에서도 posterior 100%가 나온다는 것은 백본 능력에 유리한 증거), "train exact 1.0 vs eval 0"의
간극은 **EMA lag와 prior/posterior 경로 차이가 섞여 있다**. step_3417(초기, EMA가 init 쪽으로 강하게 지연)에서
posterior조차 0%인 것, 그리고 그 지점 KL/sample=192,380인 것이 EMA-lag 가설과 정합한다.
"peak step_3417"은 prior-path eval loss 기준의 peak일 뿐, 실제로는 EMA-지연된 미성숙 상태다.

### B2. 출력 보드 (육안 — 이 프로젝트 최초)

**nh512 최종, posterior** (좌 input clue / 우 prediction) — 완벽 재구성 + clue 보존:
```
#0  queens=8  valid=True consistent=True
. . . . . Q . .     . . . . . Q . .
. . . . . . . .     . . Q . . . . .
. . . . . . . .     . . . . . . Q .
. . . . . . . .     . Q . . . . . .
. . . . . . . .     . . . . . . . Q
. . . . Q . . .     . . . . Q . . .
Q . . . . . . .     Q . . . . . . .
. . . . . . . .     . . . Q . . . .
```

**nh512 최종, prior** — 범람(32 queen), clue 무시:
```
#0  queens=32  valid=False consistent=False
. . . . . Q . .     . Q Q Q . . Q Q
. . . . . . . .     . Q Q . Q . . .
. . . . . . . .     Q . . Q . . Q .
. . . . . . . .     Q Q . Q . . . Q
. . . . . . . .     . . Q . Q . . Q
. . . . Q . . .     Q Q Q Q . . Q Q
Q . . . . . . .     . . Q Q Q Q Q Q
. . . . . . . .     . . . Q Q . . .
```

**mean1 최종, prior** — 반대 극단(2~3 queen, 거의 빈 보드), clue 무시:
```
#0  queens=2  valid=False consistent=False
. . . . . Q . .     . Q . . . . . .
. . . . . . . .     . . . . . . . .
. . . . . . . .     . . . . . . . .
. . . . . . . .     . . . . . . . Q
. . . . . . . .     . . . . . . . .
. . . . Q . . .     . . . . . . . .
Q . . . . . . .     . . . . . . . .
. . . . . . . .     . . . . . . . .
```
공통 패턴: **prior는 입력 clue를 전혀 보존하지 않고**(nh512=범람, mean1=거의 빈), posterior는 clue를 정확히 보존.

### B3. 해 개수별 버킷 (nh512 최종; `experiments/diag_bucket.py`)
| #solutions | n_inputs | posterior valid | prior valid |
|---|---|---|---|
| 1 (유일해) | 494 | **99.80%** | **0.00%** |
| 2 | 172 | 100.00% | 0.00% |
| 3–4 | 92 | 100.00% | 0.00% |
| 5–8 | 9 | 100.00% | 0.00% |
| 9–18 | 5 | 100.00% | 0.00% |
→ **prior는 유일해 input(494개)에서도 0%로 균일 실패.** 다중해 모호성 문제가 아니라 prior 경로의 계통적 실패.
posterior는 전 버킷 ~100%.

---

## Task C — evaluator 정합성 — **통과**

`evaluators/nqueens.py` 확인/드라이런:
- **헤드라인 = valid**: `result()`가 `nqueens/accuracy = acc[1]/total = valid_rate`(`nqueens.py:123`)를 논문 Table 1 "Accuracy"로,
  strict/consistent는 자체 진단 지표로 분리 출력. → 논문(consistency 미요구)과 정합.
- **인터페이스**: `required_outputs={'inputs','preds'}`, `begin_eval/update_batch/result` 존재.
  `create_evaluators`가 `evaluators="[nqueens@NQueens]"`로 **실제 로드됨**(드라이런, 학습 미실행):
  `loaded evaluators: ['NQueens'], n=8`.
- **라벨 sanity**: test 라벨 772개 → `valid=1.0, consistent=1.0, strict=1.0`,
  `nqueens/accuracy=1.0, queen_count_mean=8.0`. 재확인 완료.

---

## 발견한 이상 / 하드코딩 잔재

1. **`experiments/gram_harness.py:30` `SEQ_LEN = 81` (Sudoku 하드코딩).** `rollout()`의 패딩 배치·토큰 마스크에
   쓰여 N-Queens(64)에서 조용히 틀린 형상을 만든다. 지시대로 **고치지 않고 우회**했다(진단 스크립트가 seq_len을
   데이터에서 받음). gram_harness를 nqueens에 직접 쓰려면 `SEQ_LEN`을 config 유래로 바꿔야 한다.
   (`experiments/eval_harness.py`의 `SudokuTask.verify`에도 `assert L == 81`이 있으나 task별 분기라 무해.)
2. **`experiments/eval_harness.py`의 `_forward_once`가 `use_posterior=False` 하드코딩** — 그래서 posterior 진단에는
   못 쓰고 별도 스크립트를 만들었다(이상은 아님, 설계상 prior 전용).
3. **EMA만 저장, live 미저장**(§A7). `train exact=1.0`과 eval의 간극을 디스크에서 분리 불가능하게 만드는 구조적 제약.
   재현/디버깅 시 live 가중치도 저장하도록 `save_train_state`를 바꾸는 것을 고려할 만하다(이번엔 미변경).
4. **step_3417을 "peak"로 부른 근거(prior-path eval loss)가 EMA-지연 아티팩트일 수 있음.** 실제 posterior 능력은
   최종 checkpoint에서만 드러난다(step_3417 posterior=0% vs step_17085 posterior=100%).

---

## 임의로 정한 것 (전부 명시)

1. **B1 대상에 mean1/step_17085(최종)을 추가**했다. 지시는 nh512/{3417,17085}·mean1/3417 3개였지만,
   nh512 최종에서 posterior=99.87%가 나와 "mean1도 최종에서 그런가"를 확인해야 판정이 완결된다고 판단(→ 100%).
2. **posterior/prior 모두 K=1, seed=0** 단일 샘플. 판정(백본 능력 유무)에는 K=1로 충분하고 다샘플은 §B3/coverage 담당.
3. **B3 버킷 경계** {1,2,3-4,5-8,9-18}는 임의. sidecar 해 개수 분포(대부분 1~2)에 맞춘 구간.
4. **σ 측정을 로그가 아니라 checkpoint에서** 했다(mean 런들은 wandb/로그에 sigma 미기록). `sigma_probe.py`,
   nqueens test 256샘플. EMA 사본 기준이라 §B1과 동일 조건.
5. **gram_harness.load_model이 현재 gram.py로 인스턴스화**한다(as-run 스냅샷 아님). all_config.yaml의
   `noise_head_expansion`(nh512=1.5, mean1=키 없음→기본4.0)·`kl_reduction=mean` 키가 현재 코드와 맞아 로드 성공.
   state_dict 불일치 에러 없었음(로드 성공이 곧 정합 확인).
6. 진단 스크립트 3종 신규 생성: `diag_posterior_prior.py`, `diag_bucket.py` + 기존 `sigma_probe.py` 재사용.
   `pretrain.py` 등 학습 경로는 **일절 미변경**.

## 수치 → 원본 경로 색인
- B1: `experiments/results/diag_pp_*.json`
- B2 보드: `logs/diag_nh512_s17085.log`, `logs/diag_mean1_s17085.log`, 화면 출력
- B3: `experiments/diag_bucket.py` 실행 출력(재현 커맨드 아래)
- A5/A6 sum-run: `wandb/offline-run-*` (b* 런), A6 mean-run: `experiments/results/sigma_nh512_*.json`
- A: `models/recursive_reasoning/gram.py`, `models/losses.py`, `pretrain.py`, `models/ema.py` (인용 라인 명시)
- 참조: sudoku halt_only σ = `checkpoints/.../gram_sudoku_halt_only/experiments/e4_sigma_step_65100_k20_n1000.json`

### 재현 커맨드
```bash
cd /workspace/TinyRecursiveModels
CUDA_VISIBLE_DEVICES=3 python experiments/diag_posterior_prior.py \
  --ckpt-dir checkpoints/Nqueens-8-ACT-torch/gram_nq8_mean_nh512 --step step_17085 --show 3
CUDA_VISIBLE_DEVICES=3 python experiments/diag_bucket.py \
  --ckpt-dir checkpoints/Nqueens-8-ACT-torch/gram_nq8_mean_nh512 --step step_17085
```
