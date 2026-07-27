# REPORT_ema_test.md — EMA 원인 검증 + nq8_fix_b(EMA off) 재학습

- 일시: 2026-07-27
- 브랜치: `main`, HEAD `4c847f1`
- 인터프리터: `/home/chaeun/venv-gram/bin/python` (torch 2.5.1+cu124, Python 3.12.3)
- GPU 사용: 학습 3,4,5,6 / eval 0. launch 직전 `nvidia-smi` 전량 유휴(35–50MiB) 확인 후 점유.
  다른 실험이 쓰던 GPU는 건드리지 않았다.
- 신규 산출물
  - 스크립트: `train_nqueens8_fix.sh`, `experiments/make_raw_ema.py`
  - 결과 json: `experiments/results/ema_test_*.json`
  - 로그: `logs/gram_nq8_fix_b.log`, `logs/ema_*.log`, `logs/harness_fixb_*.log`
  - wandb: `wandb/run-20260727_083055-ckf9pcuz/`

**해석은 하지 않는다.** 아래는 측정치와, 측정을 위해 내가 정한 선택들이다.

---

## Task 1 — bias correction 검증

### 1a. 코드 확인

**raw 경로** — `models/ema.py:47-52`:
```python
47   def ema(self, module):
48       if isinstance(module, nn.DataParallel):
49           module = module.module
50       for name, param in module.named_parameters():
51           if param.requires_grad:
52               param.data.copy_(self.shadow[name].data)
```

**bias-corrected 경로** — `models/ema.py:54-66`:
```python
54   def _debiased(self, name, step):
55       mt = self.mu ** step
56       denom = 1.0 - mt
57       if denom < 1e-8:                       # step==0 or numerically ~0: no correction
58           return self.shadow[name].data
59       return (self.shadow[name].data - mt * self.w0[name].data) / denom
...
61   def ema_bias_corrected(self, module, step):
...
66               param.data.copy_(self._debiased(name, step).to(param.dtype))
```

**분기점** — `models/ema.py:68-75`:
```python
68   def ema_copy(self, module, step=None):
69       """step=None -> raw EMA (legacy). step=int -> bias-corrected EMA."""
...
71       if step is None:
72           self.ema(module_copy)
73       else:
74           self.ema_bias_corrected(module_copy, step)
```

#### denominator의 `t`가 실제 학습 step인가 — **그렇다** (단, 아래 단서 1건)

- 호출부는 `pretrain.py:659`, 인자로 `train_state.step`을 넘긴다:
```python
659              train_state_eval.model = ema_helper.ema_copy(train_state_eval.model, step=train_state.step)
```
- `train_state.step`은 `pretrain.py:294`에서 배치마다 1 증가:
```python
293  def train_batch(...):
294      train_state.step += 1
```
- `ema_helper.update()`도 배치마다 1회 호출 — `pretrain.py:648-649`:
```python
648              if config.ema:
649                  ema_helper.update(train_state.model)
```
→ 정상 구간에서 `num_updates == train_state.step`. 즉 `mu**t`의 `t`는 실제 EMA 업데이트 횟수와 일치.

- **단서(확인된 불일치 가능성, 크기는 무시 가능)**: `pretrain.py:295-296`에서
  `step > total_steps`이면 `train_batch`가 조기 return 하지만, `ema_helper.update()`는
  루프 바깥(648)이라 **그대로 호출된다**. 따라서 학습 꼬리에서 `num_updates`가 `step`을
  몇 개 앞설 수 있다. 실측 영향: `0.9999^17085 = 0.181122` vs `0.9999^17093 = 0.181077`
  → 4번째 자리 차이. 이번 판정에 영향 없음.

#### 0/1 경계 처리 — **있다. 그리고 수식상 정확하다**

- `ema.py:57-58`이 `denom < 1e-8`(즉 `step==0`)일 때 보정을 건너뛰고 raw shadow를 반환.
- `step=1` 경계를 손으로 검산하면: shadow는 `ema.py:35`에서 `w0`로 초기화되고
  (`self.shadow[name] = param.data.clone()`), `ema.py:44`의 갱신식에 따라
  `shadow_1 = (1-mu)*W_1 + mu*w0`. 이를 `ema.py:59`에 넣으면
  `(shadow_1 - mu*w0)/(1-mu) = W_1`. **정확히 live 가중치로 복원된다.**
- 즉 `(shadow - mu^t*w0)/(1-mu^t)` 는 `shadow_0 = w0` 라는 이 구현의 실제 초기화와
  정합한다. Adam식 `shadow/(1-mu^t)`(shadow_0=0 가정)와 다르며, `ema.py:12-18`의
  주석이 지적한 바와 일치한다. **코드 수준 버그는 발견하지 못했다.**

### 1b. Sudoku 앵커 sanity — 이상 없음

raw 가중치는 디스크에 없다(아래 "raw를 어떻게 얻었나" 참조). 재구성해서 측정했다.

- `mu**t = 0.9999^65100 = 0.001488` (`experiments/make_raw_ema.py` 실행 로그)
- 가중치 자체의 차이: `||raw − corrected|| / ||corrected|| = 0.005527`

| 지표 | corrected (`experiments/results/audit_regression.json`) | raw (`experiments/results/ema_test_sudoku_RAW.json`) | Δ |
|---|---|---|---|
| single | 70.60% | **69.00%** | −1.60%p |
| majority@20 | 88.40% | **89.30%** | +0.90%p |
| pass@20 | 93.10% | **93.30%** | +0.20%p |

n=1000, N=20, seed=0, sigma_scale=1.0 (양쪽 동일).

Δ의 부호가 지표마다 엇갈리고 크기가 1.6%p 이내다. 예고된 대로 `mu^t`가 0.15%라
raw/corrected가 거의 같다는 예측과 부합한다. **bias correction 버그를 시사하는
큰 괴리는 관측되지 않았다.**

주의: single의 −1.60%p는 감사 때 쓰던 ±1%p 게이트보다는 약간 크다. 이것이
가중치 차이(0.55%) 때문인지 bf16 비결정성 때문인지는 **분리하지 못했다**
(같은 seed지만 가중치가 다르므로 rollout 스트림이 갈린다).

### 1c. N-Queens raw vs corrected vs live

대상: `checkpoints/Nqueens-8-ACT-torch/gram_nq8_mean_nh512/step_17085`,
n=772, N=20, seed=0, sigma_scale=1.0, prior 경로.

> prior 경로인 근거: `experiments/eval_harness.py:196`의 `_forward_once`가
> `use_posterior=False`를 명시적으로 넘긴다.

| 가중치 | prior valid (single) | pass@20 | 출처 |
|---|---|---|---|
| corrected (`step_17085`) | **0.00%** | 0.00% | `experiments/results/ema_test_nq8_CORRECTED.json` |
| raw (`step_17085.raw`) | **0.00%** | 0.00% | `experiments/results/ema_test_nq8_RAW.json` |
| live (`step_17085.live`) | **없음 — 측정 불가** | — | 아래 참조 |

- `mu**t = 0.9999^17085 = 0.181122`, `||raw − corrected||/||corrected|| = 0.351496`.
  가중치는 실제로 크게 달랐는데도 두 결과 모두 0.00%다.
- **live 부재 확인**: `find checkpoints -name "*.live" | wc -l` → **0**.
  `.live` 저장은 `pretrain.py:689-692`에서 추가된 기능인데
  `gram_nq8_mean_nh512`는 그 이전(7월 23일) 학습이라 파일이 없다.
  지시대로 "없음" 보고하고 Task 2로 넘어갔다.

### raw 가중치를 어떻게 얻었나 (`experiments/make_raw_ema.py`)

디스크에서 읽을 수 없다. `pretrain.py:688`이 저장하는 `step_N`은 **bias-corrected**
사본이고, shadow 버퍼도 `ema.w0`(`ema.py:36`)도 저장되지 않는다.

보정이 가역이므로 역산했다:
```
corrected = (shadow − mt·w0)/(1 − mt)        # ema.py:59
⇒ shadow  = corrected·(1 − mt) + mt·w0
```
빠진 항은 `w0`(랜덤 init)뿐이다. `pretrain.py:594`가 데이터셋/모델 생성 **직전**에
`torch.random.manual_seed(config.seed + RANK)`를 부르므로 rank 0의 `w0`는
그 순서를 재생하면 재현된다. `make_raw_ema.py:build_w0()`가 그 재생을 한다.

**재현이 흔들리지 않는지 검사했다** (`--check-determinism`, 두 체크포인트 모두 동일 결과):
```
[check] rerun same settings      : max|dw0|=0.000e+00
[check] world_size 4 vs 1        : max|dw0|=0.000e+00
[check] with vs without loaders  : max|dw0|=0.000e+00
```
즉 init RNG 스트림은 world_size(=batch_size 192 vs 768)와 dataloader 생성 여부에
둔감하고, 재실행 시 bit-identical이다.

**확인 못 한 것**: 원본 `w0` 스냅샷이 저장돼 있지 않으므로, 재구성된 `w0`가
2026-07-23 학습 당시의 `w0`와 같다는 것을 **직접 대조하지는 못했다**. 위 검사는
"재구성이 내 선택들에 대해 안정적"임을 보일 뿐이고, 학습 박스의 torch/드라이버가
동일한 init 스트림을 냈다는 가정에 의존한다.

부수 확인: shadow에 안 들어가는 텐서 3개(`_orig_mod.model.inner.H_init`,
`L_init`, `puzzle_emb.weights`)는 그대로 통과시켰다. 셋 다 buffer라
`named_parameters()`에 안 잡히고(`ema.py:34,42`), EMA가 애초에 건드린 적 없다.

---

## Task 2 — nq8_fix_b (EMA off) 재학습

### 실행

새 스크립트 `train_nqueens8_fix.sh` (기존 sh 6개 미변경). 모든 손잡이가 환경변수.
`DRYRUN=1`이면 `torchrun`을 `echo`로 바꿔 조립만 확인한다.

dry-run 결과(실행 전, 지시대로 먼저 확인):
```
[preflight] GPUS=3,4,5,6 NPROC=4 PORT=29621 전부 유휴 -- launch
--nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:29621 --nnodes=1 pretrain.py
  arch=gram_nqueens8 data_paths=[data/nqueens-8] 'evaluators=[{name: nqueens@NQueens}]'
  epochs=3000 eval_interval=200 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0
  puzzle_emb_weight_decay=1.0 arch.mlp_t=False arch.pos_encodings=rope arch.L_layers=2
  arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=16 global_batch_size=768
  arch.kl_coef=0.07 arch.kl_reduction=mean arch.noise_head_expansion=1.5
  ema=False ema_rate=0.9999 lr_warmup_steps=500 +run_name=gram_nq8_fix_b
```

### config dump 발췌 (`checkpoints/Nqueens-8-ACT-torch/gram_nq8_fix_b/all_config.yaml`)

```yaml
arch:
  H_cycles: 3          # T=3
  L_cycles: 4          # K=4
  halt_max_steps: 16   # Nsup=16
  kl_balance: 0.8
  kl_coef: 0.07        # β
  kl_reduction: mean
  mlp_t: false
  noise_head_expansion: 1.5
  pos_encodings: rope
ema: false             # <-- 목표
epochs: 3000
evaluators:
- name: nqueens@NQueens
global_batch_size: 768
lr: 0.0001
lr_warmup_steps: 500   # <-- 목표
weight_decay: 1.0
run_name: gram_nq8_fix_b
seed: 0
```

`num_params = 9966085` — 논문 대응 크기와 정확히 일치.
(`pretrain.py:624`가 wandb에만 기록하므로 stdout에는 없다. wandb datastore
`wandb/run-20260727_083055-ckf9pcuz/run-ckf9pcuz.wandb`의 history에서 읽었다.)

`SWITCH TO EMA` 문자열(`pretrain.py:656`)이 로그에 **0회** 등장 → EMA 경로 미진입 확인.

### ~200 step 검증 (전부 통과)

| 항목 | 결과 | 근거 |
|---|---|---|
| NaN 없음 | ✅ | `logs/gram_nq8_fix_b.log`에 nan/inf 매치 0건 (2건은 "inference" 오탐) |
| kl_loss == kl_raw | ✅ | 로그된 1990 step 전체에서 **최대 상대오차 1.301e-07** |
| nqueens/accuracy 로그됨 | ✅ | `[EVAL @ step 1139] ... nqueens/accuracy=0.0000 nqueens/n_eval=772.0000` |
| eval kl_loss = 0.0 | ✅ | `[EVAL @ step 1139] ... all/kl_loss=0.0000` (전 15회 eval 동일) |

`kl_loss == kl_raw`는 stdout에 안 찍힌다 — `pretrain.py:646`이 키에 `loss`/`accuracy`가
든 것만 출력하는데 `diag/kl_raw`는 둘 다 없다. wandb history에서
`train/kl_loss` vs `train/diag/kl_raw`를 뽑아 비교했다. 예:
```
step=     1 kl_loss=32797.357422 kl_raw=32797.358398 rel_diff=2.978e-08
step=  1990 kl_loss=    0.170997 kl_raw=    0.170997 rel_diff=2.179e-08
```

이상 없어 중단하지 않고 완주시켰다. 소요 **36분 43초** (17085/17093 step, 7.75 it/s).

### in-loop nqueens/accuracy 곡선 (`logs/gram_nq8_fix_b.log`의 `[EVAL @ step N]` 15줄)

| step | all/accuracy(token) | all/lm_loss | **nqueens/accuracy** | consistent_rate | queen_count_mean |
|---|---|---|---|---|---|
| 1139 | 0.9184 | 0.5795 | **0.0000** | 1.0000 | 2.7772 |
| 2278 | 0.9149 | 0.6995 | **0.0000** | 1.0000 | 3.0738 |
| 3417 | 0.8352 | 1.2398 | **0.0000** | 0.9935 | 9.3808 |
| 4556 | 0.7082 | 2.4676 | **0.0000** | 0.0415 | 14.6801 |
| 5695 | 0.8642 | 1.2137 | **0.0000** | 0.8821 | 6.7539 |
| 6834 | 0.8993 | 0.9554 | **0.0000** | 0.9935 | 4.4301 |
| 7973 | 0.8661 | 1.1912 | **0.0000** | 0.7824 | 6.8238 |
| 9112 | 0.7193 | 2.6164 | **0.0000** | 0.0622 | 14.1541 |
| 10251 | 0.5538 | 4.1982 | **0.0000** | 0.0803 | 27.3355 |
| 11390 | 0.8966 | 0.9929 | **0.0000** | 1.0000 | 4.5013 |
| 12529 | 0.8873 | 1.0687 | **0.0000** | 0.9974 | 5.1671 |
| 13668 | 0.8099 | 1.8045 | **0.0000** | 0.2215 | 8.8122 |
| 14807 | 0.8914 | 1.0721 | **0.0000** | 0.9845 | 4.9806 |
| 15946 | 0.9101 | 0.9165 | **0.0000** | 1.0000 | 3.4560 |
| 17085 | 0.7377 | 2.5844 | **0.0000** | 0.0000 | 10.5803 |

곡선 형태: **처음부터 끝까지 0. 오른 적이 없다.** `all/exact_accuracy`도 전 구간 0.0000.
token accuracy와 queen_count_mean은 진동한다(정답 8개 대비 2.78~27.34 사이를 오감).

in-loop eval도 prior 경로다 — `evaluate()`가 `model.eval()` 상태로 부르고,
`gram.py:303-304`가 `use_posterior=None`을 `self.training`(=False)으로 해석한다.

### 참고: 학습 중 잠재 진단 (wandb history)

| step | σ_prior | σ_post | ‖μ_p‖ | ‖μ_q‖ | kl_raw | train/exact_acc |
|---|---|---|---|---|---|---|
| 1 | 0.1394 | 0.1702 | 13.178 | 28.397 | 32797.36 | 0.0000 |
| 3661 | 0.3120 | 0.3119 | 0.765 | 0.775 | 0.0573 | 0.9955 |
| 8541 | 0.4412 | 0.4411 | 0.693 | 0.697 | 0.0026 | 1.0000 |
| 12201 | 0.5405 | 0.5407 | 0.925 | 0.928 | 0.0189 | 1.0000 |
| 17085 | 0.6983 | 0.6983 | 2.391 | 2.396 | 0.0014 | 1.0000 |

학습 초반 이후 prior와 posterior의 σ·‖μ‖가 소수점 3자리까지 같아지고
`kl_raw`가 0.001~0.02 대로 내려간다. `train/exact_accuracy`는 1.0000으로 포화한다.
(train은 posterior 경로 + 학습 puzzle embedding, eval은 prior 경로 + 미학습 퍼즐이므로
두 수치는 같은 조건이 아니다. 이 표는 관측치로만 싣는다.)

### 완주 후 harness (prior, N=20, n=772, seed=0, sigma_scale=1.0)

| checkpoint | 선정 이유 | **prior valid (single)** | majority@20 | pass@20 | verifier_pass | json |
|---|---|---|---|---|---|---|
| `step_17085` | 최종 | **0.00%** | 0.00% | 0.00% | 0.00% | `experiments/results/ema_test_fixb_step_17085.json` |
| `step_1139` | in-loop token acc 최대(0.9184) | **0.00%** | 0.00% | 0.00% | 0.00% | `experiments/results/ema_test_fixb_step_1139.json` |
| `step_15946` | 후반부 token acc 최대(0.9101) | **0.00%** | 0.00% | 0.00% | 0.00% | `experiments/results/ema_test_fixb_step_15946.json` |

세 checkpoint 모두 772×20 = 15,440개 샘플 중 valid 보드 **0개**.

---

## 판정 재료 (해석은 사용자)

**질문: EMA off가 prior valid를 살렸는가?**

**아니오.** EMA를 완전히 끈 `gram_nq8_fix_b`에서도 prior valid는 **0.00%**다.

- in-loop `nqueens/accuracy`: 15회 eval 전부 0.0000 (오른 적 없음)
- 완주 후 harness: 최종/최고점 3개 checkpoint 전부 0.00% (772×20 샘플에서 valid 0개)
- 교차검증: EMA on인 기존 `gram_nq8_mean_nh512`도 corrected 0.00% / raw 0.00%

즉 EMA를 켜든 끄든, bias correction을 걸든 안 걸든, N-Queens prior valid는 0%다.
`fix_b`는 EMA 외에 `lr_warmup_steps`도 2000 → 500으로 함께 바꿨으므로,
이 run은 "EMA off"와 "warmup 500"을 **분리하지 않는다**. 둘 중 어느 것도
prior valid를 0%에서 끌어올리지 못했다는 것까지가 이 데이터가 말하는 범위다.

**교차검증 항목:**

| 검증 | 결과 |
|---|---|
| bias correction 코드에 수식 오류 | 발견하지 못함 (`ema.py:57-59`, step=1 손검산 통과) |
| Sudoku raw vs corrected | 최대 1.60%p 차이, 부호 엇갈림 → 큰 괴리 없음 |
| N-Queens raw vs corrected (nh512) | 둘 다 0.00% |
| N-Queens live (nh512) | **확인 못 함** — `.live` 파일 0개 |
| N-Queens EMA 완전 off 재학습 (fix_b) | 아래 harness 수치 |

---

## 임의로 정한 것 전부

1. **`evaluators` 문법을 지시와 다르게 썼다.** 지시의 `evaluators="[nqueens@NQueens]"`는
   pydantic 검증에서 실패한다 — `pretrain.py:53`이 `evaluators: List[EvaluatorConfig]`이고
   `EvaluatorConfig`(`pretrain.py:41-43`)는 `name` 필드를 가진 매핑을 요구한다.
   에러: `evaluators.0 Input should be a valid dictionary or instance of EvaluatorConfig
   [input_value='nqueens@NQueens']` (첫 launch 실패, `logs/gram_nq8_fix_b.log` 초판).
   `gram_nq8_fix.sh:30`에 있는 동작하는 형식 `'evaluators=[{name: nqueens@NQueens}]'`로
   바꿔 재launch했다. (참고: `evaluators/nqueens.py:29`의 docstring이 동작 안 하는
   형식을 사용법으로 적어두고 있다 — 수정하지 않았다.)
2. **`cd` 경로.** 기존 sh들은 `cd /workspace/TinyRecursiveModels`를 하드코딩했는데
   이 박스의 repo는 `/home/chaeun/gram-repro`다. 새 sh는 스크립트 자신의 위치를
   repo 루트로 쓴다(`train_nqueens8_fix.sh:22`).
3. **명시 override 3개 추가.** `arch.halt_max_steps=16`, `global_batch_size=768`은
   yaml/cfg 기본값과 같지만 논문 준수 항목이라 커맨드에 박아 config dump에 남게 했다.
   기존 `train_nqueens8_1.5.sh`에는 없던 인자다(값은 동일).
4. **"최고점" checkpoint 선정 기준.** in-loop `nqueens/accuracy`가 전 구간 0이라
   그것으로는 최고점을 못 고른다. 대신 `all/accuracy`(token) 최대인 `step_1139`(0.9184)와,
   후반부 최대인 `step_15946`(0.9101)을 골라 최종 `step_17085`와 함께 3개를 측정했다.
5. **raw EMA 재구성 방식** — 위 "raw 가중치를 어떻게 얻었나" 전체. 특히
   `make_raw_ema.py`가 `torch.distributed.broadcast`를 일시적으로 no-op으로 바꾼다
   (`pretrain.py:144-147`은 process group을 요구하는데, rank 0에서 `src=0` broadcast는
   항등이라 w0를 바꿀 수 없다).
6. **`--check-determinism`의 3개 대조군**(재실행 / world_size 4 vs 1 / dataloader 유무)은
   내가 고른 검사 항목이다. 원본 `w0`와의 직접 대조가 아니다.
7. **Sudoku raw 앵커를 n=1000으로** 측정했다 — 감사 Part D와 같은 설정이라
   corrected 수치를 그대로 비교 대상으로 쓸 수 있게 했다.
8. GPU 배정: 학습 3,4,5,6 / eval 0. 지시가 준 기본값(`GPUS=3,4,5,6`)을 그대로 썼고
   eval은 남는 0번을 썼다.

---

## 다음 arm 실행 커맨드

`train_nqueens8_fix.sh`는 수정 없이 환경변수만으로 arm을 바꾼다.
GPU는 launch 직전 점유 검사(각 >1000MiB면 `FATAL` exit 1)를 통과해야 뜬다.

```bash
cd /home/chaeun/gram-repro

# EMA on 대조군 (fix_b와 EMA만 다름)
RUN=gram_nq8_fix_a EMA=True  bash train_nqueens8_fix.sh

# β 축
RUN=gram_nq8_fix_c BETA=0.3  bash train_nqueens8_fix.sh
RUN=gram_nq8_fix_d BETA=1.0  bash train_nqueens8_fix.sh

# GPU/포트를 바꿔 동시에 두 arm 돌리기
RUN=gram_nq8_fix_c BETA=0.3 GPUS=0,1,2,3 PORT=29631 bash train_nqueens8_fix.sh &
RUN=gram_nq8_fix_d BETA=1.0 GPUS=4,5,6,7 PORT=29632 bash train_nqueens8_fix.sh &

# 조립만 확인 (실행 안 함)
DRYRUN=1 RUN=gram_nq8_fix_c BETA=0.3 bash train_nqueens8_fix.sh
```

지원 변수: `RUN GPUS NPROC PORT BETA KL_RED EMA EMA_RATE WARMUP NHEXP DRYRUN`.

harness 재측정:
```bash
source ~/venv-gram/bin/activate
python experiments/eval_harness.py \
  --ckpt-dir checkpoints/Nqueens-8-ACT-torch/<RUN> --step step_17085 \
  --task nqueens --num-puzzles 772 --k 20 --seed 0 \
  --out experiments/results/<RUN>_final.json
```

raw EMA 재구성(EMA on arm에만 의미 있음):
```bash
python experiments/make_raw_ema.py \
  --ckpt-dir checkpoints/Nqueens-8-ACT-torch/<RUN> --step step_17085 --check-determinism
# -> step_17085.raw 생성, --step step_17085.raw 로 harness 재실행
```
