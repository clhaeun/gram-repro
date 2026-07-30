# REPORT_audit.md — GRAM 재현 코드 전수 감사

- 일시: 2026-07-26
- 브랜치: `main` (origin/main과 동기, HEAD `4c847f1`)
- 검증 인터프리터: `/home/chaeun/venv-gram/bin/python` — torch **2.5.1+cu124**, CUDA True
  - 주의: 이 감사 박스의 파이썬은 **3.12.3**, GPU는 48GB급 8장. `requirements.txt`가 기록한
    학습 박스(Python 3.11.10 / RTX 3090 x8)와 다르다. torch 버전은 일치.
- GPU 사용: 4,5번만 사용 (지시대로). 학습 launch 없음. 전부 드라이런/eval.
- 회귀 eval 산출물: `experiments/results/audit_regression.json` / `.log`

판정 범례: **[일치]** 논문 일치 · **[이탈]** 논문 이탈 · **[미기재]** 논문 미기재(우리 임의) · **[버그]**

---

## Part A — 논문 대조

### A1. KL reduction — [일치]

**현재 상태**
- `models/losses.py:95-105` — 주석 토글이 아니라 config 키 분기, 허용값 외 **예외 발생**:
```python
95   if outputs.get("kl") is not None:
96       kl_reduction = getattr(self.model.config, "kl_reduction", None)
97       if kl_reduction not in ("sum", "mean"):
98           raise ValueError(
99               f"arch config must set kl_reduction to 'sum' or 'mean', got {kl_reduction!r}. "
...
103      kl_per_pos = outputs["kl"].sum(-1)                  # [B, L] -- D는 합
104      kl_loss = (kl_per_pos.sum(-1) if kl_reduction == "sum"
105                 else kl_per_pos.mean(-1)).sum()          # L reduction -> batch 합
```
  → `None`/오타 시 조용히 0이 되지 않고 `ValueError`. **주석 토글 흔적 없음** (grep `reduction` 결과 losses.py에 잔재 없음).
- `models/recursive_reasoning/gram.py:76` — pydantic 필드 존재, 기본값 `"sum"`:
```python
76   kl_reduction: str = "sum"   # "sum" | "mean"
```
- `config/arch/gram.yaml:31` `kl_reduction: sum`, `config/arch/gram_nqueens8.yaml:34` `kl_reduction: sum` — 둘 다 명시.
- 실측: 로드된 체크포인트의 `all_config.yaml`에 값이 남음 —
  `gram_sudoku_halt_only`=`sum`, `gram_nq8_mean1`/`gram_nq8_mean_nh512`=`mean` (아래 Part D 로드 로그).

**판정: [일치]**. 이번 사고("config만 보고는 어느 reduction으로 돌았는지 모름")는 재발 불가.

### A2. 노이즈 head 크기 — [일치] (수치 검증 통과)

**현재 상태**
- `gram.py:80` `noise_head_expansion: float = 4.0` (pydantic 기본값, 기존 checkpoint 호환).
- `gram.py:185-191` — 4개 head 전부 이 값 사용:
```python
185  nh_exp = self.config.noise_head_expansion
188  self.prior_mu     = SwiGLU(hidden_size=D, expansion=nh_exp)
189  self.prior_logvar = SwiGLU(hidden_size=D, expansion=nh_exp)
190  self.post_mu      = SwiGLU(hidden_size=D, expansion=nh_exp)
191  self.post_logvar  = SwiGLU(hidden_size=D, expansion=nh_exp)
```
- `config/arch/gram_nqueens8.yaml:38` `noise_head_expansion: 1.5`.

**검증** (인스턴스화 → `sum(p.numel())`, nqueens dims: seq_len=64, vocab=3, hidden=512, puzzle_emb_len=16):
```
noise_head_expansion=1.5 : sum(p.numel()) = 9,966,085   ← 논문 10M 대응
noise_head_expansion=4.0 : sum(p.numel()) = 16,257,541
default (필드 없음)       : 16,257,541   (= 4.0)
```
**둘 다 지정값과 정확히 일치.** `gram_0722.py`(백업)는 `expansion=self.config.expansion`(=4)를 상속하던 pre-fix 버전 — 현재 `gram.py`가 수정본임을 diff로 확인.

**판정: [일치]**.

### A3. gradient 격리 — [일치]

- ACT halt head: `gram.py:386` — `.detach()` 있음:
```python
386  q_logits = self.q_head(z_H[:, 0].detach()).to(torch.float32)
```
  → 논문 A.1 "does not propagate gradients into the recursive core" 준수.
- LPRM(`lrpm_z`) 마지막 원소는 **grad 보유**: `gram.py:350`의 append는 `torch.no_grad()` 블록 안(H_cycles-1 회),
  마지막 `gram.py:379` append는 grad 경로의 `z_H = z_H + eps`(378) 이후 → grad 있음.
```python
323  lrpm_z = []
350          lrpm_z.append(z_H[:, 0])   # no_grad 블록 내부
378  z_H = z_H + eps
379  lrpm_z.append(z_H[:, 0])           # grad 경로
```
  → 논문 A.2(Eq 16)는 LPRM에 stop-gradient 미언급. 현 상태가 논문 준수. **바꾸지 않음**(지시대로 현상 보고만).

**판정: [일치]**.

### A4. KL 위치 (Eq 14) — [일치]

- KL은 grad 경로(마지막 transition)에서만 계산됨. no_grad 루프(`gram.py:326-350`)에서는 `kl` 미계산;
  `gram.py:366`에서만 `kl = self.kl_loss(...)`, 그 외 경로는 `kl = None`(373).
  → supervision step당 **마지막 transition에만** KL. Eq 14 준수.
- `trace_kl` 계측: `gram.py:307` `trace_kl = getattr(self, "trace_kl", False)` — **기본 off**.
  on일 때도 `self._kl_trace` 리스트에 `.item()`만 append(343-345, 368-369), 손실/그래프에 영향 없음.

**판정: [일치]**.

### A5. TTS 손잡이 — [일치] (기본값에서 bit-exact)

- `sigma_scale`: `gram.py:199` `self.sigma_scale = 1.0` — 순수 float(파라미터/버퍼 아님) → **state_dict 미포함**.
  `reparameterize`(248)와 `_latent_diag`(264)에서 곱해짐.
- `use_posterior` kwarg: `gram.py:300,415,425` 살아있음. 기본 `None` → `gram.py:303-304`:
```python
303  if use_posterior is None:
304      use_posterior = self.training
```
- 기본값(`sigma_scale=1.0`, `use_posterior=self.training`)에서 학습 경로 불변 → **bit-exact**.

**판정: [일치]**.

---

## Part B — 진단 인프라

### B1. σ/KL 로깅 — [일치]

- `gram.py:253` `@torch.no_grad()` 데코레이터 + 내부 전부 `.detach()`:
```python
253  @torch.no_grad()
254  def _latent_diag(self, mu_p, logvar_p, mu_q, logvar_q, kl):
```
  반환 키(그대로 인용, `gram.py:268-282`):
  `diag/sigma_prior_std_mean`, `diag/sigma_prior_std_p50`, `diag/sigma_prior_std_p95`,
  `diag/mu_p_norm`, `diag/sigma_post_std_mean`, `diag/sigma_post_std_p50`,
  `diag/sigma_post_std_p95`, `diag/mu_q_norm`, `diag/kl_raw`.
  (posterior 경로에서만 생성 → eval 불변, `gram.py:376`.)
- `losses.py:128-129` — metrics에 병합:
```python
128  if outputs.get("diag") is not None:
129      metrics.update(outputs["diag"])
```
- `pretrain.py:348-354` — **이중 정규화 안 함**. diag/* 는 rank별 배치 평균이므로 `dist.reduce`(합)만 world_size로 되돌림:
```python
352  if k.startswith("diag/"):
353      return world_size
354  return global_batch_size if k.endswith("loss") else count
```
  → count로 다시 나누지 않음. 왜곡 없음.

**판정: [일치]**.

### B2. harness — [일치] (SEQ_LEN 상수는 잔존하나 런타임 미사용)

- `experiments/gram_harness.py:30` — 모듈 상수 `SEQ_LEN = 81`이 **정의는 남아있음**. 단, 주석 명시대로
  런타임 경로는 `inputs.shape[-1]`에서 seq_len을 받음:
```python
30   SEQ_LEN = 81            # sudoku default only; _batch/rollout now take seq_len from inputs.shape[-1]
104      seq_len = inputs.shape[-1]   # _batch()
197      seq_len = inputs.shape[-1]   # rollout()
```
  grep 결과 `SEQ_LEN` 상수를 **참조하는 코드는 없음**(정의 1건뿐, 나머지는 주석/docstring). 즉 하드코딩 오작동 경로 제거됨.
  → 완전 정리하려면 line 30 삭제 가능하나, 미사용 상수라 오작동 위험 없어 이번엔 두었다(아래 미해결 목록).
- `experiments/eval_harness.py` — seq_len을 데이터에서 받음:
  `sample_dataset:227` `n, seq_len = inputs.shape`; `_make_batch(...,seq_len)`; `main:379` `seq_len={inputs.shape[1]}`.
  (line 75 `assert L == 81`은 SudokuTask.verify 내부로 sudoku 전용 의도된 가드 — nqueens는 NQueensTask가 처리.)

**판정: [일치]** (오작동 제거 완료). 잔존 상수 1줄은 무해.

### B3. N-Queens evaluator — [일치] (라벨 772개 검증 통과)

- 헤드라인 = 제약 만족(valid_rate), `evaluators/nqueens.py:129`:
```python
129  "nqueens/accuracy": float(acc[1] / total),          # acc[1]=valid.sum()  (제약 만족만)
131  "nqueens/strict_accuracy": float(acc[3] / total),   # 우리 자체 진단(별도 키)
```
  → 논문 §4.2 "satisfies all constraints" 대응. clue 보존(consistent/strict)은 별도 키로 병기.
- 하위 제약 필드 존재: `verify_boards` 반환에 `count_ok/row_ok/col_ok/d1_ok/d2_ok`(nqueens.py:70-74). ✓
  (`right_count_but_invalid`는 명시적 필드로는 없음 — count_ok & ~valid 로 사후 유도 가능. 아래 미해결 목록에 기록.)
- **검증** (test 라벨 772개를 preds로 주입):
```
valid_rate=1.0000  consistent_rate=1.0000  strict_rate=1.0000  queen_count_mean=8.0000
evaluator.result(): nqueens/accuracy=1.0  strict_accuracy=1.0  consistent_rate=1.0
                    queen_count_mean=8.0  n_eval=772.0
```
  → 정답 라벨은 전부 valid=consistent=strict=1.0, queen_count=8.0. 규칙 검증기 정확.
- TRM 인터페이스: `required_outputs={'preds','inputs'}`, `begin_eval/update_batch/result` 전부 존재(hasattr 4/4).
  `create_evaluators` 드라이런 → `['NQueens']` 로드, `n=8` 자동 산출(seq_len=64 → √64). **실제 로드됨**.

**판정: [일치]**.

---

## Part C — 미해결/미확인 (확인만, 수정 안 함)

### C1. EMA — [이탈: bias correction은 논문 미기재]

- `ema_rate` 설정: `pretrain.py:628` `EMAHelper(mu=config.ema_rate)`. 값은 학습 sh에서 `ema_rate=0.9999`
  (`gram_sudoku_halt_only.sh:24` 등 전 스크립트). cfg 기본값은 0.999지만 sh가 0.9999로 override.
  → **논문 B.2의 0.9999와 일치**.
- checkpoint 저장 내용: `pretrain.py:688,692` — EMA on이면 `step_N` = **bias-corrected EMA**, `step_N.live` = **live(비-EMA) 가중치** 둘 다 저장:
```python
688  save_train_state(config, train_state_eval)                # step_N = (bias-corrected) EMA
692      save_train_state(config, train_state, suffix=".live") # step_N.live = live
```
- bias correction **있음** (논문 미기재): `models/ema.py:54-66`. `ema_copy(step=train_state.step)`(pretrain.py:659) 경로가
  `(shadow - mu^t·w0)/(1-mu^t)` 적용:
```python
59   return (self.shadow[name].data - mt * self.w0[name].data) / denom
```
- EMA 버퍼는 **초기 가중치 복사로 시작**(0 아님), `ema.py:35`:
```python
35   self.shadow[name] = param.data.clone()   # register() 시 init 스냅샷
```
  → step t에서 초기값 잔존 비중 = `d^t`. N-Queens 17,093 step: `0.9999^17093 ≈ 0.181` (18%). ema.py:9-22 주석도 동일 지적.
- EMA warmup: **없음**.

**판정: [이탈]**. 논문 B.2는 EMA 0.9999만 명시하고 **bias correction/디바이싱을 언급하지 않는다**.
현재 코드는 eval/저장 체크포인트에 bias-corrected EMA를 쓴다 → 논문 대비 평가 가중치가 다르다.
지시대로 **수정하지 않음**. (우선순위 높은 이탈 항목 — 아래 목록 참조)

### C2. lr warmup — [이탈/미기재]

- `config/cfg_pretrain.yaml:26` `lr_warmup_steps: 2000`, `:25` `lr_min_ratio: 1.0`. 학습 sh들은 이 둘을 **override 안 함**.
  출처: TRM/cfg 기본값(GRAM 전용 아님).
- `lr_min_ratio: 1.0` → 코사인 스케줄이 `base_lr*(1.0+0)`=상수. 즉 **warmup 후 lr 상수**(감쇠 없음).
- 총 step 대비 warmup 비중:
  - Sudoku 65,100 step → 2000/65100 = **3.07%**
  - N-Queens 17,093 step → 2000/17093 = **11.70%**

**판정: [이탈]**. 논문 27페이지 어디에도 warmup 언급 없음(사용자 확인). 우리는 TRM 기본 2000 step warmup을 쓰고 있다.

### C3. 데이터로더 순회 단위 — [확인 완료: group당 1샘플]

- `puzzle_dataset.py:_iter_train`(201) → `_sample_batch`(16): group_order를 순회하며 **group당 puzzle 하나를 샘플**:
```python
24   group_id = group_order[start_index]
25   puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])   # group당 1 puzzle
```
  그 puzzle의 예시들을 배치에 채움(각 group=unique input당 1 라벨). **전 (input,label) pair 순회 아님**.
- 실측(`data/nqueens-8/train/dataset.json`): `total_groups=4376`, `total_puzzles=7194`, `mean_puzzle_examples=1.0`.
  - epoch당 group 4376개 순회 → 3000 epoch × 4376 / 768 = **17,093.75 → 17,093 step** (관측치와 일치).
  - train pair 7194개 중 epoch당 4376개(**60.8%**)만 봄. 사용자 지적("61%만 봄")과 일치.
- pair 순회였다면 3000×7194/768 = **28,102 step**이 됐을 것 — 현재는 아님.

**판정: [확인 완료]**. group(unique input)당 1샘플 방식. EMA 오염률/학습량 계산은 17,093 step 기준이 맞다.

### C4. eval halting — [이탈, TRM 상속]

- 추론 시 항상 `halt_max_steps` 사용. `gram.py:452` `if self.training and (...)` 게이트로 q_halt 조기중단은 **학습 시에만** 적용:
```python
449  halted = is_last_step
452  if self.training and (self.config.halt_max_steps > 1):
457      if self.config.no_ACT_continue:
458          halted = halted | (q_halt_logits > 0)
```
  → eval에서는 `halted = is_last_step`만 → 항상 max steps.
- 논문 A.1은 추론 시 `σ(q_halt) > 0.5` 중단 명시. 현재는 미적용(TRM 상속). Sudoku 71.85%는 이 상태 수치.

**판정: [이탈]**. TRM 상속. 지시대로 **사실만 보고, 수정 안 함**.

### C5. 기타 하이퍼파라미터 표

| 항목 | 현재 값 | 출처 | 논문 B.2 |
|---|---|---|---|
| optimizer | AdamW | `pretrain.py:21,183` | ✓ AdamW |
| lr | 1e-4 | sh override (cfg도 1e-4) | ✓ 1e-4 |
| weight_decay | **1.0** | sh override (cfg 기본 0.1) | ✓ 1.0 |
| grad clip | 1.0 | `pretrain.py:318` | ✓ 1.0 |
| batch | 768 | cfg/sh | ✓ 768 |
| EMA rate | 0.9999 | sh | ✓ 0.9999 |
| kl_balance | 0.8 | yaml | ✓ 0.8 |
| Nsup (=halt_max_steps) | 16 | yaml | ✓ 16 |
| beta1 | 0.9 | `cfg:29` | 미기재 (Llama/TRM 기본) |
| beta2 | 0.95 | `cfg:30` | 미기재 |
| lr_min_ratio | 1.0 (lr 상수) | `cfg:25` | 미기재 |
| lr_warmup_steps | 2000 | `cfg:26` | 미기재 (논문에 warmup 없음) |
| puzzle_emb_lr | 1e-4 | sh override (cfg 기본 1e-2) | 미기재 |
| puzzle_emb_weight_decay | 1.0 | sh override (cfg 기본 0.1) | 미기재 |

**판정**: 논문 명시 항목은 전부 [일치]. beta1/beta2/lr_min_ratio/warmup/puzzle_emb_* 는 [미기재]로 TRM 상속.

---

## Part D — 회귀 검증

명령(지시대로), GPU 4에서 실행:
```
python experiments/eval_harness.py \
  --ckpt-dir checkpoints/Sudoku-extreme-1k-aug-1000-test10k-ACT-torch/gram_sudoku_halt_only \
  --step step_65100 --task sudoku --num-puzzles 1000 --k 20 --seed 0 \
  --out experiments/results/audit_regression.json
```

| 지표 | 기대(±1%p) | 실측 | Δ | 판정 |
|---|---|---|---|---|
| single | ~70.4% | **70.60%** | +0.2 | ✅ PASS |
| majority@20 | ~89.2% | **88.40%** | −0.8 | ✅ PASS |
| pass@20 | ~92.6% | **93.10%** | +0.5 | ✅ PASS |

세 지표 모두 ±1%p 게이트 안. bf16 비결정성 + 감사 박스 GPU(48GB급, 원본은 3090) 차이 감안하면 정상 재현.
**config 필드 추가(kl_reduction, noise_head_expansion)가 기존 결과를 깨지 않았다.**

**주의(내가 낸 시행착오)**: `DISABLE_COMPILE=1`로 돌리면 로드 실패한다. 체크포인트는 `torch.compile`된 모델에서 저장돼
키에 `_orig_mod.` 접두사가 있고, `load_model`이 compile 후 언랩하는 전제라서 그렇다. **정상 경로(=compile 켬)에서는 정상 로드.**

### 체크포인트 로드 테스트 (config 필드 추가 호환성)

| 이름 | 결과 | params | noise_head_exp | kl_reduction |
|---|---|---|---|---|
| gram_sudoku_halt_only | ✅ LOADED step_65100 | 14,466,565 | 4.0 | sum |
| gram_sudoku_lmean_v2 | ⚠️ **STEP 파일 없음** (all_config/gram.py/losses.py만) | — | — | — |
| gram_nq8_mean1 | ✅ LOADED step_17085 | 16,257,541 | 4.0 | mean |
| gram_nq8_mean_nh512 | ✅ LOADED step_17085 | 9,966,085 | 1.5 | mean |
| gram_nq8_b007_v2 | ⚠️ **STEP 파일 없음** (all_config/gram.py/losses.py만) | — | — | — |

- 로드 성공한 3개는 config 필드 추가로 **깨지지 않음**(각 all_config.yaml에서 필드 정상 해석).
- `gram_sudoku_lmean_v2`, `gram_nq8_b007_v2`는 디렉토리에 **가중치(step_*) 파일이 아예 없다**. 코드/설정만 남은 빈 체크포인트.
  로드 실패가 아니라 **로드할 대상이 없음**. push와 무관(가중치는 .gitignore로 어차피 추적 안 됨)하나, 재현하려면 재학습 필요.
- 부수 관찰: `gram_nq8_mean1`은 noise_head_expansion=4.0(16.2M) = **노이즈 head 수정 이전**에 학습된 체크포인트.
  all_config.yaml에 4.0이 기록돼 있어 사후 식별 가능 — 바로 이 감사 항목이 잡으려던 그 흔적이 남는다.

---

## Part E — GPU 지정 CLI화

각 학습 sh를 `GPUS`/`NPROC`/`PORT` 환경변수 오버라이드 + 점유 pre-flight로 수정.
**GPU 개수 → `--nproc-per-node` 자동 일치** (개수 어긋남 사고 방지).

수정한 파일:
- `gram_sudoku_halt_only.sh`, `gram_sudoku_lmean_v2.sh` (기본 GPUS=0,1,2,3 PORT=29503)
- `train_nqueens8.sh`, `train_nqueens8_1.5.sh` (기본 GPUS=0,1,2,3 PORT=29611)
  - 위 4개: `GPUS="${GPUS:-0,1,2,3}"`, `NPROC="${NPROC:-$(echo $GPUS | tr ',' '\n' | wc -l)}"`, `PORT` 오버라이드,
    launch 직전 `nvidia-smi`로 각 GPU >1000MiB면 `FATAL` exit 1.
- `gram_nq8_fix.sh`, `gram_nq8_sweep.sh` — 원래 positional `gpu=$1` + `--nproc-per-node 1` 하드코딩이었음.
  `gpu`가 "3,4" 같은 다중 목록이면 nproc가 1로 고정돼 어긋나는 **바로 그 사고 유형**이라, `NPROC`를 `gpu` 개수에서
  자동 계산하도록 고치고 점유 체크 추가(positional 인터페이스는 유지).

**dry-run 조립 결과** (env-var 스크립트, 실행 안 함):
```
GPUS(unset)  -> CUDA_VISIBLE_DEVICES=0,1,2,3  --nproc-per-node 4  --rdzv_endpoint=localhost:29503
GPUS=3       -> CUDA_VISIBLE_DEVICES=3        --nproc-per-node 1  --rdzv_endpoint=localhost:29503
GPUS=3,4,5,6 -> CUDA_VISIBLE_DEVICES=3,4,5,6  --nproc-per-node 4  --rdzv_endpoint=localhost:29503
```
GPU 개수와 nproc 항상 일치. 6개 스크립트 전부 `bash -n` 통과.
점유 pre-flight 동작 확인: GPU 4(당시 회귀 eval로 2935MiB 점유) → `BLOCK` 정상 판정.

**판정: [완료]**.

---

## Part F — 정리

- **백업 파일**: 지시가 말한 `*.bak_*` 9개 이상은 **현재 없음**. 실제로는 날짜 접미사 백업 2개만 존재:
  `models/losses_0722.py`, `models/recursive_reasoning/gram_0722.py`.
  - `.gitignore`는 `*.bak`, `*.bak_*`, `*.orig`, `*.rej`, `*~`, `*.swp`를 잡고 있음(패턴 존재 확인).
    **그러나 `_0722.py`는 이 패턴에 안 걸린다** → 두 파일은 **git 추적 중**(`git ls-files`에 나옴, 이미 커밋됨).
  - 삭제하지 않음(지시). 내용 확인: `losses_0722.py`는 현재 `losses.py`와 **완전 동일**;
    `gram_0722.py`는 noise-head 수정 **이전** 버전. 즉 추적 중인 pre-fix 백업이 저장소에 섞여 있다(아래 목록).
- **requirements.txt**: `torch==2.5.1+cu124`, cu124 인덱스 안내 정확. 헤더에 `Python 3.11.10 | CUDA 12.4 | RTX 3090 x8`
  로 학습 박스 기준 명시. torch 버전은 감사 박스(2.5.1+cu124)와 일치. **파이썬 minor만 3.11.10 vs 3.12.3로 다름**
  (감사 박스가 다른 머신이라 그럴 뿐, 재현 기준 문서로는 문제 없음).
- **git status**: 코드 변경분(추적 파일)은 감사 착수 시점 기준 **전부 커밋 상태**(`git diff --stat` 비어 있었음).
  Part E 수정 전 상태에서 untracked는 결과물 2개뿐:
  `experiments/results/migration_check.json`, `experiments/results/audit_regression.{json,log}`.
  → **이번 감사에서 내가 수정한 것**: 학습 sh 6개(Part E), `REPORT_audit.md`(본 문서),
    `.claude/settings.local.json`(권한 스코프, 사용자 요청). 커밋은 사용자 지시 없어 안 함.

---

## 요약

### ① 수정이 필요한데 안 한 것 (우선순위 순)

1. **[C1 · 높음] EMA bias correction이 논문 미기재인데 eval/저장 경로에 활성.**
   `pretrain.py:659` + `ema.py:61-66`가 `(shadow−μ^t·w0)/(1−μ^t)`를 적용. 논문 B.2는 EMA 0.9999만 명시,
   디바이싱 없음. 재현 충실도를 논문에 맞추려면 raw EMA(`ema()`)로 평가해야 함. **지시상 수정 금지라 보고만.**
   (특히 N-Queens는 18% init 잔존이라 raw vs bias-corrected 차이가 클 수 있음.)
2. **[C4 · 중] eval halting이 σ(q_halt)>0.5 미적용(항상 max_steps).** 논문 A.1 이탈. TRM 상속. 보고만.
3. **[C2 · 중] lr warmup 2000 step은 논문에 없음.** 논문 이탈이나 TRM 상속. 보고만.
4. **[F · 낮음] 추적 중인 pre-fix 백업 2개(`losses_0722.py`, `gram_0722.py`)가 저장소에 섞임.**
   `.gitignore`의 `*.bak_*` 패턴이 `_0722.py`를 못 잡음. push 시 다른 서버로 전파됨(혼동 소지).
   삭제/`git rm`은 지시(삭제 금지) 밖이라 안 함 — 정리 여부는 사용자 판단.
5. **[B2 · 낮음] `gram_harness.py:30`의 미사용 상수 `SEQ_LEN=81` 잔존.** 런타임 미참조라 무해하나 오해 소지.
6. **[B3 · 낮음] `right_count_but_invalid` 전용 필드 부재.** count_ok & ~valid로 유도 가능하나 명시 키는 없음.

### ② 논문 미기재라 우리가 임의로 정한 것

- beta1=0.9, beta2=0.95 (Llama/TRM 기본)
- lr_min_ratio=1.0 → warmup 후 lr 상수 (감쇠 스케줄 미채택)
- lr_warmup_steps=2000 (TRM 기본)
- puzzle_emb_lr=1e-4, puzzle_emb_weight_decay=1.0 (sh override)
- EMA bias correction 적용 여부 (①과 동일 사안)
- clue-보존(consistent/strict)을 N-Queens 자체 진단으로 추가 (헤드라인은 논문 정의=valid_rate 유지)

### ③ Part D 회귀 결과

single 70.60% / majority@20 88.40% / pass@20 93.10% — **세 지표 모두 ±1%p 안, 회귀 없음.**
로드 가능한 체크포인트 3/5는 config 필드 추가에도 정상 로드. 2개는 가중치 파일 부재(빈 dir).

### ④ push 판단

**코드/설정 관점에서 push 가능.** 근거:
- Part A(논문 대조) 전 항목 [일치], 수치 검증 2건(param count 9,966,085 / evaluator 772라벨) 정확 통과.
- Part D 회귀 무손상, 기존 체크포인트 로드 호환.
- Part E 스크립트 GPU CLI화 완료, 6개 `bash -n` 통과, nproc 자동 일치 확인.

**단, push 전에 알고 넘어갈 것 (블로커 아님, 판단 필요):**
- C1/C2/C4는 **논문 이탈이지만 의도된 현상 유지**(지시상 수정 금지). 다른 서버가 "논문 그대로"를 기대하면 이 3개가 차이를 만든다.
- `losses_0722.py`/`gram_0722.py` pre-fix 백업이 추적 중 → 같이 전파됨. 정리하고 push할지 사용자 결정 필요.
- `.claude/settings.local.json`은 전역 `~/.config/git/ignore`(`**/.claude/settings.local.json`)로 **이미 추적 제외** → push 안 됨. 조치 불필요.
- Part E로 학습 sh 6개가 modified 상태(커밋 안 함, 사용자 지시 대기). `REPORT_audit.md`와 `experiments/results/*` 결과물은 untracked.
