# GRAM 재현 — eval harness + N-Queens 종료 보고

작업일: 2026-07-21 (16:00 ~ 16:45 KST)
Launch gate: **통과** → N-Queens β sweep 3 run이 tmux에서 실행 중.

---

## 1. Task별 AC 체크리스트

### Task 0 — wandb 통합 + σ 로깅 — **통과**

`WANDB_MODE=offline` 드라이런(78 step, GPU 4, `torch.compile` 켠 채로) 후 offline
datastore를 직접 스캔해 확인. 요구된 키가 전부 기록됨:

| 요구 항목 | 실측 키 / 값 (step 78) |
|---|---|
| `lm_loss` / `q_halt_loss` / `lprm_loss` | `train/lm_loss=2.6976`, `train/q_halt_loss=0.00665`, `train/lprm_loss=4.386` |
| `kl_raw` (reduction 전 per-position 평균) | `train/diag/kl_raw=26892.73` |
| `kl_loss` (loss 실투입값) | `train/kl_loss=26892.73` |
| σ prior `std_mean/p50/p95` | `0.14229 / 0.13625 / 0.22148` |
| σ posterior `std_mean/p50/p95` | `0.15729 / 0.13389 / 0.34298` |
| `mu_p_norm` / `mu_q_norm` | `13.307 / 24.912` |
| eval `exact_accuracy` / `token_accuracy` / `kl_loss` | `eval/all/exact_accuracy=0`, `eval/all/token_accuracy=0.09168`, `eval/all/kl_loss=0` ✅ prior 경로 확인 |
| 코드 스냅샷 | wandb run `files/`에 `gram.py`, `losses.py` 업로드됨 |
| 전체 config | `wandb.init(config=config.model_dump())` — 기존 코드가 이미 수행 |

- **stdout 포맷 불변 확인**: `[step N]` 라인의 키 집합은 이전과 동일
  (`accuracy / exact_accuracy / kl_loss / lm_loss / lprm_loss / q_halt_accuracy / q_halt_loss + lr`).
  `diag/*`는 stdout 필터(`'loss' in k or 'accuracy' in k`)에 걸리지 않도록 이름을 지었고,
  `token_accuracy`는 metrics dict가 아니라 **wandb 전용 mirror**로만 추가해서 `[EVAL]` 라인도 그대로다.
- **부수 검증**: `kl_raw`와 `kl_loss`가 **정확히 일치**한다. `lmean` reduction
  (`kl.sum(-1).mean(-1).sum() / global_batch_size`)이 곧 per-position 평균이므로 수학적으로 같아야 하고,
  실제로 같게 나왔다 → reduction이 의도대로 동작한다는 독립 확인.
- 추가 forward 없음: 전부 forward가 이미 만든 텐서의 `.detach()`.

### Task 1 — 통합 eval harness — **회귀 앵커 통과 (gate open)**

`experiments/eval_harness.py`. 원본 `e1_*.py`/`e3_*.py`는 손대지 않았다.

**회귀 앵커** (`gram_sudoku_halt_only/step_65100`, n=1000 seed 0, N=20, σ=1.0):

| 지표 | 기대 | 실측 | 판정 |
|---|---|---|---|
| majority@20 | 88.9 ± 0.5 %p | **89.20%** | ✅ (+0.30) |
| pass@20 | 92.5 ± 0.5 %p | **92.60%** | ✅ (+0.10) |
| single | ≈ 70.5% | **70.40%** | ✅ (−0.10) |

→ `experiments/results/eval_harness_regression.json`

**신규 K=5 보고** (논문 Table 3의 93.96% 대조용): **majority@5 = 79.50%**,
**pass@5 = 85.50%** (unbiased 추정량) / **85.80%** (first-5 실측).

**앵커 편차에 대한 설명**: 1000문제 기준 각각 1~3문제 차이다. 나는 RNG 스트림을
`gram_harness.rollout`과 동일하게 맞췄으므로(candidate-outer / batch-inner, `seed*100003+k`,
batch_size 512 고정크기 패딩) 재현이 거의 정확하고, 남은 차이는 bf16 커널의 비결정성
(다른 GPU에서 재실행)으로 설명되는 크기다. 하네스 불일치로 보이지 않는다.
— RNG를 맞춘 이유: n=1000에서 노이즈 재추첨 1회의 표준편차가 약 1.0 %p로 ±0.5 %p 게이트보다 넓다.
독립 추첨으로 갔으면 정상 코드도 게이트를 못 넘겼을 것이다.

**AC 2 — lmean_v2 판정 (수치만, 해석 없음)** → `experiments/results/lmean_v2_eval.json`

동일 하네스 / 동일 n=1000 seed 0 서브샘플 / N=20 / σ=1.0, `step_65100` (EMA 사본):

| 지표 | halt_only | lmean_v2 |
|---|---|---|
| single | 70.40% | **1.80%** |
| majority@5 | 79.50% | 2.10% |
| pass@5 | 85.50% | 2.31% |
| majority@20 | 89.20% | 1.90% |
| pass@20 | 92.60% | 2.70% |
| distinct@20 | 6.99 | **18.36** |
| prior σ std_mean | 0.955 † | **1.523** |

† `e4_sigma_step_65100_k20_n1000.json` 기존 값.

**σ·train lm_loss 출처에 대한 정정**: 지시서는 "lmean_v2 **학습 로그에서**" 추출하라고 했지만,
그 run은 `WANDB_MODE=disabled`로 돌았고 σ 로깅(Task 0)보다 **먼저** 실행됐다.
로그에 σ도, `[step N]` 학습 손실 라인도 **존재하지 않는다**. 없는 값을 지어내지 않았고,
대신 `experiments/sigma_probe.py`를 새로 만들어 **체크포인트 가중치에서 직접 측정**했다
(로그에 남았어야 할 바로 그 양). json의 `sigma.note`에 이 사실을 명시해 두었다.
로그에서 실제로 얻을 수 있었던 건 마지막 in-training `[EVAL]` 라인뿐이다:
`eval lm_loss=4.8045, exact_accuracy=0.0158, token_accuracy=0.5192`.
측정된 posterior KL/sample = 13.08, posterior σ std_mean = 1.521 (prior와 거의 동일).

*(지시서 지시대로 판정·해석은 하지 않았다. 위 표와 σ/KL 수치가 전부다.)*

### Task 2 — N-Queens 빌더 + verifier — **통과** (서브에이전트 수행)

`dataset/nqueens_verify.py`, `dataset/build_nqueens_dataset.py`.

| 앵커 | 기대 | 실측 |
|---|---|---|
| n=8 완전해 | 92 | **92** |
| n=8 쌍 | 8,464 | **8,464** |
| n=10 완전해 | 724 | **724** |
| n=10 쌍 | 126,700 | **126,700** |

- train ∩ test input hash = ∅ (n=8: 4376 / 772 그룹, 교집합 0) — 코드에 assert.
- verifier 교차검증: sidecar 해 1,626쌍 전부 `strict=True` (FAIL 0);
  queen 1개 이동 corrupted 1,000개 전부 `valid=False` (FAIL 0).
- 해 개수 분포 히스토그램 출력됨 (n=8: 65.5%가 유일해, 평균 1.644, 최대 18).
- **실질 판정 기준(= `puzzle_dataset.py`가 로드 가능한가)**: 양 split 모두 `PuzzleDataset`
  인스턴스화 + 1 batch 순회 성공. `[B,64] int32`, 관측 토큰 집합 `{1,2}`.
- n=10도 시간이 남아 함께 생성됨.

데이터셋 통계 (n=8): train 7,194행 / 4,376그룹, test 772행 / 772그룹, seq_len 64, vocab 3.

**하네스 쪽 교차 검증(내가 추가로 수행)**: sidecar 해를 후보로 위장해 `eval_harness`의
nqueens 플러그인을 end-to-end 테스트했다. 200문제 중 50개의 후보0을 두 행 교환으로 오염시킨 결과
`single`=0.750(정확히 50/200), `verifier_pass_rate`=0.9875=(4000−50)/4000, `coverage@20`=0.997로
설계대로 동작. 또한 **strict-valid이면서 label과 다른 후보가 22.2%** — N-Queens에서
label 일치로 채점하면 안 되는 이유의 정량 근거다(= 판정 함수 분리가 필요한 이유).

### Task 3 — N-Queens config — **통과(수치 보고), 논문 10M과는 불일치**

`config/arch/gram_nqueens8.yaml`: `mlp_t=False`, `pos_encodings=rope`, `L_cycles(K)=4`,
`H_cycles(T)=3`, seq 64 / vocab 3, `kl_coef=0.07`, halt_max_steps(Nsup) 16.

| 모듈 | params | 비중 |
|---|---|---|
| L_level (attention 코어 ×2 layer) | 6,815,744 | 41.9% |
| prior_mu / prior_logvar / post_mu / post_logvar | 각 2,359,296 | 각 14.5% |
| embed_tokens / lm_head / q_head / lrpm_head | 1,536 / 1,536 / 1,026 / 513 | ~0% |
| **합계** | **16,257,541** | |

- **노이즈 head 4개 합 = 9,437,184 (58.0%)** + logvar bias 스칼라 2개.
- **논문 10M 대비 1.63배.** 원인은 명확히 노이즈 head다: 4개가 전부
  `SwiGLU(hidden=512, expansion=4)` 풀사이즈라 코어(6.8M)보다 크다. 이 4개를 단순 Linear
  (512×512)로 바꾸면 4×0.26M = 1.05M이 되어 총 ~7.9M으로 떨어진다.
  **지시서가 hidden 512 / expansion 4 / L_layers 2를 명시했으므로 임의로 줄이지 않고
  그대로 두고 불일치만 보고한다.** 논문의 10M이 노이즈 head를 Linear로 뒀거나
  공유했을 가능성이 높다 — 확인이 필요한 항목.
- `puzzle_emb`는 `CastedSparseEmbedding`이라 buffer로 잡혀 `parameters()` 집계에 0으로 나온다
  (identifier도 1개뿐).

### Task 4 — 학습 launch — **완료 (3/3 정상)**

GPU 0/1/2 사용, **GPU 3은 지시대로 비워둠**. 각각 별도 tmux 세션 + 포트 분리 + `tee`(-a 아님).

~50 step 및 이후 점검 (전부 정상):

| run | GPU | β | NaN | kl_raw vs reduction | σ prior std_mean | lm_loss |
|---|---|---|---|---|---|---|
| `gram_nq8_b007` | 0 | 0.07 | 없음 | `kl_raw`=`kl_loss`=0.5517 ✅ 일치 | 0.502 | 0.0026 |
| `gram_nq8_b0045` | 1 | 0.045 | 없음 | `kl_raw`=`kl_loss`=1.6241 ✅ 일치 | 0.532 | 0.0027 |
| `gram_nq8_b010` | 2 | 0.1 | 없음 | `kl_raw`=`kl_loss`=0.4943 ✅ 일치 | 0.400 | 0.0040 |

초기 KL 급강하(step 50 → 250에서 24093 → 40.8)는 sudoku run들과 같은 궤적이며 발산 아님.
속도 ~1.9 it/s, 총 17,093 step, ETA 약 2시간 20분 → 아침엔 전부 완료돼 있을 것.

---

## 2. 변경/생성 파일

**수정** (전부 `cp → .bak_0721_1603` 백업 후 수정, `ast.parse` 통과):
- `models/recursive_reasoning/gram.py` — `_latent_diag()` 추가, inner forward 반환값에 `diag` 추가
- `models/losses.py` — `diag`를 metrics로 전달 (6줄)
- `pretrain.py` — `diag/*` 정규화 분기, eval wandb mirror, `wandb.save` 코드 스냅샷

**신규**: `experiments/eval_harness.py`, `experiments/sigma_probe.py`,
`config/arch/gram_nqueens8.yaml`, `gram_nq8_sweep.sh`,
`dataset/nqueens_verify.py`, `dataset/build_nqueens_dataset.py`,
`data/nqueens-8/`, `data/nqueens-10/`, `experiments/results/*.json`, `logs/*`

핵심 diff (gram.py):

```python
+    @torch.no_grad()
+    def _latent_diag(self, mu_p, logvar_p, mu_q, logvar_q, kl):
+        def _std_stats(logvar, tag):
+            std = (0.5 * logvar.detach().float()).exp() * self.sigma_scale
+            flat = std.reshape(-1)
+            sub = flat[:: max(1, flat.numel() // 1000000)]   # torch.quantile은 ~2**24 상한
+            q = torch.quantile(sub, torch.tensor([0.5, 0.95], device=sub.device))
+            ...
+        if kl is not None:
+            out["diag/kl_raw"] = kl.detach().float().sum(-1).mean()
+
+        # posterior 경로(=학습)에서만 계산 → eval은 완전히 무영향
+        diag = self._latent_diag(...) if use_posterior else None
-        return ..., kl, lrpm_logits
+        return ..., kl, lrpm_logits, diag
```

pretrain.py의 핵심 한 줄 (이게 없으면 σ 값이 망가진다):

```python
+            def _denom(k):
+                if k.startswith("diag/"):
+                    return world_size          # diag는 이미 batch 평균
+                return global_batch_size if k.endswith("loss") else count
```

`sigma_scale`, `use_posterior` kwarg는 **그대로 보존**했다 (하네스가 둘 다 사용 중).

---

## 3. 지시서에 없어서 내가 임의로 정한 것 (전부 명시)

1. **`diag/` 네임스페이스와 그 정규화 규칙.** 기존 metrics 정규화는 키가 `loss`로 끝나면
   `global_batch_size`, 아니면 `count`로 나눈다. σ 통계는 이미 평균이라 둘 다 틀리다.
   특히 **`count`는 학습 초기에 0이 될 수 있다**(드라이런 step 78에서 실제로 `train/count=0`이었다)
   — 그대로 뒀으면 σ가 쓰레기 값이 됐을 것. `diag/`는 `world_size`로만 나눈다.
2. **`token_accuracy`를 metrics dict에 넣지 않고 wandb 전용 mirror(`eval/*`)로 추가.**
   metrics에 넣으면 `[EVAL]`/`[step]` 출력 라인이 바뀌어 "stdout 포맷 유지" 요구와 충돌한다.
   기존 `accuracy`가 이미 token accuracy이므로 값은 동일한 별칭이다.
3. **`kl_raw`에 `kl_balance`(β=0.8)는 포함된 채로 기록.** 지시서의 "β 적용 전"을 `kl_coef` 미적용으로
   해석했다. `kl_balance`를 벗겨내려면 KL을 한 번 더 계산해야 하는데 "추가 forward 금지" 취지에
   어긋난다고 판단. `kl_balance`는 볼록결합이라 크기 스케일은 보존된다. gram.py 주석에 명시.
4. **σ 통계에 `sigma_scale`을 곱한 값(=실제 샘플링된 std)을 기록.** 학습 중엔 항상 1.0이라 무관하지만
   기존 `capture_sigma` 프로브와 정의를 맞췄다.
5. **eval harness의 `pass@K`를 unbiased 추정량으로 정의**하고 `pass_first_k`(단순 첫 K개)를 함께 보고.
   K=N에서는 둘이 동일하므로 앵커에는 영향 없다. e1의 `by_j` semantics와 일치.
6. **`verifier_selected@K`의 sudoku verifier는 label 비교가 아니라 진짜 sudoku 규칙 검사**
   (행/열/3×3 박스 + 단서 일치)로 구현. 유일해 태스크라 통과=정답이어야 하는데,
   실측 `verifier_precision = 100.00%`로 확인됐다(하네스 자체 검증이 된 셈).
7. **`experiments/sigma_probe.py` 신설** — Task 1 AC2의 σ를 로그에서 얻을 수 없어서(위 참조)
   체크포인트에서 측정하는 경로를 만들었다.
8. **eval_interval=200** (지시서 미지정). `epochs % eval_interval == 0` 제약을 만족하고
   3000 epoch에 15회 eval / 체크포인트가 남는다.
9. **`WANDB_MODE=offline`으로 학습 launch.** 이 머신에 wandb 자격증명이 없다
   (`~/.netrc` 없음, `WANDB_API_KEY` 미설정). 내가 사용자 계정으로 로그인할 수는 없으므로
   offline으로 돌렸다 — **기록은 전부 로컬에 남아 있고 나중에 sync하면 URL이 생긴다.** (§5 참조)
10. **단일 GPU × 3 run** (run당 `--nproc-per-node 1`). β sweep 3개를 GPU 3개에 하나씩 배치하는
    가장 단순한 해석이고, `global_batch_size=768`이 3090 한 장에 들어가는지 GPU 6에서 미리 확인했다(15.7GB).
11. lmean_v2의 "최종 checkpoint"는 `step_65100`(존재하는 최대 step, EMA 사본)으로 해석.

---

## 4. 발견한 이상 / 기존 코드 버그 의심

1. **🔴 최우선 — nq8 3 run 전부에서 posterior/prior 격차가 크다.**
   step 1139 기준 train `lm_loss` ≈ **0.003** (posterior 경로, label이 `y_emb`로 주입됨) 인데
   eval `lm_loss` ≈ **1.15** (prior 경로), eval `exact_accuracy` = **0.0000**.
   게다가 eval token accuracy가 **0.40**인데, **전부 empty로 찍기만 해도 56/64 = 87.5%**가 나온다
   → 현재 prior 경로는 자명한 베이스라인보다 나쁘다.
   전체 학습의 6.7% 지점이라 아직 단정할 수 없지만, 이것이 lmean_v2가 무너진 방식
   (posterior가 답을 다 들고 있고 prior가 따라오지 못함)과 같은 형태다. **아침 확인 1순위.**
2. **KL 항이 초기에 손실을 압도한다.** 드라이런 step 50에서 `kl_loss` = 28,708, `kl_coef=0.1`을
   곱하면 2,871인데 `lm_loss`는 2.72다 — 약 1,000배. 초기화 시 `var_p = exp(−4) ≈ 0.018`이라
   `(mu_q−mu_p)²/var_p`가 55배 증폭되고 D=512로 합해지기 때문. 수백 step 안에 급강하하고
   (24093 → 40.8) 성공한 sudoku run들도 동일했으므로 **기존 동작이지 내 변경 때문이 아니다**.
   다만 β sweep 해석 시 "초기 수백 step은 사실상 KL만 최적화된다"는 점을 감안해야 한다.
3. **`logs/gram_sudoku_halt_only.log`의 마지막 `[EVAL]`이 step 31248(exact 0.1533)에서 끊긴다** —
   체크포인트는 65100까지 있고 그 checkpoint의 실측 성능은 70.4%다. 로그와 체크포인트의
   진행 구간이 안 맞는다(재시작/`tee -a` 이력 추정). 이번 작업 결론에는 영향 없지만
   그 로그로 학습 곡선을 그릴 때 주의.
4. **`losses.py`의 `lrpm` / `lprm` 철자 혼용.** `outputs` 키는 일관되게 `lrpm_logits`이고
   손실 변수만 `lprm_loss`라서 **현재는 정상 동작한다.** 다만 한 글자만 어긋나도 LPRM 손실이
   조용히 0이 되는 구조라 위험하다. 이번엔 건드리지 않았다(동작 중인 코드).
5. `expandable_segments not supported on this platform` 경고가 뜬다 — 무해하지만
   `PYTORCH_CUDA_ALLOC_CONF` 설정이 실제로는 적용되지 않고 있다.

---

## 5. 실행 중인 tmux 세션 / 아침에 확인할 것

```
tmux attach -t nq8_b007     # GPU 0, β=0.07,  logs/gram_nq8_b007.log
tmux attach -t nq8_b0045    # GPU 1, β=0.045, logs/gram_nq8_b0045.log
tmux attach -t nq8_b010     # GPU 2, β=0.1,   logs/gram_nq8_b010.log
```
(GPU 3은 아침 eval용으로 비어 있음. GPU 4~7도 전부 유휴.)

**wandb URL — 아직 없다.** 자격증명이 없어 offline으로 기록했다. 로그인 후 sync하면 URL이 생긴다:

```bash
wandb login                       # 사용자 본인이 실행해야 함
wandb sync wandb/offline-run-20260721_161930-6mc6q9b9   # gram_nq8_b007
wandb sync wandb/offline-run-20260721_161930-ddrungec   # gram_nq8_b0045
wandb sync wandb/offline-run-20260721_161930-ak9sahul   # gram_nq8_b010
```

### 아침 체크리스트

1. **(최우선) §4-1이 해소됐는지.** 각 run의 `[EVAL]` 추이에서 `exact_accuracy`가 0에서 올라왔는지,
   token accuracy가 최소한 0.875(자명 베이스라인)를 넘겼는지:
   ```bash
   for r in gram_nq8_b007 gram_nq8_b0045 gram_nq8_b010; do
     echo "== $r"; tr '\r' '\n' < logs/$r.log | grep -a "\[EVAL" | tail -5; done
   ```
   여전히 0이면 β sweep 자체가 아니라 posterior 누수/prior 미학습이 병목이라는 뜻이다.
2. 3 run 정상 종료 여부 (`tmux ls`로 세션이 사라졌으면 완료) 및 `train/diag/sigma_*` 추이 —
   lmean_v2처럼 σ가 1.5 근처로 발산했는지, halt_only의 0.95 근처에 머무는지.
3. 최종 체크포인트에 하네스 돌리기 (sidecar coverage@K 포함, GPU 3 사용):
   ```bash
   CUDA_VISIBLE_DEVICES=3 python experiments/eval_harness.py \
     --ckpt-dir checkpoints/Nqueens-8-ACT-torch/gram_nq8_b007 --step <최종step> \
     --task nqueens --nq-n 8 --sidecar data/nqueens-8/sidecar_8.json \
     --num-puzzles 772 --k 20 --ks 1,5,10,20
   ```
   (test split이 772행뿐이므로 `--num-puzzles 772`가 전수다.)
4. Task 3의 **파라미터 16.26M vs 논문 10M** — 노이즈 head 4개(9.44M)를 Linear로 줄일지 결정 필요.
5. lmean_v2 수치(§Task 1 AC2)에 대한 판정 — 지시대로 해석은 남겨두었다.
