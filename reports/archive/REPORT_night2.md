# 긴급 진단 보고 — N-Queens KL reduction

작업: 2026-07-21 16:50 ~ 23:10 KST
**결론 요약**: 지적은 **맞았다**. reduction은 L-MEAN으로 돌고 있었다(= lmean_v2와 동일 사망 경로).
분기 A로 가서 3 run kill → 명시 키로 수정 → v2 재launch. **v2 3개 모두 완주(17,093 step)**했고
100 step 검증 3종을 통과했다. 다만 **reduction을 고쳐도 N-Queens는 여전히 실패**한다.
실패 양상이 "베이스라인 이하"에서 "자명해 고착 + posterior collapse"로 바뀌었다.

---

## 1. Step 1 — 진단 증거와 판정

### 1.1 config에 reduction 키가 없었다
`config/arch/gram_nqueens8.yaml`의 KL 관련 키는 `kl_coef`, `kl_balance` **둘뿐**. reduction 키 없음.

### 1.2 `getattr` 전수 — 예상하신 메커니즘과는 달랐다
`models/losses.py`의 `getattr`는 **정확히 2개**:
- `losses.py:132` `kl_coef = getattr(self.model.config, "kl_coef", 0.0)`
- `losses.py:133` `lprm_coef = getattr(self.model.config, "lprm_coef", 1.0)`

**reduction 관련 `getattr`는 존재하지 않았다.** 즉 "getattr 기본값 fallback으로 L-mean"이 아니라,
**losses.py에서 두 줄 중 하나를 주석 처리해 고르는 구조**였다 (수정 전 `losses.py:94-95`):

```python
# kl_loss = outputs["kl"].sum(-1).sum(-1).sum()  # D는 합, L은 평균(x) L을 합으로 (o), batch는 합
kl_loss = outputs["kl"].sum(-1).mean(-1).sum()   # <- 활성. L-MEAN
```

결과적 증상(=config만 봐서는 어느 reduction인지 알 수 없음)은 지적하신 그대로지만
원인은 config fallback이 아니라 **소스 주석 토글**이다. 이 차이 때문에 "yaml에 키를 넣는다"만으로는
부족하고 코드 쪽도 같이 고쳐야 했다.

### 1.3 as-run 스냅샷 대조 — 결정적 증거
체크포인트 디렉터리에 각 run의 as-run `losses.py`가 보존돼 있어 직접 대조했다:

| run | 활성 라인 | reduction | 결과 |
|---|---|---|---|
| `gram_sudoku_halt_only` | `.sum(-1).sum(-1).sum()` | **L-SUM** | single 70.4% / majority@20 89.2% |
| `gram_sudoku_lsum` | `.sum(-1).sum(-1).sum()` | L-SUM | — |
| `gram_sudoku_lmean_v2` | `.sum(-1).mean(-1).sum()` | **L-MEAN** | single 1.8% (붕괴) |
| **수정 전 작업트리 (= nq8 v1 3 run)** | `.sum(-1).mean(-1).sum()` | **L-MEAN** | 붕괴 run과 동일 |

### 1.4 `kl_raw`/`kl_loss` 로깅 중복 여부
중복 아님. 서로 다른 두 지점에서 서로 다른 식으로 계산된다
(`gram.py:266` `kl.sum(-1).mean()` / `losses.py:111` reduction 적용값).
**L-MEAN일 때만 수학적으로 동일해진다.** 따라서 `kl_raw == kl_loss`는 확인이 아니라
**L-MEAN 양성 검출기**다 — 어제 보고에서 이걸 "reduction 정합 확인"이라고 쓴 것은 명백한 오독이다.

### 1.5 결정적 런타임 테스트 → **판정: L-MEAN**
`experiments/test_kl_reduction.py` (GPU 3). 실제 forward에서 `return_keys=["kl"]`로 원본 kl 텐서를
꺼내 loss head가 반환한 값과 수동 L-SUM/L-MEAN을 대조:

```
arch=gram_nqueens8  kl tensor (8, 80, 512)  L=80 (= seq_len 64 + puzzle_emb_len 16)
  head-reported kl_loss/B = 26211.146484
  manual  L-SUM   /B      = 2096891.750000   ratio to reported = 80.0000
  manual  L-MEAN  /B      = 26211.146484   ratio to reported = 1.0000
  diag/kl_raw             = 26211.146484
  VERDICT: active reduction is L-MEAN
  kl_raw == kl_loss ? True   (True <=> L-MEAN)
```
→ `experiments/results/kl_reduction_before.json`

---

## 2. Step 2 — 분기 A (L-mean 확정) 실행 내역

### 2.1 3 run kill
```
gram_nq8_b007    killed at step 4444/17093
gram_nq8_b0045   killed at step 4392/17093
gram_nq8_b010    killed at step 4013/17093
```

### 2.2 명시 키로 수정 (getattr 기본값 의존 제거)
백업: `*.bak_0721_1659`. `ast.parse` 통과.

`models/recursive_reasoning/gram.py` — pydantic config에 필드 추가:
```python
+    kl_reduction: str = "sum"   # "sum" | "mean"
```

`models/losses.py` — 주석 토글 제거, 명시 키 + **실패 시 예외**:
```python
         if outputs.get("kl") is not None:
             kl_reduction = getattr(self.model.config, "kl_reduction", None)
             if kl_reduction not in ("sum", "mean"):
                 raise ValueError(
                     f"arch config must set kl_reduction to 'sum' or 'mean', got {kl_reduction!r}. ...")
             kl_per_pos = outputs["kl"].sum(-1)                  # [B, L] -- D는 합
             kl_loss = (kl_per_pos.sum(-1) if kl_reduction == "sum"
                        else kl_per_pos.mean(-1)).sum()          # L reduction -> batch 합
```
`config/arch/gram_nqueens8.yaml`, `config/arch/gram.yaml` 양쪽에 `kl_reduction: sum` 명시.
launch 스크립트에도 `arch.kl_reduction=sum`을 CLI로 한 번 더 박아 로그에 남게 했다.

**수정 후 재검증** (`experiments/results/kl_reduction_after.json`):
```
  head-reported kl_loss/B = 2382356.0   manual L-SUM /B = 2382356.0  ratio 1.0000
  VERDICT: active reduction is L-SUM      kl_raw == kl_loss ? False
```
제어 실험으로 키를 `mean`으로 바꾸면 다시 L-MEAN 판정이 나오는 것도 확인(키가 실제로 경로를 지배함).

**β 정합 확인**: `halt_only`의 as-run config는 `kl_coef=0.1` + L-SUM. 지시서의 sweep
{0.045, 0.07, 0.1}은 L-SUM 기준 값으로 정합하므로 β는 조정하지 않았다.

### 2.3 재launch 커맨드 전문
```bash
tmux new-session -d -s nq8_b007_v2  "bash gram_nq8_sweep.sh 0 0.07  gram_nq8_b007_v2  29521"
tmux new-session -d -s nq8_b0045_v2 "bash gram_nq8_sweep.sh 1 0.045 gram_nq8_b0045_v2 29522"
tmux new-session -d -s nq8_b010_v2  "bash gram_nq8_sweep.sh 2 0.1   gram_nq8_b010_v2  29523"
```
`gram_nq8_sweep.sh`가 실행하는 실체:
```bash
CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc-per-node 1 --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:${port} --nnodes=1 pretrain.py \
  arch=gram_nqueens8 data_paths="[data/nqueens-8]" evaluators="[]" \
  global_batch_size=768 epochs=3000 eval_interval=200 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.kl_coef=${beta} arch.kl_reduction=sum \
  +run_name=${run_name} ema=True ema_rate=0.9999 \
  2>&1 | tee logs/${run_name}.log
```

---

## 3. 재launch 100 step 검증 3종 — **전부 통과**

| 검증 | 기준 | 실측 | 판정 |
|---|---|---|---|
| (i) `kl_loss ≠ kl_raw`, 배수 ≈ 시퀀스 위치 수 | ~80 | **80.00** (3 run 전부) | ✅ |
| (ii) NaN 없음 | — | nan/inf/traceback 매치 0건 (3 run) | ✅ |
| (iii) eval 후 eval kl_loss = 0.0 (prior 경로) | 0.0 | `all/kl_loss=0.0000` (전 eval) | ✅ |

step 100 손실: `kl_loss` ≈ 1.03~1.06e6 (L-mean 때 1.3e4의 약 80배), `lm_loss` 0.94~1.00, NaN 없음.

**세 run 모두 17,093 step 완주** (17:01 시작 → 19:32 종료, 약 2h31m). tmux 세션은 정상 종료로 사라졌다.

---

## 4. 결과 — reduction을 고쳐도 실패한다

### 4.1 in-training eval (최종 step 17085)

| run | β | token acc | exact acc |
|---|---|---|---|
| `gram_nq8_b007_v2` | 0.07 | **0.8750** | 0.0000 |
| `gram_nq8_b010_v2` | 0.1 | **0.8750** | 0.0000 |
| `gram_nq8_b0045_v2` | 0.045 | 0.8922 (최고 0.9102) | 0.0026 |

**0.8750 = 56/64**, 즉 **퀸을 한 개도 놓지 않고 전부 empty로 찍는 자명해**와 정확히 일치한다.

### 4.2 strict verifier + coverage@20 (test 772문항 전수, N=20, σ=1.0)

| run | β | single | pass@20 | coverage@20 | **distinct@20** |
|---|---|---|---|---|---|
| `b0045_v2` | 0.045 | 0.65% | **3.50%** | 2.67% | **12.11** |
| `b007_v2` | 0.07 | 0.00% | 0.00% | 0.00% | **1.00** |
| `b010_v2` | 0.1 | 0.00% | 0.00% | 0.00% | **1.00** |

→ `experiments/results/nq8_{b0045,b007,b010}_v2_eval.json`

**`distinct@20 = 1.00`이 결정적이다**: b007/b010은 20번 독립 샘플링해도 **완전히 같은 보드**를 내놓는다.
GRAM의 확률적 latent이 출력에 아무 영향을 못 준다는 뜻이고, 그러면 majority@K·pass@K 같은
test-time scaling 지표 자체가 의미를 잃는다. verifier precision은 통과 사례가 있는 b0045에서 100%로,
하네스 쪽 문제는 아니다.

### 4.3 원인: posterior collapse (최종 diag 수치)

| run | `kl_raw` | `sigma_prior` | `sigma_post` | `mu_p_norm` | `mu_q_norm` | train `lm_loss` |
|---|---|---|---|---|---|---|
| b0045_v2 | 0.0007 | 0.6685 | 0.6684 | 0.311 | 0.311 | 0.0022 |
| b007_v2 | 0.0022 | 0.6613 | 0.6612 | 0.122 | 0.132 | 0.2300 |
| b010_v2 | 0.0003 | 0.6608 | 0.6607 | 0.172 | 0.174 | 0.2269 |

KL이 0으로 내려앉고 prior와 posterior가 소수점 3~4자리까지 일치한다 = **완전한 posterior collapse**.
Task 0에서 넣은 σ 로깅이 없었으면 이 진단은 불가능했다.

**β와 붕괴 정도가 단조 관계다**: β가 낮을수록(0.045) latent이 덜 죽고(distinct 12.11) 유일하게
정답을 낸다. β가 높을수록(0.07, 0.1) 완전 붕괴(distinct 1.00). **sweep이 준 유일한 신호는
"β를 더 내려야 한다"는 방향**이며, 주어진 {0.045, 0.07, 0.1} 범위는 전부 너무 높았을 가능성이 크다.

**추가 가설(미검증)**: 64칸 중 퀸이 8칸뿐인 7:1 클래스 불균형에서 token-level CE는
"전부 empty"에 강한 국소최소를 갖는다. b007/b010의 `q_halt_loss`가 17까지 발산한 것도 같은 그림이다.
이건 가설이며, 재launch 권한(1회)을 이미 썼으므로 검증용 추가 run은 돌리지 않았다.

---

## 5. 임의로 정한 것 (전부 명시)

1. **`kl_reduction`의 pydantic 기본값을 `"sum"`으로 뒀다.** 지시는 "getattr 기본값 의존 금지"였고
   기본값 없는 필수 필드가 가장 강하지만, 그러면 **기존 체크포인트 10개의 `all_config.yaml`이
   전부 로드 불가**가 되어 회귀 앵커와 lmean_v2 eval 재현이 깨진다. 타협으로 (a) 기본값을
   성공 run의 값(`sum`)으로 두고 (b) 값이 `sum`/`mean`이 아니면 **예외를 던지고**
   (c) yaml·CLI 양쪽에 명시했다. 새 run이 조용히 잘못된 reduction으로 도는 경로는 없앴다.
2. **`config/arch/gram.yaml`에도 `kl_reduction: sum`을 넣었다** (nqueens yaml만이 아니라).
   이 파일을 쓰는 기존 스크립트가 작업트리의 L-mean을 물려받는 걸 막기 위해서다.
   `gram_sudoku_lmean_v2.sh`를 재현하려면 이제 `arch.kl_reduction=mean`을 명시해야 한다.
3. **β 값은 그대로 뒀다** (0.045/0.07/0.1). halt_only가 L-SUM+0.1이라 sweep이 정합한다고 봤다.
4. **v1 run들의 로그·체크포인트를 지우지 않았다.** `logs/gram_nq8_b*.log`(v2 아님)와
   `checkpoints/Nqueens-8-ACT-torch/gram_nq8_b*`가 L-mean 대조군으로 남아 있다.
5. **완주 후 하네스 eval을 3 run 전부에 직접 돌렸다** (분기 B의 지시였지만, 결과가 나와 있는데
   아침까지 미룰 이유가 없다고 판단). GPU 3 사용.
6. `test_kl_reduction.py`에서 `DISABLE_COMPILE=1` 사용 — 진단 속도 목적이며 수치에는 무관.

---

## 6. 아침에 볼 것

**남은 ETA: 없음.** 실행 중인 학습이 없다. GPU 0~7 전부 유휴, tmux에 `trm` 세션만 남아 있다.

1. **결정 필요 — β를 더 내릴지.** §4.3의 단조 관계가 유일한 실측 신호다.
   β ∈ {0.01, 0.02, 0.03} 재sweep이 자연스러운 다음 수인데, 이건 지시 범위를 벗어나므로
   내가 임의로 돌리지 않았다.
2. **결정 필요 — 클래스 불균형 가설 검증 여부** (§4.3). 자명해 고착이 β와 무관한 병목이라면
   β sweep을 더 돌려도 소용없다. 확인 최소 비용은 b0045_v2 체크포인트의 예측 분포에서
   퀸 토큰 비율을 세어 보는 것.
3. 수치 재확인용 커맨드:
   ```bash
   cd /workspace/TinyRecursiveModels
   cat experiments/results/nq8_b0045_v2_eval.json          # 최고 성능 run
   cat experiments/results/kl_reduction_{before,after}.json # 진단 증거
   for r in gram_nq8_b007_v2 gram_nq8_b0045_v2 gram_nq8_b010_v2; do
     tr '\r' '\n' < logs/$r.log | grep -a "\[EVAL" | tail -3; done
   ```
4. **wandb URL은 여전히 없다** (자격증명 부재). 로그인 후 sync:
   ```bash
   wandb login
   wandb sync wandb/offline-run-20260721_170107-rpuio1y0   # gram_nq8_b007_v2
   wandb sync wandb/offline-run-20260721_170107-iza6t0a4   # gram_nq8_b0045_v2
   wandb sync wandb/offline-run-20260721_170107-twheigro   # gram_nq8_b010_v2
   ```
5. 어제 보고(`REPORT.md`)의 Task 0~3 결과는 그대로 유효하다. 단 §1-Task0의
   "kl_raw = kl_loss 일치 → reduction 정합 확인"은 **틀렸으므로** 본 문서 §1.4로 대체해 읽어야 한다.
