# [B] 결과 — puzzle_emb 는 EMA 에 **한 번도 반영되지 않았다**. 버그 발현 확정, 단 붕괴 시점의 원인은 아니다

실행: 2026-08-06, 62번(jurgen)
대상: 셀 P `nq10_A3p_ii_a7525_lr3e4_a2_05_s0`
상태: **확정**. 코드 레벨 + 체크포인트 실측 이중 확인.

## 한 줄

> **`puzzle_emb.weights` 는 EMA 사본과 live 사본이 bit-identical(상대차 0.000000) 이다.
> 원인은 `nn.Buffer` 라 `named_parameters()` 에 안 잡히는 것 — 구조적이고 전 구간 상시다.
> 그래서 "이것이 121k 붕괴를 일으켰다" 고는 말할 수 없다. EMA 가 멀쩡했던 92k 에서도 똑같이 0 이다.**

## ★ 먼저 — CHECKLIST §2.6 은 이 repo 에 없다

지시서가 참조한 "CHECKLIST §2.6 puzzle_emb 누락 버그" 는 **로컬 `CHECKLIST.md` 에 존재하지 않는다.**
절 구성은 §1–§9 이고 §2 는 "yaml 파일 끝 개행" 이다. `puzzle_emb` 문자열도 0건이다.
원격 `origin/main` (3커밋 앞)에도 없다.

→ 판정은 §2.6 인용 없이 **코드와 체크포인트만으로** 수행했다. 결론은 그대로 성립한다.

## 로그 / 경로

| 항목 | 경로 |
|---|---|
| EMA 사본 | `checkpoints/Nqueens-10-ACT-torch/nq10_A3p_ii_a7525_lr3e4_a2_05_s0/step_138534` |
| live 사본 | 같은 디렉토리 `step_138534.live` |
| 대조군 (EMA 정상 구간) | `step_92356` / `step_92356.live` |
| 진단 스크립트 | `scripts/diag_ema_vs_live.py` |

```bash
CK=checkpoints/Nqueens-10-ACT-torch/nq10_A3p_ii_a7525_lr3e4_a2_05_s0
python3 scripts/diag_ema_vs_live.py $CK/step_138534 $CK/step_138534.live --top 5
```

지표는 지시된 식 그대로 `mean(|ema-live| / (|live|+1e-8))`.
이 식은 `|live|` 가 0 근처인 원소에서 발산하므로(예: `q_head.bias` 369.45),
해석용 보조로 **노름비 `‖ema-live‖ / ‖live‖`** 를 같이 낸다.

---

## 1. 메커니즘 — 코드만으로 이미 확정된다

`models/sparse_embedding.py:19-21`:
```python
self.weights = nn.Buffer(
    trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
)
```
`puzzle_emb.weights` 는 **`nn.Parameter` 가 아니라 `nn.Buffer`** 다.

`models/ema.py:13-22`:
```python
def register(self, module):
    for name, param in module.named_parameters():      # <- Buffer 는 여기 안 잡힌다
        if param.requires_grad:
            self.shadow[name] = param.data.clone()

def update(self, module):
    for name, param in module.named_parameters():      # <- 동일
        ...
```

`named_parameters()` 는 버퍼를 내놓지 않는다. 따라서
`shadow` 에 `puzzle_emb.weights` 항목이 **생성된 적이 없고**, `ema()` 도 그 자리를 건드리지 않는다.
`ema_copy` 는 `deepcopy(live) → ema()` 순서이므로, 결과 EMA 모델의 `puzzle_emb.weights` 는
**deepcopy 로 복사된 live 값 그대로** 남는다.

→ EMA 모델 = (몸통은 ~10k step 지연된 EMA) + (puzzle_emb 는 **현재 시점 live**) 의 키메라.

**단, 이 경로는 학습 첫 step 부터 항상 참이다.** 특정 시점에 "발현" 하는 종류가 아니다.

---

## 2. 실측 — step_138534 (EMA 사망, `nqueens/accuracy`=0.0000)

### ★ 지정 항목

| 상대차(지정식) | 노름비 ‖Δ‖/‖live‖ | bit-identical | shape | 파라미터 |
|---:|---:|:---:|---|---|
| **0.000000** | **0.000000** | **예** | (1, 512) | `_orig_mod.model.inner.puzzle_emb.weights` |
| 4.564663 | 0.637804 | 아니오 | (3072, 512) | `_orig_mod.model.inner.L_level.layers.0.mlp.gate_up_proj.weight` |
| 0.000000 | 0.000000 | 예 | (3072, 512) | `…layers.0.mlp.gate_up_proj.mask` (상수 버퍼, 아래 참조) |

정상 비교군 `L_level.layers.0.mlp.gate_up_proj.weight` 는 **4.564663 / 노름비 0.638** —
EMA 와 live 가 정상적으로 벌어져 있다. 대조가 성립한다.

### 상위 5 (EMA 가 가장 많이 벌어진 쪽)

| 상대차 | 노름비 | 파라미터 |
|---:|---:|---|
| 369.452410 | 1.043093 | `q_head.bias` (numel=2, 분모 발산 사례) |
| 5.854789 | 0.651910 | `prior_logvar.down_proj.weight` |
| 5.525305 | 0.612037 | `post_logvar.gate_up_proj.weight` |
| 4.793381 | 0.685269 | `L_level.layers.0.mlp.down_proj.weight` |
| 4.640641 | 0.685269 | `L_level.layers.1.self_attn.o_proj.weight` |

### 하위 5 (EMA 가 안 움직인 쪽)

전부 상대차 0.000000 / bit-identical. 내역은 다음 절.

---

## 3. ★ 0 인 텐서 11/37 의 내역 — 의미 있는 것은 하나뿐이다

| 분류 | 개수 | 텐서 | 0 이 정상인가 |
|---|---:|---|---|
| dropout mask 버퍼 | 8 | `layers.{0,1}.{mlp.down_proj, mlp.gate_up_proj, self_attn.o_proj, self_attn.qkv_proj}.mask` | **정상.** 학습 대상이 아닌 상수 버퍼 (`weight_dropout=0.0`) |
| 초기 은닉 상수 | 2 | `H_init`, `L_init` | **정상.** 버퍼, 학습되지 않음 |
| **puzzle embedding** | **1** | **`puzzle_emb.weights`** | **★ 비정상.** 학습되는 값인데 EMA 만 못 받았다 |

**즉 "학습되는 텐서 중 상대차 0" 은 `puzzle_emb.weights` 단 하나다.**
나머지 26개 학습 파라미터는 상대차 0.172541 ~ 369.452410 (중앙값 3.221009) 로 전부 정상 범위.

→ 지시서 판정 기준 **"puzzle_emb 만 0 이고 나머지가 정상 범위"** 에 정확히 해당.
→ **버그 발현 확정. 전 파라미터 이상(별개 병리) 은 아니다.**

---

## 4. ★ 대조군 — EMA 가 멀쩡했던 step_92356 에서도 똑같이 0 이다

| 체크포인트 | EMA `nqueens/accuracy` | `puzzle_emb.weights` 상대차 | bit-identical | 0 인 텐서 수 |
|---|---:|---:|:---:|---:|
| step_92356 | **0.6863** (전 구간 정점) | **0.000000** | 예 | 11 / 37 |
| step_138534 | **0.0000** (사망) | **0.000000** | 예 | 11 / 37 |

**이것이 이번 판정의 핵심 제약이다.**

버그는 92k(정상)와 138k(사망)에서 **완전히 동일하게** 존재한다.
따라서 이 버그는 **붕괴의 상시 배경 조건**이지, **121k 라는 시점을 만든 원인이 아니다.**
"puzzle_emb 누락이 EMA 를 죽였다" 는 단일 원인 서술은 이 대조로 **기각된다.**

---

## 5. puzzle_emb 궤적 — 배경 조건이 언제 위험해지는가

EMA 몸통은 `ema_rate=0.9999` 로 유효 지연 ~10,000 step 이다.
puzzle_emb 만 현재값이므로, **puzzle_emb 이 크게 움직이는 구간에서 키메라 불일치가 커진다.**

| step | ‖puzzle_emb‖ | 직전 대비 상대이동 | cos(직전) | EMA `nqueens/acc` |
|---:|---:|---:|---:|---:|
| 88,158 | 0.4432 | 0.6165 | 0.832873 | 0.4050 |
| **92,356** | 0.4800 | 0.5304 | 0.873339 | **0.6863** |
| **96,554** | **0.9559** ← 2.0배 | **1.3726** | 0.773910 | **0.3044** |
| 100,752 | 0.7527 | 0.3389 | 0.955769 | 0.2151 |
| 113,346 | 0.5733 | 0.3776 | 0.946693 | 0.0750 |
| 117,544 | 0.4462 | 0.5029 | 0.869060 | 0.0475 |
| 121,742 | 0.5843 | 0.8343 | 0.770792 | **0.0005** |
| **125,940** | **1.3258** ← 2.3배 | **1.5223** | 0.844228 | 0.0000 |
| 138,534 | 0.7275 | 0.3790 | 0.928507 | 0.0000 |

두 번의 노름 급등(96,554 · 125,940)이 각각 **EMA 정점 직후 급락**, **live 국면 전환**과 겹친다.
다만 상대이동은 전 구간 0.33–1.52 로 **항상 크다**. 96k/125k 만 특별하다고 볼 근거는 약하다.
→ **상관은 있으나 인과 미확정.**

## 6. [A] 와의 접합

- [A]: z_H 폭주는 붕괴 **이후**(120k+). 붕괴 구간 자체에서 z_H 는 최저.
- [B]: puzzle_emb 누락은 **상시**. 붕괴 시점을 설명 못 함.

**두 후보가 모두 "시점" 을 설명하지 못한다.** 121k 에서 무엇이 바뀌었는지는 아직 미해결이다.
같은 시점에 live `nqueens/accuracy_raw` 도 0.0044 로 급락했다가 회복했다([A] §5) —
**live 궤적의 국면 전환이 실제로 있었고, EMA 만 못 넘겼다.**
다음 축은 z_H 나 puzzle_emb 이 아니라 **121k 부근 live 궤적 자체**다.

## 7. 수정 방향 (★ 62번 승인 필요, 미착수)

`EMAHelper` 가 버퍼도 shadow 에 넣도록 고치면 해결된다. 그러나:

- ★ **진행 중인 run 의 재현성이 깨진다.** `reports/TODO_audit_62.md` 의 "처리 금지" 방침과 충돌.
- ★ 기존 체크포인트 해석이 바뀐다 (지금까지의 모든 EMA 곡선이 이 키메라 기준이었다).
- 고칠 거면 `fp_layers.py` 분리와 같은 패턴 — **새 클래스명 + 새 파일**로 분기해야 한다.

**논문 마감 전에는 고치지 말고, 대신 "EMA 곡선은 puzzle_emb 만 live 인 키메라" 라는
단서를 모든 EMA 보고에 병기하는 것**을 권고한다.

## 8. 확인 못 함

- **`puzzle_emb` 를 EMA 에 포함시켰을 때 붕괴가 사라지는가**: 코드 수정이 필요해 미실행.
- **121k 국면 전환의 정체**: 본 [B] 범위 밖.
- **CHECKLIST §2.6 원문**: 로컬·원격 모두 존재하지 않음.
