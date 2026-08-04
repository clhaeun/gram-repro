# REPORT_ema_collapse.md — EMA shadow 붕괴 원인 규명

- 일시: 2026-08-04
- 대상: 같은 step 에서 `EMA strict_accuracy = 0.0000` / `live 정상` 이 반복 관측된 건
  (nq10 A3′ `rs_s0` 5 eval, 구데이터 `s1` 8 eval, nq8 A3′ 2회, 50번 nq10 `catnd_s0`)
- 결론: **원인 2건 확정. FPRM 무관, repo 전역 문제.**

---

## 0. 결론

**원인 ① (구조적 결함) — `puzzle_emb` 가 EMA 대상에서 통째로 빠져 있다.**
`models/sparse_embedding.py:18` 이 `self.weights` 를 `nn.Parameter` 가 아니라 **`nn.Buffer`**
로 선언한다. `EMAHelper.register()`/`update()` 는 `named_parameters()` 를 순회하며
`param.requires_grad` 로 거른다(`models/ema.py:33-34`, `:42-44`) → **buffer 는 절대 등록되지
않는다.** 그런데 `puzzle_emb.weights` 는 전용 옵티마이저
`CastedSparseEmbeddingSignSGD_Distributed` 로 **정상 학습된다.**

`ema_copy()` 는 `copy.deepcopy(module)` 후 파라미터만 shadow 로 덮으므로
(`models/ema.py:70-74`), 결과 모델은:

> **EMA 모델 = (약 10,000 step 평균된 transformer 가중치) + (평균 안 된 순간값 puzzle_emb)**

**원인 ② (관측 메커니즘) — 그 mismatch 가 다수클래스 붕괴로 나타난다.**
EMA 모델이 "빈 칸" 만 예측하는 퇴화해로 떨어진다. 10×10 보드는 90%가 빈 칸이라
**token accuracy 는 0.92 로 멀쩡해 보이고 `lm_loss` 는 오히려 낮아진다.**
`nqueens/strict_accuracy` 와 `queen_count_mean` 만이 이를 잡아낸다.

---

## 1. 원인 ① 실증 — checkpoint 직접 대조

`step_138534`(EMA) vs `step_138534.live` 의 텐서 37개 전수 비교:

**EMA 와 live 가 완전히 동일(= EMA 적용 안 됨)한 텐서 13개** 중 핵심:

| 텐서 | shape | 판정 |
|---|---|---|
| **`model.inner.puzzle_emb.weights`** | (1, 512) | ⚠️ **결함 — 학습되는데 EMA 안 됨** |
| `model.inner.H_init`, `L_init` | (512,) | 정상 (`gram.py:184-185` `nn.Buffer`, 미학습) |
| `L_level.alpha_1_param`, `alpha_2_param` | () | 정상 (`residual_scale=fixed` → `requires_grad=False`) |
| `*.mask` 8건 | — | 정상 (미학습) |

**EMA 가 적용된 텐서 24개** 중 상위:

| 텐서 | EMA vs live 상대차 |
|---|---|
| `q_head.weight` | 0.5109 |
| `L_level.layers.0.mlp.gate_up_proj.weight` | 0.4229 |
| `L_level.layers.0.mlp.down_proj.weight` | 0.3889 |
| `embed_tokens.embedding_weight` | 0.3554 |

**최종 step 정량:**

| | EMA vs live 상대차 |
|---|---|
| `puzzle_emb.weights` | **0.000000** ← 완전 동일 = EMA 전혀 안 됨 |
| `L_level.layers.0.mlp.gate_up_proj.weight` | **0.422878** ← 강하게 평균됨 |

### mismatch 심각도 — puzzle_emb 는 빠르게 움직인다

연속 checkpoint(`.live`, 간격 4,198 step) 간 상대변화:

| 구간 | `puzzle_emb` | transformer W |
|---|---|---|
| 125,940 → 130,138 | 0.3174 | 0.4274 |
| 130,138 → 134,336 | 0.3142 | 0.4578 |
| 134,336 → 138,534 | 0.3547 | 0.4313 |

`mu=0.9999` → EMA 유효 평균 창 `1/(1-mu)` = **약 10,000 step**.
그 창에서 `puzzle_emb` 는 최소 75% 이상 바뀐다. **10,000 step 평균된 가중치에
순간값 embedding 을 짝지우는 것**이 EMA 붕괴의 구조적 원인이다.

N-Queens 는 `num_puzzle_identifiers = 1` 이라 이 (1,512) 벡터를 **전 예제가 공유**한다.

### ★ 메커니즘 정밀화 (초판 과장 정정)

초판에 "어긋나면 전체 입력 표현이 통째로 어긋난다" 고 썼는데 **과장이다. 정정한다.**
puzzle_emb 는 116개 위치 중 **position 0 하나**만 차지한다 (§7 참조).

정확한 논리는 이렇다. 같은 step 의 EMA checkpoint 와 live checkpoint 는
**puzzle_emb 가 서로 동일**하다(상대차 0.000000). 즉 둘의 차이는 전적으로
transformer 가중치에 있다. 문제는:

> `puzzle_emb` 는 평균에 **참여하지 않으므로**, 언제나 **순간(live) 가중치에만
> 공적응(co-adapt)** 되어 있다. 거기에 10,000-step 평균 가중치를 끼워 넣으면
> 그 짝이 깨진다.

즉 "두 모델의 puzzle_emb 가 다르다" 가 아니라 **"puzzle_emb 가 둘 중 하나에만
맞춰질 수 있다"** 가 결함의 본질이다. 학습되는 텐서가 EMA 대상에서 빠지면
언제나 이 문제가 생긴다.

---

## 2. 원인 ② 실증 — 붕괴 구간 전체 지표

`nq10_fp_a3p_rs_s0`, EMA / live 병기:

| step | `all/accuracy` | `all/lm_loss` | **`queen_count_mean`** | **`strict_accuracy`** |
|---|---|---|---|---|
| 58,772 EMA | 0.9006 | 0.2982 | 10.3342 | 0.4090 |
| 58,772 live | 0.9037 | 0.4245 | 9.7561 | 0.5458 |
| **67,168 EMA** | 0.8799 | 0.2881 | 9.6508 | **0.0000** |
| 67,168 live | 0.9040 | 0.3429 | 9.3410 | 0.3742 |
| **79,762 EMA** | **0.9220** | **0.2285** | **3.4692** | **0.0000** |
| 79,762 live | 0.9016 | 0.4719 | 9.8623 | 0.5086 |
| **83,960 EMA** | **0.9212** | **0.2246** | **3.6968** | **0.0000** |
| 83,960 live | 0.9129 | 0.4125 | 9.4877 | **0.6899** |

### 읽는 법 — ★ 표준 지표가 오히려 좋아진다

step 83,960 에서 EMA 모델은:

- `queen_count_mean = 3.6968` — 10개를 놔야 하는데 **3.7개만 놓는다** (퇴화)
- `all/accuracy = 0.9212` — 10×10 보드는 100칸 중 90칸이 빈 칸이다.
  **"전부 빈 칸" 자명해의 token accuracy 가 곧 0.90** 이다. 0.9212 는 그 자명해 수준이다.
- `all/lm_loss = 0.2246` — live 의 0.4125 보다 **낮다**. 다수클래스를 확신 있게
  예측하면 CE 가 낮아지기 때문이다.
- `strict_accuracy = 0.0000`, 같은 step live 는 **0.6899**

**즉 `accuracy` 와 `lm_loss` 로는 이 실패를 절대 탐지할 수 없다 — 개선처럼 보인다.**
task 지표(`strict_accuracy`)와 `queen_count_mean` 만이 잡는다.

step 67,168 은 양상이 다르다 — `queen_count_mean = 9.65` 로 개수는 맞는데 배치가 틀렸다.
붕괴가 단일 모드가 아니라 **mismatch 정도에 따라 여러 퇴화 형태**로 나타난다.

---

## 3. 배제한 가설

| 가설 | 판정 | 근거 |
|---|---|---|
| shadow 에 NaN/Inf 유입 | **배제** | `diag/z_h_norm`/`z_l_norm` NaN/Inf **0건**, 붕괴 구간에서도 30~39 로 조용. checkpoint 텐서에도 NaN 없음 |
| bias correction 오작동 | **배제** | 붕괴 구간 보정계수 `1/(1-mu^t)` = step 67,168 에서 **1.001211**, 83,960 에서 **1.000226**. 무시 가능. (단 §5 참조 — 초반에는 2.92 로 크다) |
| EMA checkpoint 저장 시점 불일치 | **배제** | `pretrain.py:696` 이 eval 에 쓴 `train_state_eval` 을 그대로 저장. eval 값과 checkpoint 가 동일 객체 |
| `dual_eval` RNG snapshot/restore 간섭 | **배제** | `pretrain.py:673-685` 가 raw pass 전후로 CPU/CUDA RNG 를 저장·복원. EMA pass 는 raw pass **이전**에 끝나므로 영향 불가. 또한 EMA pass 는 `train_state_eval` (deepcopy) 를 쓰고 raw 는 `train_state` 를 써서 객체가 분리돼 있다 |
| FPRM/pre-norm 고유 문제 | **배제** | concat arm(`catnd_s0`)에서도 발생. `alpha_*_param` 은 `requires_grad=False` 라 EMA 와 무관 |

---

## 4. 수정안 (미적용 — 지시 대기)

### 권장: `EMAHelper` 가 학습되는 buffer 도 추적하게 한다

`models/ema.py` 의 `register()`/`update()`/`ema()`/`ema_bias_corrected()` 가
`named_parameters()` 만 보는 것을 고쳐, **명시적으로 지정된 buffer** 도 포함시킨다.
`named_buffers()` 를 무조건 전부 넣으면 `mask`·`H_init`·`local_weights`(persistent=False)
까지 들어가므로 **화이트리스트**로 `puzzle_emb.weights` 만 추가하는 쪽이 안전하다.

★ **이 수정은 EMA 곡선을 바꾼다.** 기존 EMA 기반 수치는 전부 축이 달라지므로,
적용 시점과 재측정 범위를 사수가 정해야 한다.

### 채택된 조치 — 헤드라인은 EMA 유지 (2026-08-04 사수 확정)

**초판에서 "헤드라인을 live 로 전환" 을 제안했으나 채택되지 않았다. 규약은 EMA 다.**

논문 전수 확인 결과 EMA 언급은 두 곳뿐이다 — **B.2** ("For stability, we apply EMA with
decay 0.9999, following Brock et al. and Song and Ermon"), **D.4** (무조건부 Sudoku,
decay 0.999). **평가 시 어느 가중치를 쓰는지는 명시가 없고**, "checkpoint" 는 z_0 저장
대목에 한 번 나올 뿐 선택 기준이 없다. 그럼에도 세 근거가 같은 방향을 가리킨다:

| 근거 | 내용 |
|---|---|
| (a) | 인용된 두 논문(Brock et al. / Song & Ermon)이 **평가에 EMA 를 쓴다** |
| (b) | **TRM 원본이 EMA 사본만 평가·저장한다** — `.live` 개념이 없다 (§10 원문 인용) |
| (c) | 안정화 전용이라면 그 가중치를 **저장할 이유가 없다** |

**확정 규약:**

1. **헤드라인 = 최종 step × EMA.** 정점 병기.
2. **EMA 가 다수클래스 퇴화를 보이는 run 은 그 사실을 표시하고 live 를 병기한다.**
   탐지는 `queen_count_mean`(N-Queens, n 에서 크게 벗어나면) 또는
   `num_colors_used_mean`(GC). **`accuracy`/`lm_loss` 로는 안 잡힌다** (§2).
3. **`dual_eval` 과 `.live` 는 유지한다.** ★ **논문 미언급, 우리 추가분, 진단 목적**이다.
   이것이 없었으면 EMA 퇴화를 진단하지 못했고 "학습 붕괴" 로 오독했을 것이다 (§10).

### 최소 조치 (즉시 가능): 탐지 가드

`queen_count_mean` 이 정답 개수(10)에서 크게 벗어나면 경고를 찍는다.
`accuracy`/`lm_loss` 가 못 잡는다는 것이 §2 의 요지다.

---

## 5. 부수 발견 — 초기 구간 bias correction 이 크다

`mu=0.9999` 에서 보정계수 `1/(1-mu^t)`:

| step | `mu^t` | 보정계수 |
|---|---|---|
| 4,198 | 6.5716e-01 | **2.9169** |
| 58,772 | 2.8018e-03 | 1.0028 |
| 138,534 | 9.6215e-07 | 1.0000 |

첫 eval(4,198) 에서 보정계수가 **2.92** 다. `ema.py` 주석이 경고한 그대로이며
보정 자체는 올바르게 구현돼 있으나, **`w0` 스냅샷이 파라미터에만 있고 `puzzle_emb`
에는 없어서** 초기 구간에서는 원인 ① 의 영향이 더 크다. 붕괴는 중반(67k~84k)에
나왔으므로 이 항이 붕괴의 원인은 아니다(§3).

---

## 6. 사수 지시 `[1]` — `models/ema.py` 는 TRM 원본이 아니다 (전제 정정)

`git log --follow models/ema.py`:

```
4c847f1 GRAM reproduction
8120f2b upload            <- TRM 원본 유입 시점
```

**`8120f2b`(원본) → 현재 diff 가 존재한다.** 우리가 추가한 것:

| 추가분 | 내용 |
|---|---|
| `self.w0 = {}` | init 스냅샷 (bias correction 용) |
| `self.num_updates = 0` / `+= 1` | 카운터 |
| `_debiased()`, `ema_bias_corrected()` | bias correction 신규 |
| `ema_copy(module, step=None)` | `step` 인자 추가 (None = 원본 동작) |

### 그러나 결함 지점은 원본 그대로다

`register()` / `update()` / `ema()` 의 **선택 로직**(`named_parameters()` 순회 +
`param.requires_grad` 필터)은 원본과 **바이트 단위로 동일**하다. 우리 수정은
`w0[name] = ...`, `num_updates += 1` **두 줄을 더한 것뿐**이고 어떤 텐서가
선택되는지는 전혀 바꾸지 않았다.

```
$ diff <(git show 8120f2b:models/ema.py | sed -n '/def register/,/^$/p') \
       <(sed -n '/def register/,/^$/p' models/ema.py)
6a7 >                 self.w0[name] = param.data.clone()      # 추가만
$ diff ... update ...
3a4 >         self.num_updates += 1                            # 추가만
```

**결론: 사수 판단의 결론부는 유지된다.** `puzzle_emb` 배제는 **TRM 상속**이고,
`ema.py` × `sparse_embedding.py` 조합의 문제다. 다만 "원본 그대로" 라는 **전제는
사실과 다르므로** 정정한다 — 우리가 얹은 bias correction 은 §3 에서 정량적으로
배제됐다(붕괴 구간 보정계수 1.0002~1.0012).

### 부수: `num_updates` 는 死코드다

`models/ema.py:28,41` 에서 증가시키지만 **어디서도 읽지 않는다.** `_debiased()` 는
호출자가 넘긴 `step`(=`train_state.step`)을 쓴다. 현재는 `train_state.step += 1`
(`pretrain.py:300`)과 `ema_helper.update()`(`pretrain.py:634`)가 둘 다 batch 당
1회라 값이 일치하므로 무해하다. **다만 둘이 어긋나는 변경(예: gradient
accumulation, resume)이 들어오면 bias correction 이 조용히 틀어진다.**

---

## 7. 사수 지시 `[2]` — N-Queens 에서 puzzle_emb 는 실제로 쓰인다

### 7.1 forward 경로 (`gram.py`)

```
:218  puzzle_embedding = self.puzzle_emb(puzzle_identifiers)      # (B, 512)
:220  pad_count = 16*512 - 512 = 7680
:222  puzzle_embedding = F.pad(puzzle_embedding, (0, 7680))       # (B, 8192)
:224  embedding = torch.cat((puzzle_embedding.view(-1,16,512), embedding), dim=-2)
:231  return self.embed_scale * embedding                          # embed_scale = sqrt(512)
...
:362  output = self.lm_head(z_H)[:, self.puzzle_emb_len:]          # 출력에서 앞 16 토큰 제거
:364  q_logits = self.q_head(z_H[:, 0].detach())                   # ★ position 0 = puzzle_emb 자리
```

**더해지는(add) 게 아니라 앞에 이어붙는다(concat).** L = 100 + 16 = 116.

### 7.2 ★ 실제로는 16 토큰 중 1개만 내용이 있다

`puzzle_emb_ndim = 512` = `hidden_size` 인데 `puzzle_emb_len = 16` 이라
`pad_count = 16×512 − 512 = 7680` 만큼 **0 으로 채운 뒤** (16,512) 로 reshape 한다.

| 토큰 | 노름 |
|---|---|
| t0 | **0.2331** |
| t1 ~ t15 | **0.0000** (정확히 0) |

즉 **position 0 하나만 내용이 있고, position 1~15 는 정확히 영벡터**다.
116개 위치 중 15개(13%)가 상시 0 인 채로 attention 에 참여한다.

### 7.3 크기 비교 (`embed_scale = √512 = 22.627` 적용 후)

| | raw 노름 | ×embed_scale |
|---|---|---|
| `puzzle_emb` t0 | 0.2331 | **5.27** |
| `embed_tokens` 행 평균 (vocab=3) | 0.3932 | **8.90** |

내용 토큰의 **약 59%** 크기다 — 무시할 수 없지만 "전체를 지배" 하지도 않는다.

### 7.4 ★ `q_head` 가 바로 그 자리를 읽는다

`gram.py:364` — `q_logits = self.q_head(z_H[:, 0].detach())`.
**halting 판단이 puzzle_emb 위치의 최종 상태를 직접 입력으로 받는다.**

그리고 `q_head.weight` 는 **전 텐서 중 EMA/live 상대차가 가장 크다 (0.5109)**.
붕괴 구간의 `q_halt` 거동도 이와 정합한다:

| step 83,960 | `q_halt_accuracy` | `q_halt_loss` |
|---|---|---|
| EMA | **1.0000** | **0.1021** |
| live | 0.4456 | 1.7096 |

**EMA 되지 않는 입력(position 0) + 가장 크게 평균된 소비자(`q_head`)** 라는 조합이다.
`all/steps` 는 EMA·live 모두 16.0 으로 고정이라(`halt_max_steps=16`) halting 이 조기
종료를 일으킨 것은 아니다 — q_head 는 손상의 **지표**이지 붕괴의 직접 경로는 아니다.

### 7.5 움직임 (사수 확인 요청분)

`.live` 연속 checkpoint 간 상대변화 — eval 간격(4,198 step)당 **31~35%** (§1 표).
EMA 유효 창 10,000 step 에서는 75% 이상 바뀐다.

### 7.6 논문 B.2 대비 정합성

사수 지적대로 논문 B.2 는 puzzle embedding 을 **ARC 전용**으로 명시한다.
우리는 N-Queens 에서 `puzzle_emb_len = 16`, `num_puzzle_identifiers = 1` 로 켜 둔 상태다
— **per-example 정보를 전혀 싣지 않는 전역 공유 prefix 1토큰 + 영벡터 15토큰**이다.

`puzzle_emb_len = 0` 으로 끄면 `L = 100` 이 되어 **`kl_reduction = sum` 눈금이 바뀌므로
β 재환산이 필요하다** (현재 β = 0.045 는 L = 116 전제). **지금 바꾸면 안 된다** —
진행 중인 `rs_s0`/`rs_s2` 및 29번 `s1` 과 축이 갈린다. 논문 정합성 검토 대상으로만 기록한다.

---

## 8. 아직 증명되지 않은 것 (정직한 한계)

§1·§7 은 **강한 정황**이지 인과 증명이 아니다. 확정된 사실은:

- `puzzle_emb` 가 학습되면서 EMA 에서 배제된다 (checkpoint 상대차 0.000000) — **확정**
- 붕괴 시 다수클래스 퇴화가 일어난다 (`queen_count_mean` 3.47) — **확정**
- bias correction / NaN / RNG / 저장시점은 원인이 아니다 — **확정 (배제)**
- **"puzzle_emb 배제가 붕괴의 원인이다" 는 미증명**

### 결정적 실험 (GPU 여유 시 즉시 가능)

붕괴 step 의 EMA/live checkpoint 쌍이 남아 있다:
`step_75564`, `step_79762`, `step_83960` (각각 `.live` 동반).

EMA checkpoint 의 `puzzle_emb.weights` 를 **더 이른 `.live` 의 값(= 지연된, 평균에
가까운 값)으로 교체**해 재평가한다. `strict_accuracy` 가 0.0000 에서 회복되면
인과가 확정된다. 1지점 eval 이면 되므로 GPU 1장·수분이면 끝난다.
현재 GPU 0,2,3,4 학습 / 5,6,7 harness / 1 타 작업으로 전부 사용 중이라 대기 중이다.

---

## 10. TRM 원본 checkpoint 저장 로직 (원문 인용) — `.live` 가 없을 때의 동작 확정

`git show 8120f2b:pretrain.py` 기준.

### 10.1 `save_train_state` — 저장 대상과 파일명 규칙

```python
# :235-241
def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))
```

- 저장 대상은 **모델 state_dict 하나뿐** (optimizer / EMA shadow / step 미저장)
- 파일명 **`step_{step}`** — **접미사 인자 자체가 없다** (`(config, train_state)` 2인자)

### 10.2 `ema=True` 일 때 `evaluate()` 에 넘기는 model = **EMA 사본**

```python
# :620-633
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config,
                train_state_eval,
                ...
```

raw(live) 가중치는 **평가되지 않는다. eval 이 단 1회뿐이다.**

### 10.3 raw 가중치 저장 경로 — **존재하지 않는다**

```python
# :637-645
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)      # train_state_eval = EMA 사본

            if config.ema:
                del train_state_eval
```

- `save_train_state` 호출부는 **전 파일에서 이 한 곳뿐** (정의 :235 / 호출 :642)
- 넘기는 것은 **EMA 사본**. `train_state`(raw) 저장 경로 없음
- 원본 `torch.save` 는 2곳뿐: :241(모델) + :431(`eval_save_outputs` 예측 덤프)
- 문자열 `.live` / `suffix` / `raw` 는 원본에 **0회 등장**

→ **원본에서 `ema=True` 로 돌리면 raw 가중치는 학습 종료와 동시에 영구 소실된다.**

### 10.4 `checkpoint_every_eval` — 원본에도 있다

```python
# :77
    checkpoint_every_eval: bool = False
# :641
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
```

기본 `False` = **마지막 iteration 에서만 저장.** 우리 `cfg_pretrain.yaml` 은 `True` 라
매 eval 저장하며, 덕분에 붕괴 step 의 checkpoint(`step_75564/79762/83960`)가 남아
§8 결정적 실험이 가능하다.

### 10.5 확정 — dual eval / `.live` 가 없었다면

| | 원본 동작 |
|---|---|
| eval 횟수 | **1회** (EMA 사본만) |
| stdout | `SWITCH TO EMA` / `EVALUATE` / `SAVE CHECKPOINT` 뿐 — **eval 수치는 wandb 에만** |
| 디스크 | `step_N` (EMA) 1개 |
| raw 가중치 | **저장·평가 모두 없음** |

**이번 EMA 퇴화(step 83,960: EMA 0.0000 / live 0.6899)는 원본 코드로는 관측 자체가
불가능했다.** `strict_accuracy = 0.0000` 만 보고 "학습이 붕괴했다" 로 오독했을 것이다.
§4 의 헤드라인 규약 근거 (b) 를 원문으로 확증한다.

---

## 11. 재현 방법

```bash
source ~/venv-gram/bin/activate
CK=checkpoints/Nqueens-10-ACT-torch/nq10_fp_a3p_rs_s0
python - <<'PY'
import torch, os
CK=os.environ.get("CK","checkpoints/Nqueens-10-ACT-torch/nq10_fp_a3p_rs_s0")
e=torch.load(f"{CK}/step_138534", map_location="cpu", weights_only=False)
l=torch.load(f"{CK}/step_138534.live", map_location="cpu", weights_only=False)
k="_orig_mod.model.inner.puzzle_emb.weights"
print("puzzle_emb EMA==live ?", torch.equal(e[k], l[k]))   # True 면 EMA 미적용
PY
```
